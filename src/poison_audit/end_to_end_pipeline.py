"""Component 1 - the whole poison audit, end to end, on one poisoned training set:

1. model features: the scenario manifest's `model_features` - for script 04 (`poison_rate_sweep.py`), the mRMR top-K
   chosen on the poisoned training set itself;
2. a clean model and the backdoored model (XGBoost, `backdoor_eval.model_params`) -> CDA / macro-F1 / ASR;
3. the audited (backdoored) model labels every training row;
4. BENIGN-class TreeSHAP of the audited model -> StandardScaler -> PCA(95%) -> HDBSCAN fitted on every row
   (`TwoStageAuditPipeline` stage 1, no noise cascade); with `representation="raw"` (ablation, script 12) the same
   HDBSCAN runs on the raw values of the model features instead and no TreeSHAP is computed;
5. layer A: kNN label consistency between clusters, at one or more (k, eta) settings - each its own short list and
   its own trusted rows (and the count of poisoned rows that slipped into them, the one ground-truth diagnostic here);
6. layer B and the noise rule (`PoisonRemovalBuilder`): trusted rows without noise, clean XGBoost, exemplars, record votes;
7. one removal mask per layer-A setting x variant (layer-B rule x granularity x noise threshold);
8. retrain on what each variant keeps -> CDA / macro-F1 / ASR again, next to three references: nothing removed (the
   audited model itself), every poisoned row removed (`true_poison`), and LABOR with ground-truth cluster labels
   (`labor_oracle`, the ceiling of this clustering under LABOR's rule).

Every expensive artifact (both models, the SHAP table, the clustering, the removal masks, each retrain) is written
under the scenario directory and reused when present, so a job cut off by its time limit resumes where it stopped.
For the cost table, the result records each step's wall-clock time with whether the step was computed in this run or
read from that cache (a cached step's time is not its cost), and - when a sampling interval is given - the peak memory
of the run (this process and its children).
Ground truth (the poisoned ids) is read only to score removals and to build the `true_poison`/`labor_oracle` references.
"""

from __future__ import annotations

import gc
import hashlib
import json
from contextlib import nullcontext
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from src.core.logger import get_logger
from src.poison_audit.audit_sources import AuditSourceCatalog
from src.poison_audit.backdoor_metrics import BackdoorMetricsCalculator
from src.poison_audit.cluster_label_remover import ATTACK_LIKE, ClusterLabelPoisonRemover
from src.poison_audit.cluster_neighbor_labeler import (
    KIND_POISON, ClusterProfileBuilder, NeighborLabelConsistency, NeighborLabelScorer, TrainingMatrixReader,
)
from src.poison_audit.clustering_arm_runner import ClusteringArmRunner
from src.poison_audit.clustering_evaluator import ClusteringEvaluator
from src.poison_audit.labor_backdoor_evaluator import LaborBackdoorEvaluator
from src.poison_audit.poison_removal_system import (
    DEFAULT_BENIGN_REFERENCE, LAYER_B_RULES, BenignReference, PoisonRemovalBuilder, RemovalScorer, rows_to_score,
)
from src.poison_audit.resource_monitor import PeakMemorySampler
from src.poison_audit.two_stage_audit_pipeline import TwoStageAuditPipeline

logger = get_logger(__name__)

RESULT_FILE = "end_to_end_result.json"
BACKDOOR_FILE = "backdoor_metrics.json"
REMOVAL_MASKS_FILE = "removal_masks.npz"
REMOVAL_FILE = "removal.json"
LAYER_B_CLUSTERS_FILE = "layer_b_clusters_{}.csv"
RETRAIN_DIR = "retrain"
STAGE_DIR = "stage1"
NO_REMOVAL = "no_removal"
TRUE_POISON = "true_poison"
LABOR_ORACLE = "labor_oracle"
REFERENCES = (TRUE_POISON, LABOR_ORACLE)
TREESHAP_REPRESENTATION = "treeshap"
RAW_REPRESENTATION = "raw"
REPRESENTATIONS = (TREESHAP_REPRESENTATION, RAW_REPRESENTATION)
RAW_ARM = "raw"
PIPELINE_RESULT_FILE = "pipeline_result.json"


@dataclass(frozen=True)
class RemovalVariant:
    """One way the system removes rows.

    Attributes:
        rule: Layer-B record rule (`and` / `sequential`).
        granularity: Removal granularity in the short list (`cluster` / `record` / `hybrid`).
        noise_threshold: P(attack) from which a benign noise row is removed; None = no noise rule.
    """

    rule: str
    granularity: str
    noise_threshold: Optional[float]

    @property
    def name(self) -> str:
        """Stable name, also the retrain cache key.

        Returns:
            E.g. `and_hybrid_noise0.99` or `and_cluster_nonoise`.
        """
        noise = "nonoise" if self.noise_threshold is None else f"noise{self.noise_threshold:g}"
        return f"{self.rule}_{self.granularity}_{noise}"


@dataclass(frozen=True)
class LayerASetting:
    """One (k, eta) point of layer A, with the benign rows layer B trusts after it.

    Attributes:
        k: Neighbourhood size.
        eta: Agreement threshold.
        reference: Benign reference policy of layer B; the default policy adds nothing to the name.
    """

    k: int
    eta: float
    reference: BenignReference = DEFAULT_BENIGN_REFERENCE

    @property
    def name(self) -> str:
        """Stable name, prefix of its removal variants.

        Returns:
            E.g. `k15_eta0.8`, or `k25_eta0.5_kept` with a named reference policy.
        """
        suffix = f"_{self.reference.name}" if self.reference.name else ""
        return f"k{self.k}_eta{self.eta:g}{suffix}"


class EndToEndPoisonAudit:
    """Runs the full audit on one scenario and retrains after every removal variant.

    Args:
        config: Project config (CIC-IDS2017's defaults for id/label/target/features when a manifest lacks them).
        model_params: XGBoost kwargs of the clean, backdoored and retrained models.
        clustering_params: `clustering_hdbscan` block; `min_cluster_size` comes from the scenario, `fit_sample_size`
            is forced to None (fit on every row).
        shap_batch_size: Rows per TreeSHAP call.
        purity_threshold_pct: Poison share above which a cluster counts as poison-pure (report only).
        layer_a_settings: Layer-A points to run, each with its own short list, trusted rows, layer B and removals.
        distance_block_size: Clusters per distance block of layer A (memory only).
        read_chunk_size: Rows per chunk when reading the training matrix (memory only).
        feature_set: Feature set of layers A and B (`all_raw` / `selected`).
        builder: Layer B and noise rule.
        variants: Removal variants to score and retrain.
        references: Reference removals to retrain (`true_poison`, `labor_oracle`).
        random_state: Seed of every model.
        memory_sample_interval_seconds: Sampling period of the run's peak memory; None records none.
        representation: What stage 1 clusters: the audited model's TreeSHAP attributions (`treeshap`, the method) or
            the raw values of the model features (`raw`, the representation ablation).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model_params: Dict[str, Any],
        clustering_params: Dict[str, Any],
        shap_batch_size: int,
        purity_threshold_pct: float,
        layer_a_settings: List[LayerASetting],
        distance_block_size: int,
        read_chunk_size: int,
        feature_set: str,
        builder: PoisonRemovalBuilder,
        variants: List[RemovalVariant],
        references: List[str],
        random_state: int,
        memory_sample_interval_seconds: Optional[float] = None,
        representation: str = TREESHAP_REPRESENTATION,
    ) -> None:
        unknown = [name for name in references if name not in REFERENCES]
        if unknown:
            raise ValueError(f"Invalid reference: {unknown} (one of {REFERENCES}).")
        if representation not in REPRESENTATIONS:
            raise ValueError(f"Invalid clustering representation: {representation!r} (one of {REPRESENTATIONS}).")
        self._representation = representation
        self._config = config
        self._model_params = model_params
        self._clustering_params = clustering_params
        self._shap_batch_size = shap_batch_size
        self._purity_threshold_pct = purity_threshold_pct
        if not layer_a_settings:
            raise ValueError("At least one layer-A setting is needed.")
        self._layer_a = layer_a_settings
        self._distance_block_size = distance_block_size
        self._read_chunk_size = read_chunk_size
        self._feature_set = feature_set
        self._builder = builder
        self._variants = variants
        self._references = references
        self._random_state = random_state
        self._memory_interval = memory_sample_interval_seconds

    def run(self, scenario: Dict[str, Any], output_dir: Path, clean_model_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Audit one scenario and write `end_to_end_result.json`.

        Args:
            scenario: One entry of `end_to_end.scenarios` (`name`, the five input paths, `min_cluster_size`).
            output_dir: Scenario directory (also the resume cache).
            clean_model_dir: Where the clean model is cached; scenarios sharing it and the same feature list share one
                model. Defaults to the parent of `output_dir`.

        Returns:
            The result written to `end_to_end_result.json`.
        """
        sampler = PeakMemorySampler(self._memory_interval) if self._memory_interval else None
        with sampler if sampler is not None else nullcontext():
            result = self._audit(scenario, output_dir, clean_model_dir)
        if sampler is not None:
            # Peak RAM of the whole run; it is the pipeline's cost only when every step was computed (`all_steps_computed`).
            result["peak_rss_gib"] = sampler.peak_rss_bytes / 2 ** 30
        (output_dir / RESULT_FILE).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("[%s] Done after %.1f min.", result["scenario"], result["seconds"] / 60)
        return result

    def _audit(self, scenario: Dict[str, Any], output_dir: Path, clean_model_dir: Optional[Path]) -> Dict[str, Any]:
        """The audit itself (see `run`), timing each step.

        Args:
            scenario: The scenario.
            output_dir: Scenario directory.
            clean_model_dir: Where the clean model is cached.

        Returns:
            The result, without the peak memory.
        """
        started = time.time()
        timings: Dict[str, Dict[str, Any]] = {}
        name = str(scenario["name"])
        output_dir.mkdir(parents=True, exist_ok=True)
        settings = AuditSourceCatalog.settings(
            {"manifest": scenario["manifest"], "poisoned_train": scenario["poisoned_train"]}, self._config, self._feature_set,
        )
        manifest = json.loads(Path(scenario["manifest"]).read_text(encoding="utf-8"))
        trigger_spec = {str(feature): float(value) for feature, value in manifest["trigger_spec"].items()}
        id_column, label_column, target = settings["id_column"], settings["label_column"], settings["target_label"]
        model_features = list(settings["model_features"])
        logger.info("[%s] %d model features: %s", name, len(model_features), model_features)

        model_columns = [id_column, label_column] + model_features
        clean_train = self._read(scenario["clean_train"], model_columns)
        clean_test = self._read(scenario["clean_test"], list(dict.fromkeys(model_columns + list(trigger_spec))))
        poisoned_train = self._read(scenario["poisoned_train"], model_columns)
        trigger_test = self._read(scenario["trigger_test"], model_columns)
        poisoned = np.isin(poisoned_train[id_column].to_numpy(), settings["poisoned_ids"])
        injected = ~self.already_triggered(trigger_test, clean_test, id_column, trigger_spec)

        evaluator = LaborBackdoorEvaluator(model_features, label_column, self._model_params, self._random_state)
        evaluator.fit_label_encoder(clean_train[label_column])
        # The file name carries a digest of the feature list: two scenarios share a clean model only if they select the same features.
        digest = hashlib.sha1("\n".join(model_features).encode("utf-8")).hexdigest()[:10]
        clean_path = (clean_model_dir or output_dir.parent) / f"clean_model_{len(model_features)}feat_{digest}.pkl"
        step, computed = time.time(), not clean_path.exists()
        clean_model = self._load_or_train(evaluator, clean_train, clean_path, f"clean model [{name}]")
        timings["clean_model"] = {"seconds": time.time() - step, "computed": computed}
        clean_report = evaluator.compute_class_report(clean_model, clean_test)
        clean_labels = clean_train[label_column]
        del clean_train
        gc.collect()
        audited_path = output_dir / f"dirty_model_{len(model_features)}feat.pkl"
        step, computed = time.time(), not audited_path.exists()
        audited = self._load_or_train(evaluator, poisoned_train, audited_path, f"model backdoor [{name}]")
        timings["audited_model"] = {"seconds": time.time() - step, "computed": computed}
        backdoor = BackdoorMetricsCalculator(evaluator, target, id_column).measure(
            name, audited, clean_model, clean_report, clean_test, trigger_test, poisoned_train,
            np.asarray(settings["poisoned_ids"]), ~injected,
        )
        (output_dir / BACKDOOR_FILE).write_text(json.dumps(backdoor, indent=2, ensure_ascii=False), encoding="utf-8")

        step, computed = time.time(), not (output_dir / STAGE_DIR / "arm_metrics.json").exists()
        clustering_params = {**self._clustering_params, "min_cluster_size": int(scenario["min_cluster_size"]),
                             "fit_sample_size": None}
        poisoned_ids = [int(value) for value in settings["poisoned_ids"]]
        predicted = evaluator.predict_labels(audited, poisoned_train)
        if self._representation == RAW_REPRESENTATION:
            clustering = self._cluster_raw(name, poisoned_train, model_features, settings, predicted, poisoned_ids,
                                           clustering_params, output_dir)
        else:
            clustering = TwoStageAuditPipeline(
                selected_features=model_features, model_feature_columns=model_features, label_column=label_column,
                id_column=id_column, target_label=target, model_params=self._model_params,
                clustering_params=clustering_params, shap_batch_size=self._shap_batch_size,
                purity_threshold_pct=self._purity_threshold_pct, random_state=self._random_state, run_noise_cascade=False,
            ).run(name, poisoned_train, clean_labels, clean_test, trigger_test, poisoned_ids, output_dir)
        cluster_seconds = json.loads((output_dir / STAGE_DIR / "arm_metrics.json").read_text(encoding="utf-8"))["cluster_seconds"]
        # TreeSHAP + scaling/PCA + HDBSCAN (the raw representation has no TreeSHAP); the clustering alone is `hdbscan_seconds`.
        timings[f"{self._representation}_hdbscan"] = {"seconds": time.time() - step, "computed": computed,
                                                      "hdbscan_seconds": cluster_seconds}

        step = time.time()
        removal = self._load_or_build_removals(name, output_dir, Path(scenario["poisoned_train"]), settings,
                                               poisoned_train[id_column].to_numpy(), predicted, poisoned)
        timings["layers_a_b"] = {"seconds": time.time() - step, "computed": not removal.get("from_cache", False)}
        results = [{"name": NO_REMOVAL, "kind": "reference", "n_removed": 0, "n_removed_correct": 0, "n_removed_wrong": 0,
                    "recall": 0.0, "wrong_share": 0.0, **self.model_metrics(evaluator, audited, clean_test, trigger_test,
                                                                             injected, target)}]
        retrains = []
        for item in removal["removals"]:
            mask = removal["masks"][item["name"]]
            cached = (output_dir / RETRAIN_DIR / f"{item['name']}.json").exists()
            metrics = self._retrain(item["name"], evaluator, poisoned_train, mask, clean_test, trigger_test, injected, target,
                                    output_dir / RETRAIN_DIR)
            retrains.append({"seconds": metrics["retrain_seconds"], "computed": not cached})
            results.append({**item, **metrics})
        # One entry per retraining; `retrain_seconds` is stored with the result, so it stays right when a retraining comes from the cache.
        timings["retrains"] = retrains
        result = {
            "scenario": name,
            "dataset": name.split("/")[0],
            "n_rows": int(len(poisoned_train)),
            "n_poisoned": int(poisoned.sum()),
            "n_model_features": len(model_features),
            "model_features": model_features,
            "trigger_features_in_model": sorted(feature for feature in trigger_spec if feature in model_features),
            "trigger_features": sorted(trigger_spec),
            "min_cluster_size": int(scenario["min_cluster_size"]),
            "representation": self._representation,
            "backdoor": {key: value for key, value in backdoor.items() if not key.endswith("per_class")},
            "clustering": {
                "n_clusters": clustering["stage1"]["n_clusters"], "n_noise": clustering["stage1"]["n_noise"],
                "poison_in_noise": clustering["stage1"]["placement"]["poison_in_noise"],
                "cluster_seconds": cluster_seconds,
            },
            "placement": removal["placement"],
            "layers": removal["layers"],
            "removals": results,
            "timings": timings,
            "all_steps_computed": all(step["computed"] for key, step in timings.items() if key != "retrains")
                                  and all(step["computed"] for step in timings["retrains"]),
            "seconds": time.time() - started,
        }
        return result

    def _cluster_raw(
        self, name: str, poisoned_train: pd.DataFrame, model_features: List[str], settings: Dict[str, Any],
        predicted: np.ndarray, poisoned_ids: List[int], clustering_params: Dict[str, Any], output_dir: Path,
    ) -> Dict[str, Any]:
        """Stage 1 on the raw values of the model features (representation ablation), or its cached copy.

        Same `ClusteringArmRunner` - same HDBSCAN block, scaler, PCA and evaluator - and the same stage files as the
        TreeSHAP stage, so layers A and B read it unchanged; only the clustered columns differ.

        Args:
            name: Scenario name.
            poisoned_train: Poisoned training rows (id, label, model features).
            model_features: Columns to cluster.
            settings: `AuditSourceCatalog.settings` of the scenario.
            predicted: Audited model's label per row (scores LABOR's rule with true cluster labels).
            poisoned_ids: Ground-truth poisoned ids (scoring only).
            clustering_params: HDBSCAN parameters of the scenario.
            output_dir: Scenario directory.

        Returns:
            `{"stage1": ...}` with the keys `_audit` reads from `TwoStageAuditPipeline.run`.
        """
        stage_dir = output_dir / STAGE_DIR
        if not (stage_dir / "arm_metrics.json").exists():
            ClusteringArmRunner(
                label_column=settings["label_column"], id_column=settings["id_column"], target_label=settings["target_label"],
                clustering_params=clustering_params, random_state=self._random_state,
            ).run(RAW_ARM, poisoned_train, model_features, predicted, poisoned_ids, stage_dir)
        else:
            logger.info("[%s] Reusing the existing raw-feature clustering: %s", name, stage_dir)
        metrics = json.loads((stage_dir / "arm_metrics.json").read_text(encoding="utf-8"))
        summary = pd.read_csv(stage_dir / "cluster_summary.csv")
        stage = {
            "n_clusters": metrics["n_clusters"], "n_noise": metrics["n_noise"], "removal": metrics["poison_detection"],
            "placement": ClusteringEvaluator(target_label=settings["target_label"]).poison_placement(
                summary, self._purity_threshold_pct),
        }
        (output_dir / PIPELINE_RESULT_FILE).write_text(json.dumps(
            {"scenario_name": name, "representation": RAW_REPRESENTATION, "stage1": stage}, indent=2, ensure_ascii=False,
        ), encoding="utf-8")
        return {"stage1": stage}

    def _load_or_build_removals(
        self, name: str, output_dir: Path, poisoned_train_path: Path, settings: Dict[str, Any], sample_ids: np.ndarray,
        predicted: np.ndarray, poisoned: np.ndarray,
    ) -> Dict[str, Any]:
        """Layers A and B at every layer-A setting, the noise rule and every removal mask - or their cached copy.

        Args:
            name: Scenario name.
            output_dir: Scenario directory.
            poisoned_train_path: The poisoned training CSV (read again with every raw feature for layers A and B).
            settings: `AuditSourceCatalog.settings` of the scenario.
            sample_ids: Id of every poisoned_train row, in file order.
            predicted: The audited model's label for every row.
            poisoned: Ground-truth poisoned flag per row.

        Returns:
            Dict `placement` (where the poisoned rows sit), `layers` (layer-A setting name -> `layer_a`, `layer_b`),
            `removals` (one scored entry per reference and per setting x variant) and `masks` (name -> boolean mask).

        Raises:
            ValueError: If the training matrix is not in poisoned_train's row order, or layer A has too few clusters.
        """
        masks_path, removal_path = output_dir / REMOVAL_MASKS_FILE, output_dir / REMOVAL_FILE
        wanted = list(self._references) + [f"{setting.name}_{variant.name}" for setting in self._layer_a for variant in self._variants]
        if masks_path.exists() and removal_path.exists():
            cached = json.loads(removal_path.read_text(encoding="utf-8"))
            if [item["name"] for item in cached["removals"]] == wanted:
                logger.info("[%s] Reusing the saved layers A/B and removal masks: %s", name, removal_path)
                with np.load(masks_path) as stored:
                    cached["masks"] = {key: stored[key] for key in stored.files}
                return {**cached, "from_cache": True}

        target = settings["target_label"]
        assignment = np.load(output_dir / STAGE_DIR / "cluster_assignment.npz")
        matrix = TrainingMatrixReader(self._read_chunk_size).read(
            poisoned_train_path, settings["id_column"], settings["label_column"], settings["feature_columns"],
            len(assignment["sample_id"]),
        )
        if not np.array_equal(matrix.sample_ids, sample_ids):
            raise ValueError("The layer-A/B matrix does not have the row order of poisoned_train.")
        row_clusters = AuditSourceCatalog.align_row_clusters(assignment, matrix.sample_ids)
        summary = pd.read_csv(output_dir / STAGE_DIR / "cluster_summary.csv")
        benign = matrix.rows_labelled(target)

        profiles = ClusterProfileBuilder().build(matrix, row_clusters, summary)
        labeler = NeighborLabelConsistency(max_k=max(setting.k for setting in self._layer_a), block_size=self._distance_block_size)
        labeler.fit(profiles.ranks)
        scorer = NeighborLabelScorer(target_label=target)
        scorer.fit(profiles, summary, row_clusters, predicted, poisoned)
        kinds = dict(zip(profiles.cluster_ids.astype(int), scorer.kinds))
        attack_labelled = profiles.cluster_ids[profiles.observed_attack]
        in_attack_labelled = np.isin(row_clusters, attack_labelled)
        placement = {
            "n_poisoned": int(poisoned.sum()),
            "in_noise": int((poisoned & (row_clusters < 0)).sum()),
            "in_attack_labelled_clusters": int((poisoned & in_attack_labelled).sum()),
            "in_benign_labelled_clusters": int((poisoned & (row_clusters >= 0) & ~in_attack_labelled).sum()),
        }
        audited_target = predicted == target
        rows_scored = rows_to_score([variant.granularity for variant in self._variants])

        masks: Dict[str, np.ndarray] = {}
        removals: List[Dict[str, Any]] = []
        empty = np.zeros(len(row_clusters), dtype=bool)
        if TRUE_POISON in self._references:
            masks[TRUE_POISON] = poisoned.copy()
        if LABOR_ORACLE in self._references:
            remover = ClusterLabelPoisonRemover(target_label=target)
            masks[LABOR_ORACLE] = remover.compute_removal_mask(row_clusters, remover.label_clusters(summary), predicted)
        for reference in self._references:
            removals.append({"name": reference, "kind": "reference", **RemovalScorer.score(
                {"removed": masks[reference], "attack_clusters": empty, "layer_b": empty, "noise": empty}, poisoned)})

        layers: Dict[str, Dict[str, Any]] = {}
        for setting in self._layer_a:
            if setting.k > labeler.neighbors.shape[1]:
                raise ValueError(f"Only {len(profiles.cluster_ids)} clusters, not enough for {setting.k} layer-A neighbours.")
            attack_like = NeighborLabelConsistency.relabel(
                profiles.observed_attack, labeler.attack_votes(profiles.observed_attack, setting.k), setting.k, setting.eta,
            )
            short_list = attack_like & ~profiles.observed_attack
            short_listed = profiles.cluster_ids[short_list]
            layer_a_score = scorer.score(attack_like)
            # Trusted benign rows = benign-labelled rows of the clusters outside the short list, no noise (as in TrustedTrainingSet).
            trusted_benign = benign & (row_clusters >= 0) & ~np.isin(row_clusters, short_listed)
            layer_a = {
                "k": setting.k, "eta": setting.eta, "n_clusters": int(len(profiles.cluster_ids)),
                "n_attack_labelled": int(profiles.observed_attack.sum()),
                "n_poison_clusters": layer_a_score["n_poison_clusters"],
                "n_hidden_poison_clusters": int(scorer.hidden_poison.sum()),
                "n_short_list": layer_a_score["n_short_list"],
                "n_short_list_poison": layer_a_score["n_short_list_poison"],
                "n_trusted_benign_rows": int(trusted_benign.sum()),
                "n_poisoned_in_trusted": int((trusted_benign & poisoned).sum()),
            }
            logger.info("[%s] Layer A %s: %s", name, setting.name, layer_a)
            outcome = self._builder.build(name, matrix, row_clusters, short_listed, attack_labelled, audited_target, target,
                                          rows_scored, setting.reference)
            noise = row_clusters < 0
            layer_b: Dict[str, Any] = {"reference": {
                "n_trusted_benign_rows": int(outcome.trusted_benign.sum()),
                "n_poisoned_in_trusted": int((outcome.trusted_benign & poisoned).sum()),
                "n_reference_benign_rows": int(outcome.reference_benign.sum()),
                "n_reference_noise_rows": int((outcome.reference_benign & noise).sum()),
                "n_poisoned_in_reference": int((outcome.reference_benign & poisoned).sum()),
                "reference_trained": bool(outcome.reference_trained),
            }}
            for rule in LAYER_B_RULES:
                flagged = [item.cluster for item in outcome.judgements if item.decisions[rule] == ATTACK_LIKE]
                layer_b[rule] = {
                    "n_flagged": len(flagged),
                    "n_poison_flagged": sum(kinds[cluster] == KIND_POISON for cluster in flagged),
                    "n_other_flagged": sum(kinds[cluster] != KIND_POISON for cluster in flagged),
                }
            layers[setting.name] = {"layer_a": layer_a, "layer_b": layer_b}
            pd.DataFrame([
                {"cluster": item.cluster, "truth_kind": kinds[item.cluster], "n_rows": size, "n_scored": len(votes.rows),
                 "n_poisoned": int(poisoned[row_clusters == item.cluster].sum()), "n_compared": item.n_compared,
                 "classifier_attack_share": item.classifier_attack_share, "attack_closer_share": item.attack_closer_share,
                 **{f"{rule}_votes": item.attack_votes[rule] for rule in LAYER_B_RULES},
                 **{f"{rule}_decision": item.decisions[rule] for rule in LAYER_B_RULES}}
                for item, votes, size in zip(outcome.judgements, outcome.record_votes, outcome.cluster_sizes)
            ]).to_csv(output_dir / LAYER_B_CLUSTERS_FILE.format(setting.name), index=False)
            for variant in self._variants:
                key = f"{setting.name}_{variant.name}"
                variant_masks = outcome.system.masks(outcome.judgements, variant.rule, variant.noise_threshold,
                                                     variant.granularity, outcome.record_votes)
                masks[key] = variant_masks["removed"]
                removals.append({"name": key, "kind": "system", "layer_a": setting.name, "rule": variant.rule,
                                 "granularity": variant.granularity, "noise_threshold": variant.noise_threshold,
                                 **RemovalScorer.score(variant_masks, poisoned)})
                logger.info("[%s] %s: removed %d (correct %d, wrong %d).", name, key, removals[-1]["n_removed"],
                            removals[-1]["n_removed_correct"], removals[-1]["n_removed_wrong"])
            del outcome
            gc.collect()
        np.savez_compressed(masks_path, **masks)
        cached = {"placement": placement, "layers": layers, "removals": removals}
        removal_path.write_text(json.dumps(cached, indent=2, ensure_ascii=False), encoding="utf-8")
        del matrix
        gc.collect()
        return {**cached, "masks": masks}

    def _retrain(
        self, name: str, evaluator: LaborBackdoorEvaluator, poisoned_train: pd.DataFrame, removed: np.ndarray,
        clean_test: pd.DataFrame, trigger_test: pd.DataFrame, injected: np.ndarray, target: str, cache_dir: Path,
    ) -> Dict[str, Any]:
        """Retrain on the rows a removal keeps and measure the model - or read the cached measurement.

        Args:
            name: Removal name (cache key).
            evaluator: Evaluator of the scenario (shared label encoder).
            poisoned_train: The poisoned training frame.
            removed: Rows the removal drops.
            clean_test: Untouched test set.
            trigger_test: Source-class test rows with the trigger injected.
            injected: Per trigger-test row, whether the trigger was injected (not already there).
            target: Target label.
            cache_dir: Directory of the cached measurements.

        Returns:
            CDA, macro-F1, ASR (all trigger-test rows and injected rows only), per-class recall and training seconds.
        """
        cache = cache_dir / f"{name}.json"
        if cache.exists():
            logger.info("Reusing the saved retraining result: %s", cache)
            return json.loads(cache.read_text(encoding="utf-8"))
        started = time.time()
        logger.info("Retraining after removal [%s]: keeping %d/%d rows...", name, int((~removed).sum()), len(removed))
        model = evaluator.train(poisoned_train.loc[~removed])
        metrics = {**self.model_metrics(evaluator, model, clean_test, trigger_test, injected, target),
                   "retrain_seconds": time.time() - started}
        logger.info("[%s] CDA %.4f%%, macro-F1 %.4f%%, ASR %.2f%% (triggered rows only %.2f%%).", name, 100 * metrics["cda"],
                    100 * metrics["macro_f1"], 100 * metrics["asr"], 100 * metrics["asr_injected_only"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        return metrics

    @staticmethod
    def model_metrics(
        evaluator: LaborBackdoorEvaluator, model: xgb.XGBClassifier, clean_test: pd.DataFrame, trigger_test: pd.DataFrame,
        injected: np.ndarray, target: str,
    ) -> Dict[str, Any]:
        """CDA, macro-F1, per-class recall and ASR of one model.

        Args:
            evaluator: Evaluator of the scenario.
            model: The model.
            clean_test: Untouched test set.
            trigger_test: Source-class test rows with the trigger injected.
            injected: Per trigger-test row, whether the trigger was injected.
            target: Target label.

        Returns:
            The metrics.
        """
        report = evaluator.compute_class_report(model, clean_test)
        hit = evaluator.predict_labels(model, trigger_test) == target
        return {
            "cda": report["accuracy"], "macro_f1": report["macro_f1"], "asr": float(hit.mean()),
            "asr_injected_only": float(hit[injected].mean()) if injected.any() else 0.0,
            "recall_per_class": report["recall_per_class"],
        }

    def _load_or_train(self, evaluator: LaborBackdoorEvaluator, frame: pd.DataFrame, path: Path, what: str) -> xgb.XGBClassifier:
        """Train a model, or reload it when an earlier run saved it.

        The file is written to a temporary name and then renamed, so two jobs sharing a dataset-level model never read
        a half-written file.

        Args:
            evaluator: Evaluator with its label encoder fit.
            frame: Training rows.
            path: Cache location.
            what: Description for the log.

        Returns:
            The fitted model.
        """
        if path.exists():
            logger.info("Reusing the saved %s: %s", what, path)
            with path.open("rb") as handle:
                return pickle.load(handle)
        logger.info("Training the %s on %d rows...", what, len(frame))
        started = time.time()
        model = evaluator.train(frame)
        logger.info("Trained the %s in %.1f s.", what, time.time() - started)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            pickle.dump(model, handle)
        os.replace(temporary, path)
        return model

    @staticmethod
    def _read(path: str, columns: List[str]) -> pd.DataFrame:
        """Read some columns of a CSV at full float precision.

        Args:
            path: CSV (.csv or .csv.gz).
            columns: Columns to keep.

        Returns:
            The frame, columns in file order.
        """
        logger.info("Reading %s (%d columns)...", path, len(columns))
        return pd.read_csv(path, usecols=columns, float_precision="round_trip")

    @staticmethod
    def already_triggered(
        trigger_test: pd.DataFrame, clean_test: pd.DataFrame, id_column: str, trigger_spec: Dict[str, float],
    ) -> np.ndarray:
        """Trigger-test rows that carried the whole trigger before injection (their original row in the clean test set).

        Args:
            trigger_test: Source-class test rows with the trigger injected.
            clean_test: Untouched test set.
            id_column: Row id column.
            trigger_spec: Feature -> trigger value.

        Returns:
            Boolean mask over `trigger_test`.

        Raises:
            ValueError: If a trigger-test row has no original row in the clean test set.
        """
        original = clean_test.set_index(id_column).reindex(trigger_test[id_column].to_numpy())
        if original[list(trigger_spec)].isna().all(axis=1).any():
            raise ValueError("Some trigger-test rows have no original row in the clean test set.")
        match = np.ones(len(trigger_test), dtype=bool)
        for feature, value in trigger_spec.items():
            match &= original[feature].to_numpy() == value
        return match


def build_variants(entries: List[Dict[str, Any]]) -> List[RemovalVariant]:
    """Removal variants from config.

    Args:
        entries: `end_to_end.variants`.

    Returns:
        The variants, in config order.
    """
    return [
        RemovalVariant(rule=str(entry["rule"]), granularity=str(entry["granularity"]),
                       noise_threshold=None if entry.get("noise_threshold") is None else float(entry["noise_threshold"]))
        for entry in entries
    ]


def build_layer_a_settings(
    entries: List[Dict[str, Any]], references: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[LayerASetting]:
    """Layer-A settings from config.

    Args:
        entries: `{k, eta}` items, optionally with `reference` naming one of `references`.
        references: Benign reference policies by name: `{attack_labelled_clusters, noise}`.

    Returns:
        The settings, in config order.

    Raises:
        KeyError: If an entry names an unknown reference policy.
    """
    settings = []
    for entry in entries:
        reference = DEFAULT_BENIGN_REFERENCE
        if entry.get("reference"):
            name = str(entry["reference"])
            policy = (references or {})[name]
            reference = BenignReference(name=name, attack_labelled_clusters=bool(policy["attack_labelled_clusters"]),
                                        noise=str(policy["noise"]))
        settings.append(LayerASetting(k=int(entry["k"]), eta=float(entry["eta"]), reference=reference))
    return settings


def scenario_dir(output_root: Path, name: str) -> Path:
    """Directory of one scenario under the run's output root.

    Args:
        output_root: `end_to_end.output_dir`.
        name: Scenario name (`dataset/scenario`).

    Returns:
        `output_root/dataset/scenario`.
    """
    return output_root.joinpath(*name.split("/"))


def read_results(output_root: Path, names: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Results of the scenarios that have finished.

    Args:
        output_root: `end_to_end.output_dir`.
        names: Scenario names, in report order.

    Returns:
        Tuple (finished results, names still missing).
    """
    finished, missing = [], []
    for name in names:
        path = scenario_dir(output_root, name) / RESULT_FILE
        if path.exists():
            finished.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            missing.append(name)
    return finished, missing
