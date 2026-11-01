"""Component 1 - poisoned-record removal baselines run on the poison-rate sweep's cells, each retrained after removal
like the system's own variants.

Detectors on the representation the system clusters, so the comparison isolates the detector: the target-class
TreeSHAP vector of the audited model, for the rows labelled with the target class (the poisoned rows carry that label):
- `spectral_signature` (Tran, Li & Madry, NeurIPS 2018): centre the representations, score each row by its squared
  projection on the top right singular vector, remove the round(budget_factor x n_poisoned) highest scores. The paper
  removes 1.5 x the number of poisoned rows, which it assumes known; the TRUE count is used here - information the
  system does not get, and every report says so. It is not always an advantage: when the poisoned rows rank high but
  not at the very top (Isolation Forest on TON-IoT), a budget this tight stops short of them;
- `isolation_forest` (Liu, Ting & Zhou, ICDM 2008): same representation and the same oracle budget, rows ranked by
  anomaly score (sklearn `-score_samples`);
- `isolation_forest_auto`: the rows sklearn's default decision (`contamination="auto"`) calls outliers - no oracle.

LABOR - `labor_kmeans` (Luo et al., Computer Communications 2025): k-means on every raw feature (StandardScaler, k = the
number of classes, LABOR's k-means parameters), each cluster labelled by its true composition (standing in for LABOR's
human reviewer, i.e. its best case), then LABOR's removal rule: rows of ATTACK_LIKE clusters the audited model predicts
as the target class. It differs from the sweep's `labor_oracle` reference only in the clustering (k-means on raw
features instead of HDBSCAN on TreeSHAP).

Ground truth enters the oracle budget, the cluster labels of `labor_kmeans` and the scoring - never the scores.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.core.logger import get_logger
from src.poison_audit.audit_sources import FEATURE_SET_ALL_RAW, AuditSourceCatalog
from src.poison_audit.cluster_label_remover import ClusterLabelPoisonRemover
from src.poison_audit.confident_learning import removal_scores
from src.poison_audit.end_to_end_pipeline import RESULT_FILE, EndToEndPoisonAudit
from src.poison_audit.labor_backdoor_evaluator import LaborBackdoorEvaluator
from src.poison_audit.labor_clusterer import LaborKMeansClusterer
from src.poison_audit.poison_rate_sweep import MANIFEST_FILE, POISONED_TRAIN_FILE, TRIGGER_TEST_FILE, SweepTrigger, rate_tag
from src.poison_audit.treeshap_contribution import TreeShapContributionExtractor

logger = get_logger(__name__)

SPECTRAL_SIGNATURE = "spectral_signature"
ISOLATION_FOREST = "isolation_forest"
ISOLATION_FOREST_AUTO = "isolation_forest_auto"
LABOR_KMEANS = "labor_kmeans"
BASELINES = (SPECTRAL_SIGNATURE, ISOLATION_FOREST, ISOLATION_FOREST_AUTO, LABOR_KMEANS)
SHAP_FILE = "shap_target_rows.npz"
KMEANS_FILE = "kmeans_assignment.npz"
MASKS_FILE = "baseline_masks.npz"
RETRAIN_DIR = "retrain"
BASELINE_RESULT_FILE = "baseline_result.json"


class SpectralSignatureScorer:
    """Spectral signature outlier score (Tran et al. 2018, Algorithm 1)."""

    @staticmethod
    def score(representations: np.ndarray) -> np.ndarray:
        """Squared projection of each centred representation on the top right singular vector.

        Args:
            representations: Rows x dimensions.

        Returns:
            One score per row; larger is more suspicious.
        """
        centred = representations - representations.mean(axis=0)
        _, _, right = np.linalg.svd(centred, full_matrices=False)
        return (centred @ right[0]) ** 2


class IsolationForestScorer:
    """Isolation Forest anomaly score and default outlier decision (Liu et al. 2008, sklearn).

    Args:
        n_estimators: Number of trees.
        max_samples: Sub-sample size per tree (`"auto"` = min(256, n)).
        random_state: Seed.
    """

    def __init__(self, n_estimators: int, max_samples: Any, random_state: int) -> None:
        self._params = {"n_estimators": n_estimators, "max_samples": max_samples, "random_state": random_state}

    def fit_score(self, representations: np.ndarray) -> Dict[str, np.ndarray]:
        """Fit on the rows and score them.

        Args:
            representations: Rows x dimensions.

        Returns:
            `score` (larger is more anomalous) and `outlier` (sklearn's default decision, `contamination="auto"`).
        """
        forest = IsolationForest(**self._params).fit(representations)
        return {"score": -forest.score_samples(representations), "outlier": forest.predict(representations) == -1}


def top_k_mask(scores: np.ndarray, rows: np.ndarray, k: int, n_rows: int) -> np.ndarray:
    """Mask over all rows that removes the k highest-scoring rows of a subset.

    Args:
        scores: Score per row of the subset.
        rows: Row index of each subset row in the full set.
        k: Rows to remove.
        n_rows: Rows in the full set.

    Returns:
        Boolean mask over the full set.
    """
    mask = np.zeros(n_rows, dtype=bool)
    mask[rows[np.argsort(-scores, kind="stable")[:max(0, min(k, len(rows)))]]] = True
    return mask


class BaselineDefenseSweep:
    """Runs the baselines on the sweep's cells, scores them and retrains after each removal.

    Args:
        config: Project config.
        sweep_root: Root of the sweep's cells for this seed.
        output_root: Root of this baseline's cells for this seed.
        model_params: XGBoost kwargs (`backdoor_eval.model_params`) of every retrained model.
        shap_batch_size: Rows per TreeSHAP call.
        budget_factor: Rows removed by the ranking baselines, as a multiple of the true poisoned count.
        isolation_forest: Isolation Forest scorer.
        kmeans_params: LABOR k-means parameters (`clustering` block: `init`, `max_iter`, `tol`).
        compare_with: Removal names of the sweep result scored next to the baselines.
        random_state: Seed of every model.
        drop_cache: Delete a cell's TreeSHAP cache once its result is written (~110 MB per CIC-IDS2017 cell; regenerable).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        sweep_root: Path,
        output_root: Path,
        model_params: Dict[str, Any],
        shap_batch_size: int,
        budget_factor: float,
        isolation_forest: IsolationForestScorer,
        kmeans_params: Dict[str, Any],
        compare_with: List[str],
        random_state: int,
        drop_cache: bool = False,
    ) -> None:
        self._config = config
        self._sweep_root = sweep_root
        self._output_root = output_root
        self._model_params = model_params
        self._shap_batch_size = shap_batch_size
        self._budget_factor = budget_factor
        self._forest = isolation_forest
        self._kmeans = kmeans_params
        self._compare_with = compare_with
        self._seed = random_state
        self._drop_cache = drop_cache

    def cell_paths(self, trigger: SweepTrigger, rate: float) -> Dict[str, Path]:
        """Sweep cell, its trigger-test file and this baseline's cell.

        Args:
            trigger: The trigger.
            rate: Poison rate.

        Returns:
            `sweep`, `trigger_test`, `output`.
        """
        relative = Path(trigger.dataset) / trigger.name
        return {"sweep": self._sweep_root / relative / rate_tag(rate),
                "trigger_test": self._sweep_root / relative / TRIGGER_TEST_FILE,
                "output": self._output_root / relative / rate_tag(rate)}

    def load(self, trigger: SweepTrigger, rate: float) -> Optional[Dict[str, Any]]:
        """Result of one cell if already computed.

        Args:
            trigger: The trigger.
            rate: Poison rate.

        Returns:
            The result, or None.
        """
        path = self.cell_paths(trigger, rate)["output"] / BASELINE_RESULT_FILE
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def run_cell(self, trigger: SweepTrigger, rate: float) -> Optional[Dict[str, Any]]:
        """Every baseline on one cell (reused when already computed).

        Args:
            trigger: The trigger.
            rate: Poison rate.

        Returns:
            The cell result, or None when the sweep has no result for this cell yet.
        """
        started = time.time()
        paths = self.cell_paths(trigger, rate)
        name = f"{trigger.dataset}/{trigger.name}/{rate_tag(rate)}"
        if not (paths["sweep"] / RESULT_FILE).exists():
            logger.warning("[%s] The sweep has no result for this cell yet, skipped.", name)
            return None
        cached = self.load(trigger, rate)
        if cached is not None:
            logger.info("[%s] Baseline result already there, skipped.", name)
            return cached
        out = paths["output"]
        out.mkdir(parents=True, exist_ok=True)
        settings = AuditSourceCatalog.settings(
            {"manifest": (paths["sweep"] / MANIFEST_FILE).as_posix(),
             "poisoned_train": (paths["sweep"] / POISONED_TRAIN_FILE).as_posix()}, self._config, FEATURE_SET_ALL_RAW,
        )
        id_column, label_column, target = settings["id_column"], settings["label_column"], settings["target_label"]
        model_features = list(settings["model_features"])
        frame = pd.read_csv(paths["sweep"] / POISONED_TRAIN_FILE, float_precision="round_trip")
        poisoned = np.isin(frame[id_column].to_numpy(), settings["poisoned_ids"])
        n_poisoned, n_rows = int(poisoned.sum()), len(frame)

        evaluator = LaborBackdoorEvaluator(model_features, label_column, self._model_params, self._seed)
        evaluator.fit_label_encoder(frame[label_column])
        with (paths["sweep"] / f"dirty_model_{len(model_features)}feat.pkl").open("rb") as handle:
            audited = pickle.load(handle)
        predicted = evaluator.predict_labels(audited, frame)
        target_rows = np.flatnonzero(frame[label_column].to_numpy() == target)

        representations = self._target_shap(out / SHAP_FILE, audited, evaluator, frame, target_rows, model_features, target)
        budget = int(round(self._budget_factor * n_poisoned))
        forest = self._forest.fit_score(representations)
        masks = {
            SPECTRAL_SIGNATURE: top_k_mask(SpectralSignatureScorer.score(representations), target_rows, budget, n_rows),
            ISOLATION_FOREST: top_k_mask(forest["score"], target_rows, budget, n_rows),
            ISOLATION_FOREST_AUTO: np.isin(np.arange(n_rows), target_rows[forest["outlier"]]),
            LABOR_KMEANS: self._labor_kmeans(out / KMEANS_FILE, frame, id_column, label_column, target, settings, predicted),
        }
        np.savez_compressed(out / MASKS_FILE, **masks)

        clean_test = pd.read_csv(trigger.clean_test, float_precision="round_trip",
                                 usecols=list(dict.fromkeys([id_column, label_column] + model_features + list(trigger.trigger_spec))))
        trigger_test = pd.read_csv(paths["trigger_test"], float_precision="round_trip",
                                   usecols=[id_column, label_column] + model_features)
        injected = ~EndToEndPoisonAudit.already_triggered(trigger_test, clean_test, id_column, trigger.trigger_spec)
        removals = []
        for baseline, mask in masks.items():
            metrics = self._retrain(baseline, evaluator, frame, mask, clean_test, trigger_test, injected, target, out / RETRAIN_DIR)
            removals.append({"name": baseline, "kind": "baseline",
                             **removal_scores(int((mask & poisoned).sum()), int((mask & ~poisoned).sum()), n_poisoned, n_rows),
                             **{key: metrics[key] for key in ("asr", "asr_injected_only", "cda", "macro_f1")}})
        sweep_result = json.loads((paths["sweep"] / RESULT_FILE).read_text(encoding="utf-8"))
        by_name = {item["name"]: item for item in sweep_result["removals"]}
        for removal in self._compare_with:
            item = by_name[removal]
            removals.append({"name": removal, "kind": "reference" if item.get("kind") == "reference" else "system",
                             **removal_scores(int(item["n_removed_correct"]), int(item["n_removed_wrong"]), n_poisoned, n_rows),
                             **{key: item[key] for key in ("asr", "asr_injected_only", "cda", "macro_f1")}})
        result = {
            "scenario": name, "dataset": trigger.dataset, "trigger": trigger.name, "rate": rate, "n_rows": n_rows,
            "n_poisoned": n_poisoned, "n_target_rows": int(len(target_rows)), "oracle_budget": budget,
            "clean_asr_injected_only": sweep_result["backdoor"]["clean_asr_injected_only"],
            "clean_cda": sweep_result["backdoor"]["clean_cda"],
            "audited_asr_injected_only": sweep_result["backdoor"]["backdoored_asr_injected_only"],
            "removals": removals, "seconds": time.time() - started,
        }
        temporary = out / f"{BASELINE_RESULT_FILE}.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, out / BASELINE_RESULT_FILE)
        if self._drop_cache:
            (out / SHAP_FILE).unlink(missing_ok=True)
        logger.info("[%s] Baselines done after %.1f min.", name, result["seconds"] / 60)
        return result

    def _target_shap(
        self, path: Path, model: Any, evaluator: LaborBackdoorEvaluator, frame: pd.DataFrame, rows: np.ndarray,
        model_features: List[str], target: str,
    ) -> np.ndarray:
        """Target-class TreeSHAP of the target-labelled rows, from cache when present.

        Args:
            path: Cache file.
            model: The audited model.
            evaluator: Evaluator holding the label encoder the model was trained with.
            frame: Training rows.
            rows: Target-labelled row positions.
            model_features: The model's features, in its order.
            target: Target label.

        Returns:
            Rows x model features.
        """
        if not path.exists():
            logger.info("Target-class TreeSHAP for %d rows with the target label...", len(rows))
            target_idx = int(evaluator.label_encoder.transform([target])[0])
            contributions = TreeShapContributionExtractor(model_features, target_idx, self._shap_batch_size).extract(
                model, frame.iloc[rows])
            temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
            np.savez_compressed(temporary, rows=rows, shap=contributions.to_numpy(dtype=np.float32))
            os.replace(temporary, path)
        with np.load(path) as stored:
            if not np.array_equal(stored["rows"], rows):
                raise ValueError(f"{path} does not hold the same rows with the target label.")
            return stored["shap"].astype(np.float64)

    def _labor_kmeans(
        self, path: Path, frame: pd.DataFrame, id_column: str, label_column: str, target: str, settings: Dict[str, Any],
        predicted: np.ndarray,
    ) -> np.ndarray:
        """LABOR with k-means on every raw feature and ground-truth cluster labels (module docstring).

        Args:
            path: Cache of the cluster assignment.
            frame: Training rows (every raw feature).
            id_column: Row id column.
            label_column: Label column.
            target: Target label.
            settings: Source settings (raw feature columns, poisoned ids).
            predicted: The audited model's label per row.

        Returns:
            Removal mask.
        """
        n_classes = int(frame[label_column].nunique())
        clusterer = LaborKMeansClusterer(
            exclude_columns=[column for column in frame.columns if column not in settings["feature_columns"]],
            label_column=label_column, id_column=id_column, n_clusters=n_classes, init=str(self._kmeans["init"]),
            max_iter=int(self._kmeans["max_iter"]), tol=float(self._kmeans["tol"]), random_state=self._seed,
        )
        if path.exists():
            with np.load(path) as stored:
                clusters = stored["cluster"]
        else:
            logger.info("LABOR k-means, k = %d, on %d raw features...", n_classes, len(settings["feature_columns"]))
            clusters = clusterer.fit_predict(frame)
            np.savez_compressed(path, cluster=clusters)
        summary = clusterer.summarize_clusters(frame, clusters, settings["poisoned_ids"], benign_label=target)
        remover = ClusterLabelPoisonRemover(target_label=target)
        return remover.compute_removal_mask(clusters, remover.label_clusters(summary), predicted)

    def _retrain(
        self, name: str, evaluator: LaborBackdoorEvaluator, frame: pd.DataFrame, removed: np.ndarray,
        clean_test: pd.DataFrame, trigger_test: pd.DataFrame, injected: np.ndarray, target: str, cache_dir: Path,
    ) -> Dict[str, Any]:
        """Retrain on what a removal keeps and measure it, or read the cached measurement.

        Args:
            name: Baseline name (cache key).
            evaluator: Evaluator of the cell.
            frame: Training rows.
            removed: Rows the baseline drops.
            clean_test: Untouched test set.
            trigger_test: Source-class test rows with the trigger injected.
            injected: Per trigger-test row, whether the trigger was injected.
            target: Target label.
            cache_dir: Cache directory.

        Returns:
            CDA, macro-F1, ASR (all and injected rows only), per-class recall.
        """
        cache = cache_dir / f"{name}.json"
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        logger.info("Retraining after baseline [%s]: keeping %d/%d rows...", name, int((~removed).sum()), len(removed))
        model = evaluator.train(frame.loc[~removed])
        metrics = EndToEndPoisonAudit.model_metrics(evaluator, model, clean_test, trigger_test, injected, target)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        return metrics
