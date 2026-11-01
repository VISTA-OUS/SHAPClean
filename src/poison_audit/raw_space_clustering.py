"""Clusters each poisoned set of the poison-rate sweep on its raw features instead of its TreeSHAP attributions.

The audit (`EndToEndPoisonAudit`) clusters the target-class TreeSHAP attributions of the audited model. This study asks
whether that choice matters: it runs the very same `ClusteringArmRunner` (same HDBSCAN block, full fit, same scaler and
PCA, same `min_cluster_size` per dataset) on the raw values of the same top-K model features, and scores both spaces with
the same evaluator - where the poisoned rows end up (`poison_placement`) and LABOR's rule with true cluster labels
(`poison_detection_metrics`, the stage-1 removal of the audit). Only the representation differs.

Each cell reads what script 04 left in its sweep cell (`poisoned_train.csv.gz`, the audited model); when they are gone
(a cell copied back without its large files) the poisoned set is rebuilt with the same builder and seed and checked
against the manifest's poisoned ids, and the audited model is retrained with the same parameters.

`RawSpaceAblationReport` scores the other half of the question (script 12): the whole system - layers A and B, the noise
rule, retraining - with stage 1 on raw features, next to the same system on attributions, under the paper's criterion.
Script 12's stage 1 is the clustering measured here, so its cells also feed `RawSpaceReport`.
"""

from __future__ import annotations

import gc
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.logger import get_logger
from src.poison_audit.clustering_arm_runner import ClusteringArmRunner
from src.poison_audit.clustering_evaluator import ClusteringEvaluator
from src.poison_audit.fixed_trigger_poison_builder import FixedTriggerPoisonDatasetBuilder
from src.poison_audit.labor_backdoor_evaluator import LaborBackdoorEvaluator
from src.poison_audit.poison_rate_sweep import MANIFEST_FILE, POISONED_TRAIN_FILE, SweepTrigger

logger = get_logger(__name__)

RESULT_FILE = "raw_space_result.json"
PIPELINE_RESULT_FILE = "pipeline_result.json"


class RawSpaceClusteringStudy:
    """Raw-feature clustering of one sweep cell, side by side with the attribution clustering of the audit.

    Args:
        clustering_params: The `clustering_hdbscan` block; `min_cluster_size` is set per dataset and `fit_sample_size`
            to None (full fit), exactly as `EndToEndPoisonAudit` does.
        purity_threshold_pct: Poison share above which a cluster counts as poison-pure (`end_to_end` block).
        model_params: XGBoost parameters of the audited model (`backdoor_eval.model_params`), for a retrain.
        arm_name: Name of the raw arm in logs and artifacts.
    """

    def __init__(
        self, clustering_params: Dict[str, Any], purity_threshold_pct: float, model_params: Dict[str, Any], arm_name: str,
    ) -> None:
        self._clustering_params = clustering_params
        self._purity_threshold_pct = purity_threshold_pct
        self._model_params = model_params
        self._arm_name = arm_name
        self._clean_train_cache: Dict[str, pd.DataFrame] = {}

    def run_cell(self, trigger: SweepTrigger, rate: float, seed: int, sweep_cell: Path, output_cell: Path) -> Dict[str, Any]:
        """Cluster one cell on its raw features and write `raw_space_result.json`.

        Args:
            trigger: The sweep trigger of the cell.
            rate: Poison rate.
            seed: Poisoning and model seed of the cell.
            sweep_cell: The cell of script 04 (manifest, pipeline result, poisoned set and model when still there).
            output_cell: Where this study writes the cell.

        Returns:
            The result written to `raw_space_result.json`.
        """
        manifest = json.loads((sweep_cell / MANIFEST_FILE).read_text(encoding="utf-8"))
        features = [str(feature) for feature in manifest["model_features"]]
        poisoned_ids = [int(value) for value in manifest["poisoned_sample_ids"]]
        frame, frame_source = self._poisoned_train(trigger, rate, seed, sweep_cell, features, poisoned_ids)
        predicted, model_source = self._predicted_labels(trigger, seed, sweep_cell, frame, features)

        params = {**self._clustering_params, "min_cluster_size": int(trigger.min_cluster_size), "fit_sample_size": None}
        metrics = ClusteringArmRunner(
            label_column=trigger.label_column, id_column=trigger.id_column, target_label=trigger.target_label,
            clustering_params=params, random_state=seed,
        ).run(self._arm_name, frame, features, predicted, poisoned_ids, output_cell)
        summary = pd.read_csv(output_cell / "cluster_summary.csv")
        placement = ClusteringEvaluator(target_label=trigger.target_label).poison_placement(summary, self._purity_threshold_pct)
        result = {
            "dataset": trigger.dataset, "trigger": trigger.name, "rate": rate, "seed": seed,
            "n_rows": int(len(frame)), "n_poisoned": len(poisoned_ids), "features": features,
            "poisoned_train_source": frame_source, "model_source": model_source,
            "raw": self.space_summary(metrics["n_clusters"], metrics["n_noise"], placement, metrics["poison_detection"],
                                      len(poisoned_ids)),
            "raw_pca_components": metrics["n_pca_components"], "raw_cluster_seconds": metrics["cluster_seconds"],
            "attribution": self.stage_summary(sweep_cell),
        }
        (output_cell / RESULT_FILE).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        del frame
        gc.collect()
        return result

    @staticmethod
    def space_summary(
        n_clusters: int, n_noise: int, placement: Dict[str, Any], removal: Dict[str, Any], n_poisoned: int,
    ) -> Dict[str, Any]:
        """The numbers compared between the two spaces, from one clustering.

        Args:
            n_clusters: Clusters, noise excluded.
            n_noise: Noise rows.
            placement: `ClusteringEvaluator.poison_placement` output.
            removal: `ClusteringEvaluator.poison_detection_metrics` output (LABOR's rule, true cluster labels).
            n_poisoned: Poisoned rows of the set.

        Returns:
            Dict of counts and shares (fractions, not percent).
        """
        clustered = n_poisoned - placement["poison_in_noise"]
        return {
            "n_clusters": int(n_clusters), "n_noise": int(n_noise),
            "n_clusters_with_poison": int(placement["n_clusters_with_poison"]),
            "poison_in_noise_share": placement["poison_in_noise"] / n_poisoned if n_poisoned else float("nan"),
            "clustered_poison_in_pure_share": placement["poison_in_pure_clusters"] / clustered if clustered else float("nan"),
            "largest_poison_cluster_purity_pct": float(placement["largest_poison_cluster_purity_pct"]),
            "oracle_rule_precision": float(removal["precision"]), "oracle_rule_recall": float(removal["recall"]),
        }

    @staticmethod
    def stage_summary(cell: Path) -> Dict[str, Any]:
        """The same numbers for the stage-1 clustering an audit left in a cell: the TreeSHAP attributions in a cell of
        script 04, the raw features in a cell of script 12 (both write `pipeline_result.json`).

        Args:
            cell: Audit cell.

        Returns:
            `space_summary` of that clustering.
        """
        stage = json.loads((cell / PIPELINE_RESULT_FILE).read_text(encoding="utf-8"))["stage1"]
        return RawSpaceClusteringStudy.space_summary(stage["n_clusters"], stage["n_noise"], stage["placement"],
                                                     stage["removal"], int(stage["removal"]["n_total_poisoned"]))

    def _poisoned_train(
        self, trigger: SweepTrigger, rate: float, seed: int, sweep_cell: Path, features: List[str], poisoned_ids: List[int],
    ) -> Tuple[pd.DataFrame, str]:
        """The cell's poisoned training set, read from the sweep cell or rebuilt.

        Args:
            trigger: The sweep trigger.
            rate: Poison rate.
            seed: Poisoning seed.
            sweep_cell: The cell of script 04.
            features: Model features of the cell.
            poisoned_ids: The manifest's poisoned ids.

        Returns:
            (rows with id, label and `features`, `"sweep_cell"` or `"rebuilt"`).

        Raises:
            ValueError: If a rebuilt set does not poison exactly the manifest's rows.
        """
        columns = [trigger.id_column, trigger.label_column] + features
        stored = sweep_cell / POISONED_TRAIN_FILE
        if stored.exists():
            return pd.read_csv(stored, usecols=columns, float_precision="round_trip")[columns], "sweep_cell"
        logger.info("[%s/%s/%g] %s is gone from the cell of script 04 - rebuilding it from the clean set.", trigger.dataset, trigger.name,
                    rate, POISONED_TRAIN_FILE)
        builder = FixedTriggerPoisonDatasetBuilder(trigger.trigger_spec, trigger.source_label, trigger.target_label,
                                                   trigger.label_column, trigger.id_column)
        frame, rebuilt_ids = builder.build_poisoned_train(self._clean_train(trigger), poison_rate=rate, seed=seed)
        if sorted(int(value) for value in rebuilt_ids) != sorted(poisoned_ids):
            raise ValueError(f"[{trigger.dataset}/{trigger.name}/{rate:g}] The rebuilt set poisons other rows than the manifest.")
        return frame[columns].reset_index(drop=True), "rebuilt"

    def _predicted_labels(
        self, trigger: SweepTrigger, seed: int, sweep_cell: Path, frame: pd.DataFrame, features: List[str],
    ) -> Tuple[np.ndarray, str]:
        """The audited model's label for every row - the "model says benign" half of LABOR's rule.

        Args:
            trigger: The sweep trigger.
            seed: Model seed.
            sweep_cell: The cell of script 04.
            frame: The poisoned training rows.
            features: Model features.

        Returns:
            (string label per row, `"sweep_cell"` or `"retrained"`).
        """
        evaluator = LaborBackdoorEvaluator(features, trigger.label_column, self._model_params, seed)
        # A label encoder that matches the audit: fitted on the labels of the clean set (poisoning adds or removes no class).
        evaluator.fit_label_encoder(pd.read_csv(trigger.clean_train, usecols=[trigger.label_column])[trigger.label_column])
        stored = sweep_cell / f"dirty_model_{len(features)}feat.pkl"
        if stored.exists():
            with stored.open("rb") as handle:
                model, source = pickle.load(handle), "sweep_cell"
        else:
            logger.info("[%s/%s] The audited model is gone from the cell of script 04 - retraining it.", trigger.dataset, trigger.name)
            model, source = evaluator.train(frame), "retrained"
        return evaluator.predict_labels(model, frame), source

    def _clean_train(self, trigger: SweepTrigger) -> pd.DataFrame:
        """The clean training split, read once per file.

        Args:
            trigger: The sweep trigger.

        Returns:
            Every column of the clean split.
        """
        if trigger.clean_train not in self._clean_train_cache:
            self._clean_train_cache = {trigger.clean_train: pd.read_csv(trigger.clean_train, float_precision="round_trip")}
        return self._clean_train_cache[trigger.clean_train]


class RawSpaceReport:
    """Per-set rows and per-dataset medians of the two spaces, from every finished cell.

    Args:
        dataset_titles: Dataset name -> display name.
    """

    METRICS = ("n_clusters", "n_clusters_with_poison", "poison_in_noise_share", "clustered_poison_in_pure_share",
               "oracle_rule_precision", "oracle_rule_recall")

    def __init__(self, dataset_titles: Dict[str, str]) -> None:
        self._titles = dataset_titles

    def sets(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """One row per set: every compared number in both spaces.

        Args:
            results: `raw_space_result.json` of every finished cell.

        Returns:
            Frame sorted by dataset, trigger, rate and seed.
        """
        rows = []
        for result in results:
            row = {key: result[key] for key in ("dataset", "trigger", "rate", "seed", "n_rows", "n_poisoned")}
            for space in ("raw", "attribution"):
                row.update({f"{space}_{metric}": result[space][metric] for metric in self.METRICS})
            rows.append(row)
        columns = ["dataset", "trigger", "rate", "seed"]
        return pd.DataFrame(rows).sort_values(columns).reset_index(drop=True) if rows else pd.DataFrame(columns=columns)

    def summary(self, sets: pd.DataFrame) -> pd.DataFrame:
        """Per dataset and space: number of sets, then the median and range of every compared number.

        Args:
            sets: Output of `sets`.

        Returns:
            One row per (dataset, metric).
        """
        rows = []
        for dataset, part in sets.groupby("dataset", sort=False):
            for metric in self.METRICS:
                row: Dict[str, Any] = {"dataset": dataset, "metric": metric, "n_sets": int(len(part))}
                for space in ("raw", "attribution"):
                    values = part[f"{space}_{metric}"].dropna()
                    row.update({f"{space}_median": values.median(), f"{space}_min": values.min(), f"{space}_max": values.max()})
                rows.append(row)
        return pd.DataFrame(rows)

    def markdown(self, summary: pd.DataFrame, expected: Dict[str, int], missing: Optional[List[str]] = None) -> str:
        """Markdown report: one table per dataset, a row per compared number.

        Args:
            summary: Output of `summary`.
            expected: Dataset -> number of sets the report should cover.
            missing: Cells not finished yet, listed at the top.

        Returns:
            Markdown.
        """
        labels = {
            "n_clusters": ("Clusters", False), "n_clusters_with_poison": ("Clusters holding poison", False),
            "poison_in_noise_share": ("Poison left as noise (%)", True),
            "clustered_poison_in_pure_share": ("Clustered poison in pure clusters (%)", True),
            "oracle_rule_precision": ("LABOR rule, true cluster labels: precision (%)", True),
            "oracle_rule_recall": ("LABOR rule, true cluster labels: recall (%)", True),
        }
        lines = ["# Clustering on raw features against clustering on TreeSHAP attributions", "",
                 "- The same HDBSCAN (block `clustering_hdbscan`, full fit, StandardScaler + PCA, per-dataset"
                 " `min_cluster_size`) on the same 30 mRMR features of the model; only the space differs: the raw feature"
                 " values (raw) or the target-class TreeSHAP attributions of the audited model (attribution, the audit's own"
                 " clustering step).",
                 "- Pure cluster: poison makes up at least `purity_threshold_pct` of the cluster. LABOR rule with true cluster"
                 " labels: remove the rows of clusters that are mostly attack by true label and that the audited model calls"
                 " benign (stage 1 of the audit, the LABOR-HDBSCAN reference).",
                 "- Each cell: median (min - max) over the sets.", ""]
        if missing:
            lines += [f"- {len(missing)} cells missing: {', '.join(missing[:20])}{' ...' if len(missing) > 20 else ''}.", ""]
        for dataset, part in summary.groupby("dataset", sort=False):
            n_sets = int(part["n_sets"].iloc[0])
            lines += [f"## {self._titles.get(dataset, dataset)} ({n_sets}/{expected.get(dataset, n_sets)} sets)", "",
                      "| | raw features | TreeSHAP attributions |", "|---|---|---|"]
            for _, row in part.iterrows():
                label, percent = labels[row["metric"]]
                cells = [self._cell(row, space, percent) for space in ("raw", "attribution")]
                lines.append(f"| {label} | {cells[0]} | {cells[1]} |")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _cell(row: pd.Series, space: str, percent: bool) -> str:
        """One table cell: median (min - max).

        Args:
            row: Summary row.
            space: `raw` or `attribution`.
            percent: Print fractions as percent.

        Returns:
            Formatted cell.
        """
        scale, digits = (100.0, 1) if percent else (1.0, 0)
        values = [row[f"{space}_{stat}"] for stat in ("median", "min", "max")]
        if any(pd.isna(value) for value in values):
            return "--"
        median, low, high = (scale * value for value in values)
        return f"{median:,.{digits}f} ({low:,.{digits}f} - {high:,.{digits}f})"


class RawSpaceAblationReport:
    """The "raw features" row of the ablation table: the full system with stage 1 on raw features (script 12), next to
    the same system on TreeSHAP attributions (script 04), on the same poisoned sets.

    A set counts when the retrained model's attack success rate is at most the oracle's plus `margin` and its clean
    accuracy at most `max_cda_drop` below the clean model's - criterion (1) of the paper. The benign-recall drop is
    against the oracle (`true_poison`) of the same cell, as in the ablation table.

    Args:
        variant: Name of the system's removal in both results (e.g. `k25_eta0.5_kept_and_cluster_noise0.5`).
        reference: Oracle removal name (`true_poison`).
        margin: ASR margin above the oracle (fraction).
        max_cda_drop: Largest clean-accuracy loss against the clean model (fraction).
        dataset_titles: Dataset name -> display name.
    """

    SPACES = ("raw", "treeshap")

    def __init__(self, variant: str, reference: str, margin: float, max_cda_drop: float, dataset_titles: Dict[str, str]) -> None:
        self._variant = variant
        self._reference = reference
        self._margin = margin
        self._max_cda_drop = max_cda_drop
        self._titles = dataset_titles

    def outcome(self, result: Dict[str, Any], target_label: str) -> Dict[str, Any]:
        """Criterion, FPR, precision, recall and benign-recall drop of the system in one `end_to_end_result.json`.

        Args:
            result: The cell's result.
            target_label: Benign class of the dataset.

        Returns:
            Dict of the compared numbers (fractions; the benign-recall drop in points).
        """
        removals = {item["name"]: item for item in result["removals"]}
        system, oracle = removals[self._variant], removals[self._reference]
        n_clean = result["n_rows"] - result["n_poisoned"]
        layer = result["layers"][system["layer_a"]]["layer_b"]["reference"]
        return {
            # Older results of script 04 lack this field: every one of their cells trained the reference model.
            "reference_trained": bool(layer.get("reference_trained", True)),
            "removes_backdoor": bool(system["asr_injected_only"] <= oracle["asr_injected_only"] + self._margin
                                     and result["backdoor"]["clean_cda"] - system["cda"] <= self._max_cda_drop),
            "fpr": system["n_removed_wrong"] / n_clean if n_clean else float("nan"),
            "precision": system["precision"], "recall": system["recall"],
            "benign_recall_drop_points": 100.0 * (oracle["recall_per_class"][target_label]
                                                  - system["recall_per_class"][target_label]),
            "asr": system["asr_injected_only"], "cda": system["cda"],
        }

    def sets(self, pairs: List[Dict[str, Any]]) -> pd.DataFrame:
        """One row per set with both spaces.

        Args:
            pairs: Dicts with `dataset`, `trigger`, `rate`, `seed`, `target_label`, `raw` and `treeshap` results.

        Returns:
            Frame sorted by dataset, trigger, rate and seed.
        """
        rows = []
        for pair in pairs:
            row = {key: pair[key] for key in ("dataset", "trigger", "rate", "seed")}
            for space in self.SPACES:
                row.update({f"{space}_{key}": value for key, value in self.outcome(pair[space], pair["target_label"]).items()})
            rows.append(row)
        columns = ["dataset", "trigger", "rate", "seed"]
        return pd.DataFrame(rows).sort_values(columns).reset_index(drop=True) if rows else pd.DataFrame(columns=columns)

    def summary(self, sets: pd.DataFrame) -> pd.DataFrame:
        """Per dataset and space: sets on which the backdoor is removed, median FPR / precision / recall / benign drop.

        Args:
            sets: Output of `sets`.

        Returns:
            One row per (dataset, space).
        """
        rows = []
        for dataset, part in sets.groupby("dataset", sort=False):
            for space in self.SPACES:
                rows.append({
                    "dataset": dataset, "space": space, "n_sets": int(len(part)),
                    "sets_backdoor_removed": int(part[f"{space}_removes_backdoor"].sum()),
                    "sets_without_reference": int((~part[f"{space}_reference_trained"].astype(bool)).sum()),
                    "median_fpr_pct": 100 * part[f"{space}_fpr"].median(),
                    "median_precision_pct": 100 * part[f"{space}_precision"].median(),
                    "median_recall_pct": 100 * part[f"{space}_recall"].median(),
                    "median_benign_recall_drop_points": part[f"{space}_benign_recall_drop_points"].median(),
                    "max_benign_recall_drop_points": part[f"{space}_benign_recall_drop_points"].max(),
                })
        return pd.DataFrame(rows)

    def markdown(self, summary: pd.DataFrame, expected: Dict[str, int], missing: List[str]) -> str:
        """Markdown report: one table per dataset, a row per space.

        Args:
            summary: Output of `summary`.
            expected: Dataset -> number of sets the report should cover.
            missing: Sets not finished yet.

        Returns:
            Markdown.
        """
        lines = ["# The \"raw features\" row of the ablation table: the whole system, stage 1 clustering raw features", "",
                 f"- System `{self._variant}`; backdoor-removal criterion (1) against `{self._reference}` of the same cell"
                 f" (ASR margin {100 * self._margin:g} pp, largest CDA drop {100 * self._max_cda_drop:g} pp); ΔBen = benign"
                 " recall of the reference minus that of the system (pp).",
                 "- `treeshap`: the results of script 04 on exactly the sets that have `raw`.",
                 "- No reference model: the trusted set holds one class only, so layer B confirms no cluster and the noise"
                 " rule removes no row (only the LABOR rule on attack-labelled clusters is left).", ""]
        if missing:
            lines += [f"- {len(missing)} sets missing: {', '.join(missing[:20])}{' ...' if len(missing) > 20 else ''}.", ""]
        names = {"raw": "raw features", "treeshap": "TreeSHAP attributions"}
        for dataset, part in summary.groupby("dataset", sort=False):
            n_sets = int(part["n_sets"].iloc[0])
            lines += [f"## {self._titles.get(dataset, dataset)} ({n_sets}/{expected.get(dataset, n_sets)} sets)", "",
                      "| Space | Sets | FPR (%) | Precision (%) | Recall (%) | median ΔBen (max) |"
                      " No reference model |",
                      "|---|---|---|---|---|---|---|"]
            for _, row in part.iterrows():
                lines.append(f"| {names[row['space']]} | {row['sets_backdoor_removed']}/{n_sets} | {row['median_fpr_pct']:.3f} |"
                             f" {row['median_precision_pct']:.1f} | {row['median_recall_pct']:.1f} |"
                             f" {row['median_benign_recall_drop_points']:.2f} ({row['max_benign_recall_drop_points']:.2f}) |"
                             f" {row['sets_without_reference']} |")
            lines.append("")
        return "\n".join(lines)
