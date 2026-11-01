"""Component 1 - runs ONE clustering ("arm") end to end (cluster -> evaluate -> persist), so every
representation goes through identical code and is scored by the identical `ClusteringEvaluator` metrics.
Arms differ ONLY in what frame/columns they are given: the target-class TreeSHAP attributions of the
model's features (stage 1 of the audit), or the raw values of the same features (the raw-feature
ablation of scripts 11 and 12).

Every arm reuses the SAME `component1_poison_audit.clustering_hdbscan` hyperparameters and the SAME
`random_state`, and is handed the FULL `poisoned_train` row set in its original order. When HDBSCAN is
fitted on a sample (`fit_sample_size`), `HdbscanClusterer.fit_predict`'s internal
`np.random.RandomState(seed).choice` depends only on row count and seed - never on feature values - so
every arm fits on the exact same rows and their scores stay directly comparable.

`multiclass_accuracy` and `poison_detection_metrics` are computed over ALL rows - the pipeline's
actual output, and the scope LABOR's own 89.81% figure covers.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from src.core.logger import get_logger
from src.poison_audit.clustering_evaluator import ClusteringEvaluator
from src.poison_audit.hdbscan_clusterer import HdbscanClusterer

logger = get_logger(__name__)


class ClusteringArmRunner:
    """Clusters one representation, scores it with the shared evaluator, and writes the artifacts.

    Args:
        label_column: Ground-truth multiclass label column name (`data.label_column`).
        id_column: Stable row-identifier column name (`data.sample_id_column`).
        target_label: Label the backdoor flips predictions to (`BENIGN`), passed to
            `ClusteringEvaluator`.
        clustering_params: The whole `component1_poison_audit.clustering_hdbscan` block. Only the
            five `HdbscanClusterer` hyperparameters are read from it (`pca_variance_retained`,
            `fit_sample_size`, `min_cluster_size`, `min_samples`, `core_dist_n_jobs`); the block's
            path keys (`input_path`, `manifest_path`, `output_dir`) are ignored, since each arm is
            handed its own frame and output directory instead.
        random_state: Seed shared by every arm, so all arms fit on the same rows.
    """

    def __init__(
        self,
        label_column: str,
        id_column: str,
        target_label: str,
        clustering_params: Dict[str, Any],
        random_state: int,
    ) -> None:
        self._label_column = label_column
        self._id_column = id_column
        self._clustering_params = clustering_params
        self._random_state = random_state
        self._target_label = target_label
        self._evaluator = ClusteringEvaluator(target_label=target_label)

    def run(
        self,
        arm_name: str,
        clustering_frame: pd.DataFrame,
        feature_columns: List[str],
        predicted_labels: np.ndarray,
        poisoned_sample_ids: Iterable[int],
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Clusters `clustering_frame`, scores the result, and writes artifacts to `output_dir`.

        Args:
            arm_name: Short arm identifier used in logs/report (e.g. `raw30`).
            clustering_frame: Full row set to cluster - must contain `label_column`, `id_column`
                and every column in `feature_columns`, in the SAME row order as every other arm.
            feature_columns: Columns this arm clusters on (what makes this arm different).
            predicted_labels: Model-predicted STRING label per row, same order as
                `clustering_frame` - the "model says BENIGN" half of the removal rule. Held FIXED
                across arms (same classifier) so the ablation varies only the clustering space.
            poisoned_sample_ids: Ground-truth poisoned `id_column` values.
            output_dir: Directory to write `cluster_assignment.npz`, `cluster_summary.csv`,
                `arm_metrics.json` and `arm_report.md` into.

        Returns:
            Metrics dict (also written to `arm_metrics.json`): cluster/noise counts, timings, and
            the three `ClusteringEvaluator` metric groups.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        clusterer = HdbscanClusterer(
            feature_columns=feature_columns,
            label_column=self._label_column,
            id_column=self._id_column,
            target_label=self._target_label,
            random_state=self._random_state,
            pca_variance_retained=(
                None if self._clustering_params.get("pca_variance_retained") is None
                else float(self._clustering_params["pca_variance_retained"])
            ),
            fit_sample_size=(
                None if self._clustering_params.get("fit_sample_size") is None
                else int(self._clustering_params["fit_sample_size"])
            ),
            min_cluster_size=int(self._clustering_params["min_cluster_size"]),
            min_samples=int(self._clustering_params["min_samples"]),
            core_dist_n_jobs=int(self._clustering_params["core_dist_n_jobs"]),
            # `use_scaler` is the older name (only switches StandardScaler on or off); `scaler_kind` is more general
            # and wins when present. With neither, StandardScaler is used.
            scaler_kind=self._clustering_params.get(
                "scaler_kind", "standard" if self._clustering_params.get("use_scaler", True) else None,
            ),
            row_normalize=bool(self._clustering_params.get("row_normalize", False)),
        )

        logger.info(
            "[%s] Starting fit_predict on %d rows, %d features...", arm_name, len(clustering_frame), len(feature_columns),
        )
        start_time = time.time()
        cluster_ids = clusterer.fit_predict(clustering_frame)
        cluster_seconds = time.time() - start_time
        n_noise = int((cluster_ids == -1).sum())
        n_clusters = len(set(cluster_ids.tolist())) - (1 if n_noise else 0)
        n_fit_rows, n_predicted_rows = clusterer.fit_vs_predicted_counts
        logger.info(
            "[%s] fit_predict done after %.1fs: %d clusters, %d noise (%.2f%%), PCA keeps %d dimensions.",
            arm_name, cluster_seconds, n_clusters, n_noise,
            100.0 * n_noise / len(clustering_frame), clusterer.n_pca_components,
        )

        summary_frame = clusterer.summarize_clusters(clustering_frame, cluster_ids, poisoned_sample_ids)

        accuracy = self._evaluator.multiclass_accuracy(cluster_ids, clustering_frame[self._label_column])
        logger.info(
            "[%s] multiclass accuracy: excl_noise=%.4f incl_noise=%.4f (noise=%.4f)",
            arm_name, accuracy["accuracy_excl_noise"], accuracy["accuracy_incl_noise"], accuracy["noise_fraction"],
        )

        poison = self._evaluator.poison_detection_metrics(
            cluster_summary=summary_frame,
            cluster_ids=cluster_ids,
            predicted_labels=predicted_labels,
            poisoned_sample_ids=poisoned_sample_ids,
            sample_ids=clustering_frame[self._id_column].to_numpy(),
        )
        logger.info(
            "[%s] poison: removed=%d correct=%d precision=%.4f recall=%.4f (missed=%d/%d)",
            arm_name, poison["n_removed"], poison["n_removed_correct"], poison["precision"],
            poison["recall"], poison["n_missed"], poison["n_total_poisoned"],
        )

        metrics: Dict[str, Any] = {
            "arm_name": arm_name,
            "n_rows": int(len(clustering_frame)),
            "n_feature_columns": len(feature_columns),
            "n_pca_components": clusterer.n_pca_components,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "n_fit_rows": n_fit_rows,
            "n_predicted_rows": n_predicted_rows,
            "cluster_seconds": round(cluster_seconds, 1),
            "multiclass_accuracy": accuracy,
            "poison_detection": poison,
        }

        np.savez_compressed(
            output_dir / "cluster_assignment.npz",
            sample_id=clustering_frame[self._id_column].to_numpy(),
            cluster=cluster_ids,
        )
        summary_frame.to_csv(output_dir / "cluster_summary.csv", index=False)
        (output_dir / "arm_metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        (output_dir / "arm_report.md").write_text(
            self._build_report(arm_name, feature_columns, metrics), encoding="utf-8",
        )
        logger.info("[%s] Wrote cluster_assignment.npz, cluster_summary.csv, arm_metrics.json, arm_report.md to %s.", arm_name, output_dir)
        return metrics

    def _build_report(self, arm_name: str, feature_columns: List[str], metrics: Dict[str, Any]) -> str:
        """Renders the per-arm markdown report.

        Args:
            arm_name: Short arm identifier.
            feature_columns: Columns this arm clustered on.
            metrics: The metrics dict assembled by `run`.

        Returns:
            Report text.
        """
        accuracy = metrics["multiclass_accuracy"]
        poison = metrics["poison_detection"]
        return "\n".join([
            f"# Clustering arm `{arm_name}`",
            "",
            f"- Clustering space: {metrics['n_feature_columns']} columns -> PCA keeps "
            f"{metrics['n_pca_components']} dimensions.",
            f"- Rows: {metrics['n_rows']:,} (HDBSCAN fit {metrics['n_fit_rows']:,}, "
            f"approximate_predict {metrics['n_predicted_rows']:,}).",
            f"- Clusters (noise excluded): {metrics['n_clusters']:,}; noise: {metrics['n_noise']:,} "
            f"({100.0 * metrics['n_noise'] / metrics['n_rows']:.2f}%).",
            f"- Clustering time: {metrics['cluster_seconds']}s.",
            "",
            "## Metrics (measured by the `ClusteringEvaluator` shared by every arm)",
            "",
            "| Metric | Value | Scope |",
            "|---|---:|---|",
            f"| multiclass accuracy (noise excluded) | {accuracy['accuracy_excl_noise'] * 100:.2f}% | every non-noise row |",
            f"| multiclass accuracy (noise counted as wrong) | {accuracy['accuracy_incl_noise'] * 100:.2f}% | every row |",
            f"| noise fraction | {accuracy['noise_fraction'] * 100:.2f}% | every row |",
            f"| poison precision | {poison['precision'] * 100:.2f}% | every row |",
            f"| poison recall | {poison['recall'] * 100:.2f}% | every row |",
            "",
            f"- Removed: {poison['n_removed']:,} rows, {poison['n_removed_correct']:,} of them poisoned, "
            f"{poison['n_removed_wrong']:,} wrongly.",
            f"- Missed: {poison['n_missed']:,}/{poison['n_total_poisoned']:,} poisoned rows.",
            "",
            f"- Features: `{', '.join(feature_columns[:5])}`"
            + (f", ... (+{len(feature_columns) - 5} more)" if len(feature_columns) > 5 else ""),
            "",
        ]) + "\n"
