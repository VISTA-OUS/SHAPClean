"""Component 1 - shared evaluation harness for every clustering (TreeSHAP attributions or raw features), so every
experiment is scored by the SAME metrics instead of each script inventing its own:

1. `multiclass_accuracy` - LABOR-comparable clustering accuracy. Luo et al. (2025) report 89.81%
   (+-0.39%) k-means accuracy on CIC-IDS2017 with k set to the number of label categories. HDBSCAN's
   cluster count is data-driven, not fixed to that category count, so a 1:1 Hungarian match (what a
   k=n_categories k-means setup could use) does not generalize here; this method uses the standard
   many-to-one majority-vote generalization instead - each cluster gets the true label of its own
   majority, then accuracy = the fraction of rows whose cluster-majority label matches their own
   true label. This is an ASSUMPTION about LABOR's own unstated exact accuracy formula, not a
   confirmed fact - the paper never states the formula behind its 89.81% figure.
2. `poison_detection_metrics` - precision/recall of the ACTUAL removal rule already implemented in
   `ClusterLabelPoisonRemover` (cluster composition says ATTACK_LIKE, but the model predicts
   BENIGN), reused end-to-end rather than reimplemented, so these numbers mean exactly what that rule
   would actually remove - not an approximation of it.

DBCV (Moulavi et al., SIAM SDM 2014) is deliberately not among them: it scores how well separated the
clusters are, which says nothing about whether the removal rule removes poison, and every configuration
choice here is made on the latter.

Noise (`-1`) gets no policy here (the removal system's noise rule handles noise rows later):
`multiclass_accuracy` reports both a noise-excluded and a noise-counted-as-wrong variant plus the raw noise fraction,
without picking a policy between them; `poison_detection_metrics` inherits
`ClusterLabelPoisonRemover`'s existing, unchanged behaviour of never flagging NOISE-cluster rows for
removal.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

from src.core.logger import get_logger
from src.poison_audit.cluster_label_remover import NOISE_CLUSTER_ID, ClusterLabelPoisonRemover

logger = get_logger(__name__)


class ClusteringEvaluator:
    """Scores one clustering result the same way, regardless of which ablation arm produced it.

    Args:
        target_label: The label the backdoor is meant to flip predictions to (`BENIGN`) - passed
            straight through to the internal `ClusterLabelPoisonRemover`.
    """

    def __init__(self, target_label: str) -> None:
        self._target_label = target_label
        self._remover = ClusterLabelPoisonRemover(target_label)

    def multiclass_accuracy(self, cluster_ids: np.ndarray, true_labels: pd.Series) -> Dict[str, float]:
        """Majority-vote clustering accuracy against the FULL multiclass `Label` column (not
        collapsed to binary BENIGN/ATTACK) - the LABOR-comparable metric.

        Args:
            cluster_ids: Cluster id per row.
            true_labels: Ground-truth multiclass `Label` column, same row order as `cluster_ids`.

        Returns:
            Dict with `accuracy_excl_noise` (majority-vote accuracy computed only among non-noise
            rows), `accuracy_incl_noise` (same numerator, but every noise row counts as wrong and
            the denominator is the full row count), and `noise_fraction`.
        """
        labels_array = np.asarray(true_labels)
        non_noise_mask = cluster_ids != NOISE_CLUSTER_ID
        n_total = len(cluster_ids)
        n_correct = 0
        for cluster_id in np.unique(cluster_ids[non_noise_mask]):
            member_mask = cluster_ids == cluster_id
            majority_label = pd.Series(labels_array[member_mask]).mode().iloc[0]
            n_correct += int((labels_array[member_mask] == majority_label).sum())
        n_non_noise = int(non_noise_mask.sum())
        return {
            "accuracy_excl_noise": n_correct / n_non_noise if n_non_noise else 0.0,
            "accuracy_incl_noise": n_correct / n_total if n_total else 0.0,
            "noise_fraction": 1.0 - n_non_noise / n_total if n_total else 0.0,
        }

    def poison_placement(self, cluster_summary: pd.DataFrame, purity_threshold_pct: float) -> Dict[str, Any]:
        """Where the poisoned rows ended up - the primary decision criterion for a clustering arm.

        Precision/recall already say whether the removal rule fired, but not why it failed when it
        did. The two failure modes look identical in the metrics and need different fixes: poison
        sitting in the noise cluster (never eligible for removal) versus poison diluted across
        attack-majority clusters (eligible, but only removable together with everything around it).

        Args:
            cluster_summary: `HdbscanClusterer.summarize_clusters` output for one arm.
            purity_threshold_pct: Poison share above which a cluster counts as poison-pure.

        Returns:
            Dict with poison in noise, poison inside poison-pure clusters, the number of
            poison-bearing clusters, and the largest poison-bearing cluster's size/poison count/
            purity.
        """
        real_clusters = cluster_summary[cluster_summary["cluster"] != NOISE_CLUSTER_ID]
        noise_rows = cluster_summary[cluster_summary["cluster"] == NOISE_CLUSTER_ID]
        bearing = real_clusters[real_clusters["n_poisoned"] > 0].copy()
        placement: Dict[str, Any] = {
            "poison_in_noise": int(noise_rows["n_poisoned"].sum()),
            "poison_in_pure_clusters": 0,
            "n_clusters_with_poison": int(len(bearing)),
            "largest_poison_cluster_n_records": 0,
            "largest_poison_cluster_n_poisoned": 0,
            "largest_poison_cluster_purity_pct": 0.0,
        }
        if bearing.empty:
            return placement
        bearing["purity_pct"] = 100.0 * bearing["n_poisoned"] / bearing["n_records"]
        placement["poison_in_pure_clusters"] = int(
            bearing[bearing["purity_pct"] >= purity_threshold_pct]["n_poisoned"].sum(),
        )
        largest = bearing.loc[bearing["n_poisoned"].idxmax()]
        placement["largest_poison_cluster_n_records"] = int(largest["n_records"])
        placement["largest_poison_cluster_n_poisoned"] = int(largest["n_poisoned"])
        placement["largest_poison_cluster_purity_pct"] = round(float(largest["purity_pct"]), 2)
        return placement

    def purity_breakdown(self, cluster_ids: np.ndarray, true_labels: pd.Series) -> Dict[str, Any]:
        """Per-cluster purity, reported SEPARATELY for BENIGN-majority and ATTACK-majority clusters.

        `multiclass_accuracy` returns one number over all clusters, which hides the asymmetry that
        matters operationally: an impure BENIGN-majority cluster hides attacks inside traffic a
        human reviewer would wave through, while an impure ATTACK-majority cluster costs
        legitimate records when its rows are removed. Those two failure modes have different
        consequences, so they are counted apart rather than averaged together.

        Purity is measured two ways per cluster: binary (`target_label` vs everything else, the
        split the removal rule actually acts on) and multiclass (largest single `Label` value, the
        stricter reading).

        Args:
            cluster_ids: Cluster id per row; `-1` rows are excluded from every purity figure and
                reported only as the noise count.
            true_labels: Ground-truth `Label` per row, same order as `cluster_ids`.

        Returns:
            Dict with overall counts, row-weighted binary/multiclass purity over clustered rows, and
            a `benign_majority`/`attack_majority` sub-dict each holding that group's cluster count,
            row count, row-weighted and unweighted mean binary purity, the number of clusters at
            100%/>=99%/>=95% binary purity, and the least pure cluster in the group.
        """
        frame = pd.DataFrame({
            "cluster": np.asarray(cluster_ids),
            "label": np.asarray(true_labels),
        })
        n_noise = int((frame["cluster"] == NOISE_CLUSTER_ID).sum())
        real_rows = frame[frame["cluster"] != NOISE_CLUSTER_ID]
        if real_rows.empty:
            return {"n_clusters": 0, "n_noise": n_noise, "n_rows": int(len(frame))}

        real_rows = real_rows.assign(is_benign=real_rows["label"] == self._target_label)
        per_cluster = real_rows.groupby("cluster").agg(
            n_records=("label", "size"), n_benign=("is_benign", "sum"),
        )
        per_cluster["n_attack"] = per_cluster["n_records"] - per_cluster["n_benign"]
        label_counts = real_rows.groupby(["cluster", "label"]).size()
        per_cluster["n_majority_label"] = label_counts.groupby(level=0).max()
        per_cluster["binary_purity"] = (
            per_cluster[["n_benign", "n_attack"]].max(axis=1) / per_cluster["n_records"]
        )
        per_cluster["multiclass_purity"] = (
            per_cluster["n_majority_label"] / per_cluster["n_records"]
        )
        per_cluster["majority"] = np.where(
            per_cluster["n_benign"] >= per_cluster["n_attack"], "benign", "attack",
        )

        n_clustered_rows = int(per_cluster["n_records"].sum())
        breakdown: Dict[str, Any] = {
            "n_clusters": int(len(per_cluster)),
            "n_rows": int(len(frame)),
            "n_noise": n_noise,
            "noise_pct": round(100.0 * n_noise / len(frame), 2),
            "n_clustered_rows": n_clustered_rows,
            "binary_purity_weighted_pct": round(
                100.0 * float(per_cluster[["n_benign", "n_attack"]].max(axis=1).sum()) / n_clustered_rows, 2,
            ),
            "multiclass_purity_weighted_pct": round(
                100.0 * float(per_cluster["n_majority_label"].sum()) / n_clustered_rows, 2,
            ),
        }
        for majority in ("benign", "attack"):
            group = per_cluster[per_cluster["majority"] == majority]
            if group.empty:
                breakdown[f"{majority}_majority"] = {"n_clusters": 0, "n_rows": 0}
                continue
            group_rows = int(group["n_records"].sum())
            breakdown[f"{majority}_majority"] = {
                "n_clusters": int(len(group)),
                "n_rows": group_rows,
                "binary_purity_weighted_pct": round(
                    100.0 * float(group[["n_benign", "n_attack"]].max(axis=1).sum()) / group_rows, 2,
                ),
                "binary_purity_mean_pct": round(100.0 * float(group["binary_purity"].mean()), 2),
                "multiclass_purity_weighted_pct": round(
                    100.0 * float(group["n_majority_label"].sum()) / group_rows, 2,
                ),
                "n_clusters_purity_100": int((group["binary_purity"] >= 1.0).sum()),
                "n_clusters_purity_ge_99": int((group["binary_purity"] >= 0.99).sum()),
                "n_clusters_purity_ge_95": int((group["binary_purity"] >= 0.95).sum()),
                "min_binary_purity_pct": round(100.0 * float(group["binary_purity"].min()), 2),
            }
        return breakdown

    def poison_detection_metrics(
        self,
        cluster_summary: pd.DataFrame,
        cluster_ids: np.ndarray,
        predicted_labels: np.ndarray,
        poisoned_sample_ids: Iterable[int],
        sample_ids: np.ndarray,
    ) -> Dict[str, Any]:
        """Precision/recall of the actual poison-removal rule, not an invented proxy metric.

        Args:
            cluster_summary: Output of `HdbscanClusterer.summarize_clusters` /
                `LaborKMeansClusterer.summarize_clusters`.
            cluster_ids: Cluster id per row, same order as `predicted_labels`/`sample_ids`.
            predicted_labels: Model-predicted STRING label per row (e.g.
                `LaborBackdoorEvaluator.predict_labels`'s output), same order as `cluster_ids`.
            poisoned_sample_ids: Ground-truth poisoned `id_column` values.
            sample_ids: `id_column` value per row, same order as `cluster_ids`.

        Returns:
            `ClusterLabelPoisonRemover.summarize_removal`'s dict (`n_removed`, `n_removed_correct`,
            `n_removed_wrong`, `n_missed`, `n_total_poisoned`), plus `precision`
            (`n_removed_correct / n_removed`) and `recall` (`n_removed_correct / n_total_poisoned`).
        """
        cluster_labels = self._remover.label_clusters(cluster_summary)
        removal_mask = self._remover.compute_removal_mask(cluster_ids, cluster_labels, predicted_labels)
        is_poisoned = np.isin(sample_ids, np.fromiter(poisoned_sample_ids, dtype=sample_ids.dtype))
        outcome = self._remover.summarize_removal(removal_mask, is_poisoned)
        outcome["precision"] = outcome["n_removed_correct"] / outcome["n_removed"] if outcome["n_removed"] else 0.0
        outcome["recall"] = (
            outcome["n_removed_correct"] / outcome["n_total_poisoned"] if outcome["n_total_poisoned"] else 0.0
        )
        return outcome
