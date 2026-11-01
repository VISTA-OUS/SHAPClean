"""Component 1 - second-stage ("cascade") clustering of the rows stage-1 HDBSCAN called noise, plus
the bookkeeping that merges both stages' removal decisions into one precision/recall figure.

Why this exists (measured 2026-09-11 on the best stage-1 configuration, SHAP-30 + StandardScaler +
PCA(95%), `min_cluster_size=100`): recall sits at 65.04% because 358 of the 1,024 poisoned rows land
in the noise cluster. Those 358 are NOT diffuse outliers - in the stage-1 PCA space they form a
block tighter than the poison cluster that WAS found (radius 0.984 around their own centroid vs 1.000
for the 666-row cluster, against 3.341 for ordinary noise), sitting 14.957 away from it, and the
backdoored model predicts BENIGN for all 358. So the structure is there and is separable; HDBSCAN
simply never saw enough of it. The reason is sampling, not geometry: `fit_sample_size=300000` out of
2,262,300 rows captured only 56 of those 358 rows, and 56 < `min_cluster_size`, so no cluster could
be born (confirmed by the `min_cluster_size` sweep - even at 50, EOM stability did not extract it).

Re-clustering only the noise rows changes that arithmetic instead of changing HDBSCAN's parameters:
the 358 make up 0.070% of the 514,392 noise rows versus 0.019% of the original fit sample, a 3.7x
density gain, so the same 300,000-row fit sample is expected to contain ~209 of them - comfortably
above `min_cluster_size=100`. Refitting StandardScaler/PCA on the noise subset (which
`HdbscanClusterer` does automatically) is part of the intended effect: stage-1's principal
components were fit on a population dominated by the large BENIGN/DDoS masses that are no longer
present.

This class deliberately does NOT re-implement cluster labeling or the removal rule - it reuses
`ClusterLabelPoisonRemover` so both stages are judged by exactly the same criterion, and only adds
what is genuinely new: recovering a stage's removal mask, unioning the two stages, and scoring the
union against ground truth.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.poison_audit.cluster_label_remover import ClusterLabelPoisonRemover


class NoiseCascadeAuditor:
    """Builds per-stage removal masks with the stage-1 rule and merges them into combined metrics.

    Args:
        target_label: Label the backdoor flips predictions to (`BENIGN`) - passed through to
            `ClusterLabelPoisonRemover`, so both stages apply an identical removal rule.
    """

    def __init__(self, target_label: str) -> None:
        self._remover = ClusterLabelPoisonRemover(target_label=target_label)

    def removal_mask(
        self,
        cluster_summary: pd.DataFrame,
        cluster_ids: np.ndarray,
        predicted_labels: np.ndarray,
    ) -> np.ndarray:
        """Recover one stage's row-level removal decision from its persisted artifacts.

        `ClusteringArmRunner` writes cluster assignments and metrics but not the mask itself; the
        mask is a pure function of those artifacts, so it is recomputed here rather than stored.

        Args:
            cluster_summary: That stage's `cluster_summary.csv` contents.
            cluster_ids: That stage's cluster id per row.
            predicted_labels: Model-predicted string label per row, same order as `cluster_ids`.

        Returns:
            Boolean array, True for rows the stage flags for removal.
        """
        cluster_labels = self._remover.label_clusters(cluster_summary)
        return self._remover.compute_removal_mask(cluster_ids, cluster_labels, predicted_labels)

    def combine_masks(
        self,
        stage1_mask: np.ndarray,
        stage2_mask: np.ndarray,
        stage2_positions: np.ndarray,
    ) -> np.ndarray:
        """Union stage 2's decisions (defined over the noise subset only) into stage 1's full-length
        mask.

        A union is the correct operator, not an override: stage 2 only ever sees rows stage 1 left
        as noise, and noise rows are never flagged by stage 1, so the two masks are disjoint by
        construction. The union is asserted to add exactly as many removals as stage 2 found.

        Args:
            stage1_mask: Boolean mask over all rows.
            stage2_mask: Boolean mask over the noise subset, in that subset's row order.
            stage2_positions: Positions of the noise subset's rows within the full row set, in the
                same order as `stage2_mask`.

        Returns:
            Boolean mask over all rows, True where either stage flags the row.
        """
        if len(stage2_mask) != len(stage2_positions):
            raise ValueError(
                f"stage2_mask ({len(stage2_mask)}) and stage2_positions ({len(stage2_positions)}) "
                "must have the same length.",
            )
        combined = stage1_mask.copy()
        combined[stage2_positions] |= stage2_mask
        expected = int(stage1_mask.sum()) + int(stage2_mask.sum())
        if int(combined.sum()) != expected:
            raise ValueError(
                f"The two stages overlap: the union holds {int(combined.sum())} but the separate total is {expected} "
                "- stage 2 should only receive the rows stage 1 marks as noise.",
            )
        return combined

    def score_mask(self, removal_mask: np.ndarray, is_poisoned: np.ndarray) -> Dict[str, Any]:
        """Score a removal mask against ground truth, matching `ClusteringEvaluator`'s output shape.

        Args:
            removal_mask: Boolean mask over all rows.
            is_poisoned: Ground-truth poisoned flag per row, same order as `removal_mask`.

        Returns:
            `ClusterLabelPoisonRemover.summarize_removal`'s counts plus `precision` and `recall`
            (both 0.0 when their denominator is zero).
        """
        metrics: Dict[str, Any] = dict(self._remover.summarize_removal(removal_mask, is_poisoned))
        n_removed = metrics["n_removed"]
        n_total_poisoned = metrics["n_total_poisoned"]
        metrics["precision"] = metrics["n_removed_correct"] / n_removed if n_removed else 0.0
        metrics["recall"] = metrics["n_removed_correct"] / n_total_poisoned if n_total_poisoned else 0.0
        return metrics
