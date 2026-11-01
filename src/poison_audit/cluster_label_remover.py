"""Component 1 (poison audit) - labels each cluster BENIGN_LIKE/ATTACK_LIKE from its TRUE
composition (poisoned rows count as ATTACK, since they are attack traffic whose label was flipped by
the poisoning step), then flags for removal any row whose cluster says ATTACK_LIKE while the
(backdoored) model predicts BENIGN. This is LABOR's "cluster label vs. classifier prediction"
mismatch rule (Luo et al. 2025), with the true composition standing in for LABOR's human reviewer.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

BENIGN_LIKE = "BENIGN_LIKE"
ATTACK_LIKE = "ATTACK_LIKE"
NOISE = "NOISE"
NOISE_CLUSTER_ID = -1


class ClusterLabelPoisonRemover:
    """Labels clusters by true (poison-excluded) composition and flags label/prediction mismatches
    for removal.

    Args:
        target_label: The label the backdoor is meant to flip predictions to (`BENIGN`).
    """

    def __init__(self, target_label: str) -> None:
        self._target_label = target_label

    def label_clusters(self, cluster_summary: pd.DataFrame) -> Dict[int, str]:
        """Assign BENIGN_LIKE/ATTACK_LIKE to every cluster by majority of its true composition:
        benign side `n_benign - n_poisoned`, attack side `n_attack_total + n_poisoned`. The noise
        cluster (`-1`) is tagged `NOISE` and excluded from labeling.

        This stands in for a reviewer who sees the records' behaviour, not their flipped label,
        so poisoned rows must move to the attack side rather than merely leave the benign side.
        Removing them from benign without adding them to attack lets a handful of genuine benign rows
        outvote an attack side of zero, flipping a near-pure poison cluster to BENIGN_LIKE (observed:
        306 poisoned + 5 benign rows labeled BENIGN_LIKE).

        Args:
            cluster_summary: Output of `HdbscanClusterer.summarize_clusters` /
                `LaborKMeansClusterer.summarize_clusters` - one row per cluster with `cluster`,
                `n_benign`, `n_attack_total`, `n_poisoned` columns.

        Returns:
            Mapping of cluster id to `"BENIGN_LIKE"`, `"ATTACK_LIKE"`, or `"NOISE"`. Ties (true
            benign count exactly equal to attack count) resolve to `ATTACK_LIKE` (conservative).
        """
        labels: Dict[int, str] = {}
        for _, row in cluster_summary.iterrows():
            cluster_id = int(row["cluster"])
            if cluster_id == NOISE_CLUSTER_ID:
                labels[cluster_id] = NOISE
                continue
            true_benign = row["n_benign"] - row["n_poisoned"]
            true_attack = row["n_attack_total"] + row["n_poisoned"]
            labels[cluster_id] = BENIGN_LIKE if true_benign > true_attack else ATTACK_LIKE
        return labels

    def compute_removal_mask(
        self, cluster_ids: np.ndarray, cluster_labels: Dict[int, str], predicted_labels: np.ndarray,
    ) -> np.ndarray:
        """Flag rows whose cluster is `ATTACK_LIKE` while the model predicts `target_label`.

        Args:
            cluster_ids: Cluster id per row (same order as `predicted_labels`).
            cluster_labels: Output of `label_clusters`.
            predicted_labels: Model-predicted label per row (string labels, same order as
                `cluster_ids`).

        Returns:
            Boolean array, True for rows to remove. `NOISE`-cluster rows are never flagged (they
            never map to `ATTACK_LIKE`).
        """
        row_cluster_label = pd.Series(cluster_ids).map(cluster_labels).to_numpy()
        return (row_cluster_label == ATTACK_LIKE) & (predicted_labels == self._target_label)

    def summarize_removal(self, removal_mask: np.ndarray, is_poisoned: np.ndarray) -> Dict[str, int]:
        """Count removal outcomes against ground truth.

        Args:
            removal_mask: Output of `compute_removal_mask`.
            is_poisoned: Ground-truth poisoned flag per row (same order as `removal_mask`).

        Returns:
            Dict with `n_removed`, `n_removed_correct` (removed and truly poisoned),
            `n_removed_wrong` (removed but not poisoned), `n_missed` (poisoned but not removed),
            and `n_total_poisoned`.
        """
        n_removed = int(removal_mask.sum())
        n_removed_correct = int((removal_mask & is_poisoned).sum())
        n_removed_wrong = int((removal_mask & ~is_poisoned).sum())
        n_missed = int((~removal_mask & is_poisoned).sum())
        return {
            "n_removed": n_removed,
            "n_removed_correct": n_removed_correct,
            "n_removed_wrong": n_removed_wrong,
            "n_missed": n_missed,
            "n_total_poisoned": int(is_poisoned.sum()),
        }
