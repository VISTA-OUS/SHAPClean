"""Component 1 - K-means clustering step of the LABOR pipeline (Luo et al. 2025, Section 3.2
"Unsupervised Training" + Section 4.1.3 "Clustering algorithm").

Only the clustering step itself - cluster -> ground-truth label tagging ("Human Intelligence
Feedback", Section 3.2) and the poisoned-vs-benign consistency check (Section 3.2 "Identify
Poisoned Sample") are separate, later steps of the LABOR pipeline, not implemented here.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class LaborKMeansClusterer:
    """Fits paper-parameter k-means and summarizes cluster composition against ground truth.

    Paper parameters (Section 4.1.3): k-means++ initialization, `max_iter=200`, `tol=0.0001`,
    Euclidean distance (sklearn `KMeans` default), `k` set to the dataset's number of categories.
    The paper does not state whether features were scaled before clustering; this class applies
    `StandardScaler` first since the 78 raw CICFlowMeter features span wildly different scales
    (e.g. `Flow Bytes/s` in the millions vs. 0/1 flag counts) and Euclidean-distance k-means on
    unscaled features would be dominated by the largest-magnitude columns alone.

    Args:
        exclude_columns: Non-feature columns to drop before clustering (identifier/label columns).
        label_column: Name of the ground-truth label column.
        id_column: Name of the stable row-identifier column.
        n_clusters: Number of clusters (`component1_poison_audit.clustering.n_clusters`).
        init: KMeans initialization strategy (`"k-means++"`).
        max_iter: Maximum iterations (`component1_poison_audit.clustering.max_iter`).
        tol: Convergence tolerance (`component1_poison_audit.clustering.tol`).
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        exclude_columns: List[str],
        label_column: str,
        id_column: str,
        n_clusters: int,
        init: str,
        max_iter: int,
        tol: float,
        random_state: int,
    ) -> None:
        self._exclude_columns = exclude_columns
        self._label_column = label_column
        self._id_column = id_column
        self._scaler = StandardScaler()
        self._model = KMeans(
            n_clusters=n_clusters, init=init, max_iter=max_iter, tol=tol, random_state=random_state,
        )

    def fit_predict(self, train_frame: pd.DataFrame) -> np.ndarray:
        """Scale features and fit k-means, returning one cluster id per row.

        Args:
            train_frame: Training data containing `label_column`, `id_column`, and all feature
                columns.

        Returns:
            1D array of cluster ids (length `len(train_frame)`), aligned to `train_frame`'s row
            order.
        """
        feature_columns = [c for c in train_frame.columns if c not in self._exclude_columns]
        features = train_frame[feature_columns].to_numpy(dtype=float)
        scaled_features = self._scaler.fit_transform(features)
        return self._model.fit_predict(scaled_features)

    def summarize_clusters(
        self, train_frame: pd.DataFrame, cluster_ids: np.ndarray, poisoned_sample_ids: Iterable[int],
        benign_label: str = "BENIGN",
    ) -> pd.DataFrame:
        """Build the per-cluster composition table: total / benign / attack / poisoned counts.

        Args:
            train_frame: Same frame passed to `fit_predict` (row order must match `cluster_ids`).
            cluster_ids: Output of `fit_predict`.
            poisoned_sample_ids: Ground-truth poisoned `id_column` values (from the LABOR-poisoning
                manifest) - independent of the label column, since poisoned rows now carry the
                BENIGN label.
            benign_label: The benign (target) label - CIC-IDS2017's `BENIGN` by default; the other datasets name it
                differently (`Normal`, `normal`).

        Returns:
            One row per cluster, columns: `cluster`, `n_records`, `n_benign`, `n_attack_total`,
            `n_poisoned`; sorted by `cluster` ascending.
        """
        poisoned_id_set = set(poisoned_sample_ids)
        working_frame = train_frame[[self._id_column, self._label_column]].copy()
        working_frame["cluster"] = cluster_ids
        working_frame["is_benign"] = working_frame[self._label_column] == benign_label
        working_frame["is_poisoned"] = working_frame[self._id_column].isin(poisoned_id_set)

        rows = []
        for cluster_id, group in working_frame.groupby("cluster"):
            rows.append({
                "cluster": int(cluster_id),
                "n_records": int(len(group)),
                "n_benign": int(group["is_benign"].sum()),
                "n_attack_total": int((~group["is_benign"]).sum()),
                "n_poisoned": int(group["is_poisoned"].sum()),
            })
        return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
