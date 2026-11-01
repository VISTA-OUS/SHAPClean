"""Component 1 - PCA + HDBSCAN clustering, an alternative to `LaborKMeansClusterer`
(Luo et al. 2025, Section 4.1.3's paper-specified k-means).

Adopted after an empirical comparison on a stratified 150k-row sample of a poisoned CIC-IDS2017
training set: 8 fixed-`k` cluster-shape variants (`LaborKMeansClusterer`'s raw-78-feature k-means,
k-means on a 30-feature selection with/without PCA, GMM diag/full covariance, Bisecting K-Means,
BIRCH) all landed in the 86-90% BENIGN/ATTACK binary cluster-purity range, while density-based
HDBSCAN (no fixed `k`) on the same 30-feature selection reached ~95% regardless of PCA.
PCA(95% variance) + HDBSCAN was picked over plain HDBSCAN because it is ~5x faster for a
0.3-0.4pp purity cost, and over a UMAP+HDBSCAN variant that took 2238s just to embed the same
150k-row sample (infeasible at the 2.26M-row full-dataset scale) for *lower* purity than PCA. This
is a deviation from LABOR's own clustering spec, kept alongside (not replacing)
`LaborKMeansClusterer`, which remains the LABOR baseline.

**Fit-on-sample + `approximate_predict`, not a direct fit on all 2,262,300 rows** (added
2026-09-04, after a direct `sklearn.cluster.HDBSCAN.fit_predict` on the full dataset was killed
after 6+ hours stuck in a single step with no progress signal): HDBSCAN's tree-based
nearest-neighbor acceleration (KD-tree/Ball-tree) degrades sharply past ~10-15 dimensions ("curse
of dimensionality" - each query ends up visiting most of the tree, approaching brute-force cost),
and its minimum-spanning-tree construction step does not parallelize across `n_jobs`/
`core_dist_n_jobs` the way the core-distance computation does - the killed run's CPU-time tracked
wall-clock time almost exactly (effectively single-core) despite `n_jobs=-1`. Using the standalone
`hdbscan` package (not `sklearn.cluster.HDBSCAN`) instead: fits the full HDBSCAN hierarchy on a
bounded-size random sample (`fit_sample_size` - small enough to finish in bounded time, validated
empirically rather than guessed) with `prediction_data=True`, then assigns cluster ids to every
remaining row via `hdbscan.approximate_predict` - an approximate nearest-neighbor query against
the already-built tree, much cheaper per row than the initial fit. This is the standard way to
scale density-based clustering past the row count its exact fit can handle in bounded time; the
resulting cluster structure is only as good as the sample's ability to represent the full
dataset's density regions, which has not been separately validated here.

`fit_sample_size=None` fits the hierarchy on EVERY row instead (no sample, no `approximate_predict`).
The six-hour failure above was `sklearn.cluster.HDBSCAN` (Prim's MST); the switch to this library's
Boruvka KD-tree happened together with the switch to fit-on-sample, so the sampling was never shown
to be necessary for this library; the experiments fit on every row.
`prediction_data` is only built when sampling, since it exists solely to serve `approximate_predict`.

`row_normalize=True` L2-normalizes each row between scaling and PCA, which makes the pipeline EXACTLY
Kernel PCA with a cosine kernel - the variant the Active Paths paper uses - because the cosine kernel
is the linear kernel on L2-normalized vectors, and linear-kernel PCA is ordinary PCA by the
Gram-covariance duality. That second step is what matters here: the kernel formulation needs an n x n
Gram matrix (41 TB at 2,262,300 rows), this one stays O(n x d). Verified against
`KernelPCA(n_components=10, kernel="cosine")` on this project's SHAP table: max absolute difference
2.99e-14, and no row has zero norm (the only case where the two would part ways).
"""

from __future__ import annotations

import time
from typing import Iterable, List, Optional, Tuple

import hdbscan
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer, RobustScaler, StandardScaler

from src.core.logger import get_logger

logger = get_logger(__name__)


class HdbscanClusterer:
    """Fits StandardScaler -> PCA -> HDBSCAN on a random sample of rows, then extends cluster
    assignment to every remaining row via `hdbscan.approximate_predict`. Summarizes cluster
    composition against ground truth, matching `LaborKMeansClusterer`'s report shape.

    Args:
        feature_columns: Feature columns to cluster on (in the audit, the TreeSHAP attributions of
            the model's features, rather than every raw column as `LaborKMeansClusterer` uses).
        label_column: Name of the ground-truth label column.
        id_column: Name of the stable row-identifier column.
        target_label: Label counted as benign when summarizing a cluster's true composition - the
            label the backdoor flips predictions TO (`BENIGN` on CIC-IDS2017, `Normal`/`normal` on
            the three new datasets). Required rather than defaulted: a wrong value here does not
            fail, it silently marks every cluster as attack-like.
        pca_variance_retained: Minimum fraction of variance PCA must retain
            (`component1_poison_audit.clustering_hdbscan.pca_variance_retained`).
        fit_sample_size: Row count to fit the HDBSCAN hierarchy on
            (`component1_poison_audit.clustering_hdbscan.fit_sample_size`) - the rest of the
            dataset is assigned via `approximate_predict`, not refit. `None` fits on every row.
        min_cluster_size: HDBSCAN `min_cluster_size`
            (`component1_poison_audit.clustering_hdbscan.min_cluster_size`).
        min_samples: HDBSCAN `min_samples`
            (`component1_poison_audit.clustering_hdbscan.min_samples`).
        core_dist_n_jobs: Parallel worker count for HDBSCAN's core-distance computation
            (`component1_poison_audit.clustering_hdbscan.core_dist_n_jobs`) - does not parallelize
            the MST-construction step, so is not a substitute for bounding `fit_sample_size`.
        random_state: Seed for the fit-sample draw and PCA reproducibility (HDBSCAN itself takes
            no seed - the library is deterministic given its inputs).
        scaler_kind: `"standard"` (mean/std), `"robust"` (median/IQR, so heavy-tailed SHAP columns are
            not compressed by their own outliers), or `None` for no scaling.
        row_normalize: L2-normalize each row after scaling, turning the reduction step into cosine
            Kernel PCA (see the module docstring).
    """

    def __init__(
        self,
        feature_columns: List[str],
        label_column: str,
        id_column: str,
        target_label: str,
        pca_variance_retained: Optional[float],
        fit_sample_size: Optional[int],
        min_cluster_size: int,
        min_samples: int,
        core_dist_n_jobs: int,
        random_state: int,
        scaler_kind: Optional[str] = "standard",
        row_normalize: bool = False,
    ) -> None:
        self._feature_columns = feature_columns
        self._label_column = label_column
        self._id_column = id_column
        self._target_label = target_label
        self._fit_sample_size = fit_sample_size
        self._random_state = random_state
        if scaler_kind not in ("standard", "robust", None):
            raise ValueError(f"Invalid scaler_kind: {scaler_kind!r} (only 'standard', 'robust', None).")
        self._scaler = {"standard": StandardScaler, "robust": RobustScaler}[scaler_kind]() if scaler_kind else None
        self._normalizer = Normalizer(norm="l2") if row_normalize else None
        self._pca = (
            PCA(n_components=pca_variance_retained, random_state=random_state)
            if pca_variance_retained is not None else None
        )
        self._model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            core_dist_n_jobs=core_dist_n_jobs,
            prediction_data=fit_sample_size is not None,
        )
        self._n_fit_rows = 0
        self._n_predicted_rows = 0

    def fit_predict(self, train_frame: pd.DataFrame) -> np.ndarray:
        """Scale -> PCA-reduce -> fit HDBSCAN on a random sample, then `approximate_predict` the rest.

        Args:
            train_frame: Training data containing `label_column`, `id_column`, and all
                `feature_columns`.

        Returns:
            1D array of cluster ids (length `len(train_frame)`), aligned to `train_frame`'s row
            order. `-1` marks HDBSCAN noise points (no stable cluster membership), not a real
            cluster.
        """
        n_rows = len(train_frame)
        features = train_frame[self._feature_columns].to_numpy(dtype=float)

        if self._fit_sample_size is None:
            sample_size = n_rows
            fit_mask = np.ones(n_rows, dtype=bool)
        else:
            rng = np.random.RandomState(self._random_state)
            sample_size = min(self._fit_sample_size, n_rows)
            fit_positions = rng.choice(n_rows, size=sample_size, replace=False)
            fit_mask = np.zeros(n_rows, dtype=bool)
            fit_mask[fit_positions] = True
        self._n_fit_rows = int(sample_size)

        scaled_fit_features = (
            self._scaler.fit_transform(features[fit_mask]) if self._scaler is not None else features[fit_mask]
        )
        if self._normalizer is not None:
            scaled_fit_features = self._normalizer.transform(scaled_fit_features)
        reduced_fit_features = (
            self._pca.fit_transform(scaled_fit_features) if self._pca is not None else scaled_fit_features
        )
        logger.info(
            "HDBSCAN fit starting on %d sampled rows (%d dimensions after PCA)...", sample_size, reduced_fit_features.shape[1],
        )
        fit_start_time = time.time()
        cluster_ids = np.empty(n_rows, dtype=int)
        cluster_ids[fit_mask] = self._model.fit_predict(reduced_fit_features)
        logger.info("HDBSCAN fit done after %.1fs.", time.time() - fit_start_time)

        remaining_positions = np.where(~fit_mask)[0]
        self._n_predicted_rows = int(len(remaining_positions))
        if self._n_predicted_rows > 0:
            logger.info("approximate_predict starting on the %d remaining rows...", self._n_predicted_rows)
            predict_start_time = time.time()
            scaled_remaining_features = (
                self._scaler.transform(features[remaining_positions])
                if self._scaler is not None else features[remaining_positions]
            )
            if self._normalizer is not None:
                scaled_remaining_features = self._normalizer.transform(scaled_remaining_features)
            reduced_remaining_features = (
                self._pca.transform(scaled_remaining_features)
                if self._pca is not None else scaled_remaining_features
            )
            predicted_ids, _strengths = hdbscan.approximate_predict(self._model, reduced_remaining_features)
            cluster_ids[remaining_positions] = predicted_ids
            logger.info("approximate_predict done after %.1fs.", time.time() - predict_start_time)

        return cluster_ids

    @property
    def n_pca_components(self) -> int:
        """Dimensionality actually fed to HDBSCAN (only valid after `fit_predict`).

        With PCA enabled this is the component count kept to satisfy `pca_variance_retained`; with
        PCA disabled (`pca_variance_retained=None`) it is simply the input feature count, since the
        features go to HDBSCAN unreduced.

        Returns:
            Dimension count.
        """
        if self._pca is None:
            return len(self._feature_columns)
        return int(self._pca.n_components_)

    @property
    def fit_vs_predicted_counts(self) -> Tuple[int, int]:
        """Row counts split between the HDBSCAN fit sample and the `approximate_predict`-assigned
        remainder (only valid after `fit_predict` has been called).

        Returns:
            `(n_fit_rows, n_predicted_rows)`.
        """
        return self._n_fit_rows, self._n_predicted_rows

    def summarize_clusters(
        self, train_frame: pd.DataFrame, cluster_ids: np.ndarray, poisoned_sample_ids: Iterable[int],
    ) -> pd.DataFrame:
        """Build the per-cluster composition table: total / benign / attack / poisoned counts.

        Args:
            train_frame: Same frame passed to `fit_predict` (row order must match `cluster_ids`).
            cluster_ids: Output of `fit_predict` (may include the `-1` noise cluster).
            poisoned_sample_ids: Ground-truth poisoned `id_column` values (from the LABOR-poisoning
                manifest) - independent of the label column, since poisoned rows now carry the
                BENIGN label.

        Returns:
            One row per cluster (including `-1` for noise, if present), columns: `cluster`,
            `n_records`, `n_benign`, `n_attack_total`, `n_poisoned`; sorted by `cluster` ascending.
        """
        poisoned_id_set = set(poisoned_sample_ids)
        working_frame = train_frame[[self._id_column, self._label_column]].copy()
        working_frame["cluster"] = cluster_ids
        # target_label, NOT the fixed string "BENIGN": the IoT datasets name their benign class differently
        # (Normal/normal). With "BENIGN" hardcoded, n_benign would be 0 in EVERY cluster, so
        # ClusterLabelPoisonRemover.label_clusters would call every cluster ATTACK_LIKE and the removal rule would remove
        # every row the model calls benign, although the poisoned clusters themselves are separated perfectly.
        working_frame["is_benign"] = working_frame[self._label_column] == self._target_label
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
