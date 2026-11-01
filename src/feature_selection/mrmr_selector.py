"""mRMR feature ranking (Peng, Long, Ding, 2005, IEEE TPAMI) over a frame held in memory.

Mutual information for BOTH relevance and redundancy (Peng et al.'s original MID formulation, not
the `mrmr-selection` package's default F-statistic/Pearson shortcuts), one stratified subsample per
seed, and Borda count (mean rank position) to aggregate the per-seed rankings. Each seed also
reports how many poisoned rows its subsample happened to draw.

The callback parameter NAMES matter: `mrmr-selection` 0.2.8 invokes them by keyword as
`relevance(X=, y=)` and `redundancy(X=, target_column=, features=)`. A copy of this method that
names them anything else raises `TypeError` before computing anything.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from mrmr import mrmr_classif
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from src.core.logger import get_logger

logger = get_logger(__name__)


class MrmrFeatureSelector:
    """Ranks features by mRMR across calibration seeds and aggregates with Borda count.

    Args:
        label_column: Name of the label column (the classification target).
        sample_size: Max rows per seed, drawn stratified by label. mRMR's redundancy term calls
            `mutual_info_regression` once per remaining feature at every step, so cost grows
            super-linearly in rows; the value used is `selection_feature.techniques.mrmr.sample_size`.
        denominator: MID-vs-MIQ denominator aggregation over already-selected features
            (`"mean"` or `"max"`), from the same config block.
        n_jobs: Parallel jobs for the mutual-information estimators.
    """

    def __init__(self, label_column: str, sample_size: int, denominator: str, n_jobs: int) -> None:
        self._label_column = label_column
        self._sample_size = int(sample_size)
        self._denominator = denominator
        self._n_jobs = int(n_jobs)

    def stratified_sample(self, frame: pd.DataFrame, seed: int) -> pd.DataFrame:
        """Draw at most `sample_size` rows, stratified by label.

        Args:
            frame: Source frame.
            seed: Seed for the draw.

        Returns:
            The sample, or `frame` itself when it is already small enough.
        """
        if len(frame) <= self._sample_size:
            return frame.reset_index(drop=True)
        fraction = self._sample_size / len(frame)
        sampled = (
            frame.groupby(self._label_column, group_keys=False)
            .apply(lambda group: group.sample(frac=fraction, random_state=seed))
        )
        return sampled.reset_index(drop=True)

    def rank(
        self,
        frame: pd.DataFrame,
        feature_columns: List[str],
        seeds: Sequence[int],
        poisoned_ids: Optional[Sequence[int]] = None,
        id_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Rank every feature, once per seed, then aggregate by Borda count.

        Args:
            frame: Data to rank on - clean or poisoned; mRMR reads whatever labels it carries.
            feature_columns: Candidate features.
            seeds: Calibration seeds, one subsample and one ranking each.
            poisoned_ids: Ground-truth poisoned row ids, for the per-seed enrichment diagnostic.
            id_column: Identifier column, required when `poisoned_ids` is given.

        Returns:
            Dict with `consensus_ranking` (best first), `per_seed_rankings` (seed -> ranking) and
            `per_seed_sample` (one record per seed: rows drawn, poisoned rows drawn, enrichment
            over the frame's own poison rate, seconds).

        Raises:
            ValueError: If `poisoned_ids` is given without `id_column`.
        """
        if poisoned_ids is not None and id_column is None:
            raise ValueError("poisoned_ids needs id_column to count the poisoned rows in the sample.")
        poison_set = set(int(sample_id) for sample_id in poisoned_ids) if poisoned_ids else set()
        base_rate = len(poison_set) / len(frame) if poison_set else 0.0

        per_seed_rankings: Dict[int, List[str]] = {}
        per_seed_sample: List[Dict[str, Any]] = []
        for seed in seeds:
            sampled = self.stratified_sample(frame, int(seed))
            n_poisoned = (
                int(sampled[id_column].isin(poison_set).sum()) if poison_set else 0
            )
            start_time = time.time()
            per_seed_rankings[int(seed)] = self._rank_one_seed(sampled, feature_columns, int(seed))
            elapsed = time.time() - start_time
            observed_rate = n_poisoned / len(sampled) if len(sampled) else 0.0
            record = {
                "seed": int(seed),
                "n_rows": int(len(sampled)),
                "n_poisoned_in_sample": n_poisoned,
                "enrichment_vs_base_rate": round(observed_rate / base_rate, 2) if base_rate else None,
                "seconds": round(elapsed, 1),
            }
            per_seed_sample.append(record)
            logger.info("mRMR seed=%d done after %.1fs | sample %s", seed, elapsed, record)

        return {
            "consensus_ranking": self._aggregate_borda(per_seed_rankings, feature_columns),
            "per_seed_rankings": per_seed_rankings,
            "per_seed_sample": per_seed_sample,
        }

    def _rank_one_seed(self, sampled: pd.DataFrame, feature_columns: List[str], seed: int) -> List[str]:
        """Run mRMR once, returning the full ranking for this seed.

        Args:
            sampled: Already-subsampled data for this seed.
            feature_columns: Candidate features.
            seed: Seed, also passed into the mutual-information estimators.

        Returns:
            Every feature, best first.
        """
        def relevance_func(X: pd.DataFrame, y: pd.Series, _seed: int = seed) -> pd.Series:
            scores = mutual_info_classif(X, y, random_state=_seed, n_jobs=self._n_jobs)
            return pd.Series(scores, index=X.columns)

        def redundancy_func(
            X: pd.DataFrame, target_column: str, features: List[str], _seed: int = seed,
        ) -> pd.Series:
            scores = mutual_info_regression(
                X[features], X[target_column], random_state=_seed, n_jobs=self._n_jobs,
            )
            return pd.Series(scores, index=features)

        return mrmr_classif(
            X=sampled[feature_columns], y=sampled[self._label_column], K=len(feature_columns),
            relevance=relevance_func, redundancy=redundancy_func,
            denominator=self._denominator, show_progress=False,
        )

    @staticmethod
    def _aggregate_borda(
        per_seed_rankings: Dict[int, List[str]], feature_columns: List[str],
    ) -> List[str]:
        """Aggregate per-seed rankings into one by mean rank position.

        Args:
            per_seed_rankings: seed -> ranked feature list, best first.
            feature_columns: All candidate features.

        Returns:
            Features sorted by mean rank position, ascending.
        """
        mean_ranks: Dict[str, float] = {}
        for feature in feature_columns:
            positions = [
                ranking.index(feature) for ranking in per_seed_rankings.values() if feature in ranking
            ]
            mean_ranks[feature] = float(np.mean(positions)) if positions else float(len(feature_columns))
        return sorted(mean_ranks, key=mean_ranks.get)
