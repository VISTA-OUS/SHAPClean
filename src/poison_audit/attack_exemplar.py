"""Component 1 - how close each record comes to a TRUSTED attack record: the record-matching primitive of layer B.

Idea. A poisoned record is not "a flow that resembles DDoS in general" - it is one concrete attack flow with a few
feature values overwritten, and attack traffic is highly repetitive. So attack-labelled flows of the same kind, whose
labels the poison never forges, should include one that matches it on nearly every feature and differs only where the
trigger wrote. A benign cluster that merely resembles attack clusters by its medians should have no such near-twin.
Class-level summaries are far too coarse for this (every class spans most of each feature's range); this module works
record against record.

"Nearly equal" is judged on mid-rank percentiles over the whole training set, the scale layer A uses: identical values
get identical percentiles, so discrete features compare exactly, and continuous features (IATs, rates) that two flows
of the same tool rarely share to the microsecond compare within a tolerance `delta` of percentile.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.core.logger import get_logger
from src.poison_audit.cluster_neighbor_labeler import TrainingMatrix

logger = get_logger(__name__)


class RankTransformer:
    """Maps raw feature values to mid-rank percentiles of the training set (NaN -> 0.5, as in layer A's profiles).

    Args:
        matrix: Training set whose rows define the population.
    """

    def __init__(self, matrix: TrainingMatrix) -> None:
        self._sorted = [np.sort(matrix.features[:, column][~np.isnan(matrix.features[:, column])]).astype(np.float64)
                        for column in range(matrix.features.shape[1])]

    def population(self, column: int) -> np.ndarray:
        """Sorted non-NaN values of one feature over the training set.

        Args:
            column: Feature column.

        Returns:
            Sorted float64 array (shared, do not modify).
        """
        return self._sorted[column]

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Percentiles of some rows.

        Args:
            features: Rows in training-matrix column order.

        Returns:
            Same shape, float32 values in [0, 1].
        """
        ranks = np.empty(features.shape, dtype=np.float32)
        for column, population in enumerate(self._sorted):
            values = features[:, column].astype(np.float64)
            below = np.searchsorted(population, values, side="left")
            at_or_below = np.searchsorted(population, values, side="right")
            column_ranks = (below + at_or_below) / (2.0 * len(population))
            column_ranks[np.isnan(values)] = 0.5
            ranks[:, column] = column_ranks
        return ranks


class AttackExemplarMatcher:
    """For each query record, the pool record that agrees with it on the most features.

    Args:
        tolerances: Percentile tolerances at which "nearly equal" is judged; each gets its own best match.
        block_size: Query rows per block.
        pool_block_size: Pool rows compared at once (memory is `block_size x pool_block_size x features` float32);
            None compares the whole pool at once.
    """

    def __init__(self, tolerances: List[float], block_size: int, pool_block_size: Optional[int] = None) -> None:
        self._tolerances = sorted(float(value) for value in tolerances)
        self._block_size = block_size
        self._pool_block_size = pool_block_size

    def match(self, queries: np.ndarray, pool: np.ndarray) -> Dict[float, Dict[str, np.ndarray]]:
        """Best-agreeing pool record per query, at every tolerance.

        Ties on the number of agreeing features go to the pool record with the smaller L1 percentile distance, then
        to the earlier one.

        Args:
            queries: Query rows as percentiles.
            pool: Trusted attack rows as percentiles.

        Returns:
            Tolerance -> {`n_agree` (per query), `best` (pool index per query), `mismatch` (bool, query x feature,
            features that differ from that best record)}.
        """
        n_queries, n_features = queries.shape
        pool_step = self._pool_block_size or len(pool)
        results = {
            tolerance: {"n_agree": np.zeros(n_queries, dtype=np.int32), "best": np.zeros(n_queries, dtype=np.int64),
                        "mismatch": np.zeros((n_queries, n_features), dtype=bool)}
            for tolerance in self._tolerances
        }
        for start in range(0, n_queries, self._block_size):
            stop = min(start + self._block_size, n_queries)
            rows = np.arange(stop - start)
            # Score = matching features; ties go to the closer row in L1 (the distance is < n_features, so it never outweighs a match).
            best_score = {tolerance: np.full(stop - start, -np.inf) for tolerance in self._tolerances}
            for pool_start in range(0, len(pool), pool_step):
                difference = np.abs(queries[start:stop, None, :] - pool[None, pool_start:pool_start + pool_step, :])
                distance = difference.sum(axis=2)
                for tolerance, result in results.items():
                    agree = (difference <= tolerance).sum(axis=2)
                    score = agree * (n_features + 1.0) - distance
                    best = np.argmax(score, axis=1)
                    better = score[rows, best] > best_score[tolerance]
                    best_score[tolerance][better] = score[rows, best][better]
                    result["n_agree"][start:stop][better] = agree[rows, best][better]
                    result["best"][start:stop][better] = pool_start + best[better]
        for tolerance, result in results.items():
            result["mismatch"] = np.abs(queries - pool[result["best"]]) > tolerance
        return results
