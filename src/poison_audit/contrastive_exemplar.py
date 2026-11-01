"""Component 1 - the exemplars of layer B: for every record of a short-listed cluster, the most similar TRUSTED attack
record and the most similar REFERENCE benign record, and the features where each one differs.

Why records, and why both sides.
- A poisoned record is one concrete attack flow, relabelled BENIGN and perhaps with a few feature values
  overwritten. Attack traffic is repetitive and the poison never forges an attack label, so some attack-labelled
  record matches it on every feature the attacker left alone. When a pattern was overwritten, the SAME features
  differ in every record of the group, and hold the same values there.
- Resembling attack records is not enough. The benign clusters that reach layer B were short-listed precisely
  because they resemble attack clusters, and many have near-identical attack records. What separates them is the
  other side: a genuine benign record has a benign record, from the part of the training set that raised no
  suspicion, that matches it at least as closely.
- Class-level summaries are far too coarse: every class spans most of each feature's range.

Pools.
- Trusted attack: every attack-labelled row.
- Reference benign: benign-labelled rows outside every short-listed cluster, so that no poison fragment layer A
  found serves as the benign reference of its sibling fragments. Whether noise rows (-1) may serve too is the
  caller's choice (`include_noise`, or a per-row `benign_reference` mask).

"Nearly equal" is judged on mid-rank percentiles of the whole training set, the scale of layer A
(`RankTransformer`), so discrete features compare exactly and continuous ones within `tolerance` percentile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.core.logger import get_logger
from src.poison_audit.attack_exemplar import AttackExemplarMatcher, RankTransformer
from src.poison_audit.cluster_neighbor_labeler import TrainingMatrix

logger = get_logger(__name__)


@dataclass(frozen=True)
class RecordExemplars:
    """Nearest trusted attack row and nearest reference benign row of some query rows, with their values, so that
    nothing downstream needs the training matrix.

    Attributes:
        rows: Query rows (positions in the training matrix).
        values: `(n_queries, n_features)` raw values of the queries.
        attack_rows: Nearest attack-labelled row of each query.
        attack_classes: Its observed attack class.
        attack_values: Its raw values.
        attack_mismatch: `(n_queries, n_features)` - features that differ from it.
        benign_rows: Nearest reference benign row of each query.
        benign_values: Its raw values.
        benign_mismatch: `(n_queries, n_features)` - features that differ from it.
    """

    rows: np.ndarray
    values: np.ndarray
    attack_rows: np.ndarray
    attack_classes: np.ndarray
    attack_values: np.ndarray
    attack_mismatch: np.ndarray
    benign_rows: np.ndarray
    benign_values: np.ndarray
    benign_mismatch: np.ndarray

    @property
    def attack_distance(self) -> np.ndarray:
        """Features differing from the nearest attack row, per query.

        Returns:
            Integer array.
        """
        return self.attack_mismatch.sum(axis=1)

    @property
    def benign_distance(self) -> np.ndarray:
        """Features differing from the nearest reference benign row, per query.

        Returns:
            Integer array.
        """
        return self.benign_mismatch.sum(axis=1)

    def select(self, rows: np.ndarray) -> "RecordExemplars":
        """The exemplars of some of the query rows.

        Args:
            rows: Training-matrix positions, each one of `self.rows`.

        Returns:
            Exemplars in the order of `rows`.

        Raises:
            KeyError: If a row was not queried.
        """
        index = {int(row): position for position, row in enumerate(self.rows)}
        positions = np.array([index[int(row)] for row in rows], dtype=int)
        return RecordExemplars(**{name: getattr(self, name)[positions] for name in self.__dataclass_fields__})


class ContrastiveExemplarFinder:
    """Finds, for query rows, the nearest trusted attack row and the nearest reference benign row.

    Args:
        matrix: The source's poisoned training set.
        ranker: Percentile transform of the same set (shared with the trusted rows of layer B).
        row_clusters: Cluster id per row (`-1` = noise).
        short_listed: Cluster ids of layer A's short list (never used as benign references).
        target_label: The benign/target label.
        include_noise: Keep benign-labelled noise rows (`-1`) among the benign references. Noise can hold poison
            that clustering left out, which then serves as the "benign" reference of its own sibling rows.
        tolerance: Percentile difference up to which two values count as nearly equal.
        query_block_size: Query rows compared at once.
        pool_block_size: Pool rows compared at once (bounds memory with `query_block_size`).
        benign_reference: Per row, whether it may serve as a benign reference. When given it replaces the rule above
            (`short_listed`, `include_noise`), so a caller can apply its own reference policy; None keeps the rule.
    """

    def __init__(
        self,
        matrix: TrainingMatrix,
        ranker: RankTransformer,
        row_clusters: np.ndarray,
        short_listed: np.ndarray,
        target_label: str,
        include_noise: bool,
        tolerance: float,
        query_block_size: int,
        pool_block_size: int,
        benign_reference: Optional[np.ndarray] = None,
    ) -> None:
        self._matrix = matrix
        self._tolerance = tolerance
        self._ranker = ranker
        benign = matrix.rows_labelled(target_label)
        self._attack_pool_rows = np.flatnonzero(~benign)
        if benign_reference is None:
            reference = benign & ~np.isin(row_clusters, short_listed)
            if not include_noise:
                reference &= row_clusters >= 0
        else:
            reference = benign & benign_reference
        self._benign_pool_rows = np.flatnonzero(reference)
        self._attack_pool = self._ranker.transform(matrix.features[self._attack_pool_rows])
        self._benign_pool = self._ranker.transform(matrix.features[self._benign_pool_rows])
        self._matcher = AttackExemplarMatcher([tolerance], query_block_size, pool_block_size)
        logger.info(
            "Exemplar pools: %d attack-labelled rows, %d benign reference rows.",
            len(self._attack_pool_rows), len(self._benign_pool_rows),
        )

    def find(self, rows: np.ndarray) -> RecordExemplars:
        """Both exemplars of some rows.

        Args:
            rows: Query rows (training-matrix positions).

        Returns:
            Their exemplars.
        """
        rows = np.asarray(rows)
        features = self._matrix.features
        queries = self._ranker.transform(features[rows])
        attack = self._matcher.match(queries, self._attack_pool)[self._tolerance]
        benign = self._matcher.match(queries, self._benign_pool)[self._tolerance]
        attack_rows = self._attack_pool_rows[attack["best"]]
        benign_rows = self._benign_pool_rows[benign["best"]]
        return RecordExemplars(
            rows=rows, values=features[rows],
            attack_rows=attack_rows,
            attack_classes=np.array(self._matrix.label_names)[self._matrix.label_codes[attack_rows]],
            attack_values=features[attack_rows], attack_mismatch=attack["mismatch"],
            benign_rows=benign_rows, benign_values=features[benign_rows], benign_mismatch=benign["mismatch"],
        )
