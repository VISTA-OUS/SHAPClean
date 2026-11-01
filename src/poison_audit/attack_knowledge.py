"""Component 1 - what layer B learns "an attack record looks like" from, induced from ONE dataset's own trusted rows.

Why induced, not written. Layer B has to run on every dataset the pipeline audits, each with its own feature set and
attack classes, so nothing about a dataset's attacks is written by hand: the reference model is learned from the
audited dataset itself.

What the defender can trust (dirty-label threat model: the poison relabels attack flows TO the benign label):
- every attack-labelled row;
- benign-labelled rows outside layer A's short list - the short list holds every poison fragment layer A found.
`TrustedTrainingSet` holds exactly these rows; `XgbAttackClassifier` is learned from them only. Record-level retrieval
(nearest trusted attack / benign flow of each record) lives in `contrastive_exemplar.py`.

All features are handled as mid-rank percentiles of the whole training set (`RankTransformer`, the scale of layer A),
so NaN and infinite values need no dataset-specific handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import xgboost as xgb

from src.core.logger import get_logger
from src.poison_audit.attack_exemplar import RankTransformer
from src.poison_audit.cluster_neighbor_labeler import TrainingMatrix

logger = get_logger(__name__)


@dataclass(frozen=True)
class TrustedTrainingSet:
    """Trusted rows of one training set, as percentiles, with their observed labels.

    Attributes:
        rows: Training-matrix positions.
        ranks: `(n_rows, n_features)` percentiles.
        labels: Observed label per row (attack classes and the target label).
        target_label: The benign/target label.
    """

    rows: np.ndarray
    ranks: np.ndarray
    labels: np.ndarray
    target_label: str

    @property
    def is_attack(self) -> np.ndarray:
        """Whether each row is attack-labelled.

        Returns:
            Boolean array.
        """
        return self.labels != self.target_label

    @classmethod
    def build(
        cls,
        matrix: TrainingMatrix,
        ranker: RankTransformer,
        row_clusters: np.ndarray,
        short_listed: np.ndarray,
        target_label: str,
        include_noise: bool,
        max_rows_per_class: Optional[int],
        random_state: int,
    ) -> "TrustedTrainingSet":
        """Every attack-labelled row and every benign-labelled row outside the short list, capped per class.

        Args:
            matrix: Training set.
            ranker: Percentile transform of the same set.
            row_clusters: Cluster id per row (`-1` = noise).
            short_listed: Layer A's short-listed cluster ids.
            target_label: The benign/target label.
            include_noise: Keep benign-labelled noise rows (they may hold poison that clustering left out).
            max_rows_per_class: Rows kept per class at most (a random subset beyond that - runtime only); None keeps
                every row.
            random_state: Seed of the subset.

        Returns:
            The trusted set.
        """
        benign = matrix.rows_labelled(target_label)
        keep = ~benign | (benign & ~np.isin(row_clusters, short_listed))
        if not include_noise:
            keep &= ~benign | (row_clusters >= 0)
        rng = np.random.RandomState(random_state)
        chosen = []
        for code in np.unique(matrix.label_codes[keep]):
            rows = np.flatnonzero(keep & (matrix.label_codes == code))
            capped = max_rows_per_class is not None and len(rows) > max_rows_per_class
            chosen.append(rng.choice(rows, max_rows_per_class, replace=False) if capped else rows)
        rows = np.sort(np.concatenate(chosen))
        labels = np.array(matrix.label_names, dtype=object)[matrix.label_codes[rows]]
        logger.info("Trusted set: %d rows (%d with an attack label).", len(rows), int((labels != target_label).sum()))
        return cls(rows=rows, ranks=ranker.transform(matrix.features[rows]), labels=labels, target_label=target_label)


class XgbAttackClassifier:
    """Attack probability from an XGBoost model trained on trusted rows, with the architecture of the audited model
    (`backdoor_eval.model_params`) and the two classes weighted equally, so that 0.5 is the balanced decision point.

    Args:
        model_params: XGBoost constructor kwargs.
        random_state: Seed.
    """

    def __init__(self, model_params: Dict[str, object], random_state: int) -> None:
        self._model = xgb.XGBClassifier(**model_params, random_state=random_state, eval_metric="logloss")

    def fit(self, trusted: TrustedTrainingSet) -> "XgbAttackClassifier":
        """Fit attack vs benign on every trusted row.

        Args:
            trusted: Trusted rows.

        Returns:
            Self.
        """
        attack = trusted.is_attack
        share = float(attack.mean())
        weights = np.where(attack, 0.5 / share, 0.5 / (1.0 - share))
        self._model.fit(trusted.ranks, attack.astype(int), sample_weight=weights)
        logger.info("Reference XGBoost: %d trusted rows (%.1f%% with an attack label).", len(attack), 100 * share)
        return self

    def attack_probability(self, ranks: np.ndarray) -> np.ndarray:
        """P(attack) per record.

        Args:
            ranks: Records as percentiles.

        Returns:
            Probabilities.
        """
        return self._model.predict_proba(ranks)[:, 1]
