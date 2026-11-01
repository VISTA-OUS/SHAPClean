"""Component 1 (poison audit) - builds a fixed-value trigger backdoor dataset: a specified
feature -> value dict (`trigger_spec` of a trigger block in `configs/config.yaml`) is injected
verbatim into a random subset of one source label's training rows (dirty-label, source-specific),
plus the label flipped to target_label - the BadNets design (Gu, Dolan-Gavitt, Garg 2017) with a
concrete multi-feature trigger. Every trigger family of the experiments (representative,
distribution-matching, explanation-guided) differs only in the values of `trigger_spec`.

Every row selected here has its `trigger_spec` feature values overwritten - selection is a plain
random draw over `source_label` rows, since injection means every such row is equally eligible.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


class FixedTriggerPoisonDatasetBuilder:
    """Injects a fixed `trigger_spec` (feature -> value) into `source_label` rows and flips their
    label to `target_label`.

    Args:
        trigger_spec: Feature name -> value to overwrite on every poisoned/triggered row.
        source_label: The single attack class eligible for poisoning.
        target_label: Label assigned to poisoned rows (the backdoor's intended misclassification).
        label_column: Name of the label column.
        sample_id_column: Name of the stable row-id column (used to address rows to poison).
    """

    def __init__(
        self,
        trigger_spec: Dict[str, float],
        source_label: str,
        target_label: str,
        label_column: str,
        sample_id_column: str,
    ) -> None:
        self._trigger_spec = trigger_spec
        self._source_label = source_label
        self._target_label = target_label
        self._label_column = label_column
        self._sample_id_column = sample_id_column

    def compute_target_label_reference_medians(self, train_frame: pd.DataFrame) -> Dict[str, float]:
        """Median of `target_label` rows on each trigger feature - a sanity reference to compare
        against the (externally-specified) `trigger_spec`; never used to select or filter rows.

        Args:
            train_frame: Training data containing `label_column` and every `trigger_spec` feature.

        Returns:
            Mapping of feature name to its target_label-distribution median.
        """
        target_frame = train_frame[train_frame[self._label_column] == self._target_label]
        return {feature: float(target_frame[feature].median()) for feature in self._trigger_spec}

    def count_natural_matches(self, train_frame: pd.DataFrame) -> Dict[str, Any]:
        """How many rows already carry the whole `trigger_spec` before anything is injected.

        Two numbers decide whether a candidate trigger is usable at all. A source-label row that
        already matches would end up in `build_trigger_test` without ever having been injected, which
        makes ASR count rows the attacker never touched. A target-label (BENIGN) row that matches is
        harmless - it is what makes the trigger's values realistic rather than alien - but its count
        says how much genuine traffic the model must learn to keep classifying as BENIGN.

        Args:
            train_frame: Clean training data containing `label_column` and every trigger feature.

        Returns:
            Dict with `n_rows_all`, `n_rows_source`, `n_rows_target`, `pct_of_source`, `pct_of_target`,
            `other_labels` (label -> count for matching rows that are neither source nor target), and
            `per_feature` (feature -> its own source/target match percentages).
        """
        match = pd.Series(True, index=train_frame.index)
        for feature, value in self._trigger_spec.items():
            match &= train_frame[feature] == value
        is_source = train_frame[self._label_column] == self._source_label
        is_target = train_frame[self._label_column] == self._target_label
        n_source, n_target = int(is_source.sum()), int(is_target.sum())
        return {
            "n_rows_all": int(match.sum()),
            "n_rows_source": int((match & is_source).sum()),
            "n_rows_target": int((match & is_target).sum()),
            "pct_of_source": round(100.0 * float((match & is_source).sum()) / n_source, 6) if n_source else 0.0,
            "pct_of_target": round(100.0 * float((match & is_target).sum()) / n_target, 6) if n_target else 0.0,
            "other_labels": {
                str(label): int(count)
                for label, count in train_frame.loc[match & ~is_source & ~is_target, self._label_column]
                .value_counts().items()
            },
            "per_feature": {
                feature: {
                    "pct_of_source": round(100.0 * float(((train_frame[feature] == value) & is_source).sum()) / n_source, 6) if n_source else 0.0,
                    "pct_of_target": round(100.0 * float(((train_frame[feature] == value) & is_target).sum()) / n_target, 6) if n_target else 0.0,
                }
                for feature, value in self._trigger_spec.items()
            },
        }

    def build_poisoned_train(
        self, train_frame: pd.DataFrame, poison_rate: float, seed: int,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Randomly select `poison_rate` of `source_label` training rows, overwrite their
        `trigger_spec` feature values, and flip their label to `target_label`.

        Args:
            train_frame: Clean training data.
            poison_rate: Fraction of `source_label` rows in `train_frame` to poison (e.g. 0.01 = 1%).
            seed: Seed for the row-selection draw.

        Returns:
            Tuple of (poisoned training frame, sorted array of `sample_id_column` values that were
            poisoned - the ground truth for downstream CDA/ASR/clustering evaluation).

        Raises:
            ValueError: If `train_frame` has no rows labeled `source_label` to sample from.
        """
        source_mask = train_frame[self._label_column] == self._source_label
        source_ids = train_frame.loc[source_mask, self._sample_id_column].to_numpy()
        if len(source_ids) == 0:
            raise ValueError(
                f"No row with label '{self._source_label}' in train_frame to poison."
            )

        n_poison = round(poison_rate * len(source_ids))
        rng = np.random.RandomState(seed)
        poisoned_ids = np.sort(rng.choice(source_ids, size=n_poison, replace=False))

        poisoned_frame = train_frame.copy()
        poison_mask = poisoned_frame[self._sample_id_column].isin(poisoned_ids)
        for feature, value in self._trigger_spec.items():
            # Cast the column to float BEFORE assigning: on CIC-IDS2017 every trigger column is already float, so this
            # does nothing, but the IoT datasets (TON_IoT, Edge-IIoTset) have true integer count columns (e.g.
            # dst_port). Assigning a value such as 53.5 to an int64 column only raises a FutureWarning today, with
            # pandas upcasting to float, but later pandas versions raise an error, and truncating the decimals would
            # silently change the trigger value. Casting explicitly rules out both.
            poisoned_frame[feature] = poisoned_frame[feature].astype(float)
            poisoned_frame.loc[poison_mask, feature] = value
        poisoned_frame.loc[poison_mask, self._label_column] = self._target_label
        return poisoned_frame, poisoned_ids

    def build_trigger_test(self, test_frame: pd.DataFrame) -> pd.DataFrame:
        """Inject `trigger_spec` into EVERY `source_label` row in `test_frame` (100%, no sampling) -
        the ASR evaluation set. Ground-truth label is left untouched (still `source_label`); ASR is
        measured by how often a backdoored model predicts `target_label` on this frame.

        Args:
            test_frame: Held-out test data (untouched by poisoning).

        Returns:
            A new frame: every `source_label` row from `test_frame`, with `trigger_spec` features
            overwritten.
        """
        trigger_test = test_frame.loc[test_frame[self._label_column] == self._source_label].copy()
        for feature, value in self._trigger_spec.items():
            trigger_test[feature] = value
        return trigger_test.reset_index(drop=True)
