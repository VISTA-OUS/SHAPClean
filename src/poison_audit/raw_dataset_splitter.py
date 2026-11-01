"""Component 1 - builds a train/test split directly from the raw combined CIC-IDS2017 CSV
(`component1_poison_audit.raw_split` in `configs/config.yaml`).

CIC-IDS2017 has a well-known exact-duplicate-flow characteristic (~11% of all rows dataset-wide, and
far more within some subsets - one 57,045-row subset holds only 1,854 truly unique flows). A plain row-level
`train_test_split` never repeats a `sample_id` across splits, but says nothing about CONTENT: two
rows with identical feature values can land on opposite sides, so the model effectively sees
"test" rows (under a different id) during training. This splitter groups rows by exact feature
content first and keeps every group entirely on one side, eliminating that leakage.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


class RawDatasetSplitter:
    """Group-aware stratified train/test split - no two rows with identical feature values ever
    land on opposite sides, and `id_column` = the row's original position in the raw source CSV.

    Args:
        label_column: Name of the label column to stratify on.
        id_column: Name of the identifier column to add.
        train_ratio: Fraction of rows assigned to the train split (must be `(n-1)/n` for some
            integer `n` - internally implemented as 1 held-out fold of `n = round(1/(1-train_ratio))`
            `StratifiedGroupKFold` folds, since that is what lets a group-respecting split also
            target a specific ratio).
        random_state: Seed for the split.
    """

    def __init__(self, label_column: str, id_column: str, train_ratio: float, random_state: int) -> None:
        self._label_column = label_column
        self._id_column = id_column
        self._train_ratio = train_ratio
        self._random_state = random_state

    def split(self, raw_frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split `raw_frame` so no exact-duplicate-content group crosses train/test, and stamp
        `id_column` with each row's original index.

        Args:
            raw_frame: Raw data as read from the source CSV (`pd.read_csv`'s default RangeIndex is
                required - it becomes `id_column`).

        Returns:
            Tuple of (train_frame, test_frame), each with `id_column` added as the first column and
            a fresh 0..n-1 row index.
        """
        feature_columns = [c for c in raw_frame.columns if c != self._label_column]
        content_hash = pd.util.hash_pandas_object(raw_frame[feature_columns], index=False)

        n_splits = round(1.0 / (1.0 - self._train_ratio))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self._random_state)
        train_positions, test_positions = next(
            splitter.split(raw_frame, raw_frame[self._label_column], groups=content_hash)
        )

        train_frame = raw_frame.iloc[train_positions].copy()
        test_frame = raw_frame.iloc[test_positions].copy()
        train_frame.insert(0, self._id_column, train_frame.index)
        test_frame.insert(0, self._id_column, test_frame.index)
        return train_frame.reset_index(drop=True), test_frame.reset_index(drop=True)
