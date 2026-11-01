"""Component 1 - trains an XGBoost classifier on a (clean or backdoored) training set and measures
CDA/ASR (Luo et al. 2025, Section 4.1.4). Generic over how the training set was poisoned
(`fixed_trigger_poison_builder.FixedTriggerPoisonDatasetBuilder` here): this class only ever sees
plain (features, label) frames, with no LABOR-specific assumptions.

The classifier is a plain XGBoost with common-practice parameters, not a tuned production IDS - see
`configs/config.yaml` `component1_poison_audit.backdoor_eval.model_params`.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, recall_score
from sklearn.preprocessing import LabelEncoder


class LaborBackdoorEvaluator:
    """Trains one XGBoost model and reports CDA (clean accuracy) + ASR (attack success rate).

    Args:
        feature_columns: Feature column names (all columns except id/label).
        label_column: Name of the label column.
        model_params: XGBoost constructor kwargs (`component1_poison_audit.backdoor_eval.model_params`).
        random_state: Seed for the classifier.
    """

    def __init__(
        self, feature_columns: List[str], label_column: str, model_params: Dict[str, Any], random_state: int,
    ) -> None:
        self._feature_columns = feature_columns
        self._label_column = label_column
        self._model_params = model_params
        self._random_state = random_state
        self._label_encoder = LabelEncoder()

    @property
    def label_encoder(self) -> LabelEncoder:
        """The shared label encoder (fit via `fit_label_encoder`)."""
        return self._label_encoder

    @property
    def feature_columns(self) -> List[str]:
        """The feature space this evaluator's models are trained and scored on."""
        return self._feature_columns

    def fit_label_encoder(self, all_labels: pd.Series) -> None:
        """Fit the shared label encoder once, so clean and backdoored models use identical class ids.

        Args:
            all_labels: The full set of label values that must be encodable (typically the clean,
                unpoisoned train frame's label column - a superset of any poisoned variant's labels).

        Returns:
            None.
        """
        self._label_encoder.fit(all_labels)

    def train(self, train_frame: pd.DataFrame) -> xgb.XGBClassifier:
        """Fit an XGBoost multiclass classifier on `train_frame`.

        Args:
            train_frame: Training data containing `feature_columns` and `label_column`.

        Returns:
            The fitted classifier.
        """
        features = train_frame[self._feature_columns].to_numpy(dtype=float)
        targets = self._label_encoder.transform(train_frame[self._label_column])
        model = xgb.XGBClassifier(**self._model_params, random_state=self._random_state)
        model.fit(features, targets)
        return model

    def compute_cda(self, model: xgb.XGBClassifier, test_frame: pd.DataFrame) -> float:
        """Clean Data Accuracy: overall accuracy on an untouched, trigger-free test set.

        Args:
            model: A model returned by `train`.
            test_frame: Held-out test data (untouched by poisoning).

        Returns:
            Accuracy in [0, 1].
        """
        features = test_frame[self._feature_columns].to_numpy(dtype=float)
        targets = self._label_encoder.transform(test_frame[self._label_column])
        predictions = model.predict(features)
        return float((predictions == targets).mean())

    def compute_class_report(self, model: xgb.XGBClassifier, test_frame: pd.DataFrame) -> Dict[str, Any]:
        """Accuracy, macro-F1 and per-class recall on a test set.

        CDA alone cannot answer "is a narrower model as good as a wider one" on this dataset: BENIGN
        is 80% of the rows, so accuracy stays above 99% even when a rare attack class collapses
        entirely. Macro-F1 weights every class equally and per-class recall names which class paid.

        Args:
            model: A model returned by `train`.
            test_frame: Held-out test data.

        Returns:
            Dict with `accuracy`, `macro_f1` and `recall_per_class` (label -> recall).
        """
        features = test_frame[self._feature_columns].to_numpy(dtype=float)
        targets = self._label_encoder.transform(test_frame[self._label_column])
        predictions = model.predict(features)
        class_ids = list(range(len(self._label_encoder.classes_)))
        recalls = recall_score(targets, predictions, average=None, labels=class_ids, zero_division=0)
        return {
            "accuracy": float((predictions == targets).mean()),
            "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
            "recall_per_class": {
                str(label): float(recall)
                for label, recall in zip(self._label_encoder.classes_, recalls)
            },
        }

    def predict_labels(self, model: xgb.XGBClassifier, frame: pd.DataFrame) -> np.ndarray:
        """Predicts and decodes back to string labels.

        `ClusterLabelPoisonRemover.compute_removal_mask` compares predictions against a string
        `target_label` (`"BENIGN"`), not an encoded class id - this is the decode step that makes
        that comparison possible for any caller outside this class.

        Args:
            model: A model returned by `train`.
            frame: Rows to predict on - must contain `feature_columns`.

        Returns:
            1D array of string labels, same order as `frame`'s rows.
        """
        features = frame[self._feature_columns].to_numpy(dtype=float)
        predictions = model.predict(features)
        return self._label_encoder.inverse_transform(predictions)

    def compute_asr(self, model: xgb.XGBClassifier, trigger_test_frame: pd.DataFrame, target_label: str) -> float:
        """Attack Success Rate: fraction of trigger-carrying attack rows misclassified as `target_label`.

        Args:
            model: A model returned by `train`.
            trigger_test_frame: Test rows carrying the trigger the backdoor is meant to respond to
                - either natural rows already satisfying a threshold condition (see
                `LaborPoisonDatasetBuilder.select_poison_candidates`) or rows with trigger values
                directly injected (see `FixedTriggerPoisonDatasetBuilder.build_trigger_test`).
            target_label: The label the backdoor is meant to flip predictions to (`BENIGN`).

        Returns:
            Fraction in [0, 1] predicted as `target_label`.
        """
        features = trigger_test_frame[self._feature_columns].to_numpy(dtype=float)
        predictions = model.predict(features)
        target_id = self._label_encoder.transform([target_label])[0]
        return float((predictions == target_id).mean())
