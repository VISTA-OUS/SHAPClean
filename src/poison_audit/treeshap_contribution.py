"""Component 1 - TreeSHAP contribution vector for the FIXED BENIGN class, computed for every row
regardless of that row's own label (Lundberg, Erion, Chen, DeGrave, Prutkin, Nair, Katz,
Himmelfarb, Bansal, Lee (2020), *From local explanations to global understanding with explainable
AI for trees*, Nature Machine Intelligence).

Every row - genuinely BENIGN, genuinely ATTACK, and ATTACK rows relabeled BENIGN by the poisoning
step - is explained against the SAME class (BENIGN), never each row's own label. This mirrors how
Hoyheim et al. (Active Paths, ICMCIS 2026) build their clustering input: all training data is
passed through one shared contribution extraction, not a per-row class-specific one. A per-row
own-class scheme would place genuinely-ATTACK rows and BENIGN/poisoned rows in different semantic
spaces (e.g. "why DDoS" vs. "why BENIGN"), not directly comparable, making clustering across them
meaningless. Fixing the class to BENIGN gives every row a comparable answer to the same question:
how much does each feature push this row's BENIGN logit up or down?

Values live in the model's raw margin (logit) space, not probability space: for every row,
`TreeExplainer.expected_value[benign_class_idx]` plus that row's contributions equals the model's
BENIGN logit exactly. The BENIGN logit alone does not decide the prediction - the softmax over all
classes does, so a genuinely DDoS row can have a positive BENIGN logit and still P(BENIGN) ~ 0.

What the BENIGN slice captures of a backdoor, measured 2026-09-13 on the fixed-trigger scenario
(1,024 poisoned rows sharing identical trigger values, 30-feature dirty model): the trigger acts
mainly by suppressing the poisoned rows' source class, not by boosting BENIGN. In the DDoS slice the
three trigger features sum to -6.22 on average and have the largest |SHAP| in 100% of poisoned
rows; in the BENIGN slice they sum to +2.60 and lead in only 65.0% of them. Those 666 rows are
exactly the poison cluster HDBSCAN recovers at `min_cluster_size=100` (trigger sum +3.74). In the
other 358, `Average Packet Size` leads (356 of 358) and the trigger sum is +0.47, which is why those
rows form a second, separate mode. So the BENIGN slice separates poison through the overall shape of
the contribution vector, not through trigger dominance. A source-class slice carries the trigger far
more uniformly, but choosing it requires knowing which attack class was poisoned - information a
deployed audit does not have.

Do NOT replace this with |SHAP| averaged across ALL classes: that erases both the sign and the
class-specific signal this class depends on.

`extract` processes `records_df` in batches of `batch_size` rows to bound peak memory:
`shap_values` returns `n_rows x n_features x n_classes` float32 values, which for all 2,262,300
training rows and 15 classes is 4.1 GB at 30 features and 10.6 GB at 78, and XGBoost fills its own
contribution buffer (one extra bias column per class) during the same call. Batch boundaries do not
change the result, since each row's SHAP value depends only on that row and the fixed model.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src.core.logger import get_logger

logger = get_logger(__name__)


class TreeShapContributionExtractor:
    """Computes signed TreeSHAP contribution toward one fixed class, for every row of a frame.

    Args:
        feature_columns: Exactly the columns `model` was fit on, in the same order (in the
            sweep, the mRMR top-K of the poisoned training set). Not validated here. A wrong column COUNT fails inside `shap_values` with
            an XGBoost shape-mismatch error, but a wrong ORDER does not fail at all: for XGBoost
            models `shap` delegates to `predict(pred_contribs=True, validate_features=False)`, so
            column names are never checked - even for a model fit on a named DataFrame - and
            reordered columns silently yield contributions attributed to the wrong features.
        benign_class_idx: Encoded id of the BENIGN class in the model's label space, from the same
            `LabelEncoder` the model was trained with (`LaborBackdoorEvaluator.label_encoder`).
        batch_size: Max rows per `TreeExplainer.shap_values` call
            (`component1_poison_audit.shap_batch_size`) - bounds peak memory (see the module
            docstring); a technical limit, not an ML hyperparameter.
    """

    def __init__(self, feature_columns: List[str], benign_class_idx: int, batch_size: int) -> None:
        self._feature_columns = feature_columns
        self._benign_class_idx = benign_class_idx
        self._batch_size = batch_size

    def extract(self, model: xgb.XGBClassifier, records_df: pd.DataFrame) -> pd.DataFrame:
        """Computes the BENIGN-class contribution vector for every row of `records_df`.

        Args:
            model: The (backdoored) XGBoost model to explain - must be the exact model whose
                learned trigger->BENIGN association is under investigation, not a clean reference
                model (the backdoor signal only exists in the dirty model's weights).
            records_df: Rows to explain - ANY rows, regardless of their own label. Genuinely
                BENIGN, genuinely ATTACK, and ATTACK-relabeled-BENIGN-by-poisoning rows all get a
                BENIGN-class vector computed the exact same way.

        Returns:
            DataFrame with the same index as `records_df`, one `shap_{feature}` column per feature
            in `feature_columns`, signed (neither absolute-valued nor averaged across classes).
        """
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        n_rows = len(records_df)
        batch_frames = []
        for start in range(0, n_rows, self._batch_size):
            end = min(start + self._batch_size, n_rows)
            batch = records_df.iloc[start:end]
            logger.info("TreeSHAP batch %d-%d / %d...", start, end, n_rows)
            shap_values = explainer.shap_values(batch[self._feature_columns], check_additivity=False)
            benign_contribution = self._select_benign_class(shap_values)
            column_names = [f"shap_{feature}" for feature in self._feature_columns]
            batch_frames.append(pd.DataFrame(benign_contribution, columns=column_names, index=batch.index))
        return pd.concat(batch_frames, axis=0)

    def _select_benign_class(self, shap_values: object) -> np.ndarray:
        """Slices out the BENIGN-class values, keeping sign, from either `TreeExplainer.shap_values`
        output convention: a list of per-class 2D arrays, or one stacked 3D array.

        Args:
            shap_values: Output of `TreeExplainer.shap_values` for a multiclass model.

        Returns:
            2D array of shape `(n_records, n_features)`.
        """
        if isinstance(shap_values, list):
            return np.asarray(shap_values[self._benign_class_idx])
        array = np.asarray(shap_values)
        if array.ndim == 3:
            return array[:, :, self._benign_class_idx]
        return array
