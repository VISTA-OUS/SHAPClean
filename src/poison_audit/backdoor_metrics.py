"""Component 1 - the CDA/ASR measurement for one fixed-value-trigger scenario (Luo et al. 2025,
Section 4.1.4), factored out so every scenario is judged by exactly the same arithmetic.

It is callable on a poisoned dataset loaded back from disk, so every dataset (CIC-IDS2017, TON-IoT,
Edge-IIoTset) is measured by the same code. Re-typing the same formulas in a second place would make
the numbers of different datasets only *look* comparable - and comparing them is the point of running
the defense on several datasets.

Every ASR is reported twice, over all trigger-test rows and over only the rows this scenario actually
injected, alongside the same two numbers for a CLEAN model. A trigger specified by hand can already
occur in real traffic: rows of the attack class that carry it naturally enter the ASR test set
without the attacker touching them, so `asr` on its own stops meaning "the backdoor fired". The clean
model's ASR separates "these trigger values already look benign to any model" from "the model learned
the backdoor".
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import xgboost as xgb

from src.core.logger import get_logger
from src.poison_audit.labor_backdoor_evaluator import LaborBackdoorEvaluator

logger = get_logger(__name__)


class BackdoorMetricsCalculator:
    """Scores a backdoored model against a clean one on the same rows.

    Args:
        evaluator: Evaluator over the audited model's feature space, its label encoder already fit on
            the CLEAN train labels (a superset of any poisoned variant's labels).
        target_label: Label the backdoor flips predictions to.
        sample_id_column: Name of the stable row-id column.
    """

    def __init__(
        self, evaluator: LaborBackdoorEvaluator, target_label: str, sample_id_column: str,
    ) -> None:
        self._evaluator = evaluator
        self._target_label = target_label
        self._sample_id_column = sample_id_column

    def measure(
        self,
        name: str,
        model: xgb.XGBClassifier,
        clean_model: xgb.XGBClassifier,
        clean_report: Dict[str, Any],
        clean_test: pd.DataFrame,
        trigger_test: pd.DataFrame,
        poisoned_train: pd.DataFrame,
        poisoned_ids: np.ndarray,
        was_already_triggered: np.ndarray,
    ) -> Dict[str, Any]:
        """Measure CDA/ASR of the backdoored model against the clean model on the same rows.

        Args:
            name: Scenario name, for the log line.
            model: The backdoored model.
            clean_model: The clean reference model.
            clean_report: `compute_class_report` of `clean_model` on `clean_test`.
            clean_test: Untouched test set.
            trigger_test: Every source-class test row with the trigger injected.
            poisoned_train: The poisoned training frame.
            poisoned_ids: Ids of the poisoned rows.
            was_already_triggered: Per-row flag over `trigger_test`, True where that row already
                carried the whole trigger before injection.

        Returns:
            Dict of measured numbers - accuracy/macro-F1/per-class recall for both models, ASR over
            all trigger-test rows and over injected-only rows, the drops, and the recall ceiling.
        """
        injected = ~was_already_triggered
        backdoored_report = self._evaluator.compute_class_report(model, clean_test)
        backdoored_hit = self._evaluator.predict_labels(model, trigger_test) == self._target_label
        clean_hit = self._evaluator.predict_labels(clean_model, trigger_test) == self._target_label
        # Recall ceiling: the defence's removal rule can only remove poisoned rows the model calls target_label, so
        # the recall of a later defence run must be read against this number, not against 100%.
        train_predicted = self._evaluator.predict_labels(model, poisoned_train)
        is_poisoned = np.isin(poisoned_train[self._sample_id_column].to_numpy(), poisoned_ids)

        metrics: Dict[str, Any] = {
            "n_poisoned": int(len(poisoned_ids)),
            "n_clean_test": int(len(clean_test)),
            "n_trigger_test": int(len(trigger_test)),
            "n_trigger_test_injected": int(injected.sum()),
            "n_trigger_test_already_triggered": int(was_already_triggered.sum()),
            "clean_cda": float(clean_report["accuracy"]),
            "clean_macro_f1": float(clean_report["macro_f1"]),
            "clean_asr": float(clean_hit.mean()),
            "clean_asr_injected_only": float(clean_hit[injected].mean()) if injected.any() else 0.0,
            "backdoored_cda": float(backdoored_report["accuracy"]),
            "backdoored_macro_f1": float(backdoored_report["macro_f1"]),
            "backdoored_asr": float(backdoored_hit.mean()),
            "backdoored_asr_injected_only": (
                float(backdoored_hit[injected].mean()) if injected.any() else 0.0
            ),
            # 0% poison control: there is no poisoned row, so the recall ceiling is 0 as in TwoStageAuditPipeline (not NaN).
            "recall_ceiling": (
                float((train_predicted[is_poisoned] == self._target_label).mean()) if is_poisoned.any() else 0.0
            ),
            "clean_recall_per_class": clean_report["recall_per_class"],
            "backdoored_recall_per_class": backdoored_report["recall_per_class"],
        }
        metrics["cda_drop_pp"] = 100.0 * (metrics["clean_cda"] - metrics["backdoored_cda"])
        metrics["macro_f1_drop_pp"] = 100.0 * (metrics["clean_macro_f1"] - metrics["backdoored_macro_f1"])
        metrics["asr_increase_pp"] = 100.0 * (metrics["backdoored_asr"] - metrics["clean_asr"])
        metrics["asr_increase_injected_only_pp"] = 100.0 * (
            metrics["backdoored_asr_injected_only"] - metrics["clean_asr_injected_only"]
        )
        logger.info(
            "[%s] ASR %.2f%% (triggered rows only %.2f%%, clean model %.2f%%) | CDA %.4f%% (drop %.4fpp) | "
            "macro-F1 %.4f%% (drop %.4fpp) | recall ceiling %.2f%%.",
            name, 100 * metrics["backdoored_asr"], 100 * metrics["backdoored_asr_injected_only"],
            100 * metrics["clean_asr"], 100 * metrics["backdoored_cda"], metrics["cda_drop_pp"],
            100 * metrics["backdoored_macro_f1"], metrics["macro_f1_drop_pp"],
            100 * metrics["recall_ceiling"],
        )
        return metrics
