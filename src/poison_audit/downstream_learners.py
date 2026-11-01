"""Component 1 - is a sanitised dataset clean for any learner, not only the one that audited it?

The audit uses an XGBoost model as its instrument. Here other learner families are trained on what each removal of the
poison-rate sweep keeps (its cached `removal_masks.npz`), next to `no_removal` (does the backdoor form for that learner
at all?) and `true_poison` (the ceiling), and measured like the sweep's own retrains: CDA, macro-F1 and ASR on the
injected trigger-test rows, plus the accuracy on the kept training rows, which exposes a learner that failed to
converge. The learners use their library defaults - a check of the dataset, not a tuned benchmark - except
LightGBM's learning rate (config `downstream_learners`).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

from src.core.logger import get_logger
from src.poison_audit.end_to_end_pipeline import NO_REMOVAL, REMOVAL_MASKS_FILE, RESULT_FILE, EndToEndPoisonAudit
from src.poison_audit.labor_backdoor_evaluator import LaborBackdoorEvaluator
from src.poison_audit.poison_rate_sweep import MANIFEST_FILE, POISONED_TRAIN_FILE, TRIGGER_TEST_FILE, SweepTrigger, rate_tag

logger = get_logger(__name__)

LEARNER_BUILDERS: Dict[str, Callable[[Dict[str, Any], int], Any]] = {
    "lightgbm": lambda params, seed: LGBMClassifier(**params, random_state=seed),
    "random_forest": lambda params, seed: RandomForestClassifier(**params, random_state=seed),
}
RESULT_FILE_NAME = "downstream_result.json"
RETRAIN_DIR = "retrain"


class FactoryEvaluator(LaborBackdoorEvaluator):
    """`LaborBackdoorEvaluator` whose models come from a factory, so any classifier with `fit`/`predict` is measured
    exactly like the sweep's XGBoost retrains.

    Args:
        feature_columns: Model features.
        label_column: Label column.
        factory: Builds one unfitted classifier.
    """

    def __init__(self, feature_columns: List[str], label_column: str, factory: Callable[[], Any]) -> None:
        super().__init__(feature_columns, label_column, model_params={}, random_state=0)
        self._factory = factory

    def train(self, train_frame: pd.DataFrame) -> Any:
        """Fit a classifier from the factory on `train_frame`.

        Args:
            train_frame: Training rows with the features and the label.

        Returns:
            The fitted classifier.
        """
        model = self._factory()
        model.fit(train_frame[self.feature_columns].to_numpy(dtype=float),
                  self.label_encoder.transform(train_frame[self._label_column]))
        return model


class DownstreamLearnerSweep:
    """Retrains other learners after the sweep's removals, cell by cell.

    Args:
        sweep_root: Root of the sweep's cells for this seed.
        output_root: Root of this check's cells for this seed.
        learners: Learner name -> constructor kwargs (`LEARNER_BUILDERS` keys).
        removals: Removal names to retrain after (`no_removal` and names in the sweep's `removal_masks.npz`).
        random_state: Seed of every model.
    """

    def __init__(
        self, sweep_root: Path, output_root: Path, learners: Dict[str, Dict[str, Any]], removals: List[str], random_state: int,
    ) -> None:
        unknown = [name for name in learners if name not in LEARNER_BUILDERS]
        if unknown:
            raise ValueError(f"Invalid learner: {unknown} (one of {list(LEARNER_BUILDERS)}).")
        self._sweep_root = sweep_root
        self._output_root = output_root
        self._learners = learners
        self._removals = removals
        self._seed = random_state

    def load(self, trigger: SweepTrigger, rate: float) -> Optional[Dict[str, Any]]:
        """Result of one cell if already computed.

        Args:
            trigger: The trigger.
            rate: Poison rate.

        Returns:
            The result, or None.
        """
        path = self._output_root / trigger.dataset / trigger.name / rate_tag(rate) / RESULT_FILE_NAME
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def run_cell(self, trigger: SweepTrigger, rate: float) -> Optional[Dict[str, Any]]:
        """Every learner x removal on one cell; each retrain is cached.

        Args:
            trigger: The trigger.
            rate: Poison rate.

        Returns:
            The cell result, or None when the sweep has no result for this cell yet.
        """
        started = time.time()
        name = f"{trigger.dataset}/{trigger.name}/{rate_tag(rate)}"
        sweep_cell = self._sweep_root / trigger.dataset / trigger.name / rate_tag(rate)
        out = self._output_root / trigger.dataset / trigger.name / rate_tag(rate)
        if not (sweep_cell / RESULT_FILE).exists():
            logger.warning("[%s] The sweep has no result for this cell yet, skipped.", name)
            return None
        out.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((sweep_cell / MANIFEST_FILE).read_text(encoding="utf-8"))
        id_column, label_column, target = manifest["id_column"], manifest["label_column"], manifest["target_label"]
        features = list(manifest["model_features"])
        columns = [id_column, label_column] + features
        frame = pd.read_csv(sweep_cell / POISONED_TRAIN_FILE, usecols=columns, float_precision="round_trip")
        clean_test = pd.read_csv(trigger.clean_test, float_precision="round_trip",
                                 usecols=list(dict.fromkeys(columns + list(trigger.trigger_spec))))
        trigger_test = pd.read_csv(sweep_cell.parent / TRIGGER_TEST_FILE, usecols=columns, float_precision="round_trip")
        injected = ~EndToEndPoisonAudit.already_triggered(trigger_test, clean_test, id_column, trigger.trigger_spec)
        with np.load(sweep_cell / REMOVAL_MASKS_FILE) as stored:
            masks = {removal: (np.zeros(len(frame), dtype=bool) if removal == NO_REMOVAL else stored[removal])
                     for removal in self._removals}

        results: Dict[str, Dict[str, Any]] = {}
        for learner, params in self._learners.items():
            evaluator = FactoryEvaluator(features, label_column,
                                         lambda params=params, learner=learner: LEARNER_BUILDERS[learner](params, self._seed))
            evaluator.fit_label_encoder(frame[label_column])
            results[learner] = {}
            for removal, mask in masks.items():
                cache = out / RETRAIN_DIR / learner / f"{removal}.json"
                if not cache.exists():
                    logger.info("[%s] %s after [%s]: keeping %d/%d rows...", name, learner, removal, int((~mask).sum()), len(mask))
                    kept = frame.loc[~mask]
                    model = evaluator.train(kept)
                    metrics = EndToEndPoisonAudit.model_metrics(evaluator, model, clean_test, trigger_test, injected, target)
                    # Accuracy on the training part itself: a diverging learner (LightGBM at learning_rate 0.1) shows up here.
                    metrics["train_accuracy"] = float((evaluator.predict_labels(model, kept) == kept[label_column].to_numpy()).mean())
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
                results[learner][removal] = json.loads(cache.read_text(encoding="utf-8"))
        sweep_result = json.loads((sweep_cell / RESULT_FILE).read_text(encoding="utf-8"))
        by_name = {item["name"]: item for item in sweep_result["removals"]}
        results["xgboost"] = {removal: {key: by_name[removal][key] for key in ("asr", "asr_injected_only", "cda", "macro_f1")}
                              for removal in self._removals}
        result = {"scenario": name, "dataset": trigger.dataset, "trigger": trigger.name, "rate": rate,
                  "n_poisoned": int(len(manifest["poisoned_sample_ids"])), "learners": results,
                  "seconds": time.time() - started}
        temporary = out / f"{RESULT_FILE_NAME}.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, out / RESULT_FILE_NAME)
        logger.info("[%s] Done after %.1f min.", name, result["seconds"] / 60)
        return result
