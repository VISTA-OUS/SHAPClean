"""Component 1 - Confident Learning baseline (Northcutt, Jiang & Chuang, "Confident Learning: Estimating Uncertainty
in Dataset Labels", JAIR 70, 2021): finds label errors from out-of-fold predicted probabilities, with no clustering.

Dirty-label poisoning flips the label of triggered source-class rows to the target class, so a label-error detector is
the first baseline a reader of a poisoned-record removal method asks about. The steps follow the paper (Section 3) and
its reference library cleanlab (`filter.find_label_issues`, `count.compute_confident_joint`):

1. out-of-fold probabilities: K stratified folds, each held-out fold predicted by an XGBoost model with the audited
   model's architecture (`backdoor_eval.model_params`), trained on the other folds;
2. per-class threshold t_j: mean probability of class j over the rows labelled j;
3. confident label of a row: the class whose probability reaches its threshold, or the argmax when several do; rows
   with no such class have none;
4. two rules flag label issues:
   - `confident_joint`: rows whose confident label differs from the given one (the off-diagonal of the confident
     joint, cleanlab's `filter_by="confident_learning"`);
   - `prune_by_noise_rate`: the confident joint calibrated to the given label counts; for each off-diagonal (i, j),
     the round(C[i, j]) rows labelled i with the largest margin p_j - p_i (cleanlab's default `filter_by`);
   both then drop rows whose argmax is the given label, as cleanlab does.

Scopes: `all` removes every flagged row (CL as published); `target_label` keeps only flagged rows labelled with the
target class - the knowledge the clustering pipeline also uses (it audits target-class TreeSHAP).

What the measurement checks rather than assumes: the other folds hold poisoned rows with the same trigger, so the
out-of-fold model may have learnt the backdoor and still predict the flipped label for the held-out poisoned rows.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.core.logger import get_logger
from src.poison_audit.audit_sources import AuditSourceCatalog
from src.poison_audit.end_to_end_pipeline import RESULT_FILE
from src.poison_audit.labor_backdoor_evaluator import LaborBackdoorEvaluator
from src.poison_audit.poison_rate_sweep import MANIFEST_FILE, POISONED_TRAIN_FILE, rate_tag

logger = get_logger(__name__)

RULE_CONFIDENT_JOINT = "confident_joint"
RULE_PRUNE_BY_NOISE_RATE = "prune_by_noise_rate"
RULES = (RULE_CONFIDENT_JOINT, RULE_PRUNE_BY_NOISE_RATE)
SCOPE_ALL = "all"
SCOPE_TARGET_LABEL = "target_label"
SCOPES = (SCOPE_ALL, SCOPE_TARGET_LABEL)
THRESHOLD_TOLERANCE = 1e-6  # cleanlab: pred_probs >= thresholds - 1e-6
FLOATING_POINT_COMPARISON = 1e-6  # cleanlab `_reduce_issues`: a tie with the given label keeps the given label
OOF_FILE = "oof_probabilities.npz"
MASKS_FILE = "cl_masks.npz"
CL_RESULT_FILE = "cl_result.json"


def removal_scores(n_correct: int, n_wrong: int, n_poisoned: int, n_rows: int) -> Dict[str, float]:
    """Record-level scores of one removal.

    Args:
        n_correct: Poisoned rows removed.
        n_wrong: Clean rows removed.
        n_poisoned: Poisoned rows in the training set.
        n_rows: Rows in the training set.

    Returns:
        `n_removed`, `n_removed_correct`, `n_removed_wrong`, `precision`, `recall`, `fpr` (clean rows removed / clean
        rows) and `wrong_share` (clean rows removed / all rows, the sweep's existing measure).
    """
    removed = n_correct + n_wrong
    return {
        "n_removed": removed, "n_removed_correct": n_correct, "n_removed_wrong": n_wrong,
        "precision": n_correct / removed if removed else 0.0,
        "recall": n_correct / n_poisoned if n_poisoned else 0.0,
        "fpr": n_wrong / (n_rows - n_poisoned) if n_rows > n_poisoned else 0.0,
        "wrong_share": n_wrong / n_rows if n_rows else 0.0,
    }


class ConfidentLearningDetector:
    """Out-of-fold probabilities and the two Confident Learning rules (module docstring).

    Args:
        n_folds: Stratified folds of the out-of-fold predictions.
        model_params: XGBoost kwargs of every fold model.
        random_state: Seed of the fold split and of every fold model.
    """

    def __init__(self, n_folds: int, model_params: Dict[str, Any], random_state: int) -> None:
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}.")
        self._n_folds = n_folds
        self._model_params = model_params
        self._random_state = random_state

    def out_of_fold_probabilities(
        self, frame: pd.DataFrame, feature_columns: List[str], label_column: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict every row with a model that never saw it.

        Args:
            frame: Training rows with `feature_columns` and `label_column`.
            feature_columns: Model features.
            label_column: Given (possibly poisoned) label.

        Returns:
            Probabilities (rows x classes), given label ids, class names (id order).
        """
        evaluator = LaborBackdoorEvaluator(feature_columns, label_column, self._model_params, self._random_state)
        evaluator.fit_label_encoder(frame[label_column])
        labels = evaluator.label_encoder.transform(frame[label_column])
        classes = np.asarray(evaluator.label_encoder.classes_).astype(str)
        features = frame[feature_columns].to_numpy(dtype=float)
        probabilities = np.zeros((len(frame), len(classes)), dtype=np.float64)
        folds = StratifiedKFold(n_splits=self._n_folds, shuffle=True, random_state=self._random_state)
        for index, (train_rows, held_out) in enumerate(folds.split(features, labels), start=1):
            model = evaluator.train(frame.iloc[train_rows])
            probabilities[held_out] = model.predict_proba(features[held_out])
            logger.info("Fold %d/%d: training on %d rows, predicting %d rows.", index, self._n_folds, len(train_rows), len(held_out))
        return probabilities, labels, classes

    @staticmethod
    def confident_labels(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Confident label of every row (step 3), -1 where no class reaches its threshold.

        Args:
            probabilities: Out-of-fold probabilities.
            labels: Given label ids.

        Returns:
            Class id per row, or -1.
        """
        n_classes = probabilities.shape[1]
        thresholds = np.array([
            probabilities[labels == k, k].mean() if (labels == k).any() else np.inf for k in range(n_classes)
        ])
        confident = probabilities >= thresholds - THRESHOLD_TOLERANCE
        n_confident = confident.sum(axis=1)
        guess = np.where(n_confident > 1, probabilities.argmax(axis=1), confident.argmax(axis=1))
        return np.where(n_confident > 0, guess, -1)

    def label_issues(self, probabilities: np.ndarray, labels: np.ndarray, rule: str) -> np.ndarray:
        """Rows flagged as label issues by one rule (step 4).

        Args:
            probabilities: Out-of-fold probabilities.
            labels: Given label ids.
            rule: `confident_joint` or `prune_by_noise_rate`.

        Returns:
            Boolean mask over rows.

        Raises:
            ValueError: For an unknown rule.
        """
        confident = self.confident_labels(probabilities, labels)
        if rule == RULE_CONFIDENT_JOINT:
            flagged = (confident >= 0) & (confident != labels)
        elif rule == RULE_PRUNE_BY_NOISE_RATE:
            flagged = self._prune_by_noise_rate(probabilities, labels, confident)
        else:
            raise ValueError(f"Invalid CL rule: {rule!r} (one of {RULES}).")
        adjusted = probabilities.copy()
        adjusted[np.arange(len(labels)), labels] += FLOATING_POINT_COMPARISON
        return flagged & (adjusted.argmax(axis=1) != labels)

    def _prune_by_noise_rate(self, probabilities: np.ndarray, labels: np.ndarray, confident: np.ndarray) -> np.ndarray:
        """`prune_by_noise_rate`: calibrated off-diagonal counts, largest margins first.

        Args:
            probabilities: Out-of-fold probabilities.
            labels: Given label ids.
            confident: Output of `confident_labels`.

        Returns:
            Boolean mask over rows.
        """
        n_classes = probabilities.shape[1]
        joint = np.zeros((n_classes, n_classes), dtype=np.float64)
        counted = confident >= 0
        np.add.at(joint, (labels[counted], confident[counted]), 1.0)
        np.fill_diagonal(joint, np.maximum(joint.diagonal(), 1.0))
        label_counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
        calibrated = (joint.T / np.maximum(joint.sum(axis=1), np.finfo(float).tiny) * label_counts).T
        calibrated = calibrated / calibrated.sum() * label_counts.sum()
        counts = self._round_preserving_row_totals(calibrated)
        flagged = np.zeros(len(labels), dtype=bool)
        for given in range(n_classes):
            rows = np.flatnonzero(labels == given)
            for true in range(n_classes):
                n_prune = int(min(counts[given, true], len(rows) - 1))
                if true == given or n_prune <= 0:
                    continue
                margin = probabilities[rows, true] - probabilities[rows, given]
                flagged[rows[np.argsort(-margin, kind="stable")[:n_prune]]] = True
        return flagged

    @staticmethod
    def _round_preserving_row_totals(matrix: np.ndarray) -> np.ndarray:
        """Round to integers, each row keeping its (rounded) total - largest remainders get the leftover units.

        Args:
            matrix: Non-negative counts.

        Returns:
            Integer matrix of the same shape.
        """
        floored = np.floor(matrix)
        rounded = floored.astype(np.int64)
        leftover = np.rint(matrix.sum(axis=1)).astype(np.int64) - rounded.sum(axis=1)
        remainders = matrix - floored
        for row, extra in enumerate(leftover):
            if extra > 0:
                rounded[row, np.argsort(-remainders[row], kind="stable")[:extra]] += 1
        return rounded


class ConfidentLearningSweep:
    """Runs the baseline on the poison-rate sweep's cells and scores it next to the sweep's own removals.

    Nothing is rebuilt: each cell's poisoned set, manifest (model features, poisoned ids) and end-to-end result come
    from the sweep directory. Out-of-fold probabilities are cached per cell, so the rules can be re-scored for free.

    Args:
        detector: The Confident Learning detector.
        config: Project config (id/label defaults when a manifest lacks them).
        sweep_root: `poison_rate_sweep.output_dir`.
        output_root: `confident_learning.output_dir`.
        rules: Rules to apply.
        scopes: Scopes to apply.
        compare_with: Removal names of the sweep result scored next to the baseline.
        drop_cache: Delete a cell's out-of-fold probabilities once its result is written (~100 MB per CIC-IDS2017 cell;
            regenerable; the result keeps every score the paper reads).
    """

    def __init__(
        self,
        detector: ConfidentLearningDetector,
        config: Dict[str, Any],
        sweep_root: Path,
        output_root: Path,
        rules: List[str],
        scopes: List[str],
        compare_with: List[str],
        drop_cache: bool = False,
    ) -> None:
        unknown = [name for name in rules if name not in RULES] + [name for name in scopes if name not in SCOPES]
        if unknown:
            raise ValueError(f"Invalid CL rule/scope: {unknown} (rules {RULES}, scopes {SCOPES}).")
        self._detector = detector
        self._config = config
        self._sweep_root = sweep_root
        self._output_root = output_root
        self._rules = rules
        self._scopes = scopes
        self._compare_with = compare_with
        self._drop_cache = drop_cache

    @property
    def variant_names(self) -> List[str]:
        """Name of every baseline variant, in report order.

        Returns:
            E.g. `cl_confident_joint_all`.
        """
        return [f"cl_{rule}_{scope}" for rule in self._rules for scope in self._scopes]

    def cell_paths(self, dataset: str, trigger: str, rate: float) -> Tuple[Path, Path]:
        """Sweep cell directory and this baseline's output directory for one cell.

        Args:
            dataset: Dataset name.
            trigger: Trigger name.
            rate: Poison rate.

        Returns:
            (sweep cell, output cell).
        """
        relative = Path(dataset) / trigger / rate_tag(rate)
        return self._sweep_root / relative, self._output_root / relative

    def load(self, dataset: str, trigger: str, rate: float) -> Optional[Dict[str, Any]]:
        """Result of one cell if it was already scored.

        Args:
            dataset: Dataset name.
            trigger: Trigger name.
            rate: Poison rate.

        Returns:
            The result, or None.
        """
        path = self.cell_paths(dataset, trigger, rate)[1] / CL_RESULT_FILE
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def run_cell(self, dataset: str, trigger: str, rate: float) -> Optional[Dict[str, Any]]:
        """Score the baseline on one cell (reused when already scored).

        Args:
            dataset: Dataset name.
            trigger: Trigger name.
            rate: Poison rate.

        Returns:
            The cell result, or None when the sweep has no result for this cell yet.
        """
        name = f"{dataset}/{trigger}/{rate_tag(rate)}"
        sweep_cell, cell = self.cell_paths(dataset, trigger, rate)
        if not (sweep_cell / RESULT_FILE).exists():
            logger.warning("[%s] The sweep has no result for this cell yet, skipped.", name)
            return None
        cached = self.load(dataset, trigger, rate)
        if cached is not None:
            logger.info("[%s] CL result already there, skipped.", name)
            return cached
        cell.mkdir(parents=True, exist_ok=True)
        settings = AuditSourceCatalog.settings(
            {"manifest": (sweep_cell / MANIFEST_FILE).as_posix(), "poisoned_train": (sweep_cell / POISONED_TRAIN_FILE).as_posix()},
            self._config, "selected",
        )
        id_column, label_column, target = settings["id_column"], settings["label_column"], settings["target_label"]
        features = list(settings["model_features"])
        frame = pd.read_csv(sweep_cell / POISONED_TRAIN_FILE, usecols=[id_column, label_column] + features,
                            float_precision="round_trip")
        poisoned = np.isin(frame[id_column].to_numpy(), settings["poisoned_ids"])
        probabilities, labels, classes = self._load_or_predict(cell / OOF_FILE, frame, features, label_column, name)
        target_id = int(np.flatnonzero(classes == target)[0])
        sweep_result = json.loads((sweep_cell / RESULT_FILE).read_text(encoding="utf-8"))
        source_label = json.loads((sweep_cell / MANIFEST_FILE).read_text(encoding="utf-8"))["source_label"]
        source_id = int(np.flatnonzero(classes == str(source_label))[0])

        masks, removals = {}, []
        for rule in self._rules:
            flagged = self._detector.label_issues(probabilities, labels, rule)
            for scope in self._scopes:
                mask = flagged if scope == SCOPE_ALL else flagged & (labels == target_id)
                variant = f"cl_{rule}_{scope}"
                masks[variant] = mask
                removals.append({"name": variant, **removal_scores(int((mask & poisoned).sum()), int((mask & ~poisoned).sum()),
                                                                     int(poisoned.sum()), len(frame))})
        by_name = {item["name"]: item for item in sweep_result["removals"]}
        for variant in self._compare_with:
            item = by_name[variant]
            removals.append({"name": variant, **removal_scores(int(item["n_removed_correct"]), int(item["n_removed_wrong"]),
                                                                 int(sweep_result["n_poisoned"]), int(sweep_result["n_rows"])),
                             "asr": item.get("asr"), "cda": item.get("cda")})
        predicted = probabilities.argmax(axis=1)
        result = {
            "scenario": name, "dataset": dataset, "trigger": trigger, "rate": rate,
            "n_rows": int(len(frame)), "n_poisoned": int(poisoned.sum()), "n_model_features": len(features),
            "oof": {
                "accuracy_clean_rows": float((predicted[~poisoned] == labels[~poisoned]).mean()),
                "poisoned_predicted_target": float((predicted[poisoned] == target_id).mean()) if poisoned.any() else 0.0,
                "poisoned_predicted_source": float((predicted[poisoned] == source_id).mean()) if poisoned.any() else 0.0,
                "poisoned_mean_p_source": float(probabilities[poisoned, source_id].mean()) if poisoned.any() else 0.0,
            },
            "removals": removals,
        }
        np.savez_compressed(cell / MASKS_FILE, **masks)
        temporary = cell / f"{CL_RESULT_FILE}.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, cell / CL_RESULT_FILE)
        if self._drop_cache:
            (cell / OOF_FILE).unlink(missing_ok=True)
        logger.info("[%s] Poisoned rows the OOF model predicts as the target label: %.2f%%.", name, 100 * result["oof"]["poisoned_predicted_target"])
        return result

    def _load_or_predict(
        self, path: Path, frame: pd.DataFrame, features: List[str], label_column: str, name: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Out-of-fold probabilities of a cell, from cache when present.

        Args:
            path: Cache file.
            frame: Training rows.
            features: Model features.
            label_column: Label column.
            name: Cell name (logging).

        Returns:
            Probabilities (the stored float32 values, so a fresh run and a cached one score alike), given label ids,
            class names.
        """
        if not path.exists():
            logger.info("[%s] Out-of-fold probabilities on %d rows, %d features...", name, len(frame), len(features))
            probabilities, labels, classes = self._detector.out_of_fold_probabilities(frame, features, label_column)
            temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
            np.savez_compressed(temporary, probabilities=probabilities.astype(np.float32), labels=labels, classes=classes)
            os.replace(temporary, path)
        cached = np.load(path)
        return cached["probabilities"].astype(np.float64), cached["labels"], cached["classes"]
