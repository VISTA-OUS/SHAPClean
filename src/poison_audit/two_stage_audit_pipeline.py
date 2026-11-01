"""Component 1 - the clustering stage of the audit, end to end, for ONE poisoned training set.

Train the audited (dirty) model, explain every row through its BENIGN-class TreeSHAP slice, cluster
with StandardScaler + PCA(95%) + HDBSCAN, label clusters by their true composition (poisoned rows count
as attack - the `labor_oracle` reference), remove rows whose cluster is ATTACK_LIKE while the model
predicts BENIGN, then optionally re-cluster stage 1's noise with the same mechanism and union both
stages' removals.

That second stage (`run_noise_cascade`, default on) pays off only when stage 1 failed to form a poison
cluster, and is pure added cost when stage 1 succeeded - see the class docstring. The end-to-end audit
(`end_to_end_pipeline.py`) runs stage 1 only and takes its clustering from here.

The pipeline is a single unit that takes a poisoned training set and nothing scenario-specific,
composed from the existing building blocks without re-implementing any of them - which guarantees
every scenario is judged by exactly the same rules.

Two numbers are reported because a scenario can fail in ways that say nothing about the defense:
- ASR/CDA of the audited model: confirms a working backdoor exists at all. A scenario whose backdoor
  never took is not evidence about the defense either way.
- The recall ceiling: the fraction of poisoned training rows the dirty model predicts as BENIGN. The
  removal rule can only ever remove those, so a low recall is only a clustering failure if it sits
  well below this ceiling.

The audited model's feature space and the clustering's feature space are separate arguments
(`model_feature_columns` vs `selected_features`), defaulting to the same columns. They only differ
when the model must be able to see a trigger the defense's representation does not contain - see
`model_feature_columns` in the class docstring.

Every expensive artifact (model, SHAP table, each stage's clustering) is written to a fixed layout
under `output_dir` and reused when present, so an interrupted multi-hour run resumes instead of
restarting.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from src.core.logger import get_logger
from src.poison_audit.clustering_arm_runner import ClusteringArmRunner
from src.poison_audit.clustering_evaluator import ClusteringEvaluator
from src.poison_audit.labor_backdoor_evaluator import LaborBackdoorEvaluator
from src.poison_audit.noise_cascade_auditor import NoiseCascadeAuditor
from src.poison_audit.treeshap_contribution import TreeShapContributionExtractor

logger = get_logger(__name__)

NOISE_CLUSTER_ID = -1
# File names of the default case (a 30-feature model). For another number of features the file names are derived from
# it (`_model_file`/`_shap_file`) and equal these two constants for 30.
MODEL_FILE = "dirty_model_30feat.pkl"
SHAP_FILE = "shap30_contribution.csv.gz"
STAGE_METRICS_FILE = "arm_metrics.json"


class TwoStageAuditPipeline:
    """Runs the two-stage SHAP-space HDBSCAN audit on one poisoned training set.

    Args:
        selected_features: SHAP columns fed to HDBSCAN, named by their feature (the model's
            features the defense explains on).
        model_feature_columns: Feature space the audited model is trained on and explained over.
            Defaults to `selected_features`, which is what the end-to-end audit passes. Passing a
            wider list separates the two: the model sees all of it, the clustering still sees only
            `selected_features` - the only way a trigger placed OUTSIDE them can both create a real
            backdoor and be absent from the defense's representation. Must be a superset of
            `selected_features`.
        label_column: Label column name.
        id_column: Stable row-identifier column name.
        target_label: Label the backdoor flips to (`BENIGN`).
        model_params: XGBoost constructor kwargs for the audited model.
        clustering_params: `HdbscanClusterer` parameters shared by BOTH stages
            (`pca_variance_retained`, `fit_sample_size`, `min_cluster_size`, `min_samples`,
            `core_dist_n_jobs`, optional `use_scaler`).
        shap_batch_size: Rows per TreeSHAP call.
        purity_threshold_pct: Poison share above which a cluster counts as poison-pure.
        random_state: Seed for model training, fit-sample draws and PCA.
        run_noise_cascade: Whether to re-cluster stage 1's noise and union its removals in.
            Default True, the configuration every earlier result was produced with. Set False to
            stop after stage 1: measured on the two specified triggers (2026-09-18), the cascade
            recovered nothing whenever stage 1 had already formed a high-purity poison cluster - it
            removed 130 further rows in each full-fit run, 0 of them poison - so it only earns its
            keep as a fallback for a stage 1 that collapsed. When False, `run`'s result carries
            `stage2: None` and `noise_cascade_ran: False`.
    """

    def __init__(
        self,
        selected_features: List[str],
        label_column: str,
        id_column: str,
        target_label: str,
        model_params: Dict[str, Any],
        clustering_params: Dict[str, Any],
        shap_batch_size: int,
        purity_threshold_pct: float,
        random_state: int,
        model_feature_columns: Optional[List[str]] = None,
        run_noise_cascade: bool = True,
    ) -> None:
        self._selected_features = selected_features
        self._model_feature_columns = list(model_feature_columns or selected_features)
        missing = [feature for feature in selected_features if feature not in set(self._model_feature_columns)]
        if missing:
            raise ValueError(
                f"model_feature_columns must hold every clustered feature; missing: {missing[:5]}",
            )
        self._model_file = f"dirty_model_{len(self._model_feature_columns)}feat.pkl"
        self._shap_file = f"shap{len(self._model_feature_columns)}_contribution.csv.gz"
        self._label_column = label_column
        self._id_column = id_column
        self._target_label = target_label
        self._model_params = model_params
        self._clustering_params = clustering_params
        self._shap_batch_size = shap_batch_size
        self._purity_threshold_pct = purity_threshold_pct
        self._random_state = random_state
        self._run_noise_cascade = run_noise_cascade
        self._auditor = NoiseCascadeAuditor(target_label=target_label)
        self._evaluator = ClusteringEvaluator(target_label=target_label)

    def run(
        self,
        scenario_name: str,
        poisoned_train: pd.DataFrame,
        clean_train_labels: pd.Series,
        clean_test: pd.DataFrame,
        trigger_test: pd.DataFrame,
        poisoned_sample_ids: List[int],
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Audit one scenario and write `pipeline_result.json` + `pipeline_report.md`.

        Args:
            scenario_name: Name used in logs and the report.
            poisoned_train: Full poisoned training frame (features, label, id).
            clean_train_labels: Label column of the clean training set, so the label encoder covers
                every class regardless of which labels poisoning removed.
            clean_test: Untouched test set, for CDA.
            trigger_test: Test rows carrying the trigger, for ASR.
            poisoned_sample_ids: Ground-truth poisoned ids.
            output_dir: Scenario output directory (also the resume cache).

        Returns:
            The result dict written to `pipeline_result.json`.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        backdoor_evaluator = LaborBackdoorEvaluator(
            feature_columns=self._model_feature_columns,
            label_column=self._label_column,
            model_params=self._model_params,
            random_state=self._random_state,
        )
        backdoor_evaluator.fit_label_encoder(clean_train_labels)
        model = self._load_or_train_model(backdoor_evaluator, poisoned_train, output_dir / self._model_file)

        sample_ids = poisoned_train[self._id_column].to_numpy()
        is_poisoned = np.isin(sample_ids, np.asarray(poisoned_sample_ids))
        predicted_labels = backdoor_evaluator.predict_labels(model, poisoned_train)
        validity = {
            "backdoored_cda": backdoor_evaluator.compute_cda(model, clean_test),
            "asr": backdoor_evaluator.compute_asr(model, trigger_test, self._target_label),
            "n_trigger_test": int(len(trigger_test)),
            # Without poisoned rows (the control scenario on a clean set) the recall ceiling does not exist - report
            # 0.0, not nan, so the result tables and JSON stay readable.
            "recall_ceiling": (
                float((predicted_labels[is_poisoned] == self._target_label).mean())
                if bool(is_poisoned.any()) else 0.0
            ),
        }
        logger.info("[%s] Scenario validity: %s", scenario_name, validity)

        benign_class_idx = int(backdoor_evaluator.label_encoder.transform([self._target_label])[0])
        if self._run_noise_cascade or not (output_dir / "stage1" / STAGE_METRICS_FILE).exists():
            shap_frame = self._load_or_build_shap(
                model, benign_class_idx, poisoned_train, output_dir / self._shap_file,
            )
        else:
            # Stage 1 is already clustered and there is no stage 2: the SHAP table only checks the row order, so it is
            # not recomputed once deleted (poison_rate_sweep.drop_shap) - TreeSHAP takes ~55 minutes per CIC-IDS2017 cell.
            logger.info("[%s] Stage 1 already done, SHAP table not needed.", scenario_name)
            shap_frame = poisoned_train[[self._id_column, self._label_column]].reset_index(drop=True)
        feature_columns = [f"shap_{feature}" for feature in self._selected_features]

        stage1_metrics, stage1_ids, stage1_summary = self._run_stage(
            f"{scenario_name}_stage1", shap_frame, feature_columns, predicted_labels,
            poisoned_sample_ids, output_dir / "stage1",
        )
        noise_positions = np.where(stage1_ids == NOISE_CLUSTER_ID)[0]
        stage1_mask = self._auditor.removal_mask(stage1_summary, stage1_ids, predicted_labels)
        stage2: Optional[Dict[str, Any]] = None
        true_labels = poisoned_train[self._label_column]

        if self._run_noise_cascade:
            stage2_metrics, stage2_ids, stage2_summary = self._run_stage(
                f"{scenario_name}_stage2", shap_frame.iloc[noise_positions].reset_index(drop=True),
                feature_columns, predicted_labels[noise_positions], poisoned_sample_ids,
                output_dir / "stage2",
            )
            stage2_mask = self._auditor.removal_mask(
                stage2_summary, stage2_ids, predicted_labels[noise_positions],
            )
            combined_mask = self._auditor.combine_masks(stage1_mask, stage2_mask, noise_positions)
            final_noise = np.zeros(len(sample_ids), dtype=bool)
            final_noise[noise_positions[stage2_ids == NOISE_CLUSTER_ID]] = True
            stage2 = {
                "n_clusters": stage2_metrics["n_clusters"],
                "n_noise": stage2_metrics["n_noise"],
                "removal": self._auditor.score_mask(stage2_mask, is_poisoned[noise_positions]),
                "placement": self._evaluator.poison_placement(stage2_summary, self._purity_threshold_pct),
                "purity": self._evaluator.purity_breakdown(
                    stage2_ids, true_labels.iloc[noise_positions].reset_index(drop=True),
                ),
            }
            n_clusters_to_label = int(stage1_metrics["n_clusters"] + stage2_metrics["n_clusters"])
        else:
            # Stage 2 did NOT run: `stage2` stays None instead of zeros. A block of zeros reads exactly like "stage 2
            # ran and recovered nothing", and the two are very different when comparing tables.
            # The noise of stage 1 is therefore the final noise - no later step clusters it again.
            combined_mask = stage1_mask
            final_noise = np.zeros(len(sample_ids), dtype=bool)
            final_noise[noise_positions] = True
            n_clusters_to_label = int(stage1_metrics["n_clusters"])
        del shap_frame
        result: Dict[str, Any] = {
            "scenario_name": scenario_name,
            "n_rows": int(len(sample_ids)),
            "n_poisoned": int(is_poisoned.sum()),
            "poison_pct": round(100.0 * float(is_poisoned.mean()), 4),
            "validity": validity,
            "stage1": {
                "n_clusters": stage1_metrics["n_clusters"],
                "n_noise": stage1_metrics["n_noise"],
                "removal": self._auditor.score_mask(stage1_mask, is_poisoned),
                "placement": self._evaluator.poison_placement(stage1_summary, self._purity_threshold_pct),
                "purity": self._evaluator.purity_breakdown(stage1_ids, true_labels),
            },
            "noise_cascade_ran": bool(self._run_noise_cascade),
            "stage2": stage2,
            "combined": self._auditor.score_mask(combined_mask, is_poisoned),
            "n_final_noise": int(final_noise.sum()),
            "n_poison_in_final_noise": int((final_noise & is_poisoned).sum()),
            "n_clusters_to_label": n_clusters_to_label,
            "wrong_removed_true_labels": {
                str(label): int(count)
                for label, count in true_labels[combined_mask & ~is_poisoned].value_counts().items()
            },
        }
        (output_dir / "pipeline_result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        (output_dir / "pipeline_report.md").write_text(self._build_report(result), encoding="utf-8")
        logger.info(
            "[%s] %s: removed %d, correct %d/%d (precision %.2f%%, recall %.2f%%), %d clusters to label.",
            scenario_name, "Both stages" if self._run_noise_cascade else "Stage 1 only",
            result["combined"]["n_removed"], result["combined"]["n_removed_correct"],
            result["n_poisoned"], 100 * result["combined"]["precision"], 100 * result["combined"]["recall"],
            result["n_clusters_to_label"],
        )
        return result

    def _load_or_train_model(
        self, backdoor_evaluator: LaborBackdoorEvaluator, poisoned_train: pd.DataFrame, model_path: Path,
    ) -> xgb.XGBClassifier:
        """Reuse the persisted audited model, or train and persist it.

        Args:
            backdoor_evaluator: Evaluator with its label encoder already fit.
            poisoned_train: Training frame.
            model_path: Where the model is cached.

        Returns:
            The audited dirty model.
        """
        if model_path.exists():
            logger.info("Reusing the saved model: %s", model_path)
            with model_path.open("rb") as model_file:
                return pickle.load(model_file)
        logger.info(
            "Training the dirty model on %d features and %d rows...",
            len(self._model_feature_columns), len(poisoned_train),
        )
        start_time = time.time()
        model = backdoor_evaluator.train(poisoned_train)
        logger.info("Training done after %.1fs.", time.time() - start_time)
        with model_path.open("wb") as model_file:
            pickle.dump(model, model_file)
        return model

    def _load_or_build_shap(
        self, model: xgb.XGBClassifier, benign_class_idx: int, poisoned_train: pd.DataFrame, shap_path: Path,
    ) -> pd.DataFrame:
        """Reuse the persisted BENIGN-class SHAP table, or compute and persist it.

        Args:
            model: The audited model to explain.
            benign_class_idx: Encoded BENIGN class id.
            poisoned_train: Rows to explain.
            shap_path: Where the table is cached.

        Returns:
            Frame with id, label and one `shap_*` column per selected feature, in `poisoned_train` order.

        Raises:
            ValueError: If a cached table's row order does not match `poisoned_train`.
        """
        if shap_path.exists():
            logger.info("Reusing the saved SHAP table: %s", shap_path)
            shap_frame = pd.read_csv(shap_path)
        else:
            logger.info("Computing BENIGN-class TreeSHAP for %d rows...", len(poisoned_train))
            start_time = time.time()
            contributions = TreeShapContributionExtractor(
                feature_columns=self._model_feature_columns,
                benign_class_idx=benign_class_idx,
                batch_size=self._shap_batch_size,
            ).extract(model, poisoned_train)
            logger.info("TreeSHAP done after %.1fs.", time.time() - start_time)
            shap_frame = pd.concat(
                [
                    poisoned_train[[self._id_column, self._label_column]].reset_index(drop=True),
                    contributions.reset_index(drop=True),
                ],
                axis=1,
            )
            shap_frame.to_csv(shap_path, index=False)
        if not np.array_equal(shap_frame[self._id_column].to_numpy(), poisoned_train[self._id_column].to_numpy()):
            raise ValueError(f"{shap_path.name} does NOT have the row order of poisoned_train.")
        return shap_frame

    def _run_stage(
        self,
        stage_name: str,
        frame: pd.DataFrame,
        feature_columns: List[str],
        predicted_labels: np.ndarray,
        poisoned_sample_ids: List[int],
        stage_dir: Path,
    ) -> Tuple[Dict[str, Any], np.ndarray, pd.DataFrame]:
        """Cluster one stage, or reload it if that stage already finished.

        Args:
            stage_name: Arm name for `ClusteringArmRunner`.
            frame: Rows to cluster (the SHAP table, or its stage-1 noise subset).
            feature_columns: SHAP columns to cluster on.
            predicted_labels: Model predictions aligned to `frame`.
            poisoned_sample_ids: Ground-truth poisoned ids.
            stage_dir: Stage output directory.

        Returns:
            Tuple of (arm metrics, cluster id per row of `frame`, cluster summary).

        Raises:
            ValueError: If a reloaded stage does not cover exactly `frame`'s rows.
        """
        if not (stage_dir / STAGE_METRICS_FILE).exists():
            ClusteringArmRunner(
                label_column=self._label_column,
                id_column=self._id_column,
                target_label=self._target_label,
                clustering_params=self._clustering_params,
                random_state=self._random_state,
            ).run(
                arm_name=stage_name,
                clustering_frame=frame,
                feature_columns=feature_columns,
                predicted_labels=predicted_labels,
                poisoned_sample_ids=poisoned_sample_ids,
                output_dir=stage_dir,
            )
        else:
            logger.info("Reusing the existing %s result: %s", stage_name, stage_dir)
        assignment = np.load(stage_dir / "cluster_assignment.npz")
        if not np.array_equal(assignment["sample_id"], frame[self._id_column].to_numpy()):
            raise ValueError(f"{stage_dir} does not match the rows of {stage_name}.")
        metrics = json.loads((stage_dir / STAGE_METRICS_FILE).read_text(encoding="utf-8"))
        return metrics, assignment["cluster"], pd.read_csv(stage_dir / "cluster_summary.csv")

    @staticmethod
    def _build_report(result: Dict[str, Any]) -> str:
        """Render one scenario's result as markdown.

        Args:
            result: Output of `run`.

        Returns:
            Markdown text.
        """
        validity, combined = result["validity"], result["combined"]
        lines = [
            f"# Two-stage pipeline - scenario `{result['scenario_name']}`",
            "",
            "## Is the scenario valid",
            "",
            f"- Poisoned rows: {result['n_poisoned']:,}/{result['n_rows']:,} ({result['poison_pct']}%)",
            f"- Audited model: CDA {100 * validity['backdoored_cda']:.2f}%, "
            f"ASR {100 * validity['asr']:.2f}% on {validity['n_trigger_test']:,} trigger-test rows",
            f"- Recall ceiling (share of poisoned rows the model predicts BENIGN - the removal rule can only remove these): "
            f"**{100 * validity['recall_ceiling']:.2f}%**",
            "",
            "## Removal results",
            "",
            "| | clusters | noise | removed | correct | precision | recall |",
            "|---|---|---|---|---|---|---|",
        ]
        for key, label in (("stage1", "Stage 1"), ("stage2", "Stage 2")):
            stage = result[key]
            if stage is None:
                lines.append(f"| {label} | _not run_ | - | - | - | - | - |")
                continue
            removal = stage["removal"]
            # Stage-2 recall counts only the poison in noise and is not comparable with stage 1 - left empty.
            recall_cell = "-" if key == "stage2" else f"{100 * removal['recall']:.2f}%"
            lines.append(
                f"| {label} | {stage['n_clusters']:,} | {stage['n_noise']:,} | {removal['n_removed']:,} | "
                f"{removal['n_removed_correct']:,} | {100 * removal['precision']:.2f}% | {recall_cell} |"
            )
        lines += [
            f"| **Total** | **{result['n_clusters_to_label']:,}** | **{result['n_final_noise']:,}** | "
            f"**{combined['n_removed']:,}** | **{combined['n_removed_correct']:,}** | "
            f"**{100 * combined['precision']:.2f}%** | **{100 * combined['recall']:.2f}%** |",
            "",
            f"- Poison left in the final noise: {result['n_poison_in_final_noise']:,}",
            "- True labels of wrongly removed rows: "
            + (", ".join(f"{label}={count:,}" for label, count in result["wrong_removed_true_labels"].items()) or "none"),
            "",
            "## Where the poison lands",
            "",
            "| | poison in noise | poison in pure clusters | clusters holding poison | largest poisoned cluster (rows / poison / pure) |",
            "|---|---|---|---|---|",
        ]
        for key, label in (("stage1", "Stage 1"), ("stage2", "Stage 2")):
            if result[key] is None:
                lines.append(f"| {label} | _not run_ | - | - | - |")
                continue
            placement = result[key]["placement"]
            lines.append(
                f"| {label} | {placement['poison_in_noise']:,} | {placement['poison_in_pure_clusters']:,} | "
                f"{placement['n_clusters_with_poison']:,} | {placement['largest_poison_cluster_n_records']:,} / "
                f"{placement['largest_poison_cluster_n_poisoned']:,} / {placement['largest_poison_cluster_purity_pct']}% |"
            )
        return "\n".join(lines) + "\n"
