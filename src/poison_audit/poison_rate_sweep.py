"""Component 1 - the poison-rate sweep (script 04): the whole audit of `end_to_end_pipeline.py` on every dataset x
trigger x poison rate, with the model's top-K features chosen on each poisoned training set.

How each cell is built:
- the poisoned sets are built here, at every rate, with `FixedTriggerPoisonDatasetBuilder` (seed
  `project.random_state`, or the evaluation seed). `RandomState.choice` without replacement takes a prefix of one
  fixed permutation, so the poisoned rows are nested across rates (0.5% within 1% within 2% ...); at the rate of an
  earlier run the poisoned ids are checked against that run's manifest when it is present;
- the model's features are not a fixed list: mRMR (`selection_feature.techniques.mrmr`) ranks every raw feature on the
  poisoned training set - what a defender retraining on received data would do - and the top-K are kept. A trigger
  feature that falls outside them cannot be seen by the model, so the backdoor may not form; the report says how many
  trigger features made the cut instead of letting that read as a defence success.
  The sampling seeds are disjoint from the poisoning seed: seed 42's stratified sample would hold nearly every
  poisoned row (`mrmr_reselection` in config).

Preparation (poisoned set + mRMR + manifest, `PoisonedCellPreparer`) is separate from the audit, so script 03 can
prepare every seed ahead of the audit and the baselines. The calibration seed (`project.random_state`) keeps the
original layout; every other poisoning seed gets its own `seed_<n>/` root (`seed_root`), and its model seed is the same
number.

Every step is cached in the cell directory (poisoned set, feature selection, models, clustering, removal masks,
retrains); a cell with a result is skipped, so a job cut off by its time limit resumes on resubmission.

The clean control (script 10) is one more cell per dataset and seed, `<dataset>/clean_control/rate_0pct`: the same
audit at poison rate 0, so every row it removes is a clean row. Nothing in the audit reads the trigger except the
trigger-test set, so one control per dataset covers all three triggers (`TriggerCatalog.clean_control`).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.logger import get_logger
from src.feature_selection.mrmr_selector import MrmrFeatureSelector
from src.poison_audit.end_to_end_pipeline import (
    RESULT_FILE, TREESHAP_REPRESENTATION, EndToEndPoisonAudit, build_layer_a_settings, build_variants,
)
from src.poison_audit.fixed_trigger_poison_builder import FixedTriggerPoisonDatasetBuilder
from src.poison_audit.poison_removal_system import PoisonRemovalBuilder, RecordVoteLabeler
from src.poison_audit.tabular_dataset_preparer import CSV_ARGS

logger = get_logger(__name__)

POISONED_TRAIN_FILE = "poisoned_train.csv.gz"
TRIGGER_TEST_FILE = "trigger_test.csv.gz"
POISON_FILE = "poison.json"
FEATURE_SELECTION_FILE = "feature_selection.json"
MANIFEST_FILE = "manifest.json"
SOURCE_FIXED = "fixed_trigger"
SOURCE_SPECIFIED = "specified_trigger"
SOURCE_NEW_DATASET = "new_dataset"
CLEAN_CONTROL = "clean_control"
# Definition of the control cell (no row is poisoned), not an experimental parameter.
CLEAN_CONTROL_RATE = 0.0


@dataclass(frozen=True)
class SweepTrigger:
    """One dataset x trigger of the sweep, resolved against the config blocks that define it.

    Attributes:
        dataset: Dataset name.
        name: Trigger family name used in the sweep (`representative`, `distribution_matching`, `explanation_guided`).
        trigger_spec: Feature -> injected value.
        source_label: Attack class whose rows are poisoned.
        target_label: Label they are flipped to.
        label_column: Label column.
        id_column: Row id column.
        clean_train: Clean training split.
        clean_test: Clean test split.
        reference_manifest: Manifest of the earlier single-rate run of this trigger (its poisoned ids are checked).
        top_k: Features the model keeps.
        min_cluster_size: HDBSCAN density floor of the dataset.
    """

    dataset: str
    name: str
    trigger_spec: Dict[str, float]
    source_label: str
    target_label: str
    label_column: str
    id_column: str
    clean_train: str
    clean_test: str
    reference_manifest: Optional[str]
    top_k: int
    min_cluster_size: int


class TriggerCatalog:
    """Resolves the sweep's dataset entries against the config blocks that already hold each trigger, so a trigger value
    lives in exactly one place.

    Args:
        config: Project config.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._audit = config["component1_poison_audit"]

    def resolve(self, entry: Dict[str, Any]) -> List[SweepTrigger]:
        """Every trigger of one sweep dataset entry.

        Args:
            entry: One item of `poison_rate_sweep.datasets`.

        Returns:
            The triggers, in config order.

        Raises:
            ValueError: For an unknown trigger source.
        """
        name, triggers = str(entry["name"]), []
        for item in entry["triggers"]:
            source = str(item["source"])
            if source == SOURCE_FIXED:
                fixed = self._audit["fixed_trigger_scenario"]
                spec, source_label, target_label = fixed["trigger_spec"], fixed["source_label"], fixed["target_label"]
                label_column, id_column = self._config["data"]["label_column"], self._config["data"]["sample_id_column"]
                clean_train, clean_test = entry["clean_train"], entry["clean_test"]
                reference = Path(fixed["output_dir"]) / "fixed_trigger_scenario_manifest.json"
            elif source == SOURCE_SPECIFIED:
                fixed = self._audit["fixed_trigger_scenario"]
                specified = fixed["specified_triggers"]
                scenario = next(s for s in specified["scenarios"] if s["name"] == item["scenario"])
                spec, source_label, target_label = scenario["trigger_spec"], specified["source_label"], fixed["target_label"]
                label_column, id_column = self._config["data"]["label_column"], self._config["data"]["sample_id_column"]
                clean_train, clean_test = entry["clean_train"], entry["clean_test"]
                reference = Path(specified["output_dir"]) / scenario["name"] / "specified_trigger_manifest.json"
            elif source == SOURCE_NEW_DATASET:
                new = self._audit["new_datasets"]
                dataset = next(d for d in new["datasets"] if d["name"] == name)
                scenario = next(s for s in dataset["scenarios"] if s["name"] == item["scenario"])
                spec, source_label, target_label = scenario["trigger_spec"], dataset["source_label"], dataset["target_label"]
                label_column, id_column = dataset["label_column"], self._config["data"]["sample_id_column"]
                root = Path(new["output_root"]) / name
                clean_train, clean_test = str(root / "clean" / "train.csv.gz"), str(root / "clean" / "test.csv.gz")
                reference = root / "poisoned" / scenario["name"] / "poison_manifest.json"
            else:
                raise ValueError(f"Invalid trigger source: {source!r} ({SOURCE_FIXED}, {SOURCE_SPECIFIED}, {SOURCE_NEW_DATASET}).")
            triggers.append(SweepTrigger(
                dataset=name, name=str(item["name"]), trigger_spec={str(k): float(v) for k, v in spec.items()},
                source_label=str(source_label), target_label=str(target_label), label_column=str(label_column),
                id_column=str(id_column), clean_train=str(clean_train), clean_test=str(clean_test),
                reference_manifest=reference.as_posix() if reference.exists() else None,
                top_k=int(entry["top_k"]), min_cluster_size=int(entry["min_cluster_size"]),
            ))
        return triggers

    def clean_control(self, entry: Dict[str, Any], trigger_name: str) -> SweepTrigger:
        """The clean control of one sweep dataset: its training set audited at poison rate 0.

        At rate 0 the builder poisons no row, so the training set is the clean split; the trigger only sets which
        trigger the trigger-test set carries (the ASR the control reports as a sanity check).

        Args:
            entry: One item of `poison_rate_sweep.datasets`.
            trigger_name: Trigger of that dataset whose trigger-test set the control uses.

        Returns:
            That trigger renamed `clean_control`, with no earlier run to check against.

        Raises:
            KeyError: If the dataset has no trigger of that name.
        """
        matches = [trigger for trigger in self.resolve(entry) if trigger.name == trigger_name]
        if not matches:
            raise KeyError(f"Dataset {entry['name']} has no trigger {trigger_name!r} for the control cell.")
        return replace(matches[0], name=CLEAN_CONTROL, reference_manifest=None)


def rate_tag(rate: float) -> str:
    """Directory name of a poison rate.

    Args:
        rate: Fraction of the source class, e.g. 0.005.

    Returns:
        E.g. `rate_0.5pct`.
    """
    return f"rate_{100 * rate:g}pct"


def seed_root(output_root: Path, seed: int, calibration_seed: int) -> Path:
    """Root of one poisoning seed's cells.

    Args:
        output_root: `poison_rate_sweep.output_dir`.
        seed: Poisoning (and model) seed.
        calibration_seed: `project.random_state` - the seed the configuration is chosen on.

    Returns:
        `output_root` itself for the calibration seed (the layout of every run so far), `output_root/seed_<n>` otherwise.
    """
    return output_root if seed == calibration_seed else output_root / f"seed_{seed}"


class PoisonedCellPreparer:
    """Builds the poisoned training set of every cell and chooses the model's top-K features on it by mRMR.

    A cell is prepared once its manifest exists; the audit (script 04) and the baselines only read prepared cells, so
    script 03 can prepare every seed ahead of them.

    Args:
        mrmr_sample_size: Rows per mRMR seed (`selection_feature.techniques.mrmr.sample_size`).
        mrmr_denominator: mRMR redundancy aggregation (same block).
        mrmr_n_jobs: Parallel jobs of the mutual-information estimators.
        selection_seeds: mRMR sampling seeds (disjoint from the poisoning seed).
        output_root: Root of this poisoning seed's cells (`seed_root`).
        random_state: Poisoning seed.
        check_reference: Compare the poisoned ids with the earlier single-rate runs' manifests - only meaningful at the
            seed those runs were built with.
    """

    def __init__(
        self,
        mrmr_sample_size: int,
        mrmr_denominator: str,
        mrmr_n_jobs: int,
        selection_seeds: List[int],
        output_root: Path,
        random_state: int,
        check_reference: bool,
    ) -> None:
        self._mrmr = (mrmr_sample_size, mrmr_denominator, mrmr_n_jobs)
        self._seeds = selection_seeds
        self._root = output_root
        self._seed = random_state
        self._check_reference_ids = check_reference

    @property
    def output_root(self) -> Path:
        """Root of this poisoning seed's cells."""
        return self._root

    def cell_dir(self, trigger: SweepTrigger, rate: float) -> Path:
        """Directory of one cell.

        Args:
            trigger: The trigger.
            rate: Poison rate.

        Returns:
            `output_root/dataset/trigger/rate_<x>pct`.
        """
        return self._root / trigger.dataset / trigger.name / rate_tag(rate)

    def trigger_test_path(self, trigger: SweepTrigger) -> Path:
        """Source-class test rows with the trigger injected, shared by every rate of a trigger.

        Args:
            trigger: The trigger.

        Returns:
            `output_root/dataset/trigger/trigger_test.csv.gz`.
        """
        return self._root / trigger.dataset / trigger.name / TRIGGER_TEST_FILE

    def is_prepared(self, trigger: SweepTrigger, rate: float) -> bool:
        """Whether a cell has its manifest (written last, after its poisoned set and feature selection).

        Args:
            trigger: The trigger.
            rate: Poison rate.

        Returns:
            True when prepared.
        """
        return (self.cell_dir(trigger, rate) / MANIFEST_FILE).exists()

    def prepare(self, trigger: SweepTrigger, rates: List[float]) -> None:
        """Prepare every rate of one trigger that is not prepared yet; each clean split is read at most once.

        Args:
            trigger: The trigger.
            rates: Poison rates.
        """
        pending = [rate for rate in rates if not self.is_prepared(trigger, rate)]
        test_path = self.trigger_test_path(trigger)
        if not pending and test_path.exists():
            return
        builder = FixedTriggerPoisonDatasetBuilder(trigger.trigger_spec, trigger.source_label, trigger.target_label,
                                                   trigger.label_column, trigger.id_column)
        if not test_path.exists():
            clean_test = pd.read_csv(trigger.clean_test, float_precision="round_trip")
            self._write_csv(builder.build_trigger_test(clean_test), test_path)
            del clean_test
        if not pending:
            return
        logger.info("[%s/%s] Reading the clean sets...", trigger.dataset, trigger.name)
        clean_train = pd.read_csv(trigger.clean_train, float_precision="round_trip")
        for rate in pending:
            self._prepare_cell(trigger, rate, builder, clean_train, self.cell_dir(trigger, rate))

    def _prepare_cell(
        self, trigger: SweepTrigger, rate: float, builder: FixedTriggerPoisonDatasetBuilder, clean_train: pd.DataFrame,
        cell: Path,
    ) -> None:
        """Poison the training set, choose the top-K features on it and write the cell's manifest.

        Args:
            trigger: The trigger.
            rate: Poison rate.
            builder: Injection builder of the trigger.
            clean_train: Clean training split.
            cell: Cell directory.

        Raises:
            ValueError: If the poisoned ids at the rate of an earlier run differ from that run's.
        """
        cell.mkdir(parents=True, exist_ok=True)
        poisoned_train, poisoned_ids = builder.build_poisoned_train(clean_train, poison_rate=rate, seed=self._seed)
        if self._check_reference_ids:
            self._check_reference(trigger, rate, poisoned_ids)
        if not (cell / POISON_FILE).exists():
            self._write_csv(poisoned_train, cell / POISONED_TRAIN_FILE)
            (cell / POISON_FILE).write_text(json.dumps({"poison_rate": rate, "n_poisoned": int(len(poisoned_ids))}), encoding="utf-8")
        selection_path = cell / FEATURE_SELECTION_FILE
        if selection_path.exists():
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
        else:
            candidates = [column for column in poisoned_train.columns if column not in (trigger.id_column, trigger.label_column)]
            logger.info("[%s/%s/%s] mRMR on %d features, seeds %s...", trigger.dataset, trigger.name, rate_tag(rate),
                        len(candidates), self._seeds)
            selector = MrmrFeatureSelector(trigger.label_column, *self._mrmr)
            ranked = selector.rank(poisoned_train, candidates, self._seeds, poisoned_ids=[int(value) for value in poisoned_ids],
                                   id_column=trigger.id_column)
            ranking = list(ranked["consensus_ranking"])
            selected = ranking[: trigger.top_k]
            selection = {
                "top_k": trigger.top_k, "selected": selected, "consensus_ranking": ranking,
                "per_seed_sample": ranked["per_seed_sample"], "seeds": self._seeds,
                "trigger_rank": {feature: ranking.index(feature) + 1 for feature in trigger.trigger_spec},
                "trigger_in_top_k": sum(feature in selected for feature in trigger.trigger_spec),
            }
            selection_path.write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("[%s/%s/%s] Top-%d features: %d/%d trigger features made it (ranks %s).", trigger.dataset, trigger.name,
                    rate_tag(rate), trigger.top_k, selection["trigger_in_top_k"], len(trigger.trigger_spec), selection["trigger_rank"])
        manifest = {
            "dataset": trigger.dataset, "scenario_id": trigger.name, "trigger_spec": trigger.trigger_spec,
            "source_label": trigger.source_label, "target_label": trigger.target_label,
            "label_column": trigger.label_column, "id_column": trigger.id_column, "model_features": selection["selected"],
            "poison_rate": rate, "poison_rate_basis": "source_class", "seed": self._seed,
            "n_poisoned": int(len(poisoned_ids)), "poisoned_sample_ids": [int(value) for value in poisoned_ids],
        }
        temporary = cell / f"{MANIFEST_FILE}.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, cell / MANIFEST_FILE)

    @staticmethod
    def _check_reference(trigger: SweepTrigger, rate: float, poisoned_ids: np.ndarray) -> None:
        """At the rate of the earlier single-rate run, the rebuilt poisoned ids must be that run's.

        Args:
            trigger: The trigger.
            rate: Poison rate.
            poisoned_ids: Rebuilt poisoned ids.

        Raises:
            ValueError: On a mismatch.
        """
        if trigger.reference_manifest is None:
            return
        reference = json.loads(Path(trigger.reference_manifest).read_text(encoding="utf-8"))
        if abs(float(reference.get("poison_rate", -1)) - rate) > 1e-12:
            return
        expected = np.sort(np.asarray(reference["poisoned_sample_ids"], dtype=np.int64))
        if not np.array_equal(np.sort(poisoned_ids.astype(np.int64)), expected):
            raise ValueError(f"[{trigger.dataset}/{trigger.name}] Rebuilding at {rate:g} poisons OTHER rows than "
                             f"{trigger.reference_manifest} - stopping instead of auditing the wrong dataset.")
        logger.info("[%s/%s] The poisoned set at %g matches the earlier build exactly (%d rows).", trigger.dataset, trigger.name, rate,
                    len(expected))

    @staticmethod
    def _write_csv(frame: pd.DataFrame, path: Path) -> None:
        """Write a CSV under a temporary name, then rename it.

        Args:
            frame: Rows to write.
            path: Destination.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        frame.to_csv(temporary, **CSV_ARGS)
        os.replace(temporary, path)


class PoisonRateSweep:
    """Audits every cell of the sweep, preparing the cells that are not prepared yet.

    Args:
        audit: The end-to-end audit, configured once for every cell.
        preparer: Builds the poisoned set and the feature selection of a cell.
        drop_shap: Delete a cell's TreeSHAP table once its result is written (it is regenerable and ~1 GB on
            CIC-IDS2017; a finished cell is never rerun).
    """

    def __init__(self, audit: EndToEndPoisonAudit, preparer: PoisonedCellPreparer, drop_shap: bool) -> None:
        self._audit = audit
        self._preparer = preparer
        self._drop_shap = drop_shap

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], output_root: Path, seed: int, audit: Optional[EndToEndPoisonAudit] = None,
    ) -> "PoisonRateSweep":
        """The sweep as the `poison_rate_sweep` block sets it up (shared by scripts 04, 10 and 13).

        Args:
            config: Project config.
            output_root: Root of this seed's cells (`seed_root`).
            seed: Poisoning seed and seed of every model.
            audit: Audit to run on each cell; None builds the block's full audit (`build_audit`). Script 13 passes one
                restricted to the locked configuration, to rebuild cells faster.

        Returns:
            The sweep.
        """
        scenario_cfg = config["component1_poison_audit"]["fixed_trigger_scenario"]
        cfg = scenario_cfg["poison_rate_sweep"]
        audit = audit if audit is not None else cls.build_audit(config, seed)
        selection_cfg = config["selection_feature"]
        mrmr_cfg = selection_cfg["techniques"]["mrmr"]
        seed_set = next(item for item in scenario_cfg["mrmr_reselection"]["seed_sets"] if item["name"] == cfg["feature_selection_seed_set"])
        preparer = PoisonedCellPreparer(
            mrmr_sample_size=int(mrmr_cfg["sample_size"]), mrmr_denominator=str(mrmr_cfg["denominator"]),
            mrmr_n_jobs=int(selection_cfg["execution_n_jobs"]), selection_seeds=[int(value) for value in seed_set["seeds"]],
            output_root=output_root, random_state=seed, check_reference=seed == int(config["project"]["random_state"]),
        )
        return cls(audit=audit, preparer=preparer, drop_shap=bool(cfg["drop_shap"]))

    @staticmethod
    def build_audit(
        config: Dict[str, Any], seed: int, representation: str = TREESHAP_REPRESENTATION,
        layer_a_names: Optional[List[str]] = None, variant_names: Optional[List[str]] = None,
        references: Optional[List[str]] = None,
    ) -> EndToEndPoisonAudit:
        """The end-to-end audit of the sweep's cells, as the `poison_rate_sweep` block sets it up.

        Args:
            config: Project config.
            seed: Seed of every model.
            representation: What stage 1 clusters (`treeshap`, or `raw` for the representation ablation of script 12).
            layer_a_names: Layer-A setting names to keep (e.g. `k25_eta0.5_kept`); None keeps the block's list.
            variant_names: Removal variant names to keep (e.g. `and_cluster_noise0.5`); None keeps the block's list.
            references: References to retrain; None keeps the block's list.

        Returns:
            The audit.

        Raises:
            ValueError: If a requested layer-A setting or variant is not in the block.
        """
        audit_cfg = config["component1_poison_audit"]
        scenario_cfg = audit_cfg["fixed_trigger_scenario"]
        cfg = scenario_cfg["poison_rate_sweep"]
        removal_cfg, layer_cfg, knn_cfg = scenario_cfg["poison_removal_system"], scenario_cfg["layer_b"], scenario_cfg["knn_label_consistency"]
        model_params = dict(audit_cfg["backdoor_eval"]["model_params"])
        ex_cfg = layer_cfg["exemplars"]
        builder = PoisonRemovalBuilder(
            model_params, float(removal_cfg["classifier_threshold"]), int(removal_cfg["max_compared_records"]),
            float(ex_cfg["tolerance"]), int(ex_cfg["query_block_size"]), int(ex_cfg["pool_block_size"]),
            RecordVoteLabeler(tie_label=str(scenario_cfg["record_voting"]["tie_break_label"])), seed,
        )
        layer_a = build_layer_a_settings(list(cfg["layer_a_settings"]), cfg.get("benign_references"))
        variants = build_variants(list(cfg["variants"]))
        if layer_a_names is not None:
            layer_a = [setting for setting in layer_a if setting.name in layer_a_names]
        if variant_names is not None:
            variants = [variant for variant in variants if variant.name in variant_names]
        if (layer_a_names is not None and len(layer_a) != len(set(layer_a_names))) or \
                (variant_names is not None and len(variants) != len(set(variant_names))):
            raise ValueError(f"Layer A {layer_a_names} / variants {variant_names} not all found in poison_rate_sweep.")
        return EndToEndPoisonAudit(
            config=config, model_params=model_params, clustering_params=dict(audit_cfg["clustering_hdbscan"]),
            shap_batch_size=int(audit_cfg["shap_batch_size"]),
            purity_threshold_pct=float(scenario_cfg["end_to_end"]["purity_threshold_pct"]),
            layer_a_settings=layer_a,
            distance_block_size=int(knn_cfg["distance_block_size"]), read_chunk_size=int(knn_cfg["read_chunk_size"]),
            feature_set=str(knn_cfg["feature_set"]), builder=builder, variants=variants,
            references=[str(name) for name in (references if references is not None else cfg["references"])],
            random_state=seed, memory_sample_interval_seconds=float(cfg["memory_sample_interval_seconds"]),
            representation=representation,
        )

    def cell_dir(self, trigger: SweepTrigger, rate: float) -> Path:
        """Directory of one cell.

        Args:
            trigger: The trigger.
            rate: Poison rate.

        Returns:
            The preparer's cell directory.
        """
        return self._preparer.cell_dir(trigger, rate)

    def run_trigger(
        self, trigger: SweepTrigger, rates: List[float], prepare_only: bool = False, refresh: bool = False,
    ) -> None:
        """Every rate of one trigger.

        Args:
            trigger: The trigger.
            rates: Poison rates to run.
            prepare_only: Stop after the poisoned set, the feature selection and the manifest (no audit) - a smoke test
                of the sweep's own steps on a machine that cannot fit HDBSCAN on CIC-IDS2017.
            refresh: Audit cells that already have a result again - after new layer-A settings or removal variants
                were added to config. Models, clustering and every retrain already measured are reused; only the
                removals and the retrains that are missing are computed.
        """
        pending = [rate for rate in rates if refresh or not (self.cell_dir(trigger, rate) / RESULT_FILE).exists()]
        for rate in sorted(set(rates) - set(pending)):
            logger.info("[%s/%s/%s] Already finished, skipped.", trigger.dataset, trigger.name, rate_tag(rate))
        if not pending:
            return
        self._preparer.prepare(trigger, pending)
        if prepare_only:
            return
        trigger_test_path = self._preparer.trigger_test_path(trigger)
        for rate in pending:
            started = time.time()
            cell = self.cell_dir(trigger, rate)
            name = f"{trigger.dataset}/{trigger.name}/{rate_tag(rate)}"
            logger.info("=== %s ===", name)
            scenario = {
                "name": name, "clean_train": trigger.clean_train, "clean_test": trigger.clean_test,
                "poisoned_train": (cell / POISONED_TRAIN_FILE).as_posix(), "trigger_test": trigger_test_path.as_posix(),
                "manifest": (cell / MANIFEST_FILE).as_posix(), "min_cluster_size": trigger.min_cluster_size,
            }
            self._audit.run(scenario, cell, clean_model_dir=self._preparer.output_root / trigger.dataset)
            if self._drop_shap:
                for shap_table in cell.glob("shap*_contribution.csv.gz"):
                    shap_table.unlink()
            logger.info("[%s] Cell done after %.1f min.", name, (time.time() - started) / 60)
