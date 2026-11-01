"""The whole system with stage 1 clustering the RAW feature values instead of TreeSHAP attributions - the "raw
features" row of the ablation table in the paper (config block `raw_space_audit`).

Each cell of script 04 (`poison_rate_sweep/seed_<n>/<dataset>/<trigger>/rate_<x>pct`) is audited again with the sweep's
own `EndToEndPoisonAudit` (`PoisonRateSweep.build_audit`) and `representation = raw`: HDBSCAN (same block, full fit,
per-dataset `min_cluster_size`) on the model's 30 mRMR features, then layer A, layer B, the noise rule and retraining
unchanged, only for the selected configuration (`raw_space_audit.layer_a`, `variants`) and the `true_poison` reference.
The inputs (poisoned set, triggered test set, manifest, audited model, clean model) come from the cell and dataset
folders of script 04, so nothing is rebuilt, those two models are not retrained and no TreeSHAP is computed.

Default seeds: the 5 `selection_feature.evaluation_seeds` - on a SLURM cluster, one job per seed (`--seed`). Finished
cells are skipped, so a job that runs out of time only has to be submitted again. The run ends by rebuilding
`raw_space_audit_report.md`, `raw_space_audit_report.csv` (per dataset, raw features next to TreeSHAP) and
`raw_space_audit_sets.csv` (one row per set) in `raw_space_audit.output_dir`.

    python scripts/component1_poison_audit/12_raw_space_audit.py --check
    python scripts/component1_poison_audit/12_raw_space_audit.py --seed 8459
    python scripts/component1_poison_audit/12_raw_space_audit.py --seed 8459 --dataset edge_iiot --rate 0.01
    python scripts/component1_poison_audit/12_raw_space_audit.py --report-only
"""

from __future__ import annotations

import argparse
import fnmatch
import gc
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.logger import get_logger
from src.core.utils import load_config, set_seed
from src.poison_audit.end_to_end_pipeline import RAW_REPRESENTATION, RESULT_FILE
from src.poison_audit.poison_rate_sweep import (MANIFEST_FILE, POISONED_TRAIN_FILE, TRIGGER_TEST_FILE, PoisonRateSweep,
                                                SweepTrigger, TriggerCatalog, rate_tag, seed_root)
from src.poison_audit.raw_space_clustering import RawSpaceAblationReport, RawSpaceClusteringStudy, RawSpaceReport

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", action="append", default=None, help="Dataset name pattern (repeatable); default: every dataset.")
    parser.add_argument("--trigger", action="append", default=None, help="Trigger name pattern (repeatable); default: every trigger.")
    parser.add_argument("--rate", action="append", type=float, default=None, help="Poison rate (repeatable), for example 0.01.")
    parser.add_argument("--seed", action="append", type=int, default=None,
                        help="Seed (repeatable); default: the 5 evaluation_seeds.")
    parser.add_argument("--sweep-dir", default=None, help="Overrides poison_rate_sweep.output_dir (where the inputs are read).")
    parser.add_argument("--output-dir", default=None, help="Overrides raw_space_audit.output_dir.")
    parser.add_argument("--report-only", action="store_true", help="Only rebuild the report from the finished cells.")
    parser.add_argument("--check", action="store_true", help="Only check the inputs and count the cells that would run.")
    return parser.parse_args()


def pick(values: List[str], patterns: Optional[List[str]]) -> List[str]:
    """The values that match at least one pattern (no pattern keeps them all).

    Args:
        values: Values.
        patterns: fnmatch patterns.

    Returns:
        The kept values, in order.
    """
    return [value for value in values if not patterns or any(fnmatch.fnmatch(value, pattern) for pattern in patterns)]


def cell_paths(sweep_root: Path, output_root: Path, trigger: SweepTrigger, rate: float) -> Tuple[Path, Path]:
    """The cell of script 04 and the cell of this script for the same set.

    Args:
        sweep_root: Root of the seed in `poison_rate_sweep.output_dir`.
        output_root: Root of the seed in `raw_space_audit.output_dir`.
        trigger: Trigger.
        rate: Poison rate.

    Returns:
        (cell of script 04, cell of this script).
    """
    relative = Path(trigger.dataset) / trigger.name / rate_tag(rate)
    return sweep_root / relative, output_root / relative


def missing_inputs(sweep_cell: Path) -> List[str]:
    """The inputs from script 04 that a cell still lacks.

    Args:
        sweep_cell: The cell of script 04.

    Returns:
        Names of the missing files (empty when complete).
    """
    needed = [sweep_cell / name for name in (MANIFEST_FILE, POISONED_TRAIN_FILE, RESULT_FILE)]
    needed.append(sweep_cell.parent / TRIGGER_TEST_FILE)
    return [path.name for path in needed if not path.exists()]


def build_report(config: Dict[str, Any], triggers: List[SweepTrigger], rates: List[float], seeds: List[int],
                 sweep_dir: Path, output_dir: Path, calibration_seed: int) -> None:
    """Rebuild the report from every finished cell of every dataset and report seed.

    Args:
        config: Project config.
        triggers: Every trigger of the sweep.
        rates: Every rate.
        seeds: Seeds of the report.
        sweep_dir: `poison_rate_sweep.output_dir`.
        output_dir: `raw_space_audit.output_dir`.
        calibration_seed: `project.random_state`.
    """
    scenario_cfg = config["component1_poison_audit"]["fixed_trigger_scenario"]
    cfg, criterion = scenario_cfg["raw_space_audit"], scenario_cfg["calibration_report"]
    clustering_cfg = scenario_cfg["raw_space_clustering"]
    titles = dict(scenario_cfg["dataset_titles"])
    variant = f"{cfg['layer_a'][0]}_{cfg['variants'][0]}"
    pairs, clusterings, missing, expected = [], [], [], {}
    for trigger in triggers:
        expected[trigger.dataset] = expected.get(trigger.dataset, 0) + len(rates) * len(seeds)
        for seed in seeds:
            for rate in rates:
                sweep_cell, cell = cell_paths(seed_root(sweep_dir, seed, calibration_seed),
                                              seed_root(output_dir, seed, calibration_seed), trigger, rate)
                if not (cell / RESULT_FILE).exists() or not (sweep_cell / RESULT_FILE).exists():
                    missing.append(f"{seed}/{trigger.dataset}/{trigger.name}/{rate_tag(rate)}")
                    continue
                raw = json.loads((cell / RESULT_FILE).read_text(encoding="utf-8"))
                pairs.append({"dataset": trigger.dataset, "trigger": trigger.name, "rate": rate, "seed": seed,
                              "target_label": trigger.target_label, "raw": raw,
                              "treeshap": json.loads((sweep_cell / RESULT_FILE).read_text(encoding="utf-8"))})
                # Stage 1 of this cell is exactly what script 11 measures: record it too, so script 11 need not run.
                clusterings.append({"dataset": trigger.dataset, "trigger": trigger.name, "rate": rate, "seed": seed,
                                    "n_rows": raw["n_rows"], "n_poisoned": raw["n_poisoned"],
                                    "raw": RawSpaceClusteringStudy.stage_summary(cell),
                                    "attribution": RawSpaceClusteringStudy.stage_summary(sweep_cell)})
    output_dir.mkdir(parents=True, exist_ok=True)
    report = RawSpaceAblationReport(variant, str(cfg["references"][0]), float(criterion["backdoor_removed_margin"]),
                                    float(criterion["max_cda_drop"]), titles)
    sets = report.sets(pairs)
    summary = report.summary(sets) if len(sets) else sets
    sets.to_csv(output_dir / str(cfg["sets_csv"]), index=False)
    summary.to_csv(output_dir / str(cfg["report_csv"]), index=False)
    (output_dir / str(cfg["report_file"])).write_text(report.markdown(summary, expected, missing) if len(sets) else
                                                      "# No finished cell yet.\n", encoding="utf-8")
    clustering_report = RawSpaceReport(titles)
    clustering_sets = clustering_report.sets(clusterings)
    clustering_summary = clustering_report.summary(clustering_sets) if len(clustering_sets) else clustering_sets
    clustering_sets.to_csv(output_dir / str(clustering_cfg["sets_csv"]), index=False)
    clustering_summary.to_csv(output_dir / str(clustering_cfg["report_csv"]), index=False)
    (output_dir / str(clustering_cfg["report_file"])).write_text(
        clustering_report.markdown(clustering_summary, expected, missing) if len(clustering_sets) else "# No finished cell yet.\n",
        encoding="utf-8")
    logger.info("Report of %d/%d sets: %s and %s", len(pairs), len(pairs) + len(missing), output_dir / str(cfg["report_file"]),
                output_dir / str(clustering_cfg["report_file"]))


def main() -> None:
    """Run the selected cells, then rebuild the report."""
    args = parse_args()
    config = load_config()
    calibration_seed = int(config["project"]["random_state"])
    evaluation_seeds = [int(value) for value in config["selection_feature"]["evaluation_seeds"]]
    seeds = args.seed or evaluation_seeds
    set_seed(seeds[0])
    scenario_cfg = config["component1_poison_audit"]["fixed_trigger_scenario"]
    sweep_cfg, cfg = scenario_cfg["poison_rate_sweep"], scenario_cfg["raw_space_audit"]
    sweep_dir = Path(args.sweep_dir or sweep_cfg["output_dir"])
    output_dir = Path(args.output_dir or cfg["output_dir"])

    catalog = TriggerCatalog(config)
    all_triggers = [trigger for entry in sweep_cfg["datasets"] for trigger in catalog.resolve(entry)]
    all_rates = [float(rate) for rate in sweep_cfg["poison_rates"]]
    datasets = pick(list(dict.fromkeys(trigger.dataset for trigger in all_triggers)), args.dataset)
    triggers = [trigger for trigger in all_triggers if trigger.dataset in datasets and pick([trigger.name], args.trigger)]
    rates = [rate for rate in all_rates if not args.rate or any(abs(rate - wanted) < 1e-12 for wanted in args.rate)]
    if not triggers or not rates:
        raise ValueError(f"No cell matches --dataset {args.dataset} --trigger {args.trigger} --rate {args.rate}.")

    order = [str(name) for name in cfg["dataset_order"]]
    triggers = sorted(triggers, key=lambda trigger: order.index(trigger.dataset) if trigger.dataset in order else len(order))
    cells = [(seed, trigger, rate, *cell_paths(seed_root(sweep_dir, seed, calibration_seed),
                                                seed_root(output_dir, seed, calibration_seed), trigger, rate))
             for seed in seeds for trigger in triggers for rate in rates]
    if args.check:
        incomplete = [(cell, missing_inputs(cell[3])) for cell in cells if missing_inputs(cell[3])]
        for (seed, trigger, rate, _, _), absent in incomplete[:10]:
            logger.info("MISSING at seed %d %s/%s/%s: %s", seed, trigger.dataset, trigger.name, rate_tag(rate), absent)
        logger.info("%d cells in total; %d with every input from script 04; %d still with the audited model; %d finished.", len(cells),
                    len(cells) - len(incomplete), sum(any(cell[3].glob("dirty_model_*feat.pkl")) for cell in cells),
                    sum((cell[4] / RESULT_FILE).exists() for cell in cells))
        return

    if not args.report_only:
        audits = {seed: PoisonRateSweep.build_audit(config, seed, RAW_REPRESENTATION, [str(name) for name in cfg["layer_a"]],
                                                    [str(name) for name in cfg["variants"]],
                                                    [str(name) for name in cfg["references"]]) for seed in seeds}
        for seed, trigger, rate, sweep_cell, output_cell in cells:
            name = f"{trigger.dataset}/{trigger.name}/{rate_tag(rate)}"
            if (output_cell / RESULT_FILE).exists():
                logger.info("[seed %d %s] Already finished, skipped.", seed, name)
                continue
            absent = missing_inputs(sweep_cell)
            if absent:
                logger.warning("[seed %d %s] Cell of script 04 lacks %s, skipped.", seed, name, absent)
                continue
            set_seed(seed)
            started = time.time()
            output_cell.mkdir(parents=True, exist_ok=True)
            # The audited model of script 04 (same set, same seed): copied over so the audit does not retrain it.
            for model in sweep_cell.glob("dirty_model_*feat.pkl"):
                if not (output_cell / model.name).exists():
                    shutil.copy2(model, output_cell / model.name)
            scenario = {
                "name": name, "clean_train": trigger.clean_train, "clean_test": trigger.clean_test,
                "poisoned_train": (sweep_cell / POISONED_TRAIN_FILE).as_posix(),
                "trigger_test": (sweep_cell.parent / TRIGGER_TEST_FILE).as_posix(),
                "manifest": (sweep_cell / MANIFEST_FILE).as_posix(), "min_cluster_size": trigger.min_cluster_size,
            }
            audits[seed].run(scenario, output_cell, clean_model_dir=seed_root(sweep_dir, seed, calibration_seed) / trigger.dataset)
            logger.info("[seed %d %s] Cell done after %.1f min.", seed, name, (time.time() - started) / 60)
            gc.collect()

    build_report(config, all_triggers, all_rates, evaluation_seeds, sweep_dir, output_dir, calibration_seed)


if __name__ == "__main__":
    main()
