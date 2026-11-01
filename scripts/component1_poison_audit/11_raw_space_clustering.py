"""Cluster each poisoned set on the RAW feature values instead of TreeSHAP attributions, to compare the two spaces
(an ablation of the paper; src/poison_audit/raw_space_clustering.py, config block `raw_space_clustering`).

Each cell of script 04 (`poison_rate_sweep/seed_<n>/<dataset>/<trigger>/rate_<x>pct`) is clustered again with the
audit's own HDBSCAN (block `clustering_hdbscan`, full fit, per-dataset `min_cluster_size`) on the 30 mRMR features of
the model, then scored by the same evaluator: where the poisoned rows land (noise, pure clusters, number of clusters
holding poison) and the LABOR rule with the true cluster labels. The attribution-space numbers come from the cell's own
`pipeline_result.json`, so the two sides differ only in the clustering space.

Default seeds: the 5 `selection_feature.evaluation_seeds`. Finished cells are skipped, so a job that runs out of time
only has to be submitted again. The run ends by rebuilding `raw_space_report.md`, `raw_space_report.csv` (medians per
dataset) and `raw_space_sets.csv` (one row per set) in `raw_space_clustering.output_dir` from every finished cell.

    python scripts/component1_poison_audit/11_raw_space_clustering.py --check
    python scripts/component1_poison_audit/11_raw_space_clustering.py --dataset edge_iiot --dataset ton_iot
    python scripts/component1_poison_audit/11_raw_space_clustering.py --dataset cicids2017 --seed 8459 --trigger representative
    python scripts/component1_poison_audit/11_raw_space_clustering.py --report-only
"""

from __future__ import annotations

import argparse
import fnmatch
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.logger import get_logger
from src.core.utils import load_config, set_seed
from src.poison_audit.poison_rate_sweep import (MANIFEST_FILE, POISONED_TRAIN_FILE, SweepTrigger, TriggerCatalog,
                                                rate_tag, seed_root)
from src.poison_audit.raw_space_clustering import PIPELINE_RESULT_FILE, RESULT_FILE, RawSpaceClusteringStudy, RawSpaceReport

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
    parser.add_argument("--output-dir", default=None, help="Overrides raw_space_clustering.output_dir.")
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
        output_root: Root of the seed in `raw_space_clustering.output_dir`.
        trigger: Trigger.
        rate: Poison rate.

    Returns:
        (cell of script 04, cell of this script).
    """
    relative = Path(trigger.dataset) / trigger.name / rate_tag(rate)
    return sweep_root / relative, output_root / relative


def is_ready(sweep_cell: Path) -> bool:
    """Whether the cell of script 04 is finished (it has the manifest and the audit's clustering result).

    Args:
        sweep_cell: The cell of script 04.

    Returns:
        True when the inputs are there.
    """
    return (sweep_cell / MANIFEST_FILE).exists() and (sweep_cell / PIPELINE_RESULT_FILE).exists()


def build_report(config: Dict[str, Any], triggers: List[SweepTrigger], rates: List[float], seeds: List[int],
                 sweep_dir: Path, output_dir: Path, calibration_seed: int) -> None:
    """Rebuild the report from every finished cell of every dataset and report seed.

    Args:
        config: Project config.
        triggers: Every trigger of the sweep.
        rates: Every rate.
        seeds: Seeds of the report.
        sweep_dir: `poison_rate_sweep.output_dir`.
        output_dir: `raw_space_clustering.output_dir`.
        calibration_seed: `project.random_state`.
    """
    scenario_cfg = config["component1_poison_audit"]["fixed_trigger_scenario"]
    cfg = scenario_cfg["raw_space_clustering"]
    results, missing, expected = [], [], {}
    for trigger in triggers:
        expected[trigger.dataset] = expected.get(trigger.dataset, 0) + len(rates) * len(seeds)
        for seed in seeds:
            for rate in rates:
                _, cell = cell_paths(seed_root(sweep_dir, seed, calibration_seed), seed_root(output_dir, seed, calibration_seed),
                                     trigger, rate)
                if (cell / RESULT_FILE).exists():
                    results.append(json.loads((cell / RESULT_FILE).read_text(encoding="utf-8")))
                else:
                    missing.append(f"{seed}/{trigger.dataset}/{trigger.name}/{rate_tag(rate)}")
    report = RawSpaceReport(dict(scenario_cfg["dataset_titles"]))
    sets = report.sets(results)
    summary = report.summary(sets) if len(sets) else sets
    output_dir.mkdir(parents=True, exist_ok=True)
    sets.to_csv(output_dir / str(cfg["sets_csv"]), index=False)
    summary.to_csv(output_dir / str(cfg["report_csv"]), index=False)
    (output_dir / str(cfg["report_file"])).write_text(report.markdown(summary, expected, missing) if len(sets) else
                                                      "# No finished cell yet.\n", encoding="utf-8")
    logger.info("Report of %d/%d sets: %s", len(results), len(results) + len(missing), output_dir / str(cfg["report_file"]))


def main() -> None:
    """Run the selected cells, then rebuild the report."""
    args = parse_args()
    config = load_config()
    calibration_seed = int(config["project"]["random_state"])
    seeds = args.seed or [int(value) for value in config["selection_feature"]["evaluation_seeds"]]
    set_seed(seeds[0])
    audit_cfg = config["component1_poison_audit"]
    scenario_cfg = audit_cfg["fixed_trigger_scenario"]
    sweep_cfg, cfg = scenario_cfg["poison_rate_sweep"], scenario_cfg["raw_space_clustering"]
    sweep_dir, output_dir = Path(sweep_cfg["output_dir"]), Path(args.output_dir or cfg["output_dir"])

    catalog = TriggerCatalog(config)
    all_triggers = [trigger for entry in sweep_cfg["datasets"] for trigger in catalog.resolve(entry)]
    all_rates = [float(rate) for rate in sweep_cfg["poison_rates"]]
    datasets = pick(list(dict.fromkeys(trigger.dataset for trigger in all_triggers)), args.dataset)
    triggers = [trigger for trigger in all_triggers if trigger.dataset in datasets and pick([trigger.name], args.trigger)]
    rates = [rate for rate in all_rates if not args.rate or any(abs(rate - wanted) < 1e-12 for wanted in args.rate)]
    if not triggers or not rates:
        raise ValueError(f"No cell matches --dataset {args.dataset} --trigger {args.trigger} --rate {args.rate}.")

    cells = [(seed, trigger, rate, *cell_paths(seed_root(sweep_dir, seed, calibration_seed),
                                                seed_root(output_dir, seed, calibration_seed), trigger, rate))
             for seed in seeds for trigger in triggers for rate in rates]
    if args.check:
        ready = [cell for cell in cells if is_ready(cell[3])]
        logger.info("%d cells in total; %d with the results of script 04; %d still with poisoned_train, %d still with the audited model; %d finished.",
                    len(cells), len(ready), sum((cell[3] / POISONED_TRAIN_FILE).exists() for cell in ready),
                    sum(any(cell[3].glob("dirty_model_*feat.pkl")) for cell in ready),
                    sum((cell[4] / RESULT_FILE).exists() for cell in cells))
        return

    if not args.report_only:
        study = RawSpaceClusteringStudy(
            clustering_params=dict(audit_cfg["clustering_hdbscan"]),
            purity_threshold_pct=float(scenario_cfg["end_to_end"]["purity_threshold_pct"]),
            model_params=dict(audit_cfg["backdoor_eval"]["model_params"]), arm_name=str(cfg["arm_name"]),
        )
        for seed, trigger, rate, sweep_cell, output_cell in cells:
            name = f"seed {seed} {trigger.dataset}/{trigger.name}/{rate_tag(rate)}"
            if (output_cell / RESULT_FILE).exists():
                logger.info("[%s] Already finished, skipped.", name)
                continue
            if not is_ready(sweep_cell):
                logger.warning("[%s] Cell of script 04 not finished (%s), skipped.", name, sweep_cell)
                continue
            set_seed(seed)
            started = time.time()
            result = study.run_cell(trigger, rate, seed, sweep_cell, output_cell)
            logger.info("[%s] Done after %.1f min: clustered poisoned rows in pure clusters %.1f%% (attribution %.1f%%), "
                        "LABOR rule with true labels precision %.1f%% (attribution %.1f%%).", name, (time.time() - started) / 60,
                        100 * result["raw"]["clustered_poison_in_pure_share"],
                        100 * result["attribution"]["clustered_poison_in_pure_share"],
                        100 * result["raw"]["oracle_rule_precision"], 100 * result["attribution"]["oracle_rule_precision"])
            gc.collect()

    build_report(config, all_triggers, all_rates, [int(value) for value in config["selection_feature"]["evaluation_seeds"]],
                 sweep_dir, output_dir, calibration_seed)


if __name__ == "__main__":
    main()
