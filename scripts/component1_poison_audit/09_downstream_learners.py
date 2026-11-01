"""Is the cleaned dataset also clean for learners other than XGBoost (src/poison_audit/downstream_learners.py, config
block `downstream_learners`)?

For each cell of the poison-rate sweep (script 04): train LightGBM and RandomForest (default parameters) on what each
removal variant keeps - the masks are read from the cell's `removal_masks.npz` - plus the two references `no_removal`
and `true_poison`, and measure CDA / macro-F1 / ASR as the XGBoost retrainings of the sweep do (the XGBoost column is
read from the sweep results). Every training is cached.

    python scripts/component1_poison_audit/09_downstream_learners.py
    python scripts/component1_poison_audit/09_downstream_learners.py --dataset ton_iot --rate 0.01
    python scripts/component1_poison_audit/09_downstream_learners.py --seed 8459 --report-only
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.logger import get_logger
from src.core.utils import load_config, set_seed
from src.poison_audit.downstream_learners import DownstreamLearnerSweep
from src.poison_audit.poison_rate_sweep import TriggerCatalog, seed_root

logger = get_logger(__name__)

REPORT_FILE = "downstream_learners_report.md"
SUMMARY_FILE = "downstream_learners_summary.csv"
DATASET_TITLES = {"cicids2017": "CIC-IDS2017", "edge_iiot": "Edge-IIoTset", "ton_iot": "TON-IoT", "unsw_nb15": "UNSW-NB15"}


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", action="append", default=None, help="Dataset name pattern (repeatable); default: from the config.")
    parser.add_argument("--trigger", action="append", default=None, help="Trigger name pattern (repeatable); default: every trigger.")
    parser.add_argument("--rate", action="append", type=float, default=None, help="Poison rate (repeatable); default: the whole grid.")
    parser.add_argument("--seed", type=int, default=None, help="Seed of the sweep; default: project.random_state.")
    parser.add_argument("--report-only", action="store_true", help="Only rebuild the report from the finished cells.")
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


def build_report(results: List[Dict[str, Any]], learners: List[str], removals: List[str], seed: int) -> str:
    """Markdown report: one table per dataset and learner, one row per (trigger, rate), one column per removal variant.

    Args:
        results: Results of the finished cells.
        learners: Learners in column order.
        removals: Removal variants in column order.
        seed: Seed of the sweep.

    Returns:
        Markdown text.
    """
    lines = [f"# The cleaned dataset with learners other than XGBoost — seed {seed}", "",
             "- Each cell: `ASR · CDA` (%) after training that learner on what the removal variant keeps. ASR counts only"
             " the rows that actually received the trigger. LightGBM, RandomForest: the library's default parameters.", ""]
    for dataset in dict.fromkeys(result["dataset"] for result in results):
        rows = [result for result in results if result["dataset"] == dataset]
        lines += [f"## {DATASET_TITLES.get(dataset, dataset)}", ""]
        for learner in learners:
            lines += [f"### {learner}", "", "| trigger | rate | poisoned rows | " + " | ".join(removals) + " |",
                      "|---|---:|---:|" + "---|" * len(removals)]
            for result in rows:
                cells = result["learners"].get(learner, {})
                lines.append(f"| {result['trigger']} | {100 * result['rate']:g}% | {result['n_poisoned']:,} | " + " | ".join(
                    f"{100 * cells[name]['asr_injected_only']:.2f} · {100 * cells[name]['cda']:.3f}" if name in cells else "—"
                    for name in removals) + " |")
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the selected cells, then rebuild the report from every finished cell."""
    args = parse_args()
    config = load_config()
    calibration_seed = int(config["project"]["random_state"])
    seed = calibration_seed if args.seed is None else int(args.seed)
    set_seed(seed)
    scenario_cfg = config["component1_poison_audit"]["fixed_trigger_scenario"]
    cfg, sweep_cfg = scenario_cfg["downstream_learners"], scenario_cfg["poison_rate_sweep"]
    output_root = seed_root(Path(cfg["output_dir"]), seed, calibration_seed)
    output_root.mkdir(parents=True, exist_ok=True)
    removals = [str(name) for name in cfg["removals"]]
    sweep = DownstreamLearnerSweep(
        sweep_root=seed_root(Path(sweep_cfg["output_dir"]), seed, calibration_seed), output_root=output_root,
        learners={str(name): dict(params) for name, params in cfg["learners"].items()}, removals=removals, random_state=seed,
    )
    catalog = TriggerCatalog(config)
    triggers = [trigger for entry in sweep_cfg["datasets"] if entry["name"] in cfg["datasets"] for trigger in catalog.resolve(entry)]
    rates = [float(rate) for rate in sweep_cfg["poison_rates"]]
    grid = [(trigger, rate) for trigger in triggers for rate in rates]
    chosen = [(trigger, rate) for trigger, rate in grid
              if pick([trigger.dataset], args.dataset) and pick([trigger.name], args.trigger)
              and (not args.rate or any(abs(rate - wanted) < 1e-12 for wanted in args.rate))]
    if not chosen:
        raise ValueError(f"No cell matches --dataset {args.dataset} --trigger {args.trigger} --rate {args.rate}.")
    if not args.report_only:
        for trigger, rate in chosen:
            sweep.run_cell(trigger, rate)

    results = [result for result in (sweep.load(trigger, rate) for trigger, rate in grid) if result]
    learners = list(cfg["learners"]) + ["xgboost"]
    (output_root / REPORT_FILE).write_text(build_report(results, learners, removals, seed), encoding="utf-8")
    pd.DataFrame([
        {"dataset": result["dataset"], "trigger": result["trigger"], "rate": result["rate"], "n_poisoned": result["n_poisoned"],
         "learner": learner, "removal": removal, **{key: metrics[key] for key in ("asr", "asr_injected_only", "cda", "macro_f1")}}
        for result in results for learner, by_removal in result["learners"].items() for removal, metrics in by_removal.items()
    ]).to_csv(output_root / SUMMARY_FILE, index=False)
    logger.info("Report of %d/%d cells: %s", len(results), len(grid), output_root / REPORT_FILE)


if __name__ == "__main__":
    main()
