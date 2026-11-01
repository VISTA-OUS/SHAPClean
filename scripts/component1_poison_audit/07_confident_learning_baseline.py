"""The Confident Learning baseline (Northcutt et al., JAIR 2021) on the finished cells of the poison-rate sweep
(script 04), compared on precision / recall / FPR with the removal variants of SHAPClean
(src/poison_audit/confident_learning.py, config block `confident_learning`).

For each cell (dataset, trigger, rate) of the sweep:
1. read the poisoned set, the cell's mRMR features and its poisoned rows from the cell folder (nothing is rebuilt);
2. out-of-fold probabilities: 5 stratified folds, XGBoost with the architecture of the audited model (cached per cell);
3. per-class thresholds -> confident labels -> two CL rules (`confident_joint`, `prune_by_noise_rate`) x two scopes
   (`all`, `target_label`);
4. score each mask with the ground truth, next to the `compare_with` variants read from the cell's
   end_to_end_result.json.

FPR = clean rows removed / clean rows. Finished cells are skipped; delete a cell's `cl_result.json` to rescore it from the
cache.

    python scripts/component1_poison_audit/07_confident_learning_baseline.py
    python scripts/component1_poison_audit/07_confident_learning_baseline.py --dataset ton_iot --rate 0.01
    python scripts/component1_poison_audit/07_confident_learning_baseline.py --report-only
    python scripts/component1_poison_audit/07_confident_learning_baseline.py --seed 8459

`--seed` reads that seed's cells of the sweep (`<output_dir>/seed_<n>/`, as in script 04) and writes to a folder of the
same name; the folds and the out-of-fold models use that seed too.
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
from src.poison_audit.confident_learning import ConfidentLearningDetector, ConfidentLearningSweep
from src.poison_audit.poison_rate_sweep import seed_root

logger = get_logger(__name__)

REPORT_FILE = "confident_learning_report.md"
SUMMARY_FILE = "confident_learning_summary.csv"
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
    parser.add_argument("--report-only", action="store_true", help="Only rebuild the report from the scored cells.")
    parser.add_argument("--seed", type=int, default=None, help="Seed of the sweep; default: project.random_state.")
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


def cell_text(item: Dict[str, Any]) -> str:
    """One table cell: precision · recall · FPR.

    Args:
        item: Scores of one removal variant.

    Returns:
        For example `25.3 · 99.4 · 0.781`.
    """
    return f"{100 * item['precision']:.1f} · {100 * item['recall']:.1f} · {100 * item['fpr']:.3f}"


def build_report(results: List[Dict[str, Any]], datasets: List[str], names: List[str]) -> str:
    """Markdown report: one table per dataset, one row per (trigger, rate).

    Args:
        results: Results of the scored cells.
        datasets: Datasets in config order.
        names: Variants in column order (baselines first, then SHAPClean).

    Returns:
        Markdown text.
    """
    lines = ["# Confident Learning baseline against SHAPClean", "",
             "- Each cell: `precision · recall · FPR` (%). FPR = clean rows removed / clean rows.",
             "- `cl_<rule>_<scope>`: rule `confident_joint` or `prune_by_noise_rate`; scope `all` (every flagged row) "
             "or `target_label` (only rows with the target label).",
             "- Column `poisoned OOF → target`: share of poisoned rows for which the out-of-fold model still predicts the"
             " flipped label (the target label).",
             f"- Scored cells: {len(results)}.", ""]
    for dataset in datasets:
        rows = [result for result in results if result["dataset"] == dataset]
        if not rows:
            continue
        lines += [f"## {DATASET_TITLES.get(dataset, dataset)}", "",
                  "| trigger | rate | poisoned rows | poisoned OOF → target | " + " | ".join(names) + " |",
                  "|---|---:|---:|---:|" + "---|" * len(names)]
        for result in rows:
            by_name = {item["name"]: item for item in result["removals"]}
            lines.append(
                f"| {result['trigger']} | {100 * result['rate']:g}% | {result['n_poisoned']:,} | "
                f"{100 * result['oof']['poisoned_predicted_target']:.1f}% | "
                + " | ".join(cell_text(by_name[name]) for name in names) + " |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Score the baseline on the selected cells, then rebuild the report from every scored cell."""
    args = parse_args()
    config = load_config()
    calibration_seed = int(config["project"]["random_state"])
    seed = calibration_seed if args.seed is None else int(args.seed)
    set_seed(seed)
    audit_cfg = config["component1_poison_audit"]
    scenario_cfg = audit_cfg["fixed_trigger_scenario"]
    cfg, sweep_cfg = scenario_cfg["confident_learning"], scenario_cfg["poison_rate_sweep"]
    output_root = seed_root(Path(cfg["output_dir"]), seed, calibration_seed)
    output_root.mkdir(parents=True, exist_ok=True)

    detector = ConfidentLearningDetector(int(cfg["n_folds"]), dict(audit_cfg["backdoor_eval"]["model_params"]), seed)
    sweep = ConfidentLearningSweep(
        detector=detector, config=config, sweep_root=seed_root(Path(sweep_cfg["output_dir"]), seed, calibration_seed), output_root=output_root,
        rules=[str(rule) for rule in cfg["rules"]], scopes=[str(scope) for scope in cfg["scopes"]],
        compare_with=[str(name) for name in cfg["compare_with"]], drop_cache=bool(cfg["drop_cache"]),
    )
    entries = [entry for entry in sweep_cfg["datasets"] if entry["name"] in cfg["datasets"]]
    rates = [float(rate) for rate in sweep_cfg["poison_rates"]]
    grid = [(str(entry["name"]), str(item["name"]), rate) for entry in entries for item in entry["triggers"] for rate in rates]
    chosen = [cell for cell in grid
              if pick([cell[0]], args.dataset) and pick([cell[1]], args.trigger)
              and (not args.rate or any(abs(cell[2] - wanted) < 1e-12 for wanted in args.rate))]
    if not chosen:
        raise ValueError(f"No cell matches --dataset {args.dataset} --trigger {args.trigger} --rate {args.rate}.")
    if not args.report_only:
        for dataset, trigger, rate in chosen:
            sweep.run_cell(dataset, trigger, rate)

    results = [result for result in (sweep.load(*cell) for cell in grid) if result]
    names = sweep.variant_names + [str(name) for name in cfg["compare_with"]]
    datasets = [str(entry["name"]) for entry in entries]
    (output_root / REPORT_FILE).write_text(build_report(results, datasets, names), encoding="utf-8")
    pd.DataFrame([
        {"dataset": result["dataset"], "trigger": result["trigger"], "rate": result["rate"], "n_rows": result["n_rows"],
         "n_poisoned": result["n_poisoned"], **{f"oof_{key}": value for key, value in result["oof"].items()}, **item}
        for result in results for item in result["removals"]
    ]).to_csv(output_root / SUMMARY_FILE, index=False)
    logger.info("Report of %d/%d cells: %s", len(results), len(grid), output_root / REPORT_FILE)


if __name__ == "__main__":
    main()
