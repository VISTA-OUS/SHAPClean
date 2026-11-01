"""Build every poisoned training set of the poison-rate grid and select its top-K features with mRMR
(`PoisonedCellPreparer` in src/poison_audit/poison_rate_sweep.py, config block `poison_rate_sweep`).

All the main results of the paper use the features that mRMR selects on the poisoned set itself, not a fixed list. This
script runs that preparation apart from the audit, once for every seed; the audit (script 04) and the baselines then
only read prepared cells.

For each cell (seed, dataset, trigger, rate):
1. build the poisoned set with `FixedTriggerPoisonDatasetBuilder` (for one seed the poisoned rows are nested across
   rates; on the calibration seed at 1% they are checked against the manifest of an earlier build);
2. rank every raw feature of that set with mRMR (parameters `selection_feature.techniques.mrmr`, sampling seeds from
   `feature_selection_seed_set`) and keep the top K;
3. write `poisoned_train.csv.gz`, `feature_selection.json` and `manifest.json` to the cell folder; the triggered test
   set is written once per trigger.

Default seeds: `project.random_state` (the calibration seed, written to the output folder itself) and
`selection_feature.evaluation_seeds` (written to `seed_<n>/`). Cells that already have a manifest are skipped. The run
ends by rebuilding the summary table of every prepared cell.

    python scripts/component1_poison_audit/03_prepare_poisoned_datasets.py --check
    python scripts/component1_poison_audit/03_prepare_poisoned_datasets.py --dataset edge_iiot --dataset ton_iot
    python scripts/component1_poison_audit/03_prepare_poisoned_datasets.py --dataset cicids2017 --seed 8459
    python scripts/component1_poison_audit/03_prepare_poisoned_datasets.py --summary-only
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.logger import get_logger
from src.core.utils import load_config, set_seed
from src.poison_audit.poison_rate_sweep import (
    FEATURE_SELECTION_FILE, MANIFEST_FILE, PoisonedCellPreparer, SweepTrigger, TriggerCatalog, rate_tag, seed_root,
)

logger = get_logger(__name__)

SUMMARY_FILE = "poisoned_datasets_summary.csv"


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", action="append", default=None, help="Dataset name pattern (repeatable); default: every dataset.")
    parser.add_argument("--trigger", action="append", default=None, help="Trigger name pattern (repeatable); default: every trigger.")
    parser.add_argument("--rate", action="append", type=float, default=None, help="Poison rate (repeatable); default: the whole grid.")
    parser.add_argument("--seed", action="append", type=int, default=None,
                        help="Poisoning seed (repeatable); default: the calibration seed and evaluation_seeds.")
    parser.add_argument("--check", action="store_true", help="Only check the inputs and count the cells left to prepare.")
    parser.add_argument("--summary-only", action="store_true", help="Only rebuild the summary table.")
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


def check_inputs(preparers: List[PoisonedCellPreparer], triggers: List[SweepTrigger], rates: List[float]) -> int:
    """Log the missing inputs and the number of cells left to prepare.

    Args:
        preparers: One preparer per seed.
        triggers: Selected triggers.
        rates: Selected rates.

    Returns:
        The number of problems found.
    """
    problems = 0
    if importlib.util.find_spec("mrmr") is None:
        problems += 1
        logger.info("MISSING package mrmr-selection in the environment (pip install mrmr-selection==0.2.8).")
    for trigger in triggers:
        missing = [path for path in (trigger.clean_train, trigger.clean_test) if not Path(path).exists()]
        problems += len(missing)
        if missing:
            logger.info("%s/%s: MISSING %s.", trigger.dataset, trigger.name, missing)
    total = len(preparers) * len(triggers) * len(rates)
    pending = sum(not preparer.is_prepared(trigger, rate) for preparer in preparers for trigger in triggers for rate in rates)
    logger.info("%d cells in total, %d not prepared yet, %d problems.", total, pending, problems)
    return problems


def build_summary(preparers: Dict[int, PoisonedCellPreparer], triggers: List[SweepTrigger], rates: List[float]) -> pd.DataFrame:
    """One row per prepared cell: the number of poisoned rows, the trigger features in the top K and the top-K list.

    Args:
        preparers: Preparer of each seed.
        triggers: Every trigger of the grid.
        rates: Every rate of the grid.

    Returns:
        The summary table.
    """
    rows: List[Dict[str, Any]] = []
    for seed, preparer in preparers.items():
        for trigger in triggers:
            for rate in rates:
                cell = preparer.cell_dir(trigger, rate)
                if not preparer.is_prepared(trigger, rate):
                    continue
                manifest = json.loads((cell / MANIFEST_FILE).read_text(encoding="utf-8"))
                selection = json.loads((cell / FEATURE_SELECTION_FILE).read_text(encoding="utf-8"))
                rows.append({
                    "seed": seed, "dataset": trigger.dataset, "trigger": trigger.name, "rate": rate,
                    "cell_dir": cell.as_posix(), "n_poisoned": manifest["n_poisoned"], "top_k": selection["top_k"],
                    "trigger_in_top_k": selection["trigger_in_top_k"], "n_trigger_features": len(trigger.trigger_spec),
                    "trigger_rank": json.dumps(selection["trigger_rank"], ensure_ascii=False),
                    "selected_features": ";".join(selection["selected"]),
                })
    return pd.DataFrame(rows)


def main() -> None:
    """Prepare the selected cells, then rebuild the summary table of every prepared cell."""
    args = parse_args()
    config = load_config()
    calibration_seed = int(config["project"]["random_state"])
    set_seed(calibration_seed)
    scenario_cfg = config["component1_poison_audit"]["fixed_trigger_scenario"]
    cfg = scenario_cfg["poison_rate_sweep"]
    selection_cfg = config["selection_feature"]
    mrmr_cfg = selection_cfg["techniques"]["mrmr"]
    selection_seeds = next(item["seeds"] for item in scenario_cfg["mrmr_reselection"]["seed_sets"]
                           if item["name"] == cfg["feature_selection_seed_set"])
    base_root = Path(cfg["output_dir"])

    all_seeds = list(dict.fromkeys([calibration_seed] + [int(seed) for seed in selection_cfg["evaluation_seeds"]]))
    seeds = [int(seed) for seed in args.seed] if args.seed else all_seeds
    catalog = TriggerCatalog(config)
    all_triggers = [trigger for entry in cfg["datasets"] for trigger in catalog.resolve(entry)]
    all_rates = [float(rate) for rate in cfg["poison_rates"]]
    datasets = pick(list(dict.fromkeys(trigger.dataset for trigger in all_triggers)), args.dataset)
    triggers = [trigger for trigger in all_triggers if trigger.dataset in datasets and pick([trigger.name], args.trigger)]
    rates = [rate for rate in all_rates if not args.rate or any(abs(rate - wanted) < 1e-12 for wanted in args.rate)]
    if not triggers or not rates:
        raise ValueError(f"No cell matches --dataset {args.dataset} --trigger {args.trigger} --rate {args.rate}.")

    def preparer_for(seed: int) -> PoisonedCellPreparer:
        """The preparer of one poisoning seed.

        Args:
            seed: Poisoning seed.

        Returns:
            A preparer that writes to the folder of that seed.
        """
        return PoisonedCellPreparer(
            mrmr_sample_size=int(mrmr_cfg["sample_size"]), mrmr_denominator=str(mrmr_cfg["denominator"]),
            mrmr_n_jobs=int(selection_cfg["execution_n_jobs"]), selection_seeds=[int(value) for value in selection_seeds],
            output_root=seed_root(base_root, seed, calibration_seed), random_state=seed,
            check_reference=seed == calibration_seed,
        )

    chosen = [preparer_for(seed) for seed in seeds]
    if args.check:
        check_inputs(chosen, triggers, rates)
        return
    if not args.summary_only:
        if check_inputs(chosen, triggers, rates):
            raise FileNotFoundError("Missing inputs (see the log above).")
        for seed, preparer in zip(seeds, chosen):
            for trigger in triggers:
                logger.info("=== seed %d: %s/%s, %s ===", seed, trigger.dataset, trigger.name,
                            ", ".join(rate_tag(rate) for rate in rates))
                preparer.prepare(trigger, rates)

    summary = build_summary({seed: preparer_for(seed) for seed in all_seeds}, all_triggers, all_rates)
    base_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(base_root / SUMMARY_FILE, index=False)
    logger.info("Prepared %d/%d cells: %s", len(summary), len(all_seeds) * len(all_triggers) * len(all_rates),
                base_root / SUMMARY_FILE)


if __name__ == "__main__":
    main()
