"""The main experiment: the whole pipeline on 3 datasets x 3 triggers x 5 poison rates, with a model on the top-K
features that mRMR selects on the poisoned set itself (src/poison_audit/poison_rate_sweep.py, config block
`poison_rate_sweep`).

For each cell (dataset, trigger, rate):
1. build the poisoned set with `FixedTriggerPoisonDatasetBuilder` (nested across rates for one seed; at 1% on the
   calibration seed it is checked against an earlier build);
2. rank every raw feature of the poisoned set with mRMR and keep the dataset's top K;
3. train the clean model and the backdoored model on those features and measure CDA / macro-F1 / ASR;
4. TreeSHAP -> StandardScaler -> PCA(95%) -> HDBSCAN full fit;
5. layer A for each setting (k, eta) in `layer_a_settings` - each setting can carry a policy for the trusted benign
   reference of layer B (`benign_references`) - and count the poisoned rows that enter that reference;
6. layer B and the noise rule, removal for each of the `variants`, retraining and new measurements; plus two
   references (removing exactly the poisoned rows, LABOR with the true cluster labels).

Every step is saved in the cell folder and finished cells are skipped, so an interrupted job only has to be submitted
again. Each job runs part of the grid (`--dataset`, `--trigger`, `--rate`) and ends by rebuilding the report from every
finished cell.

    python scripts/component1_poison_audit/04_poison_rate_sweep.py --check
    python scripts/component1_poison_audit/04_poison_rate_sweep.py --dataset cicids2017 --trigger representative
    python scripts/component1_poison_audit/04_poison_rate_sweep.py --dataset edge_iiot --rate 0.005 --rate 0.01
    python scripts/component1_poison_audit/04_poison_rate_sweep.py --report-only
    python scripts/component1_poison_audit/04_poison_rate_sweep.py --dataset cicids2017 --rate 0.01 --prepare-only
    python scripts/component1_poison_audit/04_poison_rate_sweep.py --seed 8459 --dataset edge_iiot

`--seed` sets both the poisoning seed and the seed of every model. The calibration seed (`project.random_state`, the
default) writes to the output folder itself; any other seed writes to `<output_dir>/seed_<n>/`. Cells prepared by
script 03 (poisoned set and mRMR) only run the audit.

`--refresh` audits finished cells again, after a layer-A setting or a removal variant was added to the config: the
models, the clustering and every retraining already measured are reused (TreeSHAP is not recomputed); only the
removals are rebuilt and the missing variants trained.

    python scripts/component1_poison_audit/04_poison_rate_sweep.py --dataset edge_iiot --refresh
"""

from __future__ import annotations

import argparse
import fnmatch
import gc
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.logger import get_logger
from src.core.utils import load_config, set_seed
from src.poison_audit.end_to_end_pipeline import RESULT_FILE, build_layer_a_settings
from src.poison_audit.poison_rate_sweep import (
    FEATURE_SELECTION_FILE, PoisonRateSweep, SweepTrigger, TriggerCatalog, rate_tag, seed_root,
)

logger = get_logger(__name__)

REPORT_FILE = "poison_rate_sweep_report.md"
DATASET_TITLES = {"cicids2017": "CIC-IDS2017", "edge_iiot": "Edge-IIoTset", "ton_iot": "TON-IoT"}


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", action="append", default=None, help="Dataset name pattern (repeatable); default: every dataset.")
    parser.add_argument("--trigger", action="append", default=None, help="Trigger name pattern (repeatable); default: every trigger.")
    parser.add_argument("--rate", action="append", type=float, default=None, help="Poison rate (repeatable); default: the whole grid.")
    parser.add_argument("--output-dir", default=None, help="Overrides poison_rate_sweep.output_dir.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Poisoning and model seed; default: project.random_state (the calibration seed).")
    parser.add_argument("--report-only", action="store_true", help="Only rebuild the report from the finished cells.")
    parser.add_argument("--refresh", action="store_true",
                        help="Audit finished cells again (reusing models, clustering and retrainings; only adds what is missing).")
    parser.add_argument("--check", action="store_true", help="Only check the inputs and list the cells that would run.")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Only build the poisoned sets, select features and write manifests (smoke test, no audit).")
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


def check_inputs(triggers: List[SweepTrigger], rates: List[float]) -> int:
    """Log the missing inputs and the cells that would run.

    Args:
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
        logger.info("%s/%s: %s; reference manifest at 1%%: %s; %d rates.", trigger.dataset, trigger.name,
                    "clean sets present" if not missing else f"MISSING {missing}", trigger.reference_manifest or "none", len(rates))
    logger.info("%d cells in total, %d problems.", len(triggers) * len(rates), problems)
    return problems


def pct(value: float, digits: int = 2) -> str:
    """A rate as a percentage.

    Args:
        value: Rate in [0, 1].
        digits: Number of decimals.

    Returns:
        A string such as `97.12%`.
    """
    return f"{100 * value:.{digits}f}%"


def load_cells(sweep: PoisonRateSweep, triggers: List[SweepTrigger], rates: List[float]) -> List[Dict[str, Any]]:
    """The results of every finished cell, with its feature selection.

    Args:
        sweep: The sweep (for the cell folders).
        triggers: Every trigger of the grid.
        rates: Every rate of the grid.

    Returns:
        One dict per cell: `trigger`, `rate`, `result` (None while not finished), `selection`.
    """
    cells = []
    for trigger in triggers:
        for rate in rates:
            cell = sweep.cell_dir(trigger, rate)
            result_path, selection_path = cell / RESULT_FILE, cell / FEATURE_SELECTION_FILE
            cells.append({
                "trigger": trigger, "rate": rate,
                "result": json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None,
                "selection": json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else None,
            })
    return cells


def build_report(cells: List[Dict[str, Any]], layer_names: List[str]) -> str:
    """Markdown report: three tables per dataset, one row per (trigger, rate).

    Args:
        cells: Output of `load_cells`.
        layer_names: Names of the layer-A settings, in config order.

    Returns:
        Markdown text.
    """
    done = [cell for cell in cells if cell["result"]]
    lines = ["# Poison-rate sweep: the whole pipeline, top-K features selected by mRMR on the poisoned set", "",
             f"- Finished cells: {len(done)}/{len(cells)}.",
             "- ASR is measured on every source-class attack row of the triggered test set; in brackets: only the rows"
             " that actually received the trigger.",
             "- The trusted benign reference (what the clean model learns from and where exemplars find the nearest benign"
             " flow) is by default the benign rows of the clusters outside the short list, without noise. Settings with the"
             " `_kept` suffix: drop the benign rows of attack-labelled clusters and add the noise rows the clean model calls"
             " benign (exemplars only). Poisoned rows in the reference are counted with the ground truth, for scoring"
             " only.", ""]
    missing = [f"{c['trigger'].dataset}/{c['trigger'].name}/{rate_tag(c['rate'])}" for c in cells if not c["result"]]
    if missing:
        lines += [f"- Not finished: {', '.join(missing)}.", ""]
    for dataset in dict.fromkeys(cell["trigger"].dataset for cell in cells):
        rows = [cell for cell in done if cell["trigger"].dataset == dataset]
        if not rows:
            continue
        lines += [f"## {DATASET_TITLES.get(dataset, dataset)}", "", "### Features and backdoor", "",
                  "| trigger | rate | poisoned rows | trigger features in top K (rank) | clean model: ASR · CDA | audited model: ASR (triggered rows only) · CDA |",
                  "|---|---:|---:|---|---|---|"]
        for cell in rows:
            r, b, s = cell["result"], cell["result"]["backdoor"], cell["selection"] or {}
            ranks = ", ".join(f"{feature} {rank}" for feature, rank in s.get("trigger_rank", {}).items())
            lines.append(
                f"| {cell['trigger'].name} | {100 * cell['rate']:g}% | {r['n_poisoned']:,} | "
                f"{len(r['trigger_features_in_model'])}/{len(r['trigger_features'])} ({ranks}) | "
                f"{pct(b['clean_asr'])} · {pct(b['clean_cda'], 3)} | "
                f"{pct(b['backdoored_asr'])} ({pct(b['backdoored_asr_injected_only'])}) · {pct(b['backdoored_cda'], 3)} |"
            )
        header = " | ".join(f"{name}: short list (poisoned / hidden poisoned clusters) · poisoned rows in the trusted benign reference · layer B calls attack: poisoned / other"
                            for name in layer_names)
        lines += ["", "### Clustering, layer A and trusted set", "",
                  f"| trigger | rate | clusters · noise | poisoned rows: in noise / attack-labelled clusters / benign-labelled clusters | {header} |",
                  "|---|---:|---|---|" + "---|" * len(layer_names)]
        for cell in rows:
            r, c, p = cell["result"], cell["result"]["clustering"], cell["result"]["placement"]
            parts = []
            for name in layer_names:
                a, lb = r["layers"][name]["layer_a"], r["layers"][name]["layer_b"]["and"]
                reference = r["layers"][name]["layer_b"].get("reference", {})
                in_reference = reference.get("n_poisoned_in_reference", a["n_poisoned_in_trusted"])
                parts.append(f"{a['n_short_list']} ({a['n_short_list_poison']} / {a['n_hidden_poison_clusters']}) · "
                             f"{in_reference:,} · {lb['n_poison_flagged']} / {lb['n_other_flagged']}")
            lines.append(
                f"| {cell['trigger'].name} | {100 * cell['rate']:g}% | {c['n_clusters']:,} · {pct(c['n_noise'] / r['n_rows'])} | "
                f"{p['in_noise']:,} / {p['in_attack_labelled_clusters']:,} / {p['in_benign_labelled_clusters']:,} | " + " | ".join(parts) + " |"
            )
        systems = [item for item in done[0]["result"]["removals"] if item["kind"] == "system"]
        header = " | ".join(f"{item['name']}: recall · % wrongly removed · ASR · CDA" for item in systems)
        lines += ["", "### Removal and retraining", "",
                  f"| trigger | rate | remove every poisoned row: ASR | LABOR, true cluster labels: recall · ASR | {header} |",
                  "|---|---:|---|---|" + "---|" * len(systems)]
        for cell in rows:
            by_name = {item["name"]: item for item in cell["result"]["removals"]}
            truth, oracle = by_name["true_poison"], by_name["labor_oracle"]
            parts = [f"{pct(by_name[item['name']]['recall'])} · {pct(by_name[item['name']]['wrong_share'], 3)} · "
                     f"{pct(by_name[item['name']]['asr'])} · {pct(by_name[item['name']]['cda'], 3)}" for item in systems]
            lines.append(
                f"| {cell['trigger'].name} | {100 * cell['rate']:g}% | {pct(truth['asr'])} | "
                f"{pct(oracle['recall'])} · {pct(oracle['asr'])} | " + " | ".join(parts) + " |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the selected part of the grid, then rebuild the report from every finished cell."""
    args = parse_args()
    config = load_config()
    calibration_seed = int(config["project"]["random_state"])
    seed = calibration_seed if args.seed is None else int(args.seed)
    set_seed(seed)
    cfg = config["component1_poison_audit"]["fixed_trigger_scenario"]["poison_rate_sweep"]
    output_root = seed_root(Path(args.output_dir or cfg["output_dir"]), seed, calibration_seed)
    output_root.mkdir(parents=True, exist_ok=True)

    catalog = TriggerCatalog(config)
    all_triggers = [trigger for entry in cfg["datasets"] for trigger in catalog.resolve(entry)]
    all_rates = [float(rate) for rate in cfg["poison_rates"]]
    datasets = pick(list(dict.fromkeys(trigger.dataset for trigger in all_triggers)), args.dataset)
    triggers = [trigger for trigger in all_triggers if trigger.dataset in datasets and pick([trigger.name], args.trigger)]
    rates = [rate for rate in all_rates if not args.rate or any(abs(rate - wanted) < 1e-12 for wanted in args.rate)]
    if not triggers or not rates:
        raise ValueError(f"No cell matches --dataset {args.dataset} --trigger {args.trigger} --rate {args.rate}.")

    if args.check:
        check_inputs(triggers, rates)
        return
    layer_settings = build_layer_a_settings(list(cfg["layer_a_settings"]), cfg.get("benign_references"))
    sweep = PoisonRateSweep.from_config(config, output_root, seed)
    if not args.report_only:
        if check_inputs(triggers, rates):
            raise FileNotFoundError("Missing inputs (see the log above).")
        for trigger in triggers:
            sweep.run_trigger(trigger, rates, prepare_only=args.prepare_only, refresh=args.refresh)
            gc.collect()

    cells = load_cells(sweep, all_triggers, all_rates)
    (output_root / REPORT_FILE).write_text(build_report(cells, [setting.name for setting in layer_settings]), encoding="utf-8")
    logger.info("Report of %d/%d cells: %s", sum(bool(cell["result"]) for cell in cells), len(cells), output_root / REPORT_FILE)


if __name__ == "__main__":
    main()
