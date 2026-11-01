"""The 0% poison control. Audits a CLEAN training set to measure how many rows SHAPClean removes by mistake when there
is nothing to remove (src/poison_audit/poison_rate_sweep.py, config block `poison_rate_sweep`, entry `clean_control`).

One cell per (seed, dataset), `<seed_root>/<dataset>/clean_control/rate_0pct`, running exactly the steps of a cell of
script 04:
1. the "poisoned set" at rate 0 is the clean training set (the builder poisons no row); mRMR selects the top-K features
   on that set;
2. the clean model and the audited model (same data), TreeSHAP -> HDBSCAN full fit, layer A, layer B, the noise rule;
3. every removal variant in `poison_rate_sweep.variants` and the references, retraining, CDA / macro-F1 / per-class
   recall.
Every removed row is a clean row. The audit reads the trigger nowhere except in the triggered test set (only to measure
ASR), so one cell serves all 3 triggers.

Default seeds: the 5 `selection_feature.evaluation_seeds`. Finished cells are skipped, so a job that runs out of time
only has to be submitted again. The run ends by rebuilding `clean_control_report.md` and `clean_control_report.csv` in
`poison_rate_sweep.output_dir` from every finished cell (one table per dataset, one row per seed).

    python scripts/component1_poison_audit/10_clean_control.py --check
    python scripts/component1_poison_audit/10_clean_control.py --dataset edge_iiot --dataset ton_iot
    python scripts/component1_poison_audit/10_clean_control.py --dataset cicids2017 --seed 8459
    python scripts/component1_poison_audit/10_clean_control.py --report-only
"""

from __future__ import annotations

import argparse
import fnmatch
import gc
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.logger import get_logger
from src.core.utils import load_config, set_seed
from src.poison_audit.end_to_end_pipeline import NO_REMOVAL, RESULT_FILE, TRUE_POISON
from src.poison_audit.poison_rate_sweep import CLEAN_CONTROL_RATE, PoisonRateSweep, SweepTrigger, TriggerCatalog, seed_root

logger = get_logger(__name__)

REPORT_FILE = "clean_control_report.md"
REPORT_CSV = "clean_control_report.csv"
DATASET_TITLES = {"cicids2017": "CIC-IDS2017", "edge_iiot": "Edge-IIoTset", "ton_iot": "TON-IoT"}


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", action="append", default=None, help="Dataset name pattern (repeatable); default: every dataset.")
    parser.add_argument("--seed", action="append", type=int, default=None,
                        help="Seed (repeatable); default: the 5 evaluation_seeds.")
    parser.add_argument("--output-dir", default=None, help="Overrides poison_rate_sweep.output_dir.")
    parser.add_argument("--report-only", action="store_true", help="Only rebuild the report from the finished cells.")
    parser.add_argument("--check", action="store_true", help="Only check the inputs and list the cells that would run.")
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


def check_inputs(controls: List[SweepTrigger], sweeps: Dict[int, PoisonRateSweep]) -> int:
    """Log the missing inputs and the cells that would run.

    Args:
        controls: Control cells of the selected datasets.
        sweeps: Sweep of each seed.

    Returns:
        The number of problems found.
    """
    problems = 0
    if importlib.util.find_spec("mrmr") is None:
        problems += 1
        logger.info("MISSING package mrmr-selection in the environment (pip install mrmr-selection==0.2.8).")
    for control in controls:
        missing = [path for path in (control.clean_train, control.clean_test) if not Path(path).exists()]
        problems += len(missing)
        logger.info("%s: %s; triggered test set: %s.", control.dataset, "clean sets present" if not missing else f"MISSING {missing}",
                    control.trigger_spec)
    done = sum((sweep.cell_dir(control, CLEAN_CONTROL_RATE) / RESULT_FILE).exists() for sweep in sweeps.values() for control in controls)
    logger.info("%d cells in total (%d seeds x %d datasets), %d finished, %d problems.", len(sweeps) * len(controls), len(sweeps),
                len(controls), done, problems)
    return problems


def removal_rows(seed: int, control: SweepTrigger, result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per removal variant (references included) of a cell: rows removed, their origin and the cost after
    retraining.

    A "drop" is measured against the model trained on the whole set (`no_removal`, which at rate 0 is the clean set),
    in percentage points.

    Args:
        seed: Seed of the cell.
        control: The control cell.
        result: The cell's `end_to_end_result.json`.

    Returns:
        The rows, in the order of the result (without `no_removal`).

    Raises:
        KeyError: If the per-class recall lacks the dataset's target (benign) class.
    """
    removals = {item["name"]: item for item in result["removals"]}
    base = removals[NO_REMOVAL]
    if control.target_label not in base["recall_per_class"]:
        raise KeyError(f"[{control.dataset}] recall_per_class has no target class {control.target_label!r}.")
    rows = []
    for item in result["removals"]:
        if item["name"] == NO_REMOVAL:
            continue
        drops = {label: 100 * (base["recall_per_class"][label] - recall) for label, recall in item["recall_per_class"].items()}
        worst = max(drops, key=drops.get)
        # When no class loses recall there is no "worst class".
        worst_class = worst if drops[worst] > 0 else ""
        rows.append({
            "seed": seed, "dataset": control.dataset, "removal": item["name"], "kind": item["kind"],
            "n_rows": result["n_rows"], "n_clusters": result["clustering"]["n_clusters"], "n_noise": result["clustering"]["n_noise"],
            "n_removed": item["n_removed"], "removed_pct": 100 * item["n_removed"] / result["n_rows"],
            "wrong_in_attack_clusters": item["wrong_in_attack_clusters"], "wrong_in_layer_b": item["wrong_in_layer_b"],
            "wrong_in_noise": item["wrong_in_noise"],
            "cda": item["cda"], "cda_drop_pp": 100 * (base["cda"] - item["cda"]),
            "macro_f1_drop_pp": 100 * (base["macro_f1"] - item["macro_f1"]),
            "benign_recall_drop_pp": drops[control.target_label],
            "worst_class": worst_class, "worst_class_recall_drop_pp": max(drops[worst], 0.0),
            "asr_injected_only": item["asr_injected_only"], "base_asr_injected_only": base["asr_injected_only"],
        })
    return rows


def mean_std(values: List[float], digits: int) -> str:
    """Mean ± sample standard deviation.

    Args:
        values: Values (at least one).
        digits: Number of decimals.

    Returns:
        A string such as `0.812 ± 0.021`; a single value gives the mean only.
    """
    mean = statistics.mean(values)
    if len(values) < 2:
        return f"{mean:,.{digits}f}"
    return f"{mean:,.{digits}f} ± {statistics.stdev(values):,.{digits}f}"


def build_report(results: Dict[str, Dict[int, Dict[str, Any]]], controls: List[SweepTrigger], seeds: List[int],
                 rows: pd.DataFrame) -> str:
    """Markdown report: two tables per dataset (removed rows; cost after retraining), one row per seed.

    Args:
        results: Dataset -> seed -> result of the finished cell.
        controls: Control cells of every dataset.
        seeds: Seeds of the report, in order.
        rows: Output of `removal_rows` for every finished cell.

    Returns:
        Markdown text.
    """
    n_done = sum(len(by_seed) for by_seed in results.values())
    lines = ["# 0% poison control: how much SHAPClean removes by mistake from a clean training set", "",
             f"- Finished cells: {n_done}/{len(controls) * len(seeds)} (seeds {', '.join(str(seed) for seed in seeds)}).",
             "- The training set holds no poisoned row, so **every removed row is a wrong removal**. Origin: attack-labelled"
             " clusters (the LABOR rule) / layer B / the noise rule.",
             "- \"Drop\" (percentage points) is against the model trained on the whole training set (no row removed), with"
             " the same architecture and seed. Worst class: the class whose test recall falls the most after removal.", ""]
    missing = [f"{control.dataset}/seed {seed}" for control in controls for seed in seeds if seed not in results.get(control.dataset, {})]
    if missing:
        lines += [f"- Not finished: {', '.join(missing)}.", ""]
    for control in controls:
        by_seed = results.get(control.dataset, {})
        if not by_seed:
            continue
        frame = rows[rows["dataset"] == control.dataset]
        # The true_poison reference removes nothing at rate 0: it only checks that retraining reproduces the first model.
        names = [name for name in dict.fromkeys(frame["removal"]) if name != TRUE_POISON]
        first = next(iter(by_seed.values()))
        cda_gap = max(abs(result["backdoor"]["clean_cda"] - result["backdoor"]["backdoored_cda"]) for result in by_seed.values())
        retrain_gap = frame[frame["removal"] == TRUE_POISON]["cda_drop_pp"].abs().max()
        base_asr = [100 * next(item for item in result["removals"] if item["name"] == NO_REMOVAL)["asr_injected_only"]
                    for result in by_seed.values()]
        lines += [f"## {DATASET_TITLES.get(control.dataset, control.dataset)}", "",
                  f"- {first['n_rows']:,} training rows, top-{first['n_model_features']} mRMR features, `min_cluster_size`"
                  f" {first['min_cluster_size']}. Triggered test set: {control.trigger_spec} (source class {control.source_label}).",
                  f"- Check: the clean and the audited model (same data) differ by at most {100 * cda_gap:.4f} pp in CDA;"
                  f" retraining without removing any row (reference `{TRUE_POISON}`) differs by at most {retrain_gap:.4f} pp.",
                  f"- ASR (triggered rows only) of the model trained on the whole set: {mean_std(base_asr, 2)}%;"
                  f" after the removal variants: {100 * frame['asr_injected_only'].min():.2f}–{100 * frame['asr_injected_only'].max():.2f}%.", ""]
        layer_names = list(first["layers"])
        lines += ["### Removed rows", "",
                  "Layer-A columns: clusters in the short list · clusters layer B (`and` rule) calls attack. Removal-variant"
                  " columns: rows removed (% of the training set) · attack-labelled clusters / layer B / noise rule.", "",
                  f"| seed | clusters · noise | {' | '.join(f'layer A {name}' for name in layer_names)} | {' | '.join(f'`{name}`' for name in names)} |",
                  "|---|---|" + "---|" * (len(layer_names) + len(names))]
        for seed, result in by_seed.items():
            layers = [f"{result['layers'][name]['layer_a']['n_short_list']} · {result['layers'][name]['layer_b']['and']['n_flagged']}"
                      for name in layer_names]
            cells = []
            for name in names:
                row = frame[(frame["seed"] == seed) & (frame["removal"] == name)].iloc[0]
                cells.append(f"{row['n_removed']:,} ({row['removed_pct']:.3f}%) · {row['wrong_in_attack_clusters']:,} / "
                             f"{row['wrong_in_layer_b']:,} / {row['wrong_in_noise']:,}")
            lines.append(f"| {seed} | {result['clustering']['n_clusters']:,} · {100 * result['clustering']['n_noise'] / result['n_rows']:.2f}% | "
                         + " | ".join(layers + cells) + " |")
        summary = [f"{mean_std(list(frame[frame['removal'] == name]['n_removed']), 0)} "
                   f"({mean_std(list(frame[frame['removal'] == name]['removed_pct']), 3)}%)" for name in names]
        lines += [f"| mean ± std | | {' | '.join([''] * len(layer_names))} | " + " | ".join(summary) + " |", ""]
        lines += ["### Cost after retraining", "",
                  f"No-removal column: CDA · macro-F1 · {control.target_label} recall on the test set. Removal-variant columns,"
                  f" in pp: CDA drop · macro-F1 drop · {control.target_label} recall drop · class with the largest recall drop"
                  " (— when no class drops). Negative values are gains.", "",
                  f"| seed | no removal | {' | '.join(f'`{name}`' for name in names)} |", "|---|---|" + "---|" * len(names)]
        for seed, result in by_seed.items():
            base = next(item for item in result["removals"] if item["name"] == NO_REMOVAL)
            cells = []
            for name in names:
                row = frame[(frame["seed"] == seed) & (frame["removal"] == name)].iloc[0]
                worst = f"{row['worst_class']} {row['worst_class_recall_drop_pp']:.2f}" if row["worst_class"] else "—"
                cells.append(f"{row['cda_drop_pp']:.3f} · {row['macro_f1_drop_pp']:.3f} · {row['benign_recall_drop_pp']:.3f} · {worst}")
            lines.append(f"| {seed} | {100 * base['cda']:.3f}% · {100 * base['macro_f1']:.3f}% · "
                         f"{100 * base['recall_per_class'][control.target_label]:.3f}% | " + " | ".join(cells) + " |")
        summary = [f"{mean_std(list(frame[frame['removal'] == name]['cda_drop_pp']), 3)} · "
                   f"{mean_std(list(frame[frame['removal'] == name]['macro_f1_drop_pp']), 3)} · "
                   f"{mean_std(list(frame[frame['removal'] == name]['benign_recall_drop_pp']), 3)}" for name in names]
        lines += ["| mean ± std | | " + " | ".join(summary) + " |", ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the control cells of the selected (seed, dataset) pairs, then rebuild the report from every finished cell."""
    args = parse_args()
    config = load_config()
    calibration_seed = int(config["project"]["random_state"])
    set_seed(calibration_seed)
    cfg = config["component1_poison_audit"]["fixed_trigger_scenario"]["poison_rate_sweep"]
    base_root = Path(args.output_dir or cfg["output_dir"])
    report_seeds = [int(seed) for seed in config["selection_feature"]["evaluation_seeds"]]
    seeds = [int(seed) for seed in args.seed] if args.seed else report_seeds

    catalog = TriggerCatalog(config)
    trigger_name = str(cfg["clean_control"]["trigger"])
    all_controls = [catalog.clean_control(entry, trigger_name) for entry in cfg["datasets"]]
    datasets = pick([control.dataset for control in all_controls], args.dataset)
    controls = [control for control in all_controls if control.dataset in datasets]
    if not controls:
        raise ValueError(f"No dataset matches --dataset {args.dataset}.")
    sweeps = {seed: PoisonRateSweep.from_config(config, seed_root(base_root, seed, calibration_seed), seed)
              for seed in dict.fromkeys(seeds + report_seeds)}

    if args.check:
        check_inputs(controls, {seed: sweeps[seed] for seed in seeds})
        return
    if not args.report_only:
        if check_inputs(controls, {seed: sweeps[seed] for seed in seeds}):
            raise FileNotFoundError("Missing inputs (see the log above).")
        for seed in seeds:
            set_seed(seed)
            for control in controls:
                logger.info("=== seed %d: %s/%s ===", seed, control.dataset, control.name)
                sweeps[seed].run_trigger(control, [CLEAN_CONTROL_RATE])
                gc.collect()

    results: Dict[str, Dict[int, Dict[str, Any]]] = {}
    rows: List[Dict[str, Any]] = []
    for control in all_controls:
        for seed in report_seeds:
            path = sweeps[seed].cell_dir(control, CLEAN_CONTROL_RATE) / RESULT_FILE
            if path.exists():
                result = json.loads(path.read_text(encoding="utf-8"))
                results.setdefault(control.dataset, {})[seed] = result
                rows += removal_rows(seed, control, result)
    frame = pd.DataFrame(rows)
    base_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(base_root / REPORT_CSV, index=False)
    (base_root / REPORT_FILE).write_text(build_report(results, all_controls, report_seeds, frame), encoding="utf-8")
    logger.info("Report of %d/%d cells: %s", len(frame.groupby(["dataset", "seed"])) if len(frame) else 0,
                len(all_controls) * len(report_seeds), base_root / REPORT_FILE)


if __name__ == "__main__":
    main()
