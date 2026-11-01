"""The Spectral Signatures, Isolation Forest and original LABOR k-means baselines on the finished cells of the
poison-rate sweep (script 04), each retrained after its removal (src/poison_audit/baseline_defenses.py, config block
`baseline_defenses`).

For each cell (dataset, trigger, rate):
1. read the poisoned set, the manifest (mRMR features, poisoned rows) and the audited model from the sweep cell;
2. target-class TreeSHAP for the rows with the target label (cached) -> Spectral Signatures, Isolation Forest (oracle
   budget = 1.5 x the true number of poisoned rows) and Isolation Forest with sklearn's default decision;
3. LABOR k-means on every raw feature (cached), cluster labels from the true composition, LABOR removal rule;
4. retrain after each baseline, score precision / recall / FPR and ASR / CDA, next to the `compare_with` variants.

Finished cells are skipped.

    python scripts/component1_poison_audit/08_baseline_defenses.py
    python scripts/component1_poison_audit/08_baseline_defenses.py --dataset ton_iot --trigger representative --rate 0.01
    python scripts/component1_poison_audit/08_baseline_defenses.py --report-only
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
from src.poison_audit.baseline_defenses import BASELINES, BaselineDefenseSweep, IsolationForestScorer
from src.poison_audit.poison_rate_sweep import TriggerCatalog, seed_root

logger = get_logger(__name__)

REPORT_FILE = "baseline_defenses_report.md"
SUMMARY_FILE = "baseline_defenses_summary.csv"
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


def build_report(results: List[Dict[str, Any]], names: List[str], seed: int) -> str:
    """Markdown report: two tables per dataset, one row per (trigger, rate).

    Args:
        results: Results of the finished cells.
        names: Variants in column order.
        seed: Seed of the sweep.

    Returns:
        Markdown text.
    """
    lines = [f"# Baseline Spectral Signatures, Isolation Forest, LABOR k-means — seed {seed}", "",
             "- `spectral_signature`, `isolation_forest`: target-class TreeSHAP of the rows with the target label; they remove"
             " 1.5 x the TRUE number of poisoned rows (an oracle budget - information SHAPClean does not have)."
             " `isolation_forest_auto`: sklearn's default decision.",
             "- `labor_kmeans`: k-means on every raw feature, cluster labels from the true composition (LABOR's best case),"
             " LABOR removal rule. `labor_oracle`: the same rule on SHAPClean's HDBSCAN-TreeSHAP clusters. `true_poison`:"
             " removes exactly the poisoned rows.",
             "- FPR = clean rows removed / clean rows. ASR counts only the rows that actually received the trigger.", ""]
    for dataset in dict.fromkeys(result["dataset"] for result in results):
        rows = [result for result in results if result["dataset"] == dataset]
        lines += [f"## {DATASET_TITLES.get(dataset, dataset)}", ""]
        for title, render in (("Record level: precision · recall · FPR (%)",
                               lambda item: f"{100 * item['precision']:.1f} · {100 * item['recall']:.1f} · {100 * item['fpr']:.3f}"),
                              ("After retraining: ASR · CDA (%)",
                               lambda item: f"{100 * item['asr_injected_only']:.2f} · {100 * item['cda']:.3f}")):
            lines += [f"### {title}", "", "| trigger | rate | poisoned rows | " + " | ".join(names) + " |",
                      "|---|---:|---:|" + "---|" * len(names)]
            for result in rows:
                by_name = {item["name"]: item for item in result["removals"]}
                lines.append(f"| {result['trigger']} | {100 * result['rate']:g}% | {result['n_poisoned']:,} | "
                             + " | ".join(render(by_name[name]) if name in by_name else "—" for name in names) + " |")
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the baselines on the selected cells, then rebuild the report from every finished cell."""
    args = parse_args()
    config = load_config()
    calibration_seed = int(config["project"]["random_state"])
    seed = calibration_seed if args.seed is None else int(args.seed)
    set_seed(seed)
    audit_cfg = config["component1_poison_audit"]
    scenario_cfg = audit_cfg["fixed_trigger_scenario"]
    cfg, sweep_cfg = scenario_cfg["baseline_defenses"], scenario_cfg["poison_rate_sweep"]
    output_root = seed_root(Path(cfg["output_dir"]), seed, calibration_seed)
    output_root.mkdir(parents=True, exist_ok=True)
    forest_cfg = cfg["isolation_forest"]
    sweep = BaselineDefenseSweep(
        config=config, sweep_root=seed_root(Path(sweep_cfg["output_dir"]), seed, calibration_seed), output_root=output_root,
        model_params=dict(audit_cfg["backdoor_eval"]["model_params"]), shap_batch_size=int(audit_cfg["shap_batch_size"]),
        budget_factor=float(cfg["budget_factor"]),
        isolation_forest=IsolationForestScorer(int(forest_cfg["n_estimators"]), forest_cfg["max_samples"], seed),
        kmeans_params=dict(audit_cfg["clustering"]), compare_with=[str(name) for name in cfg["compare_with"]], random_state=seed,
        drop_cache=bool(cfg["drop_cache"]),
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
    names = list(BASELINES) + [str(name) for name in cfg["compare_with"]]
    (output_root / REPORT_FILE).write_text(build_report(results, names, seed), encoding="utf-8")
    pd.DataFrame([
        {"dataset": result["dataset"], "trigger": result["trigger"], "rate": result["rate"], "n_rows": result["n_rows"],
         "n_poisoned": result["n_poisoned"], "oracle_budget": result["oracle_budget"], **item}
        for result in results for item in result["removals"]
    ]).to_csv(output_root / SUMMARY_FILE, index=False)
    logger.info("Report of %d/%d cells: %s", len(results), len(grid), output_root / REPORT_FILE)


if __name__ == "__main__":
    main()
