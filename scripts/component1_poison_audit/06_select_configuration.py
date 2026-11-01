"""Apply the configuration-selection criterion, fixed BEFORE the calibration-seed results were seen (config block
`calibration_report`). Only reads the sweep results of the calibration seed and runs nothing.

- Data: seed 42 (`project.random_state`), every dataset of `poison_rate_sweep` that has results.
- Candidates: the 4 layer-A settings fixed in `calibration_report.selection_layer_a` x every cluster-level `and`-rule
  removal variant in `poison_rate_sweep.variants` (noise-rule threshold). `layer_a_only` is an ablation, not a
  candidate. The list is not read from `poison_rate_sweep.layer_a_settings`, which now holds only the selected setting.
- Step 1 - exclude: candidates whose CDA drop (clean-model CDA - CDA after retraining) exceeds `max_cda_drop` in any
  cell.
- Step 2 - choose: the most cells where the backdoor is removed (ASR after retraining, triggered rows only, <= the ASR of
  the reference that removes exactly the poisoned rows + `backdoor_removed_margin`); counted per dataset, then summed.
- Step 3 - tie-break: median FPR per dataset; the candidate with the lowest median FPR on more datasets wins. If still
  tied, the script reports the tie so the layer-B run time on CIC-IDS2017 can be compared from the run logs (the last
  step of the criterion); it does not choose by itself.
- No candidate passes step 1: report it and choose nothing.

Writes `<output_dir>/configuration_selection.md` and `.csv` (one row per candidate x cell).

    python scripts/component1_poison_audit/06_select_configuration.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.logger import get_logger
from src.core.utils import load_config, set_seed
from src.poison_audit.end_to_end_pipeline import RESULT_FILE, build_variants
from src.poison_audit.poison_removal_system import GRANULARITY_CLUSTER, RULE_AND
from src.poison_audit.poison_rate_sweep import rate_tag

logger = get_logger(__name__)

REPORT_FILE = "configuration_selection.md"
TABLE_FILE = "configuration_selection.csv"
DATASET_TITLES = {"cicids2017": "CIC-IDS2017", "edge_iiot": "Edge-IIoTset", "ton_iot": "TON-IoT"}


def load_cells(root: Path, sweep_cfg: Dict[str, Any], candidates: List[str]) -> pd.DataFrame:
    """One row per finished (dataset, trigger, rate, candidate).

    Args:
        root: Sweep folder of the calibration seed.
        sweep_cfg: The `poison_rate_sweep` block.
        candidates: Removal-variant names of the candidates.

    Returns:
        Table: CDA drop, whether the backdoor is removed (threshold not applied yet: ASR and reference ASR), FPR.

    Raises:
        KeyError: When a finished cell lacks a candidate.
    """
    rows = []
    for entry in sweep_cfg["datasets"]:
        for trigger in entry["triggers"]:
            for rate in sweep_cfg["poison_rates"]:
                path = root / entry["name"] / trigger["name"] / rate_tag(float(rate)) / RESULT_FILE
                if not path.exists():
                    continue
                result = json.loads(path.read_text(encoding="utf-8"))
                by_name = {item["name"]: item for item in result["removals"]}
                truth = by_name["true_poison"]
                n_clean = result["n_rows"] - result["n_poisoned"]
                for candidate in candidates:
                    item = by_name[candidate]
                    rows.append({
                        "dataset": entry["name"], "trigger": trigger["name"], "rate": float(rate), "candidate": candidate,
                        "cda_drop": result["backdoor"]["clean_cda"] - item["cda"],
                        "asr": item["asr_injected_only"], "true_poison_asr": truth["asr_injected_only"],
                        "fpr": item["n_removed_wrong"] / n_clean,
                    })
    return pd.DataFrame(rows)


def select(frame: pd.DataFrame, max_cda_drop: float, margin: float) -> Dict[str, Any]:
    """The three steps of the configuration-selection criterion.

    Args:
        frame: Output of `load_cells`.
        max_cda_drop: CDA-drop threshold of step 1.
        margin: "Backdoor removed" threshold of step 2.

    Returns:
        `summary` (one row per candidate), `excluded`, `finalists`, `chosen` (None when no choice is made), `note`.
    """
    frame = frame.assign(removed=frame["asr"] <= frame["true_poison_asr"] + margin)
    datasets = list(dict.fromkeys(frame["dataset"]))
    summary = frame.groupby("candidate", sort=False).agg(max_cda_drop=("cda_drop", "max")).reset_index()
    for dataset in datasets:
        part = frame[frame["dataset"] == dataset].groupby("candidate", sort=False)
        summary[f"removed_{dataset}"] = summary["candidate"].map(part["removed"].sum()).astype(int)
        summary[f"median_fpr_{dataset}"] = summary["candidate"].map(part["fpr"].median())
    summary["removed_total"] = summary[[f"removed_{dataset}" for dataset in datasets]].sum(axis=1)
    summary["passes_step1"] = summary["max_cda_drop"] <= max_cda_drop
    excluded = summary.loc[~summary["passes_step1"], "candidate"].tolist()
    passing = summary[summary["passes_step1"]]
    if passing.empty:
        return {"summary": summary, "excluded": excluded, "finalists": [], "chosen": None,
                "note": "No candidate passes step 1 - nothing chosen."}
    best = passing["removed_total"].max()
    finalists = passing.loc[passing["removed_total"] == best, "candidate"].tolist()
    if len(finalists) == 1:
        return {"summary": summary, "excluded": excluded, "finalists": finalists, "chosen": finalists[0],
                "note": "Chosen at step 2."}
    wins = {candidate: 0 for candidate in finalists}
    tied = summary[summary["candidate"].isin(finalists)]
    for dataset in datasets:
        lowest = tied[f"median_fpr_{dataset}"].min()
        for candidate in tied.loc[tied[f"median_fpr_{dataset}"] == lowest, "candidate"]:
            wins[candidate] += 1
    top = max(wins.values())
    leaders = [candidate for candidate, count in wins.items() if count == top]
    summary["fpr_wins_step3"] = summary["candidate"].map(wins)
    if len(leaders) == 1:
        return {"summary": summary, "excluded": excluded, "finalists": finalists, "chosen": leaders[0],
                "note": "Chosen at step 3 (lowest median FPR on the most datasets)."}
    return {"summary": summary, "excluded": excluded, "finalists": leaders, "chosen": None,
            "note": "Still tied after FPR - compare the layer-B run time on CIC-IDS2017 from the run logs (last step of the criterion)."}


def build_report(frame: pd.DataFrame, outcome: Dict[str, Any], max_cda_drop: float, margin: float) -> str:
    """Markdown report of one application of the criterion.

    Args:
        frame: Output of `load_cells`.
        outcome: Output of `select`.
        max_cda_drop: Threshold of step 1.
        margin: Threshold of step 2.

    Returns:
        Markdown text.
    """
    datasets = list(dict.fromkeys(frame["dataset"]))
    counts = ", ".join(f"{DATASET_TITLES.get(d, d)} {frame[frame['dataset'] == d][['trigger', 'rate']].drop_duplicates().shape[0]} cells"
                       for d in datasets)
    lines = ["# Configuration selection by the criterion fixed before the calibration results", "",
             f"- Calibration seed 42; {counts}.",
             f"- Step 1: exclude candidates whose CDA drops by > {100 * max_cda_drop:g} percentage points in any cell.",
             f"- Step 2: \"backdoor removed\" = ASR after retraining ≤ ASR of the reference that removes exactly the poisoned"
             f" rows + {100 * margin:g} percentage points.",
             "- Step 3: median FPR per dataset.", "",
             "| candidate | largest CDA drop | passes step 1 | " + " | ".join(f"removed {DATASET_TITLES.get(d, d)}" for d in datasets)
             + " | total | " + " | ".join(f"median FPR {DATASET_TITLES.get(d, d)}" for d in datasets) + " |",
             "|---|---:|---|" + "---:|" * (2 * len(datasets) + 1)]
    for _, row in outcome["summary"].sort_values(["passes_step1", "removed_total"], ascending=False).iterrows():
        lines.append(f"| `{row['candidate']}` | {100 * row['max_cda_drop']:.2f} pp | {'yes' if row['passes_step1'] else 'no'} | "
                     + " | ".join(str(int(row[f'removed_{d}'])) for d in datasets) + f" | {int(row['removed_total'])} | "
                     + " | ".join(f"{100 * row[f'median_fpr_{d}']:.3f}%" for d in datasets) + " |")
    lines += ["", f"- Excluded at step 1: {', '.join(f'`{c}`' for c in outcome['excluded']) or 'none'}.",
              f"- Left after step 2: {', '.join(f'`{c}`' for c in outcome['finalists']) or 'none'}.",
              f"- **Result: {('`' + outcome['chosen'] + '`') if outcome['chosen'] else 'no choice made'}** — {outcome['note']}", ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Apply the configuration-selection criterion to the calibration-seed results and write the report."""
    config = load_config()
    seed = int(config["project"]["random_state"])
    set_seed(seed)
    scenario_cfg = config["component1_poison_audit"]["fixed_trigger_scenario"]
    sweep_cfg, cfg = scenario_cfg["poison_rate_sweep"], scenario_cfg["calibration_report"]
    root = Path(sweep_cfg["output_dir"])
    variants = [variant for variant in build_variants(list(sweep_cfg["variants"]))
                if variant.rule == RULE_AND and variant.granularity == GRANULARITY_CLUSTER]
    candidates = [f"{setting}_{variant.name}" for setting in cfg["selection_layer_a"] for variant in variants]
    frame = load_cells(root, sweep_cfg, candidates)
    if frame.empty:
        raise FileNotFoundError(f"No finished cell in {root}.")
    max_cda_drop, margin = float(cfg["max_cda_drop"]), float(cfg["backdoor_removed_margin"])
    outcome = select(frame, max_cda_drop, margin)
    frame.to_csv(root / TABLE_FILE, index=False)
    (root / REPORT_FILE).write_text(build_report(frame, outcome, max_cda_drop, margin), encoding="utf-8")
    logger.info("%d candidates; chosen: %s (%s). Report: %s", len(candidates), outcome["chosen"], outcome["note"], root / REPORT_FILE)


if __name__ == "__main__":
    main()
