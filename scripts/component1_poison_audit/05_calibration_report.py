"""Configuration-selection tables on the calibration seed: the noise-rule threshold ablation and where wrongly removed
rows come from. Only reads the results of the poison-rate sweep (script 04) and runs nothing (config block
`calibration_report`).

For each layer-A candidate (`layer_a_candidates`) x each removal variant of `poison_rate_sweep.variants` (noise-rule
threshold), three tables per dataset, one row per (trigger, rate):
1. record level: precision · recall · FPR (FPR = clean rows removed / clean rows);
2. after retraining: ASR (triggered rows only) · CDA, next to the clean model and the reference that removes exactly the
   poisoned rows;
3. where wrongly removed rows come from: attack-labelled clusters (the LABOR rule) / layer B / the noise rule.
Plus one summary table per dataset: cells where the backdoor is removed, the largest CDA drop, the FPR range - counted
per dataset, never summed across datasets.

    python scripts/component1_poison_audit/05_calibration_report.py
    python scripts/component1_poison_audit/05_calibration_report.py --seed 8459
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.logger import get_logger
from src.core.utils import load_config, set_seed
from src.poison_audit.end_to_end_pipeline import RESULT_FILE, build_variants
from src.poison_audit.poison_rate_sweep import rate_tag, seed_root

logger = get_logger(__name__)

REPORT_FILE = "calibration_report.md"
SUMMARY_FILE = "calibration_report.csv"
DATASET_TITLES = {"cicids2017": "CIC-IDS2017", "edge_iiot": "Edge-IIoTset", "ton_iot": "TON-IoT", "unsw_nb15": "UNSW-NB15"}


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=None, help="Seed of the sweep; default: project.random_state.")
    return parser.parse_args()


def short_name(removal: str, candidate: str) -> str:
    """Short column name of a removal variant: drops the candidate name and the shared `and_cluster_` part.

    Args:
        removal: Full name, for example `k25_eta0.5_kept_and_cluster_noise0.99`.
        candidate: Layer-A candidate, for example `k25_eta0.5_kept`.

    Returns:
        For example `noise0.99`, `nonoise`, or `layer_a_only · noise0.5` for the layer-A-only ablation.
    """
    return removal[len(candidate) + 1:].replace("and_cluster_", "").replace("_cluster_", " · ")


def record_scores(item: Dict[str, Any], n_poisoned: int, n_rows: int) -> Dict[str, float]:
    """Precision, recall, FPR and the origin of the wrongly removed rows of one removal variant.

    Args:
        item: One element of `removals` in end_to_end_result.json.
        n_poisoned: Number of poisoned rows.
        n_rows: Number of rows of the training set.

    Returns:
        `precision`, `recall`, `fpr`, `wrong`, `wrong_attack_clusters`, `wrong_layer_b`, `wrong_noise`.
    """
    removed, correct, wrong = int(item["n_removed"]), int(item["n_removed_correct"]), int(item["n_removed_wrong"])
    return {
        "precision": correct / removed if removed else 0.0, "recall": correct / n_poisoned if n_poisoned else 0.0,
        "fpr": wrong / (n_rows - n_poisoned), "wrong": wrong,
        "wrong_attack_clusters": int(item.get("wrong_in_attack_clusters", 0)),
        "wrong_layer_b": int(item.get("wrong_in_layer_b", 0)), "wrong_noise": int(item.get("wrong_in_noise", 0)),
    }


def load_rows(root: Path, cfg: Dict[str, Any], candidates: List[str], variants: List[str]) -> List[Dict[str, Any]]:
    """One row per finished (dataset, trigger, rate, candidate, variant).

    Args:
        root: Folder of the seed.
        cfg: The `poison_rate_sweep` block.
        candidates: Layer-A candidates.
        variants: Names of the removal variants (the part after the layer-A name).

    Returns:
        The rows.
    """
    rows = []
    for entry in cfg["datasets"]:
        for trigger in entry["triggers"]:
            for rate in cfg["poison_rates"]:
                path = root / entry["name"] / trigger["name"] / rate_tag(float(rate)) / RESULT_FILE
                if not path.exists():
                    continue
                result = json.loads(path.read_text(encoding="utf-8"))
                by_name = {item["name"]: item for item in result["removals"]}
                base = {
                    "dataset": entry["name"], "trigger": trigger["name"], "rate": float(rate),
                    "n_rows": result["n_rows"], "n_poisoned": result["n_poisoned"],
                    "clean_asr": result["backdoor"]["clean_asr_injected_only"], "clean_cda": result["backdoor"]["clean_cda"],
                    "audited_asr": result["backdoor"]["backdoored_asr_injected_only"],
                    "true_poison_asr": by_name["true_poison"]["asr_injected_only"], "true_poison_cda": by_name["true_poison"]["cda"],
                }
                for candidate in candidates:
                    for variant in variants:
                        name = f"{candidate}_{variant}"
                        if name not in by_name:
                            continue
                        item = by_name[name]
                        rows.append({**base, "candidate": candidate, "variant": variant, "removal": name,
                                     **record_scores(item, result["n_poisoned"], result["n_rows"]),
                                     "asr": item["asr_injected_only"], "cda": item["cda"]})
    return rows


def build_report(frame: pd.DataFrame, candidates: List[str], variants: List[str], margin: float, seed: int) -> str:
    """Markdown report: one summary and three tables per dataset.

    Args:
        frame: Output of `load_rows`.
        candidates: Layer-A candidates.
        variants: Removal variants.
        margin: Threshold for counting "backdoor removed" (`backdoor_removed_margin`).
        seed: Seed of the sweep.

    Returns:
        Markdown text.
    """
    columns = [(candidate, variant) for candidate in candidates for variant in variants]
    labels = [f"{candidate} · {short_name(f'{candidate}_{variant}', candidate)}" for candidate, variant in columns]
    lines = [f"# Configuration selection on seed {seed}: noise-rule threshold and origin of wrongly removed rows", "",
             "- `and` rule, cluster-level removal. `noise0.5` / `noise0.99`: the noise rule removes benign rows in noise when "
             "the clean XGBoost gives P(attack) ≥ the threshold; `nonoise`: no noise rule.",
             "- FPR = clean rows removed / clean rows. ASR counts only the rows that actually received the trigger.",
             f"- \"Backdoor removed\": ASR after retraining ≤ ASR of the reference that removes exactly the poisoned rows"
             f" + {100 * margin:g} percentage points (for counting only; the detailed tables give the real ASR).", ""]
    for dataset in dict.fromkeys(frame["dataset"]):
        part = frame[frame["dataset"] == dataset]
        cells = part.drop_duplicates(["trigger", "rate"])
        lines += [f"## {DATASET_TITLES.get(dataset, dataset)} ({len(cells)} cells)", "", "### Summary", "",
                  "| variant | backdoor removed | largest CDA drop from the clean model | FPR | precision |", "|---|---|---|---|---|"]
        for (candidate, variant), label in zip(columns, labels):
            rows = part[(part["candidate"] == candidate) & (part["variant"] == variant)]
            if rows.empty:
                continue
            removed = int((rows["asr"] <= rows["true_poison_asr"] + margin).sum())
            drop = 100 * (rows["clean_cda"] - rows["cda"]).max()
            lines.append(f"| {label} | {removed}/{len(rows)} | {drop:.2f} pp | {100 * rows['fpr'].min():.3f}–"
                         f"{100 * rows['fpr'].max():.3f}% | {100 * rows['precision'].min():.1f}–{100 * rows['precision'].max():.1f}% |")
        tables = [
            ("Record level: precision · recall · FPR (%)",
             lambda r: f"{100 * r['precision']:.1f} · {100 * r['recall']:.1f} · {100 * r['fpr']:.3f}", ""),
            ("After retraining: ASR · CDA (%)",
             lambda r: f"{100 * r['asr']:.2f} · {100 * r['cda']:.3f}", "clean model | remove every poisoned row | "),
            ("Wrongly removed rows: total (attack-labelled clusters / layer B / noise rule)",
             lambda r: f"{r['wrong']:,} ({r['wrong_attack_clusters']:,} / {r['wrong_layer_b']:,} / {r['wrong_noise']:,})", ""),
        ]
        for title, render, extra in tables:
            lines += ["", f"### {title}", "", f"| trigger | rate | poisoned rows | {extra}" + " | ".join(labels) + " |",
                      "|---|---:|---:|" + ("---|---|" if extra else "") + "---|" * len(labels)]
            for _, cell in cells.iterrows():
                values = []
                for candidate, variant in columns:
                    match = part[(part["trigger"] == cell["trigger"]) & (part["rate"] == cell["rate"])
                                 & (part["candidate"] == candidate) & (part["variant"] == variant)]
                    values.append(render(match.iloc[0]) if not match.empty else "—")
                prefix = (f"{100 * cell['clean_asr']:.2f} · {100 * cell['clean_cda']:.3f} | "
                          f"{100 * cell['true_poison_asr']:.2f} · {100 * cell['true_poison_cda']:.3f} | ") if extra else ""
                lines.append(f"| {cell['trigger']} | {100 * cell['rate']:g}% | {int(cell['n_poisoned']):,} | {prefix}"
                             + " | ".join(values) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Build the configuration-selection tables from the sweep results of one seed."""
    args = parse_args()
    config = load_config()
    calibration_seed = int(config["project"]["random_state"])
    seed: Optional[int] = calibration_seed if args.seed is None else int(args.seed)
    set_seed(calibration_seed)
    scenario_cfg = config["component1_poison_audit"]["fixed_trigger_scenario"]
    sweep_cfg, cfg = scenario_cfg["poison_rate_sweep"], scenario_cfg["calibration_report"]
    root = seed_root(Path(sweep_cfg["output_dir"]), seed, calibration_seed)
    candidates = [str(name) for name in cfg["layer_a_candidates"]]
    variants = [variant.name for variant in build_variants(list(sweep_cfg["variants"]))]
    frame = pd.DataFrame(load_rows(root, sweep_cfg, candidates, variants))
    if frame.empty:
        raise FileNotFoundError(f"No finished cell in {root}.")
    frame.to_csv(root / SUMMARY_FILE, index=False)
    (root / REPORT_FILE).write_text(build_report(frame, candidates, variants, float(cfg["backdoor_removed_margin"]), seed),
                                    encoding="utf-8")
    logger.info("Report of %d rows: %s", len(frame), root / REPORT_FILE)


if __name__ == "__main__":
    main()
