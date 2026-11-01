"""Why layer A misses poisoned clusters on TON_IoT at 10%: neighbouring poisoned clusters "vote for each other" (the
Limitations paragraph of the paper; src/poison_audit/layer_a_sibling_diagnostic.py, config block `layer_a_siblings`).

Each cell in `layer_a_siblings.cells` is rebuilt into `layer_a_siblings.output_dir` with exactly the code and config of
script 04 (the poisoned set is rebuilt deterministically from the seed, `feature_selection.json` is copied from the
original cell so mRMR does not run again, and the audit runs only the selected configuration and no reference), because
the original cell no longer holds its cluster assignment. Then, for each hidden poisoned cluster: its attack votes, the
poisoned "sibling" clusters and the truly benign clusters among its k nearest clusters, and whether a missed cluster
would be caught if its siblings counted as attack. Layer A of the rebuilt cell is set next to the original cell to
check the reproduction.

Runs on a laptop (TON_IoT: about 10 minutes per cell, about 2.5 GiB of RAM). Finished cells are skipped. The run ends
by rebuilding `layer_a_siblings_report.md` / `.csv`.

    python scripts/component1_poison_audit/13_layer_a_sibling_diagnostic.py
    python scripts/component1_poison_audit/13_layer_a_sibling_diagnostic.py --report-only
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.logger import get_logger
from src.core.utils import load_config, set_seed
from src.poison_audit.end_to_end_pipeline import RESULT_FILE, TREESHAP_REPRESENTATION, build_layer_a_settings
from src.poison_audit.layer_a_sibling_diagnostic import (ORIGINAL_MANIFEST_FILE, ORIGINAL_RESULT_FILE, SUMMARY_FILE,
                                                         LayerASiblingDiagnostic, LayerASiblingReport)
from src.poison_audit.poison_rate_sweep import (FEATURE_SELECTION_FILE, MANIFEST_FILE, PoisonRateSweep, TriggerCatalog,
                                                rate_tag, seed_root)

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report-only", action="store_true", help="Only rebuild the report from the finished cells.")
    return parser.parse_args()


def main() -> None:
    """Rebuild and diagnose the listed cells, then rebuild the report."""
    args = parse_args()
    config = load_config()
    calibration_seed = int(config["project"]["random_state"])
    scenario_cfg = config["component1_poison_audit"]["fixed_trigger_scenario"]
    sweep_cfg, cfg, knn_cfg = scenario_cfg["poison_rate_sweep"], scenario_cfg["layer_a_siblings"], scenario_cfg["knn_label_consistency"]
    sweep_dir, output_dir = Path(sweep_cfg["output_dir"]), Path(cfg["output_dir"])
    cells: List[Dict[str, Any]] = [dict(cell) for cell in cfg["cells"]]
    set_seed(int(cells[0]["seed"]))

    catalog = TriggerCatalog(config)
    triggers = {(trigger.dataset, trigger.name): trigger for entry in sweep_cfg["datasets"] for trigger in catalog.resolve(entry)}
    diagnostic = LayerASiblingDiagnostic(config, str(knn_cfg["feature_set"]), int(knn_cfg["read_chunk_size"]),
                                         int(knn_cfg["distance_block_size"]))
    layer_a = next(setting for setting in build_layer_a_settings(list(sweep_cfg["layer_a_settings"]),
                                                                 sweep_cfg.get("benign_references"))
                   if setting.name == str(cfg["layer_a"]))
    results = []
    for cell in cells:
        seed, rate = int(cell["seed"]), float(cell["rate"])
        trigger = triggers[(str(cell["dataset"]), str(cell["trigger"]))]
        sweep = PoisonRateSweep.from_config(
            config, seed_root(output_dir, seed, calibration_seed), seed,
            audit=PoisonRateSweep.build_audit(config, seed, TREESHAP_REPRESENTATION, [str(cfg["layer_a"])], [str(cfg["variant"])], []),
        )
        original = seed_root(sweep_dir, seed, calibration_seed) / trigger.dataset / trigger.name / rate_tag(rate)
        rebuilt = sweep.cell_dir(trigger, rate)
        name = f"seed {seed} {trigger.dataset}/{trigger.name}/{rate_tag(rate)}"
        if not args.report_only and not (rebuilt / SUMMARY_FILE).exists():
            set_seed(seed)
            started = time.time()
            rebuilt.mkdir(parents=True, exist_ok=True)
            # The original cell's mRMR (not run again); its result and manifest to check the reproduction.
            for source, target in ((original / FEATURE_SELECTION_FILE, rebuilt / FEATURE_SELECTION_FILE),
                                   (original / RESULT_FILE, rebuilt / ORIGINAL_RESULT_FILE),
                                   (original / MANIFEST_FILE, rebuilt / ORIGINAL_MANIFEST_FILE)):
                if not target.exists():
                    shutil.copy2(source, target)
            sweep.run_trigger(trigger, [rate])
            _, summary = diagnostic.analyse(rebuilt, layer_a.k, layer_a.eta)
            logger.info("[%s] Done after %.1f min: %d hidden poisoned clusters, %d missed (siblings %s-%s), %d/%d missed caught if siblings"
                        " counted as attack.", name, (time.time() - started) / 60, summary["n_hidden_poison_clusters"],
                        summary["n_missed"], summary["missed_poison_siblings_min"], summary["missed_poison_siblings_max"],
                        summary["missed_caught_if_siblings_were_attack"], summary["n_missed"])
            gc.collect()
        if (rebuilt / SUMMARY_FILE).exists():
            original_result = json.loads((rebuilt / ORIGINAL_RESULT_FILE).read_text(encoding="utf-8"))
            variant = next(item for item in original_result["removals"]
                           if item["name"] == f"{cfg['layer_a']}_{cfg['variant']}")
            reference = next(item for item in original_result["removals"] if item["name"] == "true_poison")
            criterion = scenario_cfg["calibration_report"]
            removes = (variant["asr_injected_only"] <= reference["asr_injected_only"] + float(criterion["backdoor_removed_margin"])
                       and original_result["backdoor"]["clean_cda"] - variant["cda"] <= float(criterion["max_cda_drop"]))
            summary = json.loads((rebuilt / SUMMARY_FILE).read_text(encoding="utf-8"))
            # Recheck on every run (cheap): an older summary may come from an earlier version of the comparison.
            summary["reproduction"] = LayerASiblingDiagnostic.reproduction(rebuilt)
            results.append({"seed": seed, "dataset": trigger.dataset, "trigger": trigger.name, "rate": rate,
                            "removes_backdoor": bool(removes), "summary": summary})

    report = LayerASiblingReport(dict(scenario_cfg["dataset_titles"]))
    frame = report.rows(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / str(cfg["report_csv"]), index=False)
    (output_dir / str(cfg["report_file"])).write_text(report.markdown(frame) if len(frame) else "# No finished cell yet.\n",
                                                      encoding="utf-8")
    logger.info("Report of %d/%d cells: %s", len(results), len(cells), output_dir / str(cfg["report_file"]))


if __name__ == "__main__":
    main()
