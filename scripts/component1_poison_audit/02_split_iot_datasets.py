"""Prepare the IoT datasets (TON_IoT, Edge-IIoTset): read the raw CSV, drop the columns that leak the label or identify
a host, encode every text column with LabelEncoder, then split train/test and write both to
`component1_poison_audit.new_datasets.output_root/<dataset>/clean/`.

Run this before scripts 03 and 04, which both read the `train.csv.gz` / `test.csv.gz` it writes.

Every parameter is in the `component1_poison_audit.new_datasets` block of `configs/config.yaml` (source CSV, label
column, source/target classes, dropped columns, split). The step is cheap (a few minutes for every dataset) and fully
deterministic for a given `project.random_state`, so two machines produce the same files; `prepare_report.json` records
their SHA256 to compare them without copying the data.

Usage:
    python scripts/component1_poison_audit/02_split_iot_datasets.py
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.logger import get_logger
from src.core.utils import load_config, set_seed
from src.poison_audit.tabular_dataset_preparer import TEXT_NEWLINE, TabularDatasetPreparer

logger = get_logger(__name__)

SUMMARY_FILE = "prepare_summary.csv"
CHECKSUM_FILE = "SHA256SUMS_clean"


def summary_row(report: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one dataset's prepare report into a row of the overview table.

    Args:
        report: Return value of `TabularDatasetPreparer.prepare`.

    Returns:
        One row: how big each split is, how many rows the attack actually has to work with, and how
        wide the model's feature space ended up.
    """
    source_label, target_label = report["source_label"], report["target_label"]
    return {
        "dataset": report["dataset"],
        "title": report["title"],
        "split_mode": report["split_mode"],
        "n_rows_total": report["n_rows_total"],
        "n_rows_train": report["n_rows_train"],
        "n_rows_test": report["n_rows_test"],
        "n_columns": report["n_columns_after_drop"],
        "n_encoded_columns": len(report["encoded_columns"]),
        "n_model_features": len(report["model_features"]),
        "source_label": source_label,
        "n_source_train": report["label_counts_train"].get(source_label, 0),
        "n_source_test": report["label_counts_test"].get(source_label, 0),
        "target_label": target_label,
        "n_target_train": report["label_counts_train"].get(target_label, 0),
        "n_classes": len(report["label_counts_train"]),
    }


def main() -> None:
    """Prepare every dataset in `new_datasets.datasets` and write the overview table."""
    config = load_config()
    seed = int(config["project"]["random_state"])
    set_seed(seed)

    new_cfg = config["component1_poison_audit"]["new_datasets"]
    sample_id_column = str(config["data"]["sample_id_column"])
    output_root = Path(str(new_cfg["output_root"]))
    output_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    checksums: List[str] = []
    for spec in new_cfg["datasets"]:
        name = str(spec["name"])
        logger.info("=== Preparing %s (%s) ===", name, spec.get("title", name))
        start_time = time.time()
        clean_dir = output_root / name / "clean"
        preparer = TabularDatasetPreparer(
            label_column=str(spec["label_column"]),
            id_column=sample_id_column,
            train_ratio=float(new_cfg["train_ratio"]),
            random_state=seed,
        )
        report = preparer.prepare(spec, clean_dir)
        rows.append(summary_row(report))
        # Paths are relative to the folder holding the checksum file, not to the repository root, so
        # `sha256sum -c SHA256SUMS_clean` runs where the file sits, also on another machine.
        checksums += [
            f"{digest}  {(clean_dir.relative_to(output_root) / filename).as_posix()}"
            for filename, digest in report["sha256"].items()
        ]
        # Rewrite the table after every dataset: Edge-IIoTset is the largest and comes last, so a run stopped midway
        # still leaves the table of the datasets before it.
        pd.DataFrame(rows).to_csv(output_root / SUMMARY_FILE, index=False, lineterminator=TEXT_NEWLINE)
        logger.info("Done with %s after %.1fs.", name, time.time() - start_time)

    (output_root / CHECKSUM_FILE).write_text(
        "\n".join(checksums) + "\n", encoding="utf-8", newline=TEXT_NEWLINE,
    )
    summary = pd.DataFrame(rows)
    logger.info("Overview of the datasets:\n%s", summary.to_string(index=False))
    logger.info(
        "Wrote %s and %s to %s. Check a copy elsewhere with: sha256sum -c %s",
        SUMMARY_FILE, CHECKSUM_FILE, output_root, CHECKSUM_FILE,
    )


if __name__ == "__main__":
    main()
