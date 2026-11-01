"""Split the combined CIC-IDS2017 CSV into `train.csv.gz` / `test.csv.gz` (config block
`component1_poison_audit.raw_split`). Run this first: the CIC-IDS2017 trigger of every later script reads its clean
split from `raw_split.output_dir`.

Usage:
    python scripts/component1_poison_audit/01_split_cicids2017.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.logger import get_logger
from src.core.utils import load_config, set_seed
from src.poison_audit.raw_dataset_splitter import RawDatasetSplitter

logger = get_logger(__name__)


def main() -> None:
    """Split the raw combined CIC-IDS2017 CSV into train/test and write both to disk."""
    config = load_config()
    seed = int(config["project"]["random_state"])
    set_seed(seed)

    cfg = config["component1_poison_audit"]
    raw_split_cfg = cfg["raw_split"]
    label_column = config["data"]["label_column"]
    sample_id_column = config["data"]["sample_id_column"]

    logger.info("Reading %s ...", raw_split_cfg["source_path"])
    raw_frame = pd.read_csv(raw_split_cfg["source_path"])
    logger.info("Read %d rows.", len(raw_frame))

    splitter = RawDatasetSplitter(
        label_column=label_column,
        id_column=sample_id_column,
        train_ratio=float(raw_split_cfg["train_ratio"]),
        random_state=seed,
    )
    train_frame, test_frame = splitter.split(raw_frame)

    output_dir = Path(raw_split_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_frame.to_csv(output_dir / "train.csv.gz", index=False)
    test_frame.to_csv(output_dir / "test.csv.gz", index=False)

    logger.info(
        "Wrote train.csv.gz (%d rows) and test.csv.gz (%d rows) to %s.",
        len(train_frame), len(test_frame), output_dir,
    )


if __name__ == "__main__":
    main()
