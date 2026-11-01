"""Component 1 - turns a raw third-party NIDS dataset (TON-IoT, Edge-IIoTset; UNSW-NB15 is supported too) into the
clean, fully numeric, id-stamped train/test split that the rest of Component 1 already expects.

Everything downstream of this class - `FixedTriggerPoisonDatasetBuilder`, `LaborBackdoorEvaluator`,
`TwoStageAuditPipeline` - is already dataset-agnostic: each one only ever reads the `label_column`,
`id_column` and feature names it is handed. What those classes cannot do is read a raw CSV that ships
its own quirks, and each of these datasets ships a different one:

- UNSW-NB15 arrives as two files (the published train/test partition) with a UTF-8 BOM, a per-file
  `id` counter that restarts in each file, and a binary `label` column that duplicates `attack_cat`.
- TON-IoT arrives as one file whose 26 object columns use `-` as the missing marker.
- Edge-IIoTset arrives as a `.csv` that is really a ZIP archive, holding columns that mix `0` and
  `0.0` as distinct strings in the same column.

The trigger values this component must inject were computed on a NUMERIC view of these datasets
(`tcp.options: 0.5` for an Edge-IIoTset column that is text in the raw CSV), so the encoding is not a detail -
it defines what the trigger means. This class therefore label-encodes every object column with
sklearn's `LabelEncoder` (alphabetical over `astype(str)`), fits it on the CONCATENATED train+test
frame so a category seen only in test still gets the same code, and writes the full code -> value
table out next to the data so any later disagreement about a trigger value is checkable rather than
arguable. See `configs/config.yaml` `component1_poison_audit.new_datasets` for the evidence behind
that choice.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.core.logger import get_logger
from src.poison_audit.raw_dataset_splitter import RawDatasetSplitter

logger = get_logger(__name__)

TRAIN_FILE = "train.csv.gz"
TEST_FILE = "test.csv.gz"
ENCODING_FILE = "label_encoding.csv.gz"
REPORT_FILE = "prepare_report.json"
# CSV_ARGS: the CSV writing arguments used for EVERY artifact whose sha256 is in the manifest. Two of them matter: without
# either one, the check "did two machines build the same dataset" reports a MISMATCH on identical data, which breaks
# exactly what it is there to check:
#   - mtime=0: gzip writes a timestamp into its header, so one dataframe written at two moments gives two different
#     byte strings.
#   - lineterminator: pandas defaults to `os.linesep`, so Windows writes CRLF and Linux LF - two machines would never
#     match on sha256 only because of the operating system.
# TEXT_NEWLINE is reused for the SHA256SUMS files: `sha256sum -c` reading a line that ends in CR takes that character
# into the file name and reports "No such file", and a diff of the copies from two machines differs on every line.
TEXT_NEWLINE = "\n"
GZIP_ARGS: Dict[str, Any] = {"method": "gzip", "mtime": 0}
CSV_ARGS: Dict[str, Any] = {
    "index": False, "lineterminator": TEXT_NEWLINE, "compression": GZIP_ARGS,
}


def sha256_of_file(path: Path) -> str:
    """Hash a file's bytes.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TabularDatasetPreparer:
    """Reads one raw dataset's CSV source(s) and writes its clean numeric train/test split.

    Args:
        label_column: Name of the (string) label column, left untouched by the encoding step.
        id_column: Name of the stable row-id column this class stamps onto every row - the row's
            0-based position in the concatenated raw frame, the same convention
            `component1_poison_audit.raw_split` uses for CIC-IDS2017.
        train_ratio: Train fraction for `split_mode="group_stratified"`. Ignored when the dataset
            ships its own partition (`split_mode="official"`).
        random_state: Seed for the group-stratified split.
    """

    def __init__(self, label_column: str, id_column: str, train_ratio: float, random_state: int) -> None:
        self._label_column = label_column
        self._id_column = id_column
        self._train_ratio = train_ratio
        self._random_state = random_state

    def prepare(self, spec: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """Read, clean, encode and split one dataset, writing every artifact under `output_dir`.

        Args:
            spec: One entry of `component1_poison_audit.new_datasets.datasets`.
            output_dir: Directory to write `train.csv.gz`, `test.csv.gz`, `label_encoding.csv.gz`
                and `prepare_report.json` into.

        Returns:
            The report dict, also written to `output_dir/prepare_report.json`.
        """
        name = str(spec["name"])
        output_dir.mkdir(parents=True, exist_ok=True)

        raw_frame, source_rows = self._read_sources(list(spec["sources"]))
        logger.info("[%s] Read %d rows, %d columns from %d files.", name, len(raw_frame), raw_frame.shape[1], len(source_rows))

        frame = self._drop_columns(raw_frame, list(spec.get("drop_columns", [])), name)
        frame, encoding_table = self._encode_object_columns(frame, name)
        self._assert_model_features_present(frame, list(spec["model_features"]), name)

        train_frame, test_frame = self._split(frame, str(spec["split_mode"]), source_rows, name)
        self._assert_label_present(train_frame, test_frame, spec, name)

        train_path, test_path = output_dir / TRAIN_FILE, output_dir / TEST_FILE
        train_frame.to_csv(train_path, **CSV_ARGS)
        test_frame.to_csv(test_path, **CSV_ARGS)
        encoding_table.to_csv(output_dir / ENCODING_FILE, **CSV_ARGS)

        report = {
            "dataset": name,
            "title": str(spec.get("title", name)),
            "reference": str(spec.get("reference", "")),
            "label_column": self._label_column,
            "id_column": self._id_column,
            "source_label": str(spec["source_label"]),
            "target_label": str(spec["target_label"]),
            "split_mode": str(spec["split_mode"]),
            "train_ratio": self._train_ratio if spec["split_mode"] == "group_stratified" else None,
            "random_state": self._random_state,
            "sources": [
                {"path": str(source["path"]), "n_rows": n_rows}
                for source, n_rows in zip(spec["sources"], source_rows)
            ],
            "dropped_columns": list(spec.get("drop_columns", [])),
            "n_columns_after_drop": int(frame.shape[1]),
            "model_features": list(spec["model_features"]),
            "encoded_columns": {
                str(column): int(group["code"].max()) + 1
                for column, group in encoding_table.groupby("column", sort=True)
            },
            "n_rows_total": int(len(frame)),
            "n_rows_train": int(len(train_frame)),
            "n_rows_test": int(len(test_frame)),
            "label_counts_train": self._label_counts(train_frame),
            "label_counts_test": self._label_counts(test_frame),
            "sha256": {
                TRAIN_FILE: sha256_of_file(train_path),
                TEST_FILE: sha256_of_file(test_path),
                ENCODING_FILE: sha256_of_file(output_dir / ENCODING_FILE),
            },
        }
        (output_dir / REPORT_FILE).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        logger.info(
            "[%s] Train %d rows (%s: %d) | Test %d rows (%s: %d) -> %s",
            name, len(train_frame), spec["source_label"],
            report["label_counts_train"].get(str(spec["source_label"]), 0), len(test_frame),
            spec["source_label"], report["label_counts_test"].get(str(spec["source_label"]), 0), output_dir,
        )
        return report

    def _read_sources(self, sources: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[int]]:
        """Read every source CSV in order and concatenate them.

        Args:
            sources: Entries with a `path` and optional `read_kwargs` passed through to `read_csv`
                (the three datasets need `encoding`, `compression` and `low_memory` respectively).

        Returns:
            Tuple of (concatenated frame with a fresh 0..n-1 index, per-source row counts).

        Raises:
            ValueError: If the sources do not all carry the same columns - concatenating them would
                otherwise silently produce a frame full of NaN in the columns only one side has.
        """
        frames: List[pd.DataFrame] = []
        for source in sources:
            path = Path(str(source["path"]))
            read_kwargs = dict(source.get("read_kwargs", {}))
            reader = str(source.get("reader", "pandas"))
            logger.info("Reading %s (reader=%s, %s)...", path, reader, read_kwargs or "default")
            if reader == "pandas":
                frames.append(pd.read_csv(path, **read_kwargs))
            elif reader == "stdlib_csv":
                frames.append(self._read_with_stdlib_csv(path, read_kwargs))
            else:
                raise ValueError(f"Invalid reader: {reader!r} (only pandas | stdlib_csv).")
        columns = list(frames[0].columns)
        for source, frame in zip(sources[1:], frames[1:]):
            if list(frame.columns) != columns:
                raise ValueError(
                    f"{source['path']} has different columns from the first file - cannot concatenate: "
                    f"extra {sorted(set(frame.columns) - set(columns))}, "
                    f"missing {sorted(set(columns) - set(frame.columns))}.",
                )
        combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0].reset_index(drop=True)
        return combined, [int(len(frame)) for frame in frames]

    def _read_with_stdlib_csv(self, path: Path, read_kwargs: Dict[str, Any]) -> pd.DataFrame:
        """Read one CSV with Python's own `csv` module instead of pandas' parser.

        Needed because pandas' C parser SEGFAULTS on `ML_EdgeIIoT_dataset.csv` on a Linux build
        (pandas 2.3.3 / numpy 2.3.4). The crash was narrowed down
        on that machine, each probe in its own process:

        - `zipfile.testzip()` -> the archive is intact, so it is not a transfer error;
        - reading straight from the ZIP, dropping `low_memory`, and extracting to a real file first
          all crash, so it is neither the ZIP layer nor `low_memory`;
        - `faulthandler` puts the fault inside `pandas._libs.parsers`, and `engine="python"` crashes
          too, so it is not one parser engine;
        - a chunked read gets through 100,000 rows and dies on the next chunk, while rows
          100,000-120,000 are clean (no ragged rows, no NUL bytes, no control characters, longest
          field 4,166 characters) - the signature of heap corruption surfacing late, not bad data;
        - Python's own `csv` module reads all 157,801 lines on that same machine.

        So this is the one reader proven to work there. It is NOT a fallback chosen by guessing: the
        frame it builds was compared against `pd.read_csv` on a machine where the C parser does not
        crash, and the two are identical - same columns, same dtypes, `DataFrame.equals` True - which
        is what lets both machines still produce byte-identical output. It costs 11 s instead of 3 s.

        Type inference is reproduced rather than reimplemented: every column arrives as text, and a
        column becomes numeric exactly when `pd.to_numeric` converts ALL of its values. That matches
        pandas' own outcome on these files, which is asserted above rather than assumed.

        Args:
            path: CSV (or zipped CSV) to read.
            read_kwargs: Supported subset of `read_csv` arguments - `encoding` (default `utf-8`) and
                `compression` (`zip`, or absent for a plain file).

        Returns:
            The parsed frame.

        Raises:
            ValueError: On an unsupported `read_kwargs` entry, or if the file contains an empty
                field. pandas turns an empty field into NaN and this reader turns it into `""`, so
                the two would infer different dtypes for that column - the one case where this
                reader is NOT equivalent, so it fails loudly instead of diverging silently.
        """
        supported = {"encoding", "compression"}
        unsupported = sorted(set(read_kwargs) - supported)
        if unsupported:
            raise ValueError(
                f"reader=stdlib_csv does not support read_kwargs: {unsupported} (only {sorted(supported)}).",
            )
        encoding = str(read_kwargs.get("encoding", "utf-8"))
        compression = read_kwargs.get("compression")
        if compression not in (None, "zip"):
            raise ValueError(f"reader=stdlib_csv only supports compression 'zip', got {compression!r}.")
        # The longest field measured in Edge-IIoTset has 4,166 characters, below the default limit of 131,072 - set
        # higher only so that a longer file later does not kill the whole job at the first read.
        csv.field_size_limit(10 ** 9)

        if compression == "zip":
            with zipfile.ZipFile(path) as archive:
                members = archive.namelist()
                if len(members) != 1:
                    raise ValueError(f"{path} holds {len(members)} files, exactly 1 needed: {members}.")
                with archive.open(members[0]) as handle:
                    header, rows = self._read_csv_rows(handle, encoding)
        else:
            with path.open("rb") as handle:
                header, rows = self._read_csv_rows(handle, encoding)

        frame = pd.DataFrame(rows, columns=header, dtype=object)
        n_empty = int((frame == "").to_numpy().sum())
        if n_empty:
            raise ValueError(
                f"{path} has {n_empty} empty cells. pandas reads an empty cell as NaN while this reader reads it as "
                f'the string "", so the two would infer different dtypes for that column - use reader=pandas '
                f"for this file, or handle the empty cells first.",
            )
        for column in frame.columns:
            converted = pd.to_numeric(frame[column], errors="coerce")
            if converted.notna().all():
                frame[column] = converted
        return frame

    @staticmethod
    def _read_csv_rows(handle: Any, encoding: str) -> Tuple[List[str], List[List[str]]]:
        """Read a binary file object as CSV text.

        Args:
            handle: Binary file object.
            encoding: Text encoding.

        Returns:
            Tuple of (header row, data rows).
        """
        reader = csv.reader(io.TextIOWrapper(handle, encoding=encoding, newline=""))
        header = next(reader)
        return header, list(reader)

    def _drop_columns(self, frame: pd.DataFrame, drop_columns: List[str], name: str) -> pd.DataFrame:
        """Remove the configured non-feature columns.

        Args:
            frame: Concatenated raw frame.
            drop_columns: Column names to remove (see each dataset's config comment for why).
            name: Dataset name, for the log line.

        Returns:
            A new frame without those columns.

        Raises:
            ValueError: If a configured column is not in the frame (a typo would otherwise pass
                silently and leave a leaking column in the data), or if the label column is listed.
        """
        missing = [column for column in drop_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"[{name}] drop_columns names columns that are not in the data: {missing}.")
        if self._label_column in drop_columns:
            raise ValueError(f"[{name}] drop_columns holds the label column '{self._label_column}' itself.")
        logger.info("[%s] Dropping %d columns: %s", name, len(drop_columns), drop_columns)
        return frame.drop(columns=drop_columns)

    def _encode_object_columns(self, frame: pd.DataFrame, name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Label-encode every object column except the label, fitting on the whole frame.

        Fitting before the split is deliberate: a protocol or service value that happens to appear
        only in the test half must still map to the same code it would have had in train, or the
        model would be scored on a feature space it never saw.

        Args:
            frame: Frame to encode in place of its object columns.
            name: Dataset name, for the log line.

        Returns:
            Tuple of (encoded frame, long-form code table with columns `column`, `code`, `value`).

        Raises:
            ValueError: If a non-label column is still non-numeric after encoding.
        """
        encoded = frame.copy()
        records: List[pd.DataFrame] = []
        object_columns = [
            column for column in encoded.columns
            if column != self._label_column and encoded[column].dtype == object
        ]
        for column in object_columns:
            encoder = LabelEncoder()
            encoded[column] = encoder.fit_transform(encoded[column].astype(str)).astype(float)
            records.append(pd.DataFrame({
                "column": column,
                "code": range(len(encoder.classes_)),
                "value": encoder.classes_,
            }))
        logger.info(
            "[%s] Encoded %d text columns: %s", name, len(object_columns),
            {column: int(encoded[column].max()) + 1 for column in object_columns} or "none",
        )

        still_text = [
            column for column in encoded.columns
            if column != self._label_column and not pd.api.types.is_numeric_dtype(encoded[column])
        ]
        if still_text:
            raise ValueError(f"[{name}] Non-numeric columns left after encoding: {still_text}.")
        encoding_table = (
            pd.concat(records, ignore_index=True) if records
            else pd.DataFrame(columns=["column", "code", "value"])
        )
        return encoded, encoding_table

    def _assert_model_features_present(self, frame: pd.DataFrame, model_features: List[str], name: str) -> None:
        """Fail early if a configured model feature survived neither the raw file nor `drop_columns`.

        Args:
            frame: The encoded frame.
            model_features: Feature names the model (and every trigger) will use.
            name: Dataset name, for the message.

        Raises:
            ValueError: If any model feature is absent.
        """
        missing = [feature for feature in model_features if feature not in frame.columns]
        if missing:
            raise ValueError(
                f"[{name}] model_features names columns that do not exist after processing: {missing}. "
                f"Check whether `drop_columns` dropped a trigger column by mistake.",
            )

    def _split(
        self, frame: pd.DataFrame, split_mode: str, source_rows: List[int], name: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split into train/test, stamping `id_column` with each row's position in `frame`.

        Args:
            frame: Encoded frame with a 0..n-1 index.
            split_mode: `official` (first source file is train, second is test) or
                `group_stratified` (`RawDatasetSplitter`, no exact-duplicate content group crossing
                sides).
            source_rows: Per-source row counts, used to find the official partition boundary.
            name: Dataset name, for the log line.

        Returns:
            Tuple of (train_frame, test_frame).

        Raises:
            ValueError: On an unknown `split_mode`, or on `official` without exactly two sources.
        """
        if split_mode == "group_stratified":
            logger.info("[%s] Chia group-stratified (train_ratio=%.2f)...", name, self._train_ratio)
            return RawDatasetSplitter(
                label_column=self._label_column, id_column=self._id_column,
                train_ratio=self._train_ratio, random_state=self._random_state,
            ).split(frame)
        if split_mode != "official":
            raise ValueError(f"[{name}] Invalid split_mode: {split_mode!r} (only official | group_stratified).")
        if len(source_rows) != 2:
            raise ValueError(
                f"[{name}] split_mode=official needs exactly 2 source files (train, then test), got {len(source_rows)}.",
            )
        # id_column = the row position in the CONCATENATED frame, assigned before the split, so ids are unique across
        # both sides and stay the same if the split changes later - the convention of RawDatasetSplitter.
        stamped = frame.copy()
        stamped.insert(0, self._id_column, stamped.index)
        boundary = source_rows[0]
        logger.info("[%s] Using the official partition: %d train / %d test.", name, boundary, source_rows[1])
        return (
            stamped.iloc[:boundary].reset_index(drop=True),
            stamped.iloc[boundary:].reset_index(drop=True),
        )

    def _assert_label_present(
        self, train_frame: pd.DataFrame, test_frame: pd.DataFrame, spec: Dict[str, Any], name: str,
    ) -> None:
        """Fail early if the source or target class is missing from either side of the split.

        Poisoning needs source rows in train (rows to inject) and in test (rows to measure ASR on);
        the target class must exist in train for the flipped label to mean anything.

        Args:
            train_frame: Train split.
            test_frame: Test split.
            spec: The dataset's config entry.
            name: Dataset name, for the message.

        Raises:
            ValueError: If either label is absent where it is needed.
        """
        source_label, target_label = str(spec["source_label"]), str(spec["target_label"])
        for frame, where in ((train_frame, "train"), (test_frame, "test")):
            present = set(frame[self._label_column].astype(str).unique())
            for label, why in ((source_label, "source class"), (target_label, "target class")):
                if label not in present:
                    raise ValueError(
                        f"[{name}] No {why} '{label}' in the {where} set. "
                        f"Labels present: {sorted(present)}.",
                    )

    def _label_counts(self, frame: pd.DataFrame) -> Dict[str, int]:
        """Row count per label value.

        Args:
            frame: Any split.

        Returns:
            Mapping label -> count, largest first.
        """
        return {
            str(label): int(count)
            for label, count in frame[self._label_column].value_counts().items()
        }
