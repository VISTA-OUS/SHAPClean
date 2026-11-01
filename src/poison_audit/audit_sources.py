"""Component 1 - the clustering runs ("sources") the label audit reads, and the per-source settings it needs.

A source is one stage-1 clustering of one poisoned training set: `dataset/scenario/run`, plus the paths to its
stage directory, poisoned_train, manifest and audited backdoor model. `settings` reads a source's id/label columns,
target label, model features and poisoned row ids; `align_row_clusters` maps a clustering's assignment onto the
training rows. Sources can also be found from path templates rather than listed one by one; a source whose files
are absent is reported, never guessed.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Features of layers A and B: every column except the id and the label, or exactly the features of the audited model.
FEATURE_SET_ALL_RAW = "all_raw"
FEATURE_SET_SELECTED = "selected"
PLACEHOLDER = re.compile(r"\{(dataset|scenario|run)\}")
SOURCE_FIELDS = ("dataset", "scenario", "run")
PATH_FIELDS = ("stage_dir", "poisoned_train", "manifest", "model_path")


class AuditSourceCatalog:
    """Discovers sources from path templates and resolves what each one needs.

    Args:
        groups: `knn_label_consistency.source_groups` - each a `stage_dir` template with optional
            {dataset}/{scenario}/{run} placeholders, fixed values for the fields it does not template, and the
            other paths (which may use the same placeholders).
    """

    def __init__(self, groups: List[Dict[str, Any]]) -> None:
        self._groups = groups

    def discover(self) -> List[Dict[str, Any]]:
        """Every source matching the templates.

        A group without placeholders is always listed, even when its directory does not exist yet, so the
        report says it is missing; a templated group only yields sources whose directories exist.

        Returns:
            One dict per source: `name`, `dataset`, `scenario`, `run` and the four paths, sorted by name.

        Raises:
            ValueError: When a group cannot determine dataset/scenario/run.
        """
        sources: Dict[str, Dict[str, Any]] = {}
        for group in self._groups:
            template = str(group["stage_dir"])
            fixed = {field: str(group[field]) for field in SOURCE_FIELDS if field in group}
            if PLACEHOLDER.search(template):
                regex = self._template_regex(template)
                matches = [Path(path).as_posix() for path in glob.glob(PLACEHOLDER.sub("*", template))]
                fills = [{**fixed, **match.groupdict()} for match in map(regex.match, sorted(matches)) if match]
            else:
                fills = [fixed]
            for values in fills:
                missing = [field for field in SOURCE_FIELDS if field not in values]
                if missing:
                    raise ValueError(f"Source group {template} cannot resolve {missing}.")
                source = {field: values[field] for field in SOURCE_FIELDS}
                source["name"] = "/".join(values[field] for field in SOURCE_FIELDS)
                for field in PATH_FIELDS:
                    source[field] = str(group[field]).format(**values)
                source["model_path"] = self._resolve_model_path(source["model_path"])
                sources[source["name"]] = source
        return sorted(sources.values(), key=lambda source: source["name"])

    @staticmethod
    def missing_files(source: Dict[str, Any]) -> List[str]:
        """Input files of a source that do not exist.

        Args:
            source: One source from `discover`.

        Returns:
            Missing paths (empty when complete).
        """
        stage_dir = Path(source["stage_dir"])
        required = [
            stage_dir / "cluster_assignment.npz",
            stage_dir / "cluster_summary.csv",
            stage_dir / "arm_metrics.json",
            Path(source["poisoned_train"]),
            Path(source["manifest"]),
            Path(source["model_path"]),
        ]
        return [path.as_posix() for path in required if not path.exists()]

    @staticmethod
    def settings(source: Dict[str, Any], config: Dict[str, Any], feature_set: str) -> Dict[str, Any]:
        """Id/label columns, target label, model features and poisoned row ids of a source.

        Read from the source's manifest when present (the three newer datasets differ from CIC-IDS2017 in
        all four), otherwise CIC-IDS2017's defaults from config.

        Args:
            source: One source.
            config: Project config.
            feature_set: `all_raw` or `selected`.

        Returns:
            Dict `id_column`, `label_column`, `target_label`, `model_features`, `feature_columns`, `poisoned_ids`.

        Raises:
            ValueError: If `feature_set` is unknown.
        """
        manifest = json.loads(Path(source["manifest"]).read_text(encoding="utf-8"))
        id_column = str(manifest.get("id_column") or config["data"]["sample_id_column"])
        label_column = str(manifest.get("label_column") or config["data"]["label_column"])
        model_features = list(
            manifest.get("model_features") or manifest.get("model_feature_columns") or config["features"]["selected_features"]
        )
        header = pd.read_csv(source["poisoned_train"], nrows=0).columns
        if feature_set == FEATURE_SET_SELECTED:
            feature_columns = model_features
        elif feature_set == FEATURE_SET_ALL_RAW:
            feature_columns = [column for column in header if column not in (id_column, label_column)]
        else:
            raise ValueError(f"Invalid feature_set: {feature_set}")
        return {
            "id_column": id_column,
            "label_column": label_column,
            "target_label": str(manifest.get("target_label") or config["component1_poison_audit"]["fixed_trigger_scenario"]["target_label"]),
            "model_features": model_features,
            "feature_columns": feature_columns,
            "poisoned_ids": np.asarray(manifest.get("poisoned_sample_ids", []), dtype=np.int64),
        }

    @staticmethod
    def align_row_clusters(assignment: Any, sample_ids: np.ndarray) -> np.ndarray:
        """Cluster of every row, in poisoned_train row order.

        Args:
            assignment: Contents of `cluster_assignment.npz` (`sample_id`, `cluster`).
            sample_ids: Id of every poisoned_train row, in file order.

        Returns:
            Cluster id per row of poisoned_train.

        Raises:
            ValueError: When the assignment does not cover exactly the rows of poisoned_train.
        """
        assigned_ids, clusters = assignment["sample_id"], assignment["cluster"]
        if np.array_equal(assigned_ids, sample_ids):
            return clusters
        positions = pd.Index(assigned_ids).get_indexer(sample_ids)
        if len(assigned_ids) != len(sample_ids) or (positions < 0).any():
            raise ValueError("cluster_assignment.npz does not cover exactly the rows of poisoned_train (stage 1 result needed).")
        return clusters[positions]

    @staticmethod
    def _template_regex(template: str) -> re.Pattern:
        """Regex matching a path template, capturing {dataset}/{scenario}/{run} as named groups.

        Args:
            template: Path template (`/` separators).

        Returns:
            Compiled regex.
        """
        pieces, position = [], 0
        for match in PLACEHOLDER.finditer(template):
            pieces.append(re.escape(template[position:match.start()]))
            pieces.append(f"(?P<{match.group(1)}>[^/]+)")
            position = match.end()
        pieces.append(re.escape(template[position:]))
        return re.compile("^" + "".join(pieces) + "$")

    @staticmethod
    def _resolve_model_path(pattern: str) -> str:
        """Model path; a pattern with `*` must match exactly one file, otherwise it is returned as is (and then
        reported missing).

        Args:
            pattern: Path or glob pattern.

        Returns:
            A concrete path when one can be determined.
        """
        if "*" not in pattern:
            return pattern
        matches = sorted(glob.glob(pattern))
        return Path(matches[0]).as_posix() if len(matches) == 1 else pattern
