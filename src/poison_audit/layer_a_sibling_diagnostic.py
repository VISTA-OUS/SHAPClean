"""Why layer A misses poisoned clusters (script 13).

Layer A short-lists a benign-labelled cluster when at least `ceil(eta * k)` of its k nearest clusters (raw-feature
profiles) are attack-labelled. A poisoned group that HDBSCAN splits into many fragments lying next to each other can
fail this test: each fragment counts the other fragments - labelled benign, like itself - among its neighbours, and
the attack votes it needs run out. This diagnostic measures that on a rebuilt sweep cell: for every hidden poisoned
cluster (poisoned, true-attack majority, benign-labelled) it counts its attack votes, the other poisoned clusters
("siblings") and the genuine benign clusters among its k nearest, and whether counting the siblings as attack would
have caught it. Ground truth is read for scoring only, as everywhere else in the audit.

The rebuilt cell must carry its stage-1 `cluster_assignment.npz` (script 04 keeps it) and `poisoned_train.csv.gz`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.core.logger import get_logger
from src.poison_audit.audit_sources import AuditSourceCatalog
from src.poison_audit.cluster_neighbor_labeler import (
    KIND_POISON, ClusterProfileBuilder, NeighborLabelConsistency, NeighborLabelScorer, TrainingMatrixReader,
)
from src.poison_audit.end_to_end_pipeline import RESULT_FILE, STAGE_DIR
from src.poison_audit.poison_rate_sweep import MANIFEST_FILE, POISONED_TRAIN_FILE

logger = get_logger(__name__)

CLUSTERS_FILE = "sibling_clusters.csv"
SUMMARY_FILE = "sibling_result.json"
ORIGINAL_RESULT_FILE = "original_result.json"
ORIGINAL_MANIFEST_FILE = "original_manifest.json"
# Layer A of the rebuilt cell must match the original cell on these numbers to count as reproduced.
REPRODUCED_KEYS = ("n_hidden_poison_clusters", "n_short_list_poison", "n_poisoned_in_trusted")


class LayerASiblingDiagnostic:
    """Counts poisoned "siblings" among the nearest clusters of every hidden poisoned cluster of one cell.

    Args:
        config: Project config (id/label/target defaults of `AuditSourceCatalog.settings`).
        feature_set: Feature set of layer A's profiles (`knn_label_consistency.feature_set`).
        read_chunk_size: Rows per chunk when reading the training matrix (memory only).
        distance_block_size: Clusters per distance block (memory only).
    """

    def __init__(self, config: Dict[str, Any], feature_set: str, read_chunk_size: int, distance_block_size: int) -> None:
        self._config = config
        self._feature_set = feature_set
        self._read_chunk_size = read_chunk_size
        self._distance_block_size = distance_block_size

    def analyse(self, cell: Path, k: int, eta: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Per-cluster counts and their summary for one cell; both are also written into the cell.

        Args:
            cell: Rebuilt audit cell (manifest, poisoned set, stage-1 assignment and summary).
            k: Layer A's neighbourhood size.
            eta: Layer A's agreement threshold.

        Returns:
            (one row per hidden poisoned cluster, summary dict).
        """
        poisoned_path = cell / POISONED_TRAIN_FILE
        settings = AuditSourceCatalog.settings({"manifest": str(cell / MANIFEST_FILE), "poisoned_train": str(poisoned_path)},
                                               self._config, self._feature_set)
        assignment = np.load(cell / STAGE_DIR / "cluster_assignment.npz")
        matrix = TrainingMatrixReader(self._read_chunk_size).read(
            poisoned_path, settings["id_column"], settings["label_column"], settings["feature_columns"],
            len(assignment["sample_id"]),
        )
        row_clusters = AuditSourceCatalog.align_row_clusters(assignment, matrix.sample_ids)
        summary = pd.read_csv(cell / STAGE_DIR / "cluster_summary.csv")
        profiles = ClusterProfileBuilder().build(matrix, row_clusters, summary)
        labeler = NeighborLabelConsistency(max_k=k, block_size=self._distance_block_size)
        labeler.fit(profiles.ranks)
        poisoned = np.isin(matrix.sample_ids, settings["poisoned_ids"])
        scorer = NeighborLabelScorer(target_label=settings["target_label"])
        # Only the cluster kinds and the hidden poisoned clusters are needed: neither depends on the labels the model predicts.
        scorer.fit(profiles, summary, row_clusters, np.full(len(row_clusters), settings["target_label"], dtype=object), poisoned)

        need = NeighborLabelConsistency.votes_needed(k, eta)
        votes = labeler.attack_votes(profiles.observed_attack, k)
        is_poison = scorer.kinds == KIND_POISON
        neighbours = labeler.neighbors[:, :k]
        rows = []
        for index in np.flatnonzero(scorer.hidden_poison):
            around = neighbours[index]
            siblings = int(is_poison[around].sum())
            rows.append({
                "cluster": int(profiles.cluster_ids[index]), "n_poisoned": int(scorer.n_poisoned[index]),
                "attack_votes": int(votes[index]), "caught": bool(votes[index] >= need), "poison_siblings": siblings,
                "genuine_benign_neighbours": int((~profiles.observed_attack[around] & ~is_poison[around]).sum()),
                "caught_if_siblings_were_attack": bool(votes[index] + siblings >= need),
            })
        clusters = pd.DataFrame(rows)
        result = self.summarize(clusters, need, k)
        result["reproduction"] = self.reproduction(cell)
        clusters.to_csv(cell / CLUSTERS_FILE, index=False)
        (cell / SUMMARY_FILE).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return clusters, result

    @staticmethod
    def summarize(clusters: pd.DataFrame, votes_needed: int, k: int) -> Dict[str, Any]:
        """Counts of the caught and missed hidden poisoned clusters.

        Args:
            clusters: Output rows of `analyse`.
            votes_needed: Attack votes layer A needs.
            k: Neighbourhood size.

        Returns:
            Summary dict.
        """
        result: Dict[str, Any] = {"k": k, "votes_needed": votes_needed, "n_hidden_poison_clusters": int(len(clusters))}
        for name, part in (("caught", clusters[clusters["caught"]] if len(clusters) else clusters),
                           ("missed", clusters[~clusters["caught"]] if len(clusters) else clusters)):
            result[f"n_{name}"] = int(len(part))
            result[f"{name}_poisoned_rows"] = int(part["n_poisoned"].sum()) if len(part) else 0
            for column in ("attack_votes", "poison_siblings", "genuine_benign_neighbours"):
                result[f"{name}_{column}_min"] = int(part[column].min()) if len(part) else None
                result[f"{name}_{column}_max"] = int(part[column].max()) if len(part) else None
        missed = clusters[~clusters["caught"]] if len(clusters) else clusters
        result["missed_caught_if_siblings_were_attack"] = int(missed["caught_if_siblings_were_attack"].sum()) if len(missed) else 0
        return result

    @staticmethod
    def reproduction(cell: Path) -> Dict[str, Any]:
        """Layer A and poisoned ids of the rebuilt cell next to the original cell's, when those were copied in
        (`original_result.json`, `original_manifest.json`).

        Args:
            cell: Rebuilt audit cell.

        Returns:
            Dict with both layer-A summaries of the locked setting (None when a file is missing) and whether the
            poisoned ids are the same.
        """
        def layers(path: Path) -> Dict[str, Any]:
            return json.loads(path.read_text(encoding="utf-8"))["layers"] if path.exists() else {}

        def ids(path: Path) -> Any:
            return sorted(json.loads(path.read_text(encoding="utf-8"))["poisoned_sample_ids"]) if path.exists() else None
        rebuilt, original = layers(cell / RESULT_FILE), layers(cell / ORIGINAL_RESULT_FILE)
        # The rebuilt cell runs only the selected configuration; the original ran every configuration, so take that name.
        name = next(iter(rebuilt), None)
        rebuilt_ids, original_ids = ids(cell / MANIFEST_FILE), ids(cell / ORIGINAL_MANIFEST_FILE)
        return {"layer": name, "rebuilt": rebuilt[name]["layer_a"] if name else None,
                "original": original[name]["layer_a"] if name in original else None,
                "same_poisoned_ids": rebuilt_ids is not None and rebuilt_ids == original_ids}


class LayerASiblingReport:
    """One table row per diagnosed cell.

    Args:
        dataset_titles: Dataset name -> display name.
    """

    def __init__(self, dataset_titles: Dict[str, str]) -> None:
        self._titles = dataset_titles

    @staticmethod
    def rows(results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Flatten the cells' summaries.

        Args:
            results: Dicts with `seed`, `dataset`, `trigger`, `rate`, `removes_backdoor` and `summary`.

        Returns:
            One row per cell.
        """
        rows = []
        for item in results:
            summary = item["summary"]
            original, rebuilt = summary["reproduction"]["original"], summary["reproduction"]["rebuilt"]
            row = {key: item[key] for key in ("seed", "dataset", "trigger", "rate", "removes_backdoor")}
            row.update({key: value for key, value in summary.items() if key != "reproduction"})
            row["reproduced"] = bool(summary["reproduction"]["same_poisoned_ids"] and original is not None
                                     and rebuilt is not None
                                     and all(original[key] == rebuilt[key] for key in REPRODUCED_KEYS))
            for key in REPRODUCED_KEYS:
                row[f"original_{key}"] = original[key] if original is not None else None
                row[f"rebuilt_{key}"] = rebuilt[key] if rebuilt is not None else None
            rows.append(row)
        return pd.DataFrame(rows)

    def markdown(self, frame: pd.DataFrame) -> str:
        """Markdown report: a row per cell.

        Args:
            frame: Output of `rows`.

        Returns:
            Markdown.
        """
        lines = ["# Layer A misses poisoned clusters: neighbouring poisoned clusters \"vote for each other\"", "",
                 "- Each cell is rebuilt with exactly the code and config of script 04 (selected configuration); \"reproduced\""
                 " = the same poisoned set, and the same number of hidden poisoned clusters, poisoned clusters in the short"
                 " list and poisoned rows in the trusted set as the original.",
                 "- Hidden poisoned cluster: holds poison, mostly attack by true label, benign by observed label. Neighbours:"
                 " the k nearest clusters by the layer-A profile. Siblings = the other poisoned clusters among them.", "",
                 "| Seed | Dataset | Trigger | Removed | Hidden clusters | Caught / missed | Missed poisoned rows | Missed: siblings |"
                 " Missed: votes | Caught: siblings | Missed caught if siblings were attack | Poison in trusted set (original / rebuilt) | Reproduced |",
                 "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, row in frame.iterrows():
            missed_siblings = (f"{row['missed_poison_siblings_min']}-{row['missed_poison_siblings_max']}"
                               if row["n_missed"] else "--")
            missed_votes = f"{row['missed_attack_votes_min']}-{row['missed_attack_votes_max']}" if row["n_missed"] else "--"
            caught_siblings = (f"{row['caught_poison_siblings_min']}-{row['caught_poison_siblings_max']}"
                               if row["n_caught"] else "--")
            lines.append(
                f"| {row['seed']} | {self._titles.get(row['dataset'], row['dataset'])} | {row['trigger']} |"
                f" {'yes' if row['removes_backdoor'] else 'no'} | {row['n_hidden_poison_clusters']} |"
                f" {row['n_caught']} / {row['n_missed']} | {row['missed_poisoned_rows']} | {missed_siblings} |"
                f" {missed_votes} (needs {row['votes_needed']}) | {caught_siblings} |"
                f" {row['missed_caught_if_siblings_were_attack']}/{row['n_missed']} |"
                f" {row['original_n_poisoned_in_trusted']} / {row['rebuilt_n_poisoned_in_trusted']} |"
                f" {'yes' if row['reproduced'] else 'NO'} |")
        return "\n".join(lines) + "\n"
