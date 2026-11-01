"""Component 1 - layer A: labels clusters from how the observed labels of their nearest neighbouring
clusters vote.

The rule is Algorithm 2 ("kNN-based defence") of Paudice, Muñoz-González, Lupu (2018), *Label
Sanitization against Label Flipping Poisoning Attacks*, ECML PKDD Workshops, applied to clusters instead
of individual points and in one direction only: find each cluster's k nearest other clusters; a cluster
labelled benign becomes ATTACK_LIKE when at least a fraction `eta` of them carry an attack label, and a
cluster labelled attack always stays ATTACK_LIKE, as in LABOR (see `NeighborLabelConsistency.relabel`).

Why clusters and observed labels. The poison here is label-flipped - DDoS rows relabelled BENIGN - and
HDBSCAN gathers it into a few clusters whose rows all read BENIGN. What can still betray such a cluster
is that it BEHAVES like clusters labelled DDoS, and at ~0.05% poison the observed labels of every other
cluster are almost all correct. The exploratory probe of 2026-09-24 (fit-sample stage 1, 1,092 clusters)
found all 15 nearest neighbours of the one poison cluster labelled DDoS, while only 8 of 872 benign
clusters had a majority of attack-labelled neighbours.

What a cluster is to this rule:
- profile: the median of every feature over the cluster's rows, expressed as a mid-rank percentile of
  that feature's distribution over the WHOLE training set. Ranks make features on wildly different
  scales comparable (Flow Bytes/s spans ten orders of magnitude, flag counts are 0/1) without letting a
  heavy tail dominate the distance, and the mid-rank puts a mass point where it belongs: ~80% of flows
  have `Idle Min = 0`, so 0 ranks near 40, not at 0. +/-inf is kept as a value (it marks zero-duration
  flows); only NaN is ignored.
- observed label: attack when the cluster's observed attack rows are at least its observed benign rows -
  ties go to attack, the convention `ClusterLabelPoisonRemover.label_clusters` uses. These counts come
  from the POISONED label column, i.e. what a defender actually holds, never from ground truth.

Ground truth enters only `NeighborLabelScorer`, which scores a labelling after it has been made.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from src.core.logger import get_logger
from src.poison_audit.cluster_label_remover import ClusterLabelPoisonRemover

logger = get_logger(__name__)

NOISE_CLUSTER_ID = -1
KIND_POISON = "POISON"
KIND_POISON_MINORITY = "POISON_MINORITY"
KIND_BENIGN = "BENIGN"
KIND_ATTACK = "ATTACK"


@dataclass(frozen=True)
class TrainingMatrix:
    """A poisoned training set held as arrays, in file row order.

    Attributes:
        sample_ids: Row identifier per row.
        features: float32 matrix, one column per entry of `feature_columns`.
        feature_columns: Feature names in column order.
        label_codes: Observed (poisoned) label of each row, as an index into `label_names`.
        label_names: Distinct observed labels in the file.
    """

    sample_ids: np.ndarray
    features: np.ndarray
    feature_columns: List[str]
    label_codes: np.ndarray
    label_names: List[str]

    def rows_labelled(self, label: str) -> np.ndarray:
        """Rows whose observed label is `label`.

        Args:
            label: Label name.

        Returns:
            Boolean mask over rows (all False if the label never occurs).
        """
        if label not in self.label_names:
            return np.zeros(len(self.label_codes), dtype=bool)
        return self.label_codes == self.label_names.index(label)


@dataclass(frozen=True)
class ClusterProfiles:
    """One rank profile and one observed label per non-noise cluster, in ascending cluster id order.

    Attributes:
        cluster_ids: Cluster ids.
        ranks: Mid-rank percentile (0-1) of each cluster's per-feature median, `(n_clusters, n_features)`.
        n_records: Rows per cluster.
        observed_attack: True where the cluster's observed (poisoned) labels are attack-majority.
    """

    cluster_ids: np.ndarray
    ranks: np.ndarray
    n_records: np.ndarray
    observed_attack: np.ndarray


class TrainingMatrixReader:
    """Reads a poisoned training CSV once into a preallocated float32 matrix.

    The full CIC-IDS2017 table (2.26M x 78) costs ~0.7 GB as float32. Concatenating pandas chunks would
    briefly need twice that, and the default float64 four times - more than the local machine has free -
    so the matrix is allocated once and filled chunk by chunk.

    Args:
        chunk_size: Rows per read chunk.
    """

    def __init__(self, chunk_size: int) -> None:
        self._chunk_size = chunk_size

    def read(
        self, path: Path, id_column: str, label_column: str, feature_columns: List[str], n_rows: int,
    ) -> TrainingMatrix:
        """Read ids, features and observed labels.

        Args:
            path: poisoned_train (.csv or .csv.gz).
            id_column: Row identifier column.
            label_column: Observed (poisoned) label column - differs between datasets.
            feature_columns: Features to keep, in the order they should appear as matrix columns.
            n_rows: Row count the file must have (the length of the cluster assignment).

        Returns:
            The file as a `TrainingMatrix`.

        Raises:
            ValueError: If the file does not hold exactly `n_rows` rows.
        """
        logger.info("Reading %s (%d features, %d rows)...", path, len(feature_columns), n_rows)
        features = np.empty((n_rows, len(feature_columns)), dtype=np.float32)
        sample_ids = np.empty(n_rows, dtype=np.int64)
        label_codes = np.empty(n_rows, dtype=np.int16)
        label_index: Dict[str, int] = {}
        position = 0
        dtypes = {feature: np.float32 for feature in feature_columns}
        dtypes[label_column] = str
        columns = [id_column, label_column] + feature_columns
        for chunk in pd.read_csv(path, usecols=columns, dtype=dtypes, chunksize=self._chunk_size):
            stop = position + len(chunk)
            if stop > n_rows:
                raise ValueError(f"{path} has more than the {n_rows} rows cluster_assignment.npz refers to.")
            features[position:stop] = chunk[feature_columns].to_numpy(dtype=np.float32)
            sample_ids[position:stop] = chunk[id_column].to_numpy()
            codes, uniques = pd.factorize(chunk[label_column])
            if (codes < 0).any():
                raise ValueError(f"{path} has rows without a label in column {label_column}.")
            mapping = np.array([label_index.setdefault(str(label), len(label_index)) for label in uniques], dtype=np.int16)
            label_codes[position:stop] = mapping[codes]
            position = stop
        if position != n_rows:
            raise ValueError(f"{path} has only {position}/{n_rows} of the rows cluster_assignment.npz refers to.")
        return TrainingMatrix(
            sample_ids=sample_ids,
            features=features,
            feature_columns=list(feature_columns),
            label_codes=label_codes,
            label_names=list(label_index),
        )


class ClusterProfileBuilder:
    """Summarises every non-noise cluster as the population rank of its per-feature median."""

    def build(
        self, matrix: TrainingMatrix, row_clusters: np.ndarray, cluster_summary: pd.DataFrame,
    ) -> ClusterProfiles:
        """Build the profile table.

        Args:
            matrix: The training set; its rows are the population the ranks are taken in.
            row_clusters: Cluster id per row of `matrix` (`-1` = noise, excluded).
            cluster_summary: The stage's `cluster_summary.csv` (`cluster`, `n_records`, `n_benign`,
                `n_attack_total` - observed-label counts; `n_poisoned` is not read here).

        Returns:
            One profile per non-noise cluster.

        Raises:
            ValueError: If the summary's row counts disagree with the assignment.
        """
        clustered = row_clusters != NOISE_CLUSTER_ID
        cluster_ids, codes = np.unique(row_clusters[clustered], return_inverse=True)
        summary = cluster_summary.set_index("cluster").reindex(cluster_ids)
        n_records = np.bincount(codes, minlength=len(cluster_ids))
        if summary["n_records"].isna().any() or not np.array_equal(summary["n_records"].to_numpy(), n_records):
            raise ValueError("cluster_summary.csv and cluster_assignment.npz do not come from the same clustering.")
        observed_attack = (summary["n_attack_total"] >= summary["n_benign"]).to_numpy()

        ranks = np.empty((len(cluster_ids), len(matrix.feature_columns)), dtype=np.float64)
        for position, feature in enumerate(matrix.feature_columns):
            column = matrix.features[:, position]
            medians = pd.Series(column[clustered]).groupby(codes).median().to_numpy(dtype=np.float64)
            population = np.sort(column[~np.isnan(column)]).astype(np.float64)
            ranks[:, position] = self._mid_rank(population, medians)
            if position % 20 == 0:
                logger.info("Cluster profiles: %d/%d features (%s)...", position + 1, len(matrix.feature_columns), feature)
        logger.info(
            "Built the profiles of %d clusters x %d features; %d clusters have an attack observed label.",
            len(cluster_ids), ranks.shape[1], int(observed_attack.sum()),
        )
        return ClusterProfiles(
            cluster_ids=cluster_ids, ranks=ranks, n_records=n_records, observed_attack=observed_attack,
        )

    @staticmethod
    def _mid_rank(population_sorted: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Mid-rank percentile of each value in a sorted population.

        A cluster whose rows have no value at all for a feature (NaN median) is placed at the middle,
        so that feature neither pulls it towards nor away from any other cluster.

        Args:
            population_sorted: Sorted population values (NaN removed).
            values: Values to place.

        Returns:
            Percentiles in [0, 1].
        """
        below = np.searchsorted(population_sorted, values, side="left")
        at_or_below = np.searchsorted(population_sorted, values, side="right")
        ranks = (below + at_or_below) / (2.0 * len(population_sorted))
        ranks[np.isnan(values)] = 0.5
        return ranks


class NeighborLabelConsistency:
    """Paudice et al. (2018) Algorithm 2 at cluster level, one direction only: a benign-labelled cluster
    becomes ATTACK_LIKE when enough of its neighbours carry an attack label; an attack-labelled cluster
    stays ATTACK_LIKE.

    Args:
        max_k: Largest neighbourhood that will be asked for (the top of the k grid).
        block_size: Clusters per distance block; memory is `block_size x n_clusters` float64 values.
            Results do not depend on it.
    """

    def __init__(self, max_k: int, block_size: int) -> None:
        self._max_k = max_k
        self._block_size = block_size
        self._neighbors: Optional[np.ndarray] = None

    @property
    def neighbors(self) -> np.ndarray:
        """Nearest-neighbour cluster positions from `fit`.

        Returns:
            `(n_clusters, max_k)` positions into the profile table, nearest first.

        Raises:
            RuntimeError: If `fit` has not run.
        """
        if self._neighbors is None:
            raise RuntimeError("fit() has not been called.")
        return self._neighbors

    def fit(self, ranks: np.ndarray) -> np.ndarray:
        """Find every cluster's `max_k` nearest OTHER clusters by L1 distance between rank profiles.

        Distance ties are broken by cluster order (stable sort), so a rerun gives identical
        neighbourhoods - HDBSCAN leaves many small clusters with identical median profiles.

        Args:
            ranks: Profile table, one row per cluster.

        Returns:
            The neighbour matrix (also kept for `attack_votes`).
        """
        n_clusters = len(ranks)
        width = min(self._max_k, n_clusters - 1)
        neighbors = np.empty((n_clusters, width), dtype=np.int64)
        for start in range(0, n_clusters, self._block_size):
            stop = min(start + self._block_size, n_clusters)
            distances = cdist(ranks[start:stop], ranks, metric="cityblock")
            distances[np.arange(stop - start), np.arange(start, stop)] = np.inf
            neighbors[start:stop] = np.argsort(distances, axis=1, kind="stable")[:, :width]
        self._neighbors = neighbors
        logger.info("Found the %d nearest neighbours of %d clusters.", width, n_clusters)
        return neighbors

    def attack_votes(self, observed_attack: np.ndarray, k: int) -> np.ndarray:
        """How many of each cluster's k nearest neighbours carry an attack observed label.

        Args:
            observed_attack: Observed label per cluster, aligned with the profile table.
            k: Neighbourhood size.

        Returns:
            Integer vote count per cluster.

        Raises:
            ValueError: If `k` exceeds the neighbourhood `fit` computed.
        """
        if k > self.neighbors.shape[1]:
            raise ValueError(f"k={k} is larger than the number of neighbours computed ({self.neighbors.shape[1]}).")
        return observed_attack[self.neighbors[:, :k]].sum(axis=1)

    @staticmethod
    def votes_needed(k: int, eta: float) -> int:
        """Attack votes a benign-labelled cluster needs among its k neighbours to become ATTACK_LIKE.

        Args:
            k: Neighbourhood size.
            eta: Agreement threshold in [0.5, 1].

        Returns:
            `ceil(eta * k)`.

        Raises:
            ValueError: If `eta` is outside [0.5, 1].
        """
        if not 0.5 <= eta <= 1.0:
            raise ValueError(f"eta must lie in [0.5, 1], got {eta}.")
        # Subtract a small amount before rounding up: 0.6 * 5 in floating point is 3.0000000000000004, and without the
        # subtraction the threshold would be 4 votes instead of 3.
        return math.ceil(eta * k - 1e-9)

    @staticmethod
    def relabel(observed_attack: np.ndarray, attack_votes: np.ndarray, k: int, eta: float) -> np.ndarray:
        """Paudice's decision rule, applied in ONE direction only: benign-labelled to ATTACK_LIKE.

        An observed-benign cluster becomes ATTACK_LIKE when at least `ceil(eta * k)` of its k neighbours
        carry an attack label (`eta = 0.5` on an even split counts as attack, the tie convention of
        `ClusterLabelPoisonRemover.label_clusters`). An observed-attack cluster always stays ATTACK_LIKE,
        whatever its neighbours say.

        Why one direction. The poison flips attack rows TO the
        target label, so only benign-labelled clusters can be the attacker's work; an attack label is
        never forged and there is no ground for doubting it. LABOR labels attack clusters ATTACK_LIKE, and
        so must this rule. Paudice's full rule also relabels attack clusters whose neighbours look benign
        (7-41 per CIC-IDS2017 full-fit scenario, measured 2026-09-25). That removed fewer clean rows, but
        only because it stopped the removal rule from acting inside those clusters - poison mixed into such
        an attack cluster would have been missed.

        Args:
            observed_attack: Observed label per cluster.
            attack_votes: Output of `attack_votes` for the same k.
            k: Neighbourhood size.
            eta: Agreement threshold in [0.5, 1].

        Returns:
            Boolean array, True = ATTACK_LIKE.

        Raises:
            ValueError: If `eta` is outside [0.5, 1].
        """
        return observed_attack | (attack_votes >= NeighborLabelConsistency.votes_needed(k, eta))


class NeighborLabelScorer:
    """Scores a cluster labelling against ground truth under the pipeline's own removal rule.

    Removal counts come from per-cluster aggregates - rows predicted `target_label`, and how many of
    those are poisoned - instead of re-mapping every row per grid point. `oracle_reference` checks
    that shortcut against `ClusterLabelPoisonRemover` itself before any number is trusted.

    Args:
        target_label: Label the backdoor flips predictions to.
    """

    def __init__(self, target_label: str) -> None:
        self._target_label = target_label
        self._remover = ClusterLabelPoisonRemover(target_label=target_label)
        self._fitted: Dict[str, Any] = {}

    def fit(
        self,
        profiles: ClusterProfiles,
        cluster_summary: pd.DataFrame,
        row_clusters: np.ndarray,
        predicted_labels: np.ndarray,
        is_poisoned: np.ndarray,
    ) -> None:
        """Precompute the per-cluster ground truth and removal aggregates.

        Args:
            profiles: Profile table (defines cluster order).
            cluster_summary: The stage's `cluster_summary.csv`.
            row_clusters: Cluster id per row.
            predicted_labels: Audited model's string prediction per row.
            is_poisoned: Ground-truth poisoned flag per row.
        """
        summary = cluster_summary.set_index("cluster").reindex(profiles.cluster_ids)
        n_poisoned = summary["n_poisoned"].to_numpy()
        true_benign = summary["n_benign"].to_numpy() - n_poisoned
        true_attack = summary["n_attack_total"].to_numpy() + n_poisoned
        oracle_attack = ~(true_benign > true_attack)
        kinds = np.where(
            n_poisoned > 0,
            np.where(oracle_attack, KIND_POISON, KIND_POISON_MINORITY),
            np.where(oracle_attack, KIND_ATTACK, KIND_BENIGN),
        )
        clustered = row_clusters != NOISE_CLUSTER_ID
        positions = np.searchsorted(profiles.cluster_ids, row_clusters[clustered])
        predicted_target = predicted_labels[clustered] == self._target_label
        n_predicted_target = np.bincount(
            positions, weights=predicted_target, minlength=len(profiles.cluster_ids),
        ).astype(np.int64)
        n_predicted_target_poisoned = np.bincount(
            positions, weights=predicted_target & is_poisoned[clustered], minlength=len(profiles.cluster_ids),
        ).astype(np.int64)
        self._fitted = {
            "profiles": profiles,
            "cluster_summary": cluster_summary,
            "row_clusters": row_clusters,
            "predicted_labels": predicted_labels,
            "is_poisoned": is_poisoned,
            "kinds": kinds,
            "n_poisoned": n_poisoned,
            "oracle_attack": oracle_attack,
            "n_predicted_target": n_predicted_target,
            "n_predicted_target_poisoned": n_predicted_target_poisoned,
            # Clean rows the removal rule would remove if the cluster were ATTACK_LIKE: the model predicts target_label but they are not poisoned.
            "n_predicted_target_clean": n_predicted_target - n_predicted_target_poisoned,
            "n_total_poisoned": int(is_poisoned.sum()),
        }

    @property
    def kinds(self) -> np.ndarray:
        """Ground-truth kind per cluster (`POISON`, `POISON_MINORITY`, `BENIGN`, `ATTACK`).

        Returns:
            Kind per cluster, aligned with the profile table.
        """
        return self._fitted["kinds"]

    @property
    def n_poisoned(self) -> np.ndarray:
        """Ground-truth poisoned rows per cluster.

        Returns:
            Count per cluster, aligned with the profile table.
        """
        return self._fitted["n_poisoned"]

    @property
    def oracle_attack(self) -> np.ndarray:
        """The oracle's label per cluster (true composition, poison counted as attack).

        Returns:
            True = ATTACK_LIKE, aligned with the profile table.
        """
        return self._fitted["oracle_attack"]

    @property
    def n_predicted_target(self) -> np.ndarray:
        """Rows the removal rule would remove from each cluster if it were ATTACK_LIKE (predicted `target_label`).

        Returns:
            Count per cluster, aligned with the profile table.
        """
        return self._fitted["n_predicted_target"]

    @property
    def n_predicted_target_poisoned(self) -> np.ndarray:
        """Of `n_predicted_target`, the poisoned rows.

        Returns:
            Count per cluster, aligned with the profile table.
        """
        return self._fitted["n_predicted_target_poisoned"]

    @property
    def hidden_poison(self) -> np.ndarray:
        """Poison clusters whose observed label is benign - the ones only layer A can bring to the short list.

        Returns:
            Boolean mask, aligned with the profile table.
        """
        return (self._fitted["kinds"] == KIND_POISON) & ~self._fitted["profiles"].observed_attack

    def poison_vote_margin(self, attack_votes: np.ndarray, k: int, eta: float) -> float:
        """Spare attack votes of the weakest hidden poison cluster that layer A still catches at (k, eta).

        A margin of 1 means that cluster would still be caught with one more benign-labelled neighbour, e.g.
        one more poison fragment next to it in another run.

        Args:
            attack_votes: `NeighborLabelConsistency.attack_votes` for the same k.
            k: Neighbourhood size.
            eta: Agreement threshold.

        Returns:
            Smallest `votes - votes_needed(k, eta)` over the caught hidden poison clusters; NaN when none is
            caught.
        """
        spare = attack_votes[self.hidden_poison] - NeighborLabelConsistency.votes_needed(k, eta)
        spare = spare[spare >= 0]
        return float(spare.min()) if len(spare) else float("nan")

    def max_poison_siblings(self, neighbors: np.ndarray, k: int) -> float:
        """Most other hidden poison clusters found among one hidden poison cluster's k nearest neighbours.

        Poison that HDBSCAN splits into fragments lying next to each other votes benign for itself: a
        fragment stays catchable only while its benign-labelled neighbours number at most
        `k - votes_needed(k, eta)` (measured 2026-09-25: at most 2 siblings in any source except
        labor_natural, whose 16 fragments had 5-8 among their 15 nearest).

        Args:
            neighbors: `NeighborLabelConsistency.neighbors`.
            k: Neighbourhood size.

        Returns:
            The count; NaN when the source has no hidden poison cluster.
        """
        hidden = self.hidden_poison
        if not hidden.any():
            return float("nan")
        return float(hidden[neighbors[hidden, :k]].sum(axis=1).max())

    def score(self, attack_like: np.ndarray) -> Dict[str, Any]:
        """Cluster-level and removal-level outcome of one labelling.

        Args:
            attack_like: True = ATTACK_LIKE, per cluster, aligned with the profile table.

        Returns:
            Counts at cluster level (poison clusters caught, benign clusters wrongly flagged and their
            rows, attack clusters relabelled benign, short list = observed-benign clusters relabelled
            attack) and at row level under the removal rule (removed, correct, wrong, missed,
            precision, recall, F1), plus the wrong removals split by the kind of cluster they sit in
            (`n_wrong_in_<kind>`), which is what separates two labellings with the same recall.
        """
        fitted = self._fitted
        profiles: ClusterProfiles = fitted["profiles"]
        kinds = fitted["kinds"]
        n_removed = int(fitted["n_predicted_target"][attack_like].sum())
        n_correct = int(fitted["n_predicted_target_poisoned"][attack_like].sum())
        n_total = fitted["n_total_poisoned"]
        precision = n_correct / n_removed if n_removed else 0.0
        recall = n_correct / n_total if n_total else 0.0
        short_list = attack_like & ~profiles.observed_attack
        benign_flagged = attack_like & (kinds == KIND_BENIGN)
        return {
            "n_clusters": int(len(kinds)),
            "n_poison_clusters": int((kinds == KIND_POISON).sum()),
            "n_poison_clusters_caught": int((attack_like & (kinds == KIND_POISON)).sum()),
            "n_benign_clusters": int((kinds == KIND_BENIGN).sum()),
            "n_benign_clusters_flagged": int(benign_flagged.sum()),
            "n_benign_rows_flagged": int(profiles.n_records[benign_flagged].sum()),
            "n_attack_clusters": int((kinds == KIND_ATTACK).sum()),
            "n_attack_clusters_relabelled_benign": int((~attack_like & (kinds == KIND_ATTACK)).sum()),
            "n_short_list": int(short_list.sum()),
            "n_short_list_poison": int((short_list & (kinds == KIND_POISON)).sum()),
            "n_removed": n_removed,
            "n_removed_correct": n_correct,
            "n_removed_wrong": n_removed - n_correct,
            "n_missed": n_total - n_correct,
            "n_total_poisoned": n_total,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            **{
                f"n_wrong_in_{kind.lower()}": int(fitted["n_predicted_target_clean"][attack_like & (kinds == kind)].sum())
                for kind in (KIND_ATTACK, KIND_POISON, KIND_BENIGN, KIND_POISON_MINORITY)
            },
        }

    def oracle_reference(self) -> Dict[str, Any]:
        """The ground-truth oracle's outcome, computed twice: row by row with
        `ClusterLabelPoisonRemover` and through `score`'s aggregates.

        Returns:
            `score`'s dict for the oracle labelling.

        Raises:
            ValueError: If the two computations disagree (the aggregate shortcut would be wrong).
        """
        fitted = self._fitted
        labels = self._remover.label_clusters(fitted["cluster_summary"])
        mask = self._remover.compute_removal_mask(fitted["row_clusters"], labels, fitted["predicted_labels"])
        row_level = self._remover.summarize_removal(mask, fitted["is_poisoned"])
        aggregate = self.score(fitted["oracle_attack"])
        for key in ("n_removed", "n_removed_correct", "n_removed_wrong", "n_missed", "n_total_poisoned"):
            if row_level[key] != aggregate[key]:
                raise ValueError(
                    f"The cluster-level count disagrees with ClusterLabelPoisonRemover on {key}: "
                    f"{aggregate[key]} vs {row_level[key]}.",
                )
        return aggregate
