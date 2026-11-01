"""Component 1 - the complete poison-removal system (layer B, the removal rules and the noise rule):

    HDBSCAN clusters -> layer A short list (kNN on cluster labels, `cluster_neighbor_labeler.py`)
      -> trusted rows: every attack-labelled row + benign-labelled rows of clustered, non-short-listed clusters (NO noise)
      -> clean XGBoost on the trusted rows, and contrastive exemplars whose benign references are also non-noise
      -> layer B: every short-listed cluster labelled by record votes (majority)
      -> cluster labels: attack-labelled clusters ATTACK_LIKE; layer-B positives ATTACK_LIKE; the rest BENIGN_LIKE
      -> LABOR removal with the AUDITED model: rows of ATTACK_LIKE clusters it predicts as the target label
      -> noise rule: benign-labelled noise rows the clean model calls attack while the audited model says target.

Why the trusted rows exclude noise: HDBSCAN leaves some poisoned rows in noise, and a benign
"reference" that is itself poison teaches the clean model "trigger -> benign" and hides poison from the exemplar check.

Two layer-B record rules are kept side by side:
- `sequential`: the clean model calls the record an attack, and the exemplars do not show a strictly closer benign flow
  (a tie does not veto) - recall first;
- `and`: the clean model calls it an attack AND the nearest attack flow is strictly closer than the nearest benign flow -
  precision first.

Three removal granularities for the short-listed clusters are compared (the audited model must predict the target
label in all three, as LABOR requires):
- `cluster`: every row of a layer-B ATTACK_LIKE cluster (the cluster call rests on at most `max_compared_records`);
- `record`: every short-listed row whose own vote is attack, whatever its cluster's call;
- `hybrid`: rows of layer-B ATTACK_LIKE clusters whose own vote is attack - the rows both others remove.

Benign reference policy (`BenignReference`; the default is the design above). The benign rows layer B trusts -
what the clean model learns as benign and where the exemplars look for the nearest benign flow - can be narrowed or
widened, as measured on the poison-rate sweep (the selected configuration uses the `kept` policy, both options below):
- `attack_labelled_clusters=False` drops the benign-labelled rows of attack-labelled clusters. LABOR calls those clusters
  ATTACK_LIKE and removes such rows, so they cannot be references; on Edge-IIoTset distribution_matching 10% the only
  poisoned row outside the short list sat there and was the nearest "benign" flow of every poisoned row;
- `noise="clean_benign"` admits the benign-labelled noise rows that the clean model (learnt without noise) calls
  benign (P(attack) below `classifier_threshold`). The noise rows it calls attack - what the noise rule removes - stay
  out, and with them almost every poisoned noise row; on TON-IoT the remaining noise holds the only close benign flows
  of a whole family of genuine benign clusters that a wide short list takes out of the reference.

Ground truth enters only `RemovalScorer`, after a removal has been decided.
"""

from __future__ import annotations

import gc
import zlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.core.logger import get_logger
from src.poison_audit.attack_exemplar import RankTransformer
from src.poison_audit.attack_knowledge import TrustedTrainingSet, XgbAttackClassifier
from src.poison_audit.cluster_label_remover import ATTACK_LIKE, BENIGN_LIKE
from src.poison_audit.cluster_neighbor_labeler import TrainingMatrix
from src.poison_audit.contrastive_exemplar import ContrastiveExemplarFinder

logger = get_logger(__name__)

# The vote of one record in layer B, and the cluster label when no record has a vote.
RECORD_ATTACK = "ATTACK"
RECORD_BENIGN = "BENIGN"
UNDECIDED = "UNDECIDED"

RULE_SEQUENTIAL = "sequential"
RULE_AND = "and"
LAYER_B_RULES = (RULE_SEQUENTIAL, RULE_AND)
# The "layer A only" ablation: every cluster of the short list is ATTACK_LIKE without asking layer B; the LABOR removal
# rule and the noise rule stay, so the difference from the `and` rule is exactly what layer B adds. Cluster level only.
RULE_LAYER_A_ONLY = "layer_a_only"
GRANULARITY_CLUSTER = "cluster"
GRANULARITY_RECORD = "record"
GRANULARITY_HYBRID = "hybrid"
GRANULARITIES = (GRANULARITY_CLUSTER, GRANULARITY_RECORD, GRANULARITY_HYBRID)
# Which short-listed rows get a record vote: only the sample that labels each cluster, also every row of the clusters
# layer B calls ATTACK_LIKE (what `hybrid` needs), or every short-listed row (what `record` needs).
ROWS_SAMPLE = "sample"
ROWS_ATTACK_LIKE = "attack_like"
ROWS_ALL = "all"
NOISE_CLUSTER_ID = -1
REFERENCE_NOISE_NONE = "none"
REFERENCE_NOISE_CLEAN_BENIGN = "clean_benign"
REFERENCE_NOISE_OPTIONS = (REFERENCE_NOISE_NONE, REFERENCE_NOISE_CLEAN_BENIGN)


@dataclass(frozen=True)
class BenignReference:
    """Which benign-labelled rows layer B trusts (module docstring).

    Attributes:
        name: Short name, suffix of the layer setting that uses it.
        attack_labelled_clusters: Keep the benign-labelled rows of attack-labelled clusters.
        noise: `none` - no noise row; `clean_benign` - the noise rows the clean model calls benign (exemplars only; the
            clean model itself never learns on noise).
    """

    name: str
    attack_labelled_clusters: bool
    noise: str

    def __post_init__(self) -> None:
        """Reject an unknown noise option.

        Raises:
            ValueError: If `noise` is not one of `REFERENCE_NOISE_OPTIONS`.
        """
        if self.noise not in REFERENCE_NOISE_OPTIONS:
            raise ValueError(f"Invalid noise policy of the benign reference: {self.noise!r} (one of {REFERENCE_NOISE_OPTIONS}).")


# The original design: the benign rows of every cluster outside the short list (attack-labelled ones included), no noise.
DEFAULT_BENIGN_REFERENCE = BenignReference(name="", attack_labelled_clusters=True, noise=REFERENCE_NOISE_NONE)


def draw_positions(key: str, purpose: str, n_rows: int, size: int, random_state: int) -> np.ndarray:
    """A reproducible draw of cluster-member positions without replacement - the records layer B scores in a cluster.

    Args:
        key: Cluster key `source|cluster`.
        purpose: Purpose of the draw - each gets its own stream.
        n_rows: Cluster size.
        size: Positions wanted (capped at the cluster size).
        random_state: Base seed.

    Returns:
        Positions into the cluster's members (rows of the cluster in training-matrix order).
    """
    # crc32, not hash(): Python's hash changes between processes, so a rerun would draw a different sample.
    seed = (random_state + zlib.crc32(f"{key}|{purpose}".encode("utf-8"))) % (2**32)
    return np.random.RandomState(seed).choice(n_rows, size=min(size, n_rows), replace=False)


class RecordVoteLabeler:
    """Combines per-record votes into a cluster label by majority.

    Args:
        tie_label: Cluster label when attack and benign votes are equal.
    """

    def __init__(self, tie_label: str) -> None:
        if tie_label not in (ATTACK_LIKE, BENIGN_LIKE):
            raise ValueError(f"tie_label must be {ATTACK_LIKE} or {BENIGN_LIKE}, got '{tie_label}'.")
        self._tie_label = tie_label

    def vote(self, record_labels: Iterable[Optional[str]]) -> Tuple[str, int, int, int]:
        """Majority vote over per-record labels; invalid labels abstain.

        Args:
            record_labels: Per-record labels (None = invalid).

        Returns:
            Tuple of (cluster label, attack votes, benign votes, invalid count). The label is
            `UNDECIDED` when no record produced a valid label.
        """
        labels = list(record_labels)
        n_attack = sum(label == RECORD_ATTACK for label in labels)
        n_benign = sum(label == RECORD_BENIGN for label in labels)
        n_invalid = len(labels) - n_attack - n_benign
        if n_attack == 0 and n_benign == 0:
            return UNDECIDED, n_attack, n_benign, n_invalid
        if n_attack == n_benign:
            return self._tie_label, n_attack, n_benign, n_invalid
        return (ATTACK_LIKE if n_attack > n_benign else BENIGN_LIKE), n_attack, n_benign, n_invalid


def rows_to_score(granularities: Sequence[str]) -> str:
    """The fewest short-listed rows that must be scored for some removal granularities.

    Args:
        granularities: Granularities that will be asked of `PoisonRemovalSystem.masks`.

    Returns:
        `ROWS_ALL`, `ROWS_ATTACK_LIKE` or `ROWS_SAMPLE`.
    """
    if GRANULARITY_RECORD in granularities:
        return ROWS_ALL
    if GRANULARITY_HYBRID in granularities:
        return ROWS_ATTACK_LIKE
    return ROWS_SAMPLE


@dataclass(frozen=True)
class RecordVotes:
    """Layer B's evidence on some rows, one entry per row.

    Attributes:
        rows: Training-matrix positions.
        classifier_attack: The clean model calls the row an attack.
        attack_closer: Its nearest attack flow is strictly closer than its nearest benign flow.
        benign_closer: Its nearest benign flow is strictly closer.
    """

    rows: np.ndarray
    classifier_attack: np.ndarray
    attack_closer: np.ndarray
    benign_closer: np.ndarray

    def attack(self, rule: str) -> np.ndarray:
        """Which rows vote attack under a rule.

        Args:
            rule: One of `LAYER_B_RULES`.

        Returns:
            Boolean array in the order of `rows`.

        Raises:
            ValueError: For an unknown rule.
        """
        if rule == RULE_SEQUENTIAL:
            return self.classifier_attack & ~self.benign_closer
        if rule == RULE_AND:
            return self.classifier_attack & self.attack_closer
        raise ValueError(f"Invalid layer-B rule: {rule!r} (one of {LAYER_B_RULES}).")

    def select(self, rows: np.ndarray) -> "RecordVotes":
        """The votes of some of the rows.

        Args:
            rows: Training-matrix positions, each one of `self.rows`.

        Returns:
            Votes in the order of `rows`.

        Raises:
            KeyError: If a row was not scored.
        """
        index = {int(row): position for position, row in enumerate(self.rows)}
        positions = np.array([index[int(row)] for row in rows], dtype=int)
        return RecordVotes(**{name: getattr(self, name)[positions] for name in self.__dataclass_fields__})

    @staticmethod
    def concatenate(parts: List["RecordVotes"]) -> "RecordVotes":
        """Votes of several disjoint row sets, in row order.

        Args:
            parts: Votes to join.

        Returns:
            One `RecordVotes` sorted by row.
        """
        joined = {name: np.concatenate([getattr(part, name) for part in parts]) for name in RecordVotes.__dataclass_fields__}
        order = np.argsort(joined["rows"], kind="stable")
        return RecordVotes(**{name: values[order] for name, values in joined.items()})


@dataclass(frozen=True)
class ClusterJudgement:
    """Layer B's view of one short-listed cluster.

    Attributes:
        cluster: Cluster id.
        n_compared: Records of the cluster that were judged.
        classifier_attack_share: Share the clean model calls attack.
        attack_closer_share: Share whose nearest attack flow is strictly closer than the nearest benign flow.
        benign_closer_share: Share whose nearest benign flow is strictly closer.
        attack_votes: Rule -> records voting attack.
        decisions: Rule -> `ATTACK_LIKE` / `BENIGN_LIKE` (majority; a tie follows the voter's tie label).
    """

    cluster: int
    n_compared: int
    classifier_attack_share: float
    attack_closer_share: float
    benign_closer_share: float
    attack_votes: Dict[str, int]
    decisions: Dict[str, str]


class LayerBJudge:
    """Scores records with the clean model and the exemplar check, and labels short-listed clusters by majority.

    Args:
        matrix: The source's poisoned training set.
        ranker: Its percentile transform.
        classifier: Clean model fitted on the trusted rows.
        finder: Exemplar finder whose pools are the trusted rows.
        classifier_threshold: P(attack) from which the clean model calls a record an attack.
        max_compared_records: Records judged per cluster (a reproducible random subset beyond that).
        voter: Majority vote over record labels.
        random_state: Seed of the subset.
    """

    def __init__(
        self,
        matrix: TrainingMatrix,
        ranker: RankTransformer,
        classifier: XgbAttackClassifier,
        finder: ContrastiveExemplarFinder,
        classifier_threshold: float,
        max_compared_records: int,
        voter: RecordVoteLabeler,
        random_state: int,
    ) -> None:
        self._matrix = matrix
        self._ranker = ranker
        self._classifier = classifier
        self._finder = finder
        self._threshold = classifier_threshold
        self._max_compared = max_compared_records
        self._voter = voter
        self._random_state = random_state

    def sample(self, key: str, rows: np.ndarray) -> np.ndarray:
        """The records of a cluster that decide its label.

        Args:
            key: Stable cluster key (`source|cluster`), seeds the subset.
            rows: Training-matrix positions of the cluster's rows.

        Returns:
            At most `max_compared_records` of `rows`, in row order.
        """
        return rows[np.sort(draw_positions(key, "layer_b", len(rows), self._max_compared, self._random_state))]

    def score(self, rows: np.ndarray) -> RecordVotes:
        """Record-level evidence on some rows.

        Args:
            rows: Training-matrix positions.

        Returns:
            Their votes.
        """
        classifier_attack = self._classifier.attack_probability(self._ranker.transform(self._matrix.features[rows])) >= self._threshold
        exemplars = self._finder.find(rows)
        return RecordVotes(
            rows=np.asarray(rows), classifier_attack=classifier_attack,
            attack_closer=exemplars.attack_distance < exemplars.benign_distance,
            benign_closer=exemplars.benign_distance < exemplars.attack_distance,
        )

    def judge(self, cluster: int, votes: RecordVotes) -> ClusterJudgement:
        """Label one cluster from the votes of its sampled records.

        Args:
            cluster: Cluster id.
            votes: Votes of the records returned by `sample`.

        Returns:
            The judgement.
        """
        attack_votes, decisions = {}, {}
        for rule in LAYER_B_RULES:
            decision, n_attack, _, _ = self._voter.vote([RECORD_ATTACK if flag else RECORD_BENIGN for flag in votes.attack(rule)])
            attack_votes[rule], decisions[rule] = n_attack, decision
        return ClusterJudgement(
            cluster=cluster, n_compared=len(votes.rows), classifier_attack_share=float(votes.classifier_attack.mean()),
            attack_closer_share=float(votes.attack_closer.mean()), benign_closer_share=float(votes.benign_closer.mean()),
            attack_votes=attack_votes, decisions=decisions,
        )


class PoisonRemovalSystem:
    """Turns cluster labels, record votes, the audited model's predictions and the noise scores into a removal mask.

    Args:
        row_clusters: Cluster id per row (`-1` = noise).
        attack_labelled_clusters: Clusters whose observed labels are attack-majority (always ATTACK_LIKE).
        audited_predicts_target: Per row, whether the audited model predicts the target label.
        benign_labelled: Per row, whether the observed label is the target label.
        noise_rows: Benign-labelled noise rows, in the order of `noise_attack_probability`.
        noise_attack_probability: The clean model's P(attack) for each of `noise_rows`.
    """

    def __init__(
        self,
        row_clusters: np.ndarray,
        attack_labelled_clusters: np.ndarray,
        audited_predicts_target: np.ndarray,
        benign_labelled: np.ndarray,
        noise_rows: np.ndarray,
        noise_attack_probability: np.ndarray,
    ) -> None:
        self._row_clusters = row_clusters
        self._attack_labelled = attack_labelled_clusters
        self._audited_target = audited_predicts_target
        self._benign = benign_labelled
        self._noise_rows = noise_rows
        self._noise_probability = noise_attack_probability

    def masks(
        self,
        judgements: List[ClusterJudgement],
        rule: str,
        noise_threshold: Optional[float],
        granularity: str = GRANULARITY_CLUSTER,
        record_votes: Optional[List[RecordVotes]] = None,
    ) -> Dict[str, np.ndarray]:
        """Removal masks of one variant.

        Args:
            judgements: Layer B's judgements of the short list.
            rule: One of `LAYER_B_RULES`, or `RULE_LAYER_A_ONLY` (every short-listed cluster ATTACK_LIKE; cluster
                granularity only).
            noise_threshold: P(attack) from which a noise row is removed; None disables the noise rule.
            granularity: One of `GRANULARITIES`.
            record_votes: Record votes covering every row the granularity needs: every short-listed row for `record`,
                every row of the ATTACK_LIKE clusters for `hybrid`.

        Returns:
            `removed` (every removed row) and its three origins: `attack_clusters`, `layer_b`, `noise`.

        Raises:
            ValueError: For an unknown granularity, when `record_votes` misses a row the granularity needs, or for
                `RULE_LAYER_A_ONLY` with a granularity other than cluster.
        """
        if rule == RULE_LAYER_A_ONLY:
            if granularity != GRANULARITY_CLUSTER:
                raise ValueError(f"Rule `{RULE_LAYER_A_ONLY}` only works with removal level `{GRANULARITY_CLUSTER}`, got {granularity!r}.")
            flagged = np.isin(self._row_clusters, [item.cluster for item in judgements])
        else:
            flagged = np.isin(self._row_clusters, [item.cluster for item in judgements if item.decisions[rule] == ATTACK_LIKE])
        if granularity == GRANULARITY_CLUSTER:
            layer_b = flagged
        elif granularity in (GRANULARITY_RECORD, GRANULARITY_HYBRID):
            covered = np.zeros(len(self._row_clusters), dtype=bool)
            voted = np.zeros(len(self._row_clusters), dtype=bool)
            for votes in record_votes or []:
                covered[votes.rows] = True
                voted[votes.rows] = votes.attack(rule)
            needed = np.isin(self._row_clusters, [item.cluster for item in judgements]) if granularity == GRANULARITY_RECORD else flagged
            if (needed & ~covered).any():
                raise ValueError(f"Level `{granularity}` needs the votes of {int((needed & ~covered).sum())} rows that were not scored.")
            layer_b = voted if granularity == GRANULARITY_RECORD else flagged & voted
        else:
            raise ValueError(f"Invalid removal level: {granularity!r} (one of {GRANULARITIES}).")
        in_attack = np.isin(self._row_clusters, self._attack_labelled) & self._audited_target
        in_layer_b = layer_b & self._audited_target
        noise = np.zeros(len(self._row_clusters), dtype=bool)
        if noise_threshold is not None:
            chosen = self._noise_rows[self._noise_probability >= noise_threshold]
            noise[chosen] = True
            noise &= self._audited_target
        return {"removed": in_attack | in_layer_b | noise, "attack_clusters": in_attack, "layer_b": in_layer_b, "noise": noise}


@dataclass(frozen=True)
class LayerBOutcome:
    """Everything layer B produced for one source.

    Attributes:
        judgements: One per short-listed cluster, in short-list order.
        record_votes: Votes of the scored rows of each short-listed cluster, same order.
        cluster_sizes: Rows of each short-listed cluster, same order.
        system: The removal system of the source.
        trusted_benign: Per row, whether the clean model learnt it as benign.
        reference_benign: Per row, whether the exemplars may use it as the nearest benign flow.
        reference_trained: False when the trusted rows held one class only and no reference model could be trained.
    """

    judgements: List[ClusterJudgement]
    record_votes: List[RecordVotes]
    cluster_sizes: List[int]
    system: PoisonRemovalSystem
    trusted_benign: np.ndarray
    reference_benign: np.ndarray
    reference_trained: bool = True


class PoisonRemovalBuilder:
    """Builds the trusted rows, the clean model and the exemplar finder of one source, runs layer B on its short list
    and scores its benign noise rows - everything the removal needs after layer A.

    The trusted rows and the exemplars' benign pool exclude noise (unless the reference policy admits some), and the
    clean model learns on every trusted row (no per-class cap): both are the system's design, not tunable parameters.

    Args:
        model_params: XGBoost kwargs of the clean model (the audited model's architecture).
        classifier_threshold: P(attack) from which the clean model calls a record an attack.
        max_compared_records: Records that label each short-listed cluster.
        tolerance: Percentile difference up to which two values count as nearly equal (exemplars).
        query_block_size: Exemplar query rows compared at once.
        pool_block_size: Exemplar pool rows compared at once.
        voter: Majority vote over record labels.
        random_state: Seed of the clean model and of the record samples.
    """

    def __init__(
        self,
        model_params: Dict[str, Any],
        classifier_threshold: float,
        max_compared_records: int,
        tolerance: float,
        query_block_size: int,
        pool_block_size: int,
        voter: RecordVoteLabeler,
        random_state: int,
    ) -> None:
        self._model_params = model_params
        self._threshold = classifier_threshold
        self._max_compared = max_compared_records
        self._tolerance = tolerance
        self._query_block_size = query_block_size
        self._pool_block_size = pool_block_size
        self._voter = voter
        self._random_state = random_state

    def build(
        self,
        name: str,
        matrix: TrainingMatrix,
        row_clusters: np.ndarray,
        short_listed: np.ndarray,
        attack_labelled_clusters: np.ndarray,
        audited_predicts_target: np.ndarray,
        target_label: str,
        rows_scored: str,
        reference: BenignReference = DEFAULT_BENIGN_REFERENCE,
    ) -> LayerBOutcome:
        """Run layer B on one source.

        Args:
            name: Source name; with the cluster id it seeds each cluster's record sample.
            matrix: The source's poisoned training set.
            row_clusters: Cluster id per row (`-1` = noise).
            short_listed: Layer A's short-listed cluster ids.
            attack_labelled_clusters: Clusters whose observed labels are attack-majority.
            audited_predicts_target: Per row, whether the audited model predicts the target label.
            target_label: The benign/target label.
            rows_scored: `ROWS_SAMPLE`, `ROWS_ATTACK_LIKE` or `ROWS_ALL` (see `rows_to_score`).
            reference: Which benign-labelled rows layer B trusts.

        Returns:
            The judgements, record votes and removal system.
        """
        ranker = RankTransformer(matrix)
        benign = matrix.rows_labelled(target_label)
        excluded = short_listed if reference.attack_labelled_clusters else np.union1d(short_listed, attack_labelled_clusters)
        trusted_benign = benign & (row_clusters != NOISE_CLUSTER_ID) & ~np.isin(row_clusters, excluded)
        trusted = TrustedTrainingSet.build(matrix, ranker, row_clusters, excluded, target_label, include_noise=False,
                                           max_rows_per_class=None, random_state=self._random_state)
        if trusted.is_attack.all() or not trusted.is_attack.any():
            logger.warning("[%s] The trusted set holds one class only (%d rows, %s): no reference model can be trained, layer B"
                           " confirms no cluster and the noise rule removes no row.", name, len(trusted.is_attack),
                           "all attack-labelled" if trusted.is_attack.all() else "all benign-labelled")
            return self._without_reference(short_listed, row_clusters, attack_labelled_clusters, audited_predicts_target,
                                           benign, trusted_benign)
        classifier = XgbAttackClassifier(self._model_params, self._random_state).fit(trusted)
        del trusted
        gc.collect()
        noise_rows = np.flatnonzero((row_clusters == NOISE_CLUSTER_ID) & benign)
        noise_probability = classifier.attack_probability(ranker.transform(matrix.features[noise_rows]))
        reference_benign = trusted_benign.copy()
        if reference.noise == REFERENCE_NOISE_CLEAN_BENIGN:
            reference_benign[noise_rows[noise_probability < self._threshold]] = True
        finder = ContrastiveExemplarFinder(matrix, ranker, row_clusters, short_listed, target_label, include_noise=False,
                                           tolerance=self._tolerance, query_block_size=self._query_block_size,
                                           pool_block_size=self._pool_block_size, benign_reference=reference_benign)
        judge = LayerBJudge(matrix, ranker, classifier, finder, self._threshold, self._max_compared, self._voter, self._random_state)
        judgements, record_votes, cluster_sizes = [], [], []
        for cluster in short_listed:
            members = np.flatnonzero(row_clusters == cluster)
            compared = judge.sample(f"{name}|{int(cluster)}", members)
            votes = judge.score(members if rows_scored == ROWS_ALL else compared)
            judgement = judge.judge(int(cluster), votes.select(compared))
            if rows_scored == ROWS_ATTACK_LIKE and ATTACK_LIKE in judgement.decisions.values():
                rest = np.setdiff1d(members, compared)
                if len(rest):
                    votes = RecordVotes.concatenate([votes, judge.score(rest)])
            judgements.append(judgement)
            record_votes.append(votes)
            cluster_sizes.append(len(members))
            logger.info("[%s] cluster %d (%d rows, %d scored): %s", name, int(cluster), len(members), len(votes.rows), judgement.decisions)
        system = PoisonRemovalSystem(row_clusters, attack_labelled_clusters, audited_predicts_target, benign, noise_rows,
                                     noise_probability)
        return LayerBOutcome(judgements=judgements, record_votes=record_votes, cluster_sizes=cluster_sizes, system=system,
                             trusted_benign=trusted_benign, reference_benign=reference_benign)

    @staticmethod
    def _without_reference(
        short_listed: np.ndarray, row_clusters: np.ndarray, attack_labelled_clusters: np.ndarray,
        audited_predicts_target: np.ndarray, benign: np.ndarray, trusted_benign: np.ndarray,
    ) -> LayerBOutcome:
        """Layer B when the trusted rows hold one class only, so no reference model can be trained.

        Both checks of a record vote need the reference model, so no record votes attack: every short-listed cluster is
        BENIGN_LIKE, no record is judged, and the noise rule removes nothing (P(attack) = 0). Attack-labelled clusters
        keep LABOR's rule. Never met when stage 1 clusters TreeSHAP attributions (every cell of script 04 trained one);
        met by script 12, which clusters raw features.

        Args:
            short_listed: Layer A's short-listed cluster ids.
            row_clusters: Cluster id per row.
            attack_labelled_clusters: Attack-labelled cluster ids.
            audited_predicts_target: Per row, whether the audited model predicts the target label.
            benign: Per row, whether its observed label is the target label.
            trusted_benign: Trusted benign rows (empty or the only class).

        Returns:
            The outcome, with `reference_trained` False.
        """
        empty_rows, empty_flags = np.empty(0, dtype=np.int64), np.zeros(0, dtype=bool)
        judgements = [ClusterJudgement(cluster=int(cluster), n_compared=0, classifier_attack_share=0.0,
                                       attack_closer_share=0.0, benign_closer_share=0.0,
                                       attack_votes={rule: 0 for rule in LAYER_B_RULES},
                                       decisions={rule: BENIGN_LIKE for rule in LAYER_B_RULES}) for cluster in short_listed]
        record_votes = [RecordVotes(rows=empty_rows, classifier_attack=empty_flags, attack_closer=empty_flags,
                                    benign_closer=empty_flags) for _ in short_listed]
        cluster_sizes = [int((row_clusters == cluster).sum()) for cluster in short_listed]
        noise_rows = np.flatnonzero((row_clusters == NOISE_CLUSTER_ID) & benign)
        system = PoisonRemovalSystem(row_clusters, attack_labelled_clusters, audited_predicts_target, benign, noise_rows,
                                     np.zeros(len(noise_rows)))
        return LayerBOutcome(judgements=judgements, record_votes=record_votes, cluster_sizes=cluster_sizes, system=system,
                             trusted_benign=trusted_benign, reference_benign=trusted_benign.copy(), reference_trained=False)


class RemovalScorer:
    """Scores a removal against ground truth - the only place ground truth meets the system."""

    @staticmethod
    def score(masks: Dict[str, np.ndarray], poisoned: np.ndarray) -> Dict[str, Any]:
        """Score one removal.

        Args:
            masks: Output of `PoisonRemovalSystem.masks`.
            poisoned: Which rows are poison.

        Returns:
            Rows removed, removed correctly, removed wrongly (in total and by origin), recall, precision, and the wrong
            removals as a share of the whole training set.
        """
        removed = masks["removed"]
        wrong = removed & ~poisoned
        correct = int((removed & poisoned).sum())
        return {
            "n_removed": int(removed.sum()), "n_removed_correct": correct, "n_removed_wrong": int(wrong.sum()),
            "wrong_in_attack_clusters": int((masks["attack_clusters"] & ~poisoned).sum()),
            "wrong_in_layer_b": int((masks["layer_b"] & ~poisoned).sum()),
            "wrong_in_noise": int((masks["noise"] & ~poisoned).sum()),
            "recall": correct / int(poisoned.sum()) if poisoned.any() else 0.0,
            "precision": correct / int(removed.sum()) if removed.any() else 0.0,
            "wrong_share": float(wrong.sum()) / len(removed),
        }
