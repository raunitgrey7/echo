"""Scores for the derived products: notes, action items and search.

Why: a transcript product is only worth shipping if it can be measured against a label. Notes are
scored by key-sentence recall (did the summary keep the sentences the script marks as essential,
matched by token overlap so an ASR slip does not break the match), action items by precision and
recall against labelled turns matched on time, and search by hit rate and reciprocal rank of the
turn that answers each query.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from echo.corpus.build import UtteranceLabel
from echo.metrics.text_norm import tokens
from echo.products.actions import ActionItem
from echo.products.notes import Bullet
from echo.products.search import Hit
from echo.vad.energy import SpeechRegion


@dataclass(frozen=True, slots=True)
class Counts:
    """Hits over a total, with a safe ratio."""

    hits: int
    total: int

    @property
    def ratio(self) -> float:
        """Hits divided by total, zero when nothing was asked."""
        return self.hits / self.total if self.total else 0.0

    def __add__(self, other: Counts) -> Counts:
        return Counts(self.hits + other.hits, self.total + other.total)


@dataclass(frozen=True, slots=True)
class Confusion:
    """Binary detection counts."""

    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        """TP over predicted positives."""
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        """TP over true positives."""
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def __add__(self, other: Confusion) -> Confusion:
        return Confusion(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)


def token_overlap(a: str, b: str) -> float:
    """Jaccard similarity of normalised token sets."""
    sa, sb = set(tokens(a)), set(tokens(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def key_sentence_recall(bullets: Sequence[Bullet], gold_sentences: Sequence[str], min_overlap: float = 0.5) -> Counts:
    """How many gold sentences have a summary bullet with token overlap at or above ``min_overlap``."""
    hits = sum(1 for gold in gold_sentences if any(token_overlap(gold, b.text) >= min_overlap for b in bullets))
    return Counts(hits, len(gold_sentences))


def best_utterance(start: float, end: float, utterances: Sequence[UtteranceLabel]) -> UtteranceLabel | None:
    """The labelled utterance overlapping a span the most, or None when nothing overlaps."""
    span = SpeechRegion(start, end)
    best, best_overlap = None, 0.0
    for utterance in utterances:
        shared = span.overlap(SpeechRegion(utterance.start, utterance.end))
        if shared > best_overlap:
            best, best_overlap = utterance, shared
    return best


def owner_matches(action: ActionItem, target: UtteranceLabel) -> bool:
    """Name match, or the speaker label when the commitment is the speaker's own.

    The system never learns participants' names, so for "I will ..." turns the correct owner is
    whoever is speaking; the predicted owner must then be the sentence's own speaker label.
    """
    if target.owner is None:
        return False
    if target.owner == target.speaker:
        return action.owner == action.speaker or action.owner.lower() == target.owner.lower()
    return action.owner.lower() == target.owner.lower()


def score_actions(actions: Sequence[ActionItem], utterances: Sequence[UtteranceLabel]) -> tuple[Confusion, Counts]:
    """Turn-level detection confusion and owner accuracy over matched items with a labelled owner."""
    matched: set[int] = set()
    fp = 0
    owner_hits, owner_total = 0, 0
    for action in actions:
        target = best_utterance(action.start, action.end, utterances)
        if target is None or not target.action or target.index in matched:
            fp += 1
            continue
        matched.add(target.index)
        if target.owner:
            owner_total += 1
            owner_hits += int(owner_matches(action, target))
    gold = [u.index for u in utterances if u.action]
    fn = len([g for g in gold if g not in matched])
    return Confusion(len(matched), fp, fn), Counts(owner_hits, owner_total)


@dataclass(frozen=True, slots=True)
class SearchScore:
    """Hit rates and reciprocal rank pooled over queries."""

    hit_at_1: Counts
    hit_at_3: Counts
    reciprocal_rank_sum: float
    queries: int

    @property
    def mrr(self) -> float:
        """Mean reciprocal rank over the top results."""
        return self.reciprocal_rank_sum / self.queries if self.queries else 0.0

    def __add__(self, other: SearchScore) -> SearchScore:
        return SearchScore(
            self.hit_at_1 + other.hit_at_1,
            self.hit_at_3 + other.hit_at_3,
            self.reciprocal_rank_sum + other.reciprocal_rank_sum,
            self.queries + other.queries,
        )


def score_query(hits: Sequence[Hit], gold: UtteranceLabel, tolerance_s: float = 0.5) -> int:
    """1-based rank of the first hit overlapping the gold utterance, or 0 when none does."""
    target = SpeechRegion(gold.start - tolerance_s, gold.end + tolerance_s)
    for rank, hit in enumerate(hits, start=1):
        if SpeechRegion(hit.start, hit.end).overlap(target) > 0:
            return rank
    return 0


def score_search(ranks: Sequence[int]) -> SearchScore:
    """Pool per-query ranks into hit@1, hit@3 and MRR."""
    return SearchScore(
        Counts(sum(1 for r in ranks if r == 1), len(ranks)),
        Counts(sum(1 for r in ranks if 1 <= r <= 3), len(ranks)),
        sum(1.0 / r for r in ranks if r > 0),
        len(ranks),
    )
