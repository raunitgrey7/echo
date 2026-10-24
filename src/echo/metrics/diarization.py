"""Speaker change and clustering scores.

Why: three questions matter for a meeting transcript: were the turn boundaries found (change
point F1 with a tolerance, because the exact instant inside a pause is undefined), was the number
of speakers right, and do the labels agree with the true speakers (duration-weighted purity).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from echo.vad.energy import SpeechRegion


@dataclass(frozen=True, slots=True)
class ChangeScore:
    """Change-point detection quality."""

    precision: float
    recall: float
    f1: float
    n_reference: int
    n_hypothesis: int


@dataclass(frozen=True, slots=True)
class LabelledSpan:
    """A speech span with its speaker label."""

    region: SpeechRegion
    speaker: str


def score_changes(reference: Sequence[float], hypothesis: Sequence[float], tolerance_s: float) -> ChangeScore:
    """Greedy one-to-one matching of change times within ``tolerance_s``."""
    unmatched = list(hypothesis)
    hits = 0
    for truth in reference:
        candidates = [h for h in unmatched if abs(h - truth) <= tolerance_s]
        if candidates:
            closest = min(candidates, key=lambda h: abs(h - truth))
            unmatched.remove(closest)
            hits += 1
    precision = hits / len(hypothesis) if hypothesis else (1.0 if not reference else 0.0)
    recall = hits / len(reference) if reference else (1.0 if not hypothesis else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return ChangeScore(precision, recall, f1, len(reference), len(hypothesis))


def purity(reference: Sequence[LabelledSpan], hypothesis: Sequence[LabelledSpan]) -> float:
    """Duration-weighted fraction of hypothesis time assigned to its cluster's majority speaker."""
    overlap: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    total = 0.0
    for hyp in hypothesis:
        for ref in reference:
            shared = hyp.region.overlap(ref.region)
            if shared > 0:
                overlap[hyp.speaker][ref.speaker] += shared
                total += shared
    if total <= 0:
        return 0.0
    return sum(max(by_ref.values()) for by_ref in overlap.values()) / total
