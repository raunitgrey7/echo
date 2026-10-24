"""Word and character error rates by Levenshtein alignment with operation counts.

Why: a single rate hides what went wrong. Counting substitutions, deletions and insertions
separately shows whether a condition makes the recogniser drop words (typical for low SNR) or
hallucinate them (typical for reverberation), and the corpus rate pools errors over reference
length rather than averaging per-file rates, which would over-weight short clips.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from echo.metrics.text_norm import normalize_text, tokens


@dataclass(frozen=True, slots=True)
class EditCounts:
    """Alignment outcome between one reference and one hypothesis."""

    substitutions: int
    deletions: int
    insertions: int
    reference_length: int

    @property
    def errors(self) -> int:
        """Total edit operations."""
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        """Errors divided by reference length; an empty reference scores its insertions as a count."""
        if self.reference_length == 0:
            return float(self.insertions)
        return self.errors / self.reference_length

    def __add__(self, other: EditCounts) -> EditCounts:
        return EditCounts(
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
            self.reference_length + other.reference_length,
        )


def align_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> EditCounts:
    """Dynamic-programming alignment with a backtrace that counts each operation type.

    Ties are broken in the order substitution, deletion, insertion so the counts are deterministic.
    """
    n, m = len(reference), len(hypothesis)
    cost = np.zeros((n + 1, m + 1), dtype=np.int32)
    cost[:, 0] = np.arange(n + 1)
    cost[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ref_token = reference[i - 1]
        for j in range(1, m + 1):
            same = 0 if ref_token == hypothesis[j - 1] else 1
            cost[i, j] = min(cost[i - 1, j - 1] + same, cost[i - 1, j] + 1, cost[i, j - 1] + 1)
    return _backtrace(cost, reference, hypothesis)


def _backtrace(cost: np.ndarray, reference: Sequence[str], hypothesis: Sequence[str]) -> EditCounts:
    i, j = len(reference), len(hypothesis)
    subs = dels = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            same = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            if cost[i, j] == cost[i - 1, j - 1] + same:
                subs += same
                i, j = i - 1, j - 1
                continue
        if i > 0 and cost[i, j] == cost[i - 1, j] + 1:
            dels += 1
            i -= 1
            continue
        ins += 1
        j -= 1
    return EditCounts(subs, dels, ins, len(reference))


def word_counts(reference: str, hypothesis: str) -> EditCounts:
    """Word-level alignment after normalisation."""
    return align_counts(tokens(reference), tokens(hypothesis))


def char_counts(reference: str, hypothesis: str) -> EditCounts:
    """Character-level alignment on the normalised strings, spaces included."""
    return align_counts(list(normalize_text(reference)), list(normalize_text(hypothesis)))


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER for one pair."""
    return word_counts(reference, hypothesis).rate


def char_error_rate(reference: str, hypothesis: str) -> float:
    """CER for one pair."""
    return char_counts(reference, hypothesis).rate


def corpus_counts(pairs: Iterable[tuple[str, str]], level: str = "word") -> EditCounts:
    """Pool alignments over many pairs; ``level`` is ``word`` or ``char``."""
    scorer = word_counts if level == "word" else char_counts
    total = EditCounts(0, 0, 0, 0)
    for reference, hypothesis in pairs:
        total = total + scorer(reference, hypothesis)
    return total
