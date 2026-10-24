"""Keyword extraction and the keyword timeline.

Why: a one-hour lecture is navigated by its terms, not by its sentences. The extractor needs
no model and no corpus statistics, so it has to lean on what a single transcript offers. Candidate
phrases are runs of content words between stopwords and punctuation (the RAKE idea). A word is
worth its frequency times a specificity prior that grows with word length, a Zipf-motivated proxy
for rarity: long Latinate words such as "transpiration" are rarer than short ones. Three
transcript cues multiply the score: a definitional pattern before the phrase ("called X",
"known as X", "we call X"), capitalisation (proper nouns such as Bastille) and appearance in the
opening sentence, where a lecturer names the topic. A phrase scores the mean of its word scores
times a mild length bonus, so two-word terms can beat their parts without three-word fragments
taking over. Every keyword keeps the times it was spoken, which is the timeline the dashboard draws.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from echo.metrics.text_norm import normalize_text
from echo.products.stopwords import STOPWORDS, is_verb_like
from echo.products.transcript import Transcript

_BREAK = re.compile(r"[.,;:!?()\[\]\"]")
DEFINITIONAL = ("called", "named", "termed", "known as", "we call", "so called")
LENGTH_PIVOT = 5
LENGTH_SLOPE = 0.5
PHRASE_BONUS = 0.0
CUE_BOOST = 3.0
CAPITAL_BOOST = 1.5
OPENING_BOOST = 2.0


@dataclass(frozen=True, slots=True)
class Keyword:
    """A ranked phrase with the times it occurs."""

    phrase: str
    score: float
    count: int
    timestamps: tuple[float, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class Candidate:
    """A phrase occurrence with its start time and the cues that applied to it."""

    words: tuple[str, ...]
    start: float
    defined: bool = False
    capitalised: bool = False
    opening: bool = False


def normalize_word(raw: str) -> str:
    """Lowercase alphanumerics; returns an empty string for pure punctuation."""
    return normalize_text(raw)


def is_content(token: str) -> bool:
    """Content words only: not a stopword, not a number, not a common verb, not an -ly adverb."""
    if not token or token in STOPWORDS or token.isdigit() or is_verb_like(token):
        return False
    return not (token.endswith("ly") and len(token) > 4)


def candidate_phrases(transcript: Transcript, max_words: int = 3) -> list[Candidate]:
    """Runs of content words between stopwords or punctuation, capped at ``max_words``, with cues."""
    phrases: list[Candidate] = []
    run: list[str] = []
    run_start, run_caps, run_defined = 0.0, False, False
    history: list[str] = []
    opening_end = _opening_end(transcript)
    for word in transcript.words:
        token = normalize_word(word.text)
        content = is_content(token)
        if content:
            if not run:
                run_start = word.start
                run_defined = _preceded_by_cue(history)
                run_caps = False
            run.append(token)
            run_caps = run_caps or word.text[:1].isupper()
        if (not content or _BREAK.search(word.text) or len(run) >= max_words) and run:
            phrases.append(Candidate(tuple(run), run_start, run_defined, run_caps, run_start <= opening_end))
            run = []
        if token:
            history = ([*history, token])[-2:]
    if run:
        phrases.append(Candidate(tuple(run), run_start, run_defined, run_caps, run_start <= opening_end))
    return phrases


def _opening_end(transcript: Transcript) -> float:
    """End time of the first sentence (first word carrying sentence punctuation), else of the first segment."""
    for word in transcript.words:
        if re.search(r"[.!?]$", word.text):
            return word.end
    return transcript.segments[0].end if transcript.segments else 0.0


def _preceded_by_cue(history: list[str]) -> bool:
    joined = " ".join(history)
    return any(joined.endswith(cue) for cue in DEFINITIONAL)


def word_weights(candidates: list[Candidate]) -> dict[str, float]:
    """Frequency times a length-based specificity prior."""
    frequency: dict[str, int] = defaultdict(int)
    for candidate in candidates:
        for word in candidate.words:
            frequency[word] += 1
    return {w: f * (1.0 + LENGTH_SLOPE * max(0, len(w) - LENGTH_PIVOT)) for w, f in frequency.items()}


def extract_keywords(transcript: Transcript, top_k: int = 10, max_words: int = 3) -> list[Keyword]:
    """Top phrases by weighted frequency and cues; ties break on first occurrence."""
    candidates = candidate_phrases(transcript, max_words)
    if not candidates:
        return []
    weights = word_weights(candidates)
    grouped: dict[tuple[str, ...], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.words].append(candidate)
    keywords: list[Keyword] = []
    for words, group in grouped.items():
        base = float(np.mean([weights[w] for w in words])) * (1.0 + PHRASE_BONUS * (len(words) - 1))
        if any(c.defined for c in group):
            base *= CUE_BOOST
        if any(c.capitalised for c in group) and len(words) <= 2:
            base *= CAPITAL_BOOST
        if any(c.opening for c in group):
            base *= OPENING_BOOST
        times = tuple(sorted(c.start for c in group))
        keywords.append(Keyword(" ".join(words), base, len(group), times))
    keywords.sort(key=lambda k: (-k.score, k.timestamps[0], k.phrase))
    return keywords[:top_k]


def keyword_hits(keywords: list[Keyword], gold_terms: tuple[str, ...]) -> int:
    """How many gold terms are covered by an extracted phrase (shared stem tokens), for evaluation."""

    def stems(text: str) -> set[str]:
        return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in normalize_text(text).split()}

    extracted = [stems(k.phrase) for k in keywords]
    hits = 0
    for term in gold_terms:
        wanted = stems(term)
        if any(wanted <= found or found <= wanted for found in extracted if found):
            hits += 1
    return hits
