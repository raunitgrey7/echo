"""Deterministic extractive lecture notes.

Why: students want the ten sentences that carry the lecture, in order, with the time they were
said. TextRank over TF-IDF cosine similarity gives exactly that without a model or a key: a
sentence is central when many other sentences resemble it. Power iteration is run to a fixed
tolerance from a uniform start, so the same transcript always yields the same notes. Sections
come from a TextTiling-style dip in lexical cohesion between neighbouring windows of sentences,
and each section is titled by its highest-scoring keyword phrase.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from echo.asr.base import Word
from echo.metrics.text_norm import tokens
from echo.products.keywords import Keyword
from echo.products.stopwords import STOPWORDS
from echo.products.transcript import Transcript

_SENTENCE_END = re.compile(r"[.!?]$")


@dataclass(frozen=True, slots=True)
class Sentence:
    """A sentence with its time span and speaker."""

    index: int
    text: str
    start: float
    end: float
    speaker: str


@dataclass(frozen=True, slots=True)
class Bullet:
    """A note line with its timestamp."""

    text: str
    start: float
    speaker: str
    score: float


@dataclass(frozen=True, slots=True)
class Section:
    """A topical block of the recording."""

    title: str
    start: float
    end: float
    bullets: tuple[Bullet, ...]


@dataclass(frozen=True, slots=True)
class Notes:
    """Summary bullets, sections and the terms used as the title."""

    title_terms: tuple[str, ...]
    summary: tuple[Bullet, ...]
    sections: tuple[Section, ...]
    n_sentences: int
    compression: float = field(default=0.0)


def join_words(words: list[Word]) -> str:
    """Rebuild text from words, honouring the recogniser's spacing so hyphenated words stay intact."""
    pieces = [(" " if w.prefix_space and i > 0 else "") + w.text for i, w in enumerate(words)]
    return "".join(pieces).strip()


def split_sentences(transcript: Transcript) -> list[Sentence]:
    """Split on sentence punctuation using word timings, across segment boundaries.

    A segment without word timings is one sentence. The speaker of a sentence is the speaker of
    the segment in which it started.
    """
    sentences: list[Sentence] = []
    buffer: list[Word] = []
    speaker = ""

    def flush() -> None:
        if buffer:
            sentences.append(Sentence(len(sentences), join_words(buffer), buffer[0].start, buffer[-1].end, speaker))
            buffer.clear()

    for segment in transcript.segments:
        if not segment.words:
            flush()
            if segment.text.strip():
                sentences.append(
                    Sentence(len(sentences), segment.text.strip(), segment.start, segment.end, segment.speaker)
                )
            continue
        for word in segment.words:
            if not buffer:
                speaker = segment.speaker
            buffer.append(word)
            if _SENTENCE_END.search(word.text):
                flush()
    flush()
    return sentences


def content_tokens(text: str) -> list[str]:
    """Normalised tokens without stopwords or bare numbers, lightly stemmed."""
    out: list[str] = []
    for token in tokens(text):
        if token in STOPWORDS or token.isdigit():
            continue
        out.append(token[:-1] if token.endswith("s") and len(token) > 3 else token)
    return out


def tfidf_matrix(sentences: list[Sentence]) -> np.ndarray:
    """Row-normalised TF-IDF with smoothed IDF; rows of all-stopword sentences stay zero."""
    docs = [Counter(content_tokens(s.text)) for s in sentences]
    vocab = sorted({w for d in docs for w in d})
    if not vocab:
        return np.zeros((len(sentences), 1))
    index = {w: i for i, w in enumerate(vocab)}
    df = Counter(w for d in docs for w in d)
    n = len(docs)
    matrix = np.zeros((n, len(vocab)))
    for row, doc in enumerate(docs):
        for word, count in doc.items():
            matrix[row, index[word]] = (1 + math.log(count)) * math.log((1 + n) / (1 + df[word])) + 1.0
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def textrank(similarity: np.ndarray, damping: float = 0.85, iterations: int = 200, tol: float = 1e-8) -> np.ndarray:
    """Stationary scores of the similarity graph; deterministic from a uniform start."""
    n = similarity.shape[0]
    if n == 0:
        return np.zeros(0)
    weights = similarity.copy()
    np.fill_diagonal(weights, 0.0)
    out_sum = weights.sum(axis=1, keepdims=True)
    out_sum[out_sum == 0] = 1.0
    transition = weights / out_sum
    scores = np.full(n, 1.0 / n)
    for _ in range(iterations):
        updated = (1 - damping) / n + damping * transition.T @ scores
        if np.abs(updated - scores).sum() < tol:
            scores = updated
            break
        scores = updated
    return scores


def summary_size(n_sentences: int, ratio: float, minimum: int, maximum: int) -> int:
    """How many sentences to keep."""
    return max(min(n_sentences, minimum), min(maximum, math.ceil(ratio * n_sentences)))


def select_summary(
    sentences: list[Sentence], scores: np.ndarray, ratio: float, minimum: int, maximum: int
) -> list[int]:
    """Indices of the top-scoring sentences, returned in reading order."""
    keep = summary_size(len(sentences), ratio, minimum, maximum)
    ranked = sorted(range(len(sentences)), key=lambda i: (-scores[i], i))
    return sorted(ranked[:keep])


def section_breaks(matrix: np.ndarray, window: int = 3, min_section: int = 3) -> list[int]:
    """Sentence indices where a new section starts (TextTiling depth scores on window cohesion)."""
    n = matrix.shape[0]
    if n < 2 * min_section:
        return []
    cohesion = np.array(
        [float(matrix[max(0, i - window) : i].mean(axis=0) @ matrix[i : i + window].mean(axis=0)) for i in range(1, n)]
    )
    depth = np.zeros_like(cohesion)
    for i in range(cohesion.shape[0]):
        left = cohesion[: i + 1].max()
        right = cohesion[i:].max()
        depth[i] = (left - cohesion[i]) + (right - cohesion[i])
    threshold = depth.mean() + 0.5 * depth.std()
    breaks: list[int] = []
    for i in np.argsort(-depth):
        if depth[i] < threshold or len(breaks) >= n // min_section - 1:
            break
        boundary = int(i) + 1
        if all(abs(boundary - b) >= min_section for b in breaks) and min_section <= boundary <= n - min_section:
            breaks.append(boundary)
    return sorted(breaks)


def section_title(sentences: list[Sentence], keywords: list[Keyword]) -> str:
    """The highest-ranked keyword whose first occurrence falls inside the section, else its first words."""
    start, end = sentences[0].start, sentences[-1].end
    for keyword in keywords:
        if any(start <= t <= end for t in keyword.timestamps):
            return keyword.phrase.title()
    return " ".join(sentences[0].text.split()[:5])


def build_notes(
    transcript: Transcript,
    keywords: list[Keyword],
    ratio: float = 0.3,
    min_sentences: int = 3,
    max_sentences: int = 8,
    bullets_per_section: int = 3,
) -> Notes:
    """Summary bullets plus sections, each with its own top sentences."""
    sentences = split_sentences(transcript)
    if not sentences:
        return Notes((), (), (), 0, 0.0)
    matrix = tfidf_matrix(sentences)
    scores = textrank(matrix @ matrix.T)
    chosen = select_summary(sentences, scores, ratio, min_sentences, max_sentences)
    summary = tuple(
        Bullet(sentences[i].text, sentences[i].start, sentences[i].speaker, float(scores[i])) for i in chosen
    )
    bounds = [0, *section_breaks(matrix), len(sentences)]
    sections: list[Section] = []
    for lo, hi in pairwise(bounds):
        block = sentences[lo:hi]
        local = sorted(range(lo, hi), key=lambda i: (-scores[i], i))[:bullets_per_section]
        bullets = tuple(
            Bullet(sentences[i].text, sentences[i].start, sentences[i].speaker, float(scores[i])) for i in sorted(local)
        )
        sections.append(Section(section_title(block, keywords), block[0].start, block[-1].end, bullets))
    title_terms = tuple(k.phrase for k in keywords[:3])
    return Notes(title_terms, summary, tuple(sections), len(sentences), len(chosen) / len(sentences))


def _bullet(sentence: Sentence, score: float) -> Bullet:
    return Bullet(sentence.text, sentence.start, sentence.speaker, score)


def notes_to_markdown(notes: Notes, title: str = "Lecture notes") -> str:
    """Markdown rendering with timestamps."""
    lines = [f"# {title}", ""]
    if notes.title_terms:
        lines += ["Key terms: " + ", ".join(notes.title_terms), ""]
    lines += ["## Summary", ""]
    lines += [f"- [{_mmss(b.start)}] {b.text}" for b in notes.summary]
    for section in notes.sections:
        lines += ["", f"## {section.title} ({_mmss(section.start)} to {_mmss(section.end)})", ""]
        lines += [f"- [{_mmss(b.start)}] {b.text}" for b in section.bullets]
    return "\n".join(lines) + "\n"


def _mmss(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"
