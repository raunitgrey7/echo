"""BM25 search over transcript sentences.

Why: a recording is only useful later if a question can jump to the moment it was answered.
BM25 is the standard lexical ranking function, needs no training and runs in microseconds on a
lecture; sentences are the indexed unit because they are the natural answer span. Tokens are the
same normalised, lightly stemmed content tokens the summariser uses, so "aquifers" and "aquifer"
meet. Every hit carries its start time so the player can seek to it.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from echo.products.notes import Sentence, content_tokens, split_sentences
from echo.products.transcript import Transcript


@dataclass(frozen=True, slots=True)
class Hit:
    """One ranked result."""

    sentence_index: int
    score: float
    start: float
    end: float
    text: str
    speaker: str


class SearchIndex:
    """Inverted index with BM25 scoring (k1 = 1.5, b = 0.75)."""

    def __init__(self, sentences: list[Sentence], k1: float = 1.5, b: float = 0.75) -> None:
        self.sentences = list(sentences)
        self.k1 = k1
        self.b = b
        self.term_freqs: list[Counter[str]] = [Counter(content_tokens(s.text)) for s in self.sentences]
        self.lengths = [sum(tf.values()) for tf in self.term_freqs]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.postings: dict[str, list[int]] = defaultdict(list)
        for doc_id, tf in enumerate(self.term_freqs):
            for term in tf:
                self.postings[term].append(doc_id)

    @classmethod
    def from_transcript(cls, transcript: Transcript) -> SearchIndex:
        """Index the sentences of a transcript."""
        return cls(split_sentences(transcript))

    def idf(self, term: str) -> float:
        """Robertson-Sparck Jones idf with the +1 inside the log so it stays positive."""
        n = len(self.sentences)
        df = len(self.postings.get(term, ()))
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score_doc(self, doc_id: int, query_terms: list[str]) -> float:
        """BM25 score of one sentence for the query."""
        tf = self.term_freqs[doc_id]
        norm = self.k1 * (1 - self.b + self.b * self.lengths[doc_id] / (self.avg_length or 1.0))
        return sum(
            self.idf(term) * (tf[term] * (self.k1 + 1)) / (tf[term] + norm) for term in query_terms if tf.get(term)
        )

    def search(self, query: str, k: int = 5) -> list[Hit]:
        """Top ``k`` sentences; ties break on position so results are stable."""
        terms = content_tokens(query)
        if not terms or not self.sentences:
            return []
        candidates = {doc for term in terms for doc in self.postings.get(term, ())}
        scored = sorted(((self.score_doc(d, terms), d) for d in candidates), key=lambda p: (-p[0], p[1]))
        return [
            Hit(
                d,
                float(score),
                self.sentences[d].start,
                self.sentences[d].end,
                self.sentences[d].text,
                self.sentences[d].speaker,
            )
            for score, d in scored[:k]
            if score > 0
        ]
