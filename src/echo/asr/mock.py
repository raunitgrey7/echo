"""A deterministic transcriber that needs no model.

Why: tests, CI on Linux and the dashboard demo must run without a 150 MB download. The mock is
honest about what it is: given an oracle that knows the reference text of a recording (the
benchmark manifest provides one, keyed by audio checksum) it emits that text spread over the
detected speech regions, optionally corrupted at a seeded word error rate so WER tests have
non-trivial values. Without an oracle it emits one placeholder token per speech region. Any
benchmark row produced with this backend is labelled ``mock`` and never presented as recognition.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable

import numpy as np

from echo.asr.base import Segment, TranscriptionResult, Word
from echo.audio.io import Audio, float_to_pcm16
from echo.vad.energy import DEFAULT_VAD, SpeechRegion, VadConfig, detect_speech

Oracle = Callable[[Audio], str | None]
PLACEHOLDER = "[speech]"
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def audio_digest(audio: Audio) -> str:
    """SHA-256 of the 16-bit PCM rendering; the key the benchmark manifest uses."""
    return hashlib.sha256(float_to_pcm16(audio.samples)).hexdigest()


def corrupt_words(words: list[str], error_rate: float, rng: np.random.Generator) -> list[str]:
    """Apply substitutions, deletions and insertions at ``error_rate`` per word, deterministically."""
    if error_rate <= 0.0:
        return list(words)
    out: list[str] = []
    for word in words:
        if rng.random() >= error_rate:
            out.append(word)
            continue
        op = rng.integers(3)
        if op == 0:
            out.append(word[1:] if len(word) > 3 else word + "s")
        elif op == 1:
            continue
        else:
            out.extend([word, word])
    return out


def spread_words(words: list[str], regions: list[SpeechRegion]) -> list[Segment]:
    """Distribute words over regions in proportion to region length, with uniform word timing."""
    if not regions:
        return []
    total = sum(r.duration for r in regions) or 1.0
    counts = [round(len(words) * r.duration / total) for r in regions]
    counts[-1] += len(words) - sum(counts)
    segments: list[Segment] = []
    cursor = 0
    for index, (region, count) in enumerate(zip(regions, counts, strict=True)):
        chunk = words[cursor : cursor + max(count, 0)]
        cursor += max(count, 0)
        if chunk:
            segments.append(_segment(index, region, chunk))
    return segments


def spread_sentences(sentences: list[list[str]], regions: list[SpeechRegion]) -> list[Segment]:
    """One sentence per region, which is how the corpus is composed; callers check the counts match."""
    return [
        _segment(index, region, words)
        for index, (region, words) in enumerate(zip(regions, sentences, strict=True))
        if words
    ]


def _segment(index: int, region: SpeechRegion, words: list[str]) -> Segment:
    step = region.duration / len(words)
    timed = tuple(Word(w, region.start + i * step, region.start + (i + 1) * step) for i, w in enumerate(words))
    return Segment(index, region.start, region.end, " ".join(words), timed)


def split_sentences(reference: str) -> list[list[str]]:
    """Reference text split at sentence punctuation into word lists."""
    return [part.split() for part in _SENTENCE_SPLIT.split(reference.strip()) if part.strip()]


class MockTranscriber:
    """Oracle-backed deterministic stand-in for a speech recogniser."""

    name = "mock"

    def __init__(
        self,
        oracle: Oracle | None = None,
        error_rate: float = 0.0,
        seed: int = 0,
        vad_config: VadConfig = DEFAULT_VAD,
    ) -> None:
        self.oracle = oracle
        self.error_rate = error_rate
        self.seed = seed
        self.vad_config = vad_config
        self.model = f"mock-oracle-wer{error_rate:.2f}"

    def transcribe(self, audio: Audio, language: str = "en") -> TranscriptionResult:
        """Emit the oracle text (or placeholders) over the detected speech regions.

        When the reference has exactly one sentence per speech region, each sentence lands in its
        own region so sentence spans line up with the true turns; otherwise words are spread by
        region duration.
        """
        started = time.perf_counter()
        regions = detect_speech(audio, self.vad_config)
        reference = self.oracle(audio) if self.oracle is not None else None
        if reference is None:
            segments = spread_words([PLACEHOLDER for _ in regions], regions)
        else:
            seed_material = f"{self.seed}:{audio_digest(audio)}".encode()
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big"))
            sentences = split_sentences(reference)
            if len(sentences) == len(regions):
                corrupted = [corrupt_words(words, self.error_rate, rng) for words in sentences]
                segments = spread_sentences(corrupted, regions)
            else:
                segments = spread_words(corrupt_words(reference.split(), self.error_rate, rng), regions)
        return TranscriptionResult(
            segments=tuple(segments),
            language=language,
            backend=self.name,
            model=self.model,
            elapsed_s=time.perf_counter() - started,
            audio_s=audio.duration,
        )
