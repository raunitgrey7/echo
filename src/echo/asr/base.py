"""The transcriber contract every backend implements.

Why: the pipeline, the API and the benchmark must not care which engine produced a transcript.
A tiny protocol with frozen dataclasses keeps the boundary explicit and lets the deterministic
mock stand in for whisper wherever no model is cached, including CI on Linux runners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from echo.audio.io import Audio


@dataclass(frozen=True, slots=True)
class Word:
    """One recognised word with its time span."""

    text: str
    start: float
    end: float
    probability: float = 1.0
    prefix_space: bool = True


@dataclass(frozen=True, slots=True)
class Segment:
    """A run of words the recogniser emitted together, usually a sentence or clause."""

    id: int
    start: float
    end: float
    text: str
    words: tuple[Word, ...] = field(default_factory=tuple)

    @property
    def duration(self) -> float:
        """Length in seconds."""
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Everything a backend returns, including how long it took."""

    segments: tuple[Segment, ...]
    language: str
    backend: str
    model: str
    elapsed_s: float
    audio_s: float

    @property
    def text(self) -> str:
        """Segments joined into one string."""
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    @property
    def real_time_factor(self) -> float:
        """Processing seconds per audio second; below 1.0 is faster than real time."""
        return self.elapsed_s / self.audio_s if self.audio_s > 0 else 0.0


@runtime_checkable
class Transcriber(Protocol):
    """What the pipeline needs from an ASR engine."""

    name: str
    model: str

    def transcribe(self, audio: Audio, language: str = "en") -> TranscriptionResult:
        """Recognise 16 kHz mono audio."""
        ...
