"""Shared fixtures: synthetic signals, transcripts and the two committed speech clips."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from echo.asr.base import Word
from echo.audio.io import SAMPLE_RATE, Audio, read_wav
from echo.products.transcript import Transcript, TranscriptSegment

FIXTURES = Path(__file__).parent / "fixtures"


def harmonic_voice(duration_s: float, f0: float, sr: int = SAMPLE_RATE, tilt: float = 1.0, seed: int = 0) -> np.ndarray:
    """A voiced-speech stand-in: harmonics of f0 with a 1/n^tilt spectral slope, slight jitter, unit peak."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * sr)) / sr
    f0_track = f0 * (1.0 + 0.02 * np.sin(2 * np.pi * 3.0 * t))
    phase = 2 * np.pi * np.cumsum(f0_track) / sr
    out = np.zeros_like(t)
    for n in range(1, 25):
        if n * f0 < sr / 2 - 100:
            out += np.sin(n * phase + rng.uniform(0, 2 * np.pi)) / n**tilt
    envelope = 0.5 * (1 - np.cos(2 * np.pi * np.clip(t / duration_s, 0, 1))) ** 0.3
    out *= envelope
    return (out / np.max(np.abs(out))).astype(np.float32)


def speech_like(pattern: list[tuple[float, float]], f0: float = 150.0, sr: int = SAMPLE_RATE, seed: int = 0) -> Audio:
    """Alternating (speech_s, silence_s) bursts; returns Audio."""
    parts: list[np.ndarray] = []
    for index, (speech_s, silence_s) in enumerate(pattern):
        parts.append(0.5 * harmonic_voice(speech_s, f0, sr, seed=seed + index))
        parts.append(np.zeros(int(silence_s * sr), dtype=np.float32))
    return Audio(np.concatenate(parts).astype(np.float32), sr)


def make_transcript(sentences: list[str], speakers: list[str] | None = None, gap_s: float = 1.0) -> Transcript:
    """A transcript with evenly spaced word timings, one segment per sentence."""
    segments: list[TranscriptSegment] = []
    cursor = 0.0
    for index, sentence in enumerate(sentences):
        words: list[Word] = []
        for token in sentence.split():
            words.append(Word(token, cursor, cursor + 0.3))
            cursor += 0.3
        speaker = speakers[index % len(speakers)] if speakers else "Speaker 1"
        segments.append(TranscriptSegment(index, words[0].start, words[-1].end, sentence, speaker, tuple(words)))
        cursor += gap_s
    return Transcript(tuple(segments), cursor, "en", "test", "test")


@pytest.fixture(scope="session")
def samantha() -> Audio:
    """Committed clip: one Samantha sentence from the water cycle lecture."""
    return read_wav(FIXTURES / "samantha_water_cycle.wav")


@pytest.fixture(scope="session")
def daniel() -> Audio:
    """Committed clip: one Daniel sentence from the binary search lecture."""
    return read_wav(FIXTURES / "daniel_binary_search.wav")


@pytest.fixture(scope="session")
def two_speakers(samantha: Audio, daniel: Audio) -> Audio:
    """Samantha, Daniel, Samantha, Daniel with one-second gaps."""
    gap = np.zeros(SAMPLE_RATE, dtype=np.float32)
    parts = [gap[:8000], samantha.samples, gap, daniel.samples, gap, samantha.samples, gap, daniel.samples, gap[:8000]]
    return Audio(np.concatenate(parts).astype(np.float32), SAMPLE_RATE)


@pytest.fixture
def lecture_transcript() -> Transcript:
    """Reference text of the photosynthesis lecture as a transcript."""
    from echo.corpus.scripts import lecture_by_id

    return make_transcript(list(lecture_by_id("photosynthesis").sentences))


@pytest.fixture
def meeting_transcript() -> Transcript:
    """Reference text of the course staff meeting with its speakers."""
    from echo.corpus.scripts import meeting_by_id

    meeting = meeting_by_id("course_staff")
    names = {s: f"Speaker {i + 1}" for i, s in enumerate(meeting.speakers)}
    segments: list[TranscriptSegment] = []
    cursor = 0.0
    for index, turn in enumerate(meeting.turns):
        words = []
        for token in turn.text.split():
            words.append(Word(token, cursor, cursor + 0.3))
            cursor += 0.3
        segments.append(
            TranscriptSegment(index, words[0].start, words[-1].end, turn.text, names[turn.speaker], tuple(words))
        )
        cursor += 0.8
    return Transcript(tuple(segments), cursor, "en", "test", "test")
