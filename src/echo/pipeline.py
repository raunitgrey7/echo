"""End-to-end analysis: audio in, every product out, with per-stage timings.

Why: the API, the CLI and the benchmark must run exactly the same steps in the same order so a
number measured by the benchmark describes what the API serves. The order matters: VAD regions
feed both the speaker clustering and the mock transcriber; speakers are attached to ASR segments
by time overlap; keywords feed the section titles of the notes; sentences from the transcript are
the unit of the search index.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from echo.asr.base import Transcriber, TranscriptionResult
from echo.audio.io import Audio, ensure_16k
from echo.diarization.change import DEFAULT_SPEAKER, SpeakerConfig, SpeakerResult, label_speakers
from echo.products.actions import ActionItem, detect_actions
from echo.products.keywords import Keyword, extract_keywords
from echo.products.notes import Notes, build_notes, notes_to_markdown
from echo.products.search import SearchIndex
from echo.products.transcript import Transcript, assign_speakers, to_dict
from echo.vad.energy import DEFAULT_VAD, SpeechRegion, VadConfig, detect_speech, speech_ratio


@dataclass(frozen=True, slots=True)
class PipelineOptions:
    """Knobs shared by every entry point."""

    language: str = "en"
    top_keywords: int = 10
    summary_ratio: float = 0.3
    vad: VadConfig = DEFAULT_VAD
    speakers: SpeakerConfig = DEFAULT_SPEAKER
    detect_speakers: bool = True


DEFAULT_OPTIONS = PipelineOptions()


@dataclass(frozen=True, slots=True)
class Analysis:
    """Everything echo knows about one recording."""

    transcript: Transcript
    regions: tuple[SpeechRegion, ...]
    speakers: SpeakerResult
    keywords: tuple[Keyword, ...]
    notes: Notes
    actions: tuple[ActionItem, ...]
    asr: TranscriptionResult
    timings_s: dict[str, float] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Audio length in seconds."""
        return self.transcript.duration

    @property
    def total_elapsed_s(self) -> float:
        """Wall time of the whole analysis."""
        return sum(self.timings_s.values())

    @property
    def real_time_factor(self) -> float:
        """Processing seconds per audio second for the full pipeline."""
        return self.total_elapsed_s / self.duration if self.duration > 0 else 0.0

    def search_index(self) -> SearchIndex:
        """Build the BM25 index over the transcript's sentences."""
        return SearchIndex.from_transcript(self.transcript)

    def notes_markdown(self, title: str = "Notes") -> str:
        """Rendered notes."""
        return notes_to_markdown(self.notes, title)


def _timed[T](timings: dict[str, float], name: str, step: Callable[[], T]) -> T:
    started = time.perf_counter()
    value = step()
    timings[name] = time.perf_counter() - started
    return value


def analyze(audio: Audio, transcriber: Transcriber, options: PipelineOptions = DEFAULT_OPTIONS) -> Analysis:
    """Run the full pipeline on one recording."""
    timings: dict[str, float] = {}
    audio16 = _timed(timings, "resample", lambda: ensure_16k(audio))
    regions = _timed(timings, "vad", lambda: detect_speech(audio16, options.vad))
    asr = _timed(timings, "asr", lambda: transcriber.transcribe(audio16, options.language))
    if options.detect_speakers:
        speakers = _timed(timings, "speakers", lambda: label_speakers(audio16, regions, options.speakers))
    else:
        speakers = SpeakerResult(tuple(0 for _ in regions), 1 if regions else 0, (), float("inf"))
    transcript = assign_speakers(asr, regions, speakers.labels)
    keywords = _timed(timings, "keywords", lambda: extract_keywords(transcript, options.top_keywords))
    notes = _timed(timings, "notes", lambda: build_notes(transcript, keywords, options.summary_ratio))
    actions = _timed(timings, "actions", lambda: detect_actions(transcript))
    return Analysis(transcript, tuple(regions), speakers, tuple(keywords), notes, tuple(actions), asr, timings)


def plain(value: object) -> object:
    """Recursively turn tuples into lists and numpy scalars into Python scalars for JSON."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def analysis_to_dict(analysis: Analysis) -> dict[str, Any]:
    """JSON-ready view used by the API and the CLI."""
    payload = {
        "duration": analysis.duration,
        "speech_ratio": speech_ratio(list(analysis.regions), analysis.duration),
        "transcript": to_dict(analysis.transcript),
        "regions": [asdict(r) for r in analysis.regions],
        "speakers": {
            "count": analysis.speakers.n_speakers,
            "changes": list(analysis.speakers.changes),
            "labels": list(analysis.speakers.labels),
        },
        "keywords": [asdict(k) for k in analysis.keywords],
        "notes": asdict(analysis.notes),
        "notes_markdown": analysis.notes_markdown(),
        "actions": [asdict(a) for a in analysis.actions],
        "timings_s": dict(analysis.timings_s),
        "real_time_factor": analysis.real_time_factor,
        "backend": analysis.asr.backend,
        "model": analysis.asr.model,
        "n_speakers": analysis.speakers.n_speakers,
        "n_actions": len(analysis.actions),
    }
    result = plain(payload)
    assert isinstance(result, dict)
    return result
