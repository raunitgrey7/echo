"""The timestamped transcript and its export formats.

Why: the transcript is the product every other product is built from, so it carries speakers
and word timings in one immutable structure. SRT and VTT exist for players and captioning tools,
plain text for reading, and the dict form feeds the API and the dashboard.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from echo.asr.base import TranscriptionResult, Word
from echo.vad.energy import SpeechRegion


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """A recognised segment with its speaker label."""

    id: int
    start: float
    end: float
    text: str
    speaker: str
    words: tuple[Word, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class Transcript:
    """Ordered segments plus provenance."""

    segments: tuple[TranscriptSegment, ...]
    duration: float
    language: str
    backend: str
    model: str

    @property
    def text(self) -> str:
        """All segment texts joined."""
        return " ".join(s.text for s in self.segments if s.text)

    @property
    def speakers(self) -> tuple[str, ...]:
        """Distinct speaker labels in order of first appearance."""
        seen: list[str] = []
        for segment in self.segments:
            if segment.speaker not in seen:
                seen.append(segment.speaker)
        return tuple(seen)

    @property
    def words(self) -> tuple[Word, ...]:
        """All words in time order."""
        return tuple(w for s in self.segments for w in s.words)


def speaker_name(index: int) -> str:
    """Human-facing label for a cluster index."""
    return f"Speaker {index + 1}"


def assign_speakers(result: TranscriptionResult, regions: list[SpeechRegion], labels: tuple[int, ...]) -> Transcript:
    """Give each ASR segment the label of the speech region it overlaps most."""
    segments: list[TranscriptSegment] = []
    for segment in result.segments:
        span = SpeechRegion(segment.start, segment.end)
        best, best_overlap = 0, -1.0
        for region, label in zip(regions, labels, strict=True):
            shared = span.overlap(region)
            if shared > best_overlap:
                best, best_overlap = label, shared
        segments.append(
            TranscriptSegment(segment.id, segment.start, segment.end, segment.text, speaker_name(best), segment.words)
        )
    return Transcript(tuple(segments), result.audio_s, result.language, result.backend, result.model)


def format_timestamp(seconds: float, separator: str = ",") -> str:
    """``HH:MM:SS,mmm`` (SRT) or ``HH:MM:SS.mmm`` (VTT)."""
    total_ms = round(max(seconds, 0.0) * 1000.0)
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def to_srt(transcript: Transcript) -> str:
    """SubRip captions, one cue per segment, speaker as a prefix."""
    blocks = [
        f"{i}\n{format_timestamp(s.start)} --> {format_timestamp(s.end)}\n{s.speaker}: {s.text}\n"
        for i, s in enumerate(transcript.segments, start=1)
    ]
    return "\n".join(blocks)


def to_vtt(transcript: Transcript) -> str:
    """WebVTT captions with voice tags, which browsers render natively."""
    cues = [
        f"{format_timestamp(s.start, '.')} --> {format_timestamp(s.end, '.')}\n<v {s.speaker}>{s.text}\n"
        for s in transcript.segments
    ]
    return "WEBVTT\n\n" + "\n".join(cues)


def to_text(transcript: Transcript) -> str:
    """Readable transcript; a speaker line starts whenever the speaker changes."""
    lines: list[str] = []
    current: str | None = None
    for segment in transcript.segments:
        if segment.speaker != current:
            current = segment.speaker
            lines.append(f"\n[{format_timestamp(segment.start, '.')[:-4]}] {segment.speaker}:")
        lines.append(segment.text)
    return " ".join(lines).strip()


def to_dict(transcript: Transcript) -> dict[str, object]:
    """JSON-ready form."""
    return {
        "duration": transcript.duration,
        "language": transcript.language,
        "backend": transcript.backend,
        "model": transcript.model,
        "speakers": list(transcript.speakers),
        "segments": [asdict(s) for s in transcript.segments],
    }
