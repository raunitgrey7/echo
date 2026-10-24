"""Build the benchmark dataset: scripts become recordings with sample-accurate labels.

Why: the benchmark must be reproducible and its labels must be certain. Every utterance is
synthesised once per (voice, rate, text) and cached; recordings are composed from cached
utterances with seeded silence gaps, degraded with a seeded condition, written as WAV and
described by a JSON manifest that records speech spans, speakers, action items, key terms,
queries and the SHA-256 of the audio. Inter-sentence gaps (0.7 to 1.4 s) are deliberately longer
than the pauses ``say`` puts at commas so sentence spans are a consistent reference for the VAD.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from echo.audio.degrade import CLEAN, Condition, apply_condition
from echo.audio.io import SAMPLE_RATE, Audio, float_to_pcm16, read_wav, silence, write_wav
from echo.audio.synth import DEFAULT_RATE_WPM, synthesize
from echo.corpus.scripts import LECTURES, MEETINGS, Lecture, Meeting, lecture_by_id, meeting_by_id

VOICES: tuple[str, ...] = ("Samantha", "Daniel", "Karen", "Moira", "Rishi")
NOISE_VOICES: tuple[str, ...] = ("Samantha", "Daniel", "Rishi")
MEETING_VOICE_SETS: tuple[tuple[str, ...], ...] = (
    ("Samantha", "Daniel", "Karen"),
    ("Daniel", "Moira", "Rishi"),
    ("Karen", "Rishi", "Samantha"),
    ("Moira", "Samantha", "Daniel"),
)
BABBLE_VOICES: tuple[str, ...] = ("Tessa", "Fred", "Aman")
RATES: tuple[int, ...] = (140, 220)
CONDITIONS: tuple[Condition, ...] = (
    Condition("white_20db", "white", 20.0),
    Condition("white_10db", "white", 10.0),
    Condition("white_5db", "white", 5.0),
    Condition("white_0db", "white", 0.0),
    Condition("pink_10db", "pink", 10.0),
    Condition("pink_5db", "pink", 5.0),
    Condition("babble_10db", "babble", 10.0),
    Condition("babble_5db", "babble", 5.0),
    Condition("reverb_rt60_0.4s", rt60_s=0.4),
    Condition("reverb_rt60_0.8s", rt60_s=0.8),
)
TRIM_DB = -40.0


@dataclass(frozen=True, slots=True)
class UtteranceLabel:
    """Ground truth for one synthesised sentence or turn inside a recording."""

    index: int
    speaker: str
    voice: str
    text: str
    start: float
    end: float
    action: bool = False
    owner: str | None = None
    due: str | None = None


@dataclass(frozen=True, slots=True)
class RecordingSpec:
    """What to build: a script, voices, a speaking rate, an acoustic condition and a seed."""

    id: str
    kind: str
    script_id: str
    voices: tuple[str, ...]
    rate_wpm: int = DEFAULT_RATE_WPM
    condition: Condition = CLEAN
    seed: int = 0
    axis: str = "voice"


@dataclass(frozen=True, slots=True)
class RecordingManifest:
    """Everything the benchmark needs to score one recording."""

    id: str
    kind: str
    script_id: str
    axis: str
    voices: tuple[str, ...]
    rate_wpm: int
    condition: dict[str, object]
    seed: int
    path: str
    sha256: str
    duration_s: float
    sample_rate: int
    reference_text: str
    utterances: tuple[UtteranceLabel, ...]
    speaker_changes: tuple[float, ...]
    key_terms: tuple[str, ...] = field(default_factory=tuple)
    key_sentences: tuple[int, ...] = field(default_factory=tuple)
    queries: tuple[dict[str, object], ...] = field(default_factory=tuple)

    @property
    def n_speakers(self) -> int:
        """Distinct speakers in the reference."""
        return len({u.speaker for u in self.utterances})


def utterance_bounds(samples: np.ndarray, sample_rate: int, trim_db: float = TRIM_DB) -> tuple[float, float]:
    """First and last sample above ``trim_db`` relative to the peak, in seconds."""
    magnitude = np.abs(samples.astype(np.float64))
    peak = float(magnitude.max()) if magnitude.size else 0.0
    if peak <= 0.0:
        return 0.0, 0.0
    above = np.flatnonzero(magnitude >= peak * 10.0 ** (trim_db / 20.0))
    return float(above[0]) / sample_rate, float(above[-1] + 1) / sample_rate


def script_turns(spec: RecordingSpec) -> list[tuple[str, str, str, bool, str | None, str | None]]:
    """(speaker, voice, text, action, owner, due) for each utterance of the script."""
    if spec.kind == "lecture":
        lecture = lecture_by_id(spec.script_id)
        return [("speaker_0", spec.voices[0], s, False, None, None) for s in lecture.sentences]
    meeting = meeting_by_id(spec.script_id)
    voice_of = dict(zip(meeting.speakers, spec.voices, strict=False))
    return [(t.speaker, voice_of[t.speaker], t.text, t.action, t.owner, t.due) for t in meeting.turns]


def gap_range(kind: str) -> tuple[float, float]:
    """Silence between utterances: lectures breathe longer than meeting turns."""
    return (0.7, 1.4) if kind == "lecture" else (0.5, 1.2)


def compose(spec: RecordingSpec, cache_dir: Path, rng: np.random.Generator) -> tuple[np.ndarray, list[UtteranceLabel]]:
    """Concatenate cached utterances with seeded gaps; returns clean samples and labels."""
    parts: list[np.ndarray] = [silence(float(rng.uniform(0.3, 0.6)))]
    labels: list[UtteranceLabel] = []
    cursor = parts[0].shape[0] / SAMPLE_RATE
    low, high = gap_range(spec.kind)
    for index, (speaker, voice, text, action, owner, due) in enumerate(script_turns(spec)):
        audio = synthesize(text, voice, spec.rate_wpm, cache_dir)
        first, last = utterance_bounds(audio.samples, audio.sample_rate)
        labels.append(UtteranceLabel(index, speaker, voice, text, cursor + first, cursor + last, action, owner, due))
        gap = silence(float(rng.uniform(low, high)))
        parts.extend([audio.samples, gap])
        cursor += audio.duration + gap.shape[0] / SAMPLE_RATE
    return np.concatenate(parts).astype(np.float32), labels


def babble_sources(cache_dir: Path) -> list[np.ndarray]:
    """Sentences from voices outside the benchmark set, so babble never contains the target text."""
    sources: list[np.ndarray] = []
    for voice, lecture in zip(BABBLE_VOICES, LECTURES[3:6], strict=False):
        sources.append(synthesize(lecture.text, voice, DEFAULT_RATE_WPM, cache_dir).samples)
    return sources


def speaker_changes(labels: list[UtteranceLabel]) -> tuple[float, ...]:
    """Midpoints of gaps where the speaker changes."""
    return tuple(
        (labels[i - 1].end + labels[i].start) / 2.0
        for i in range(1, len(labels))
        if labels[i].speaker != labels[i - 1].speaker
    )


def script_extras(spec: RecordingSpec) -> tuple[tuple[str, ...], tuple[int, ...], tuple[dict[str, object], ...]]:
    """Key terms, key sentences and queries for the script behind a spec."""
    script: Lecture | Meeting = (
        lecture_by_id(spec.script_id) if spec.kind == "lecture" else meeting_by_id(spec.script_id)
    )
    queries = tuple({"text": q.text, "answer_index": q.answer_index} for q in script.queries)
    if isinstance(script, Lecture):
        return script.key_terms, script.key_sentences, queries
    return (), (), queries


def build_recording(spec: RecordingSpec, data_dir: Path, babble: list[np.ndarray] | None = None) -> RecordingManifest:
    """Build one recording and its manifest, reusing files whose checksum still matches."""
    recordings_dir = data_dir / "recordings"
    wav_path = recordings_dir / f"{spec.id}.wav"
    json_path = recordings_dir / f"{spec.id}.json"
    if wav_path.exists() and json_path.exists():
        existing = load_manifest(json_path)
        if existing.sha256 == sha256_file(wav_path):
            return existing
    rng = np.random.default_rng(spec.seed)
    clean, labels = compose(spec, data_dir / "cache", rng)
    degraded = apply_condition(clean, spec.condition, SAMPLE_RATE, rng, babble)
    write_wav(wav_path, Audio(degraded, SAMPLE_RATE))
    key_terms, key_sentences, queries = script_extras(spec)
    manifest = RecordingManifest(
        id=spec.id,
        kind=spec.kind,
        script_id=spec.script_id,
        axis=spec.axis,
        voices=spec.voices,
        rate_wpm=spec.rate_wpm,
        condition=asdict(spec.condition),
        seed=spec.seed,
        path=str(wav_path.relative_to(data_dir)),
        sha256=sha256_file(wav_path),
        duration_s=degraded.shape[0] / SAMPLE_RATE,
        sample_rate=SAMPLE_RATE,
        reference_text=" ".join(label.text for label in labels),
        utterances=tuple(labels),
        speaker_changes=speaker_changes(labels),
        key_terms=key_terms,
        key_sentences=key_sentences,
        queries=queries,
    )
    json_path.write_text(json.dumps(asdict(manifest), indent=1), encoding="utf-8")
    return manifest


def sha256_file(path: Path) -> str:
    """Checksum of a file on disk."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audio_sha256(audio: Audio) -> str:
    """Checksum of the PCM rendering, matching what the mock oracle computes."""
    return hashlib.sha256(float_to_pcm16(audio.samples)).hexdigest()


def load_manifest(path: Path) -> RecordingManifest:
    """Read a manifest JSON back into dataclasses."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["utterances"] = tuple(UtteranceLabel(**u) for u in raw["utterances"])
    for key in ("voices", "speaker_changes", "key_terms", "key_sentences", "queries"):
        raw[key] = tuple(raw[key])
    return RecordingManifest(**raw)


def grid(profile: str = "full") -> list[RecordingSpec]:
    """The benchmark grid; ``quick`` is a small subset for CI and development."""
    specs: list[RecordingSpec] = []
    lectures = LECTURES if profile == "full" else LECTURES[:2]
    voices = VOICES if profile == "full" else VOICES[:2]
    for lecture in lectures:
        for voice in voices:
            specs.append(RecordingSpec(f"lec_{lecture.id}_{voice.lower()}", "lecture", lecture.id, (voice,)))
    noise_lectures = LECTURES[:2] if profile == "full" else LECTURES[:1]
    noise_voices = NOISE_VOICES if profile == "full" else VOICES[:1]
    conditions = CONDITIONS if profile == "full" else (CONDITIONS[1], CONDITIONS[6])
    for lecture in noise_lectures:
        for voice in noise_voices:
            for condition in conditions:
                specs.append(
                    RecordingSpec(
                        f"lec_{lecture.id}_{voice.lower()}_{condition.name}",
                        "lecture",
                        lecture.id,
                        (voice,),
                        condition=condition,
                        axis="noise",
                    )
                )
            if profile == "full":
                for rate in RATES:
                    specs.append(
                        RecordingSpec(
                            f"lec_{lecture.id}_{voice.lower()}_{rate}wpm",
                            "lecture",
                            lecture.id,
                            (voice,),
                            rate_wpm=rate,
                            axis="rate",
                        )
                    )
    meetings = MEETINGS if profile == "full" else MEETINGS[:1]
    for index, meeting in enumerate(meetings):
        voice_set = MEETING_VOICE_SETS[index % len(MEETING_VOICE_SETS)]
        specs.append(RecordingSpec(f"meet_{meeting.id}", "meeting", meeting.id, voice_set, axis="meeting"))
        if profile == "full":
            specs.append(
                RecordingSpec(
                    f"meet_{meeting.id}_white_10db",
                    "meeting",
                    meeting.id,
                    voice_set,
                    condition=CONDITIONS[1],
                    axis="meeting",
                )
            )
    return [RecordingSpec(**{**asdict_spec(s), "seed": seed_for(s.id)}) for s in specs]


def asdict_spec(spec: RecordingSpec) -> dict[str, object]:
    """Spec fields as a dict with the condition object kept intact."""
    return {
        "id": spec.id,
        "kind": spec.kind,
        "script_id": spec.script_id,
        "voices": spec.voices,
        "rate_wpm": spec.rate_wpm,
        "condition": spec.condition,
        "seed": spec.seed,
        "axis": spec.axis,
    }


def seed_for(recording_id: str) -> int:
    """Stable seed per recording so gaps and noise never depend on build order."""
    return int.from_bytes(hashlib.sha256(recording_id.encode()).digest()[:4], "big")


def build_dataset(profile: str, data_dir: Path) -> list[RecordingManifest]:
    """Build every recording of the profile and write ``manifest_<profile>.json``."""
    specs = grid(profile)
    babble = babble_sources(data_dir / "cache") if any(s.condition.noise == "babble" for s in specs) else None
    manifests = [build_recording(spec, data_dir, babble) for spec in specs]
    index = {
        "profile": profile,
        "sample_rate": SAMPLE_RATE,
        "recordings": [{"id": m.id, "path": m.path, "sha256": m.sha256, "duration_s": m.duration_s} for m in manifests],
        "total_audio_s": sum(m.duration_s for m in manifests),
    }
    (data_dir / f"manifest_{profile}.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return manifests


def load_dataset(profile: str, data_dir: Path) -> list[RecordingManifest]:
    """Load manifests of a built profile, verifying checksums; raises when a file is missing."""
    index = json.loads((data_dir / f"manifest_{profile}.json").read_text(encoding="utf-8"))
    manifests: list[RecordingManifest] = []
    for entry in index["recordings"]:
        manifest = load_manifest(data_dir / "recordings" / f"{entry['id']}.json")
        actual = sha256_file(data_dir / manifest.path)
        if actual != manifest.sha256:
            raise ValueError(f"checksum mismatch for {manifest.id}: {actual} != {manifest.sha256}")
        manifests.append(manifest)
    return manifests


def oracle_for(manifests: list[RecordingManifest], data_dir: Path) -> dict[str, str]:
    """Map audio checksum to reference text for the mock transcriber."""
    table: dict[str, str] = {}
    for manifest in manifests:
        table[audio_sha256(read_wav(data_dir / manifest.path))] = manifest.reference_text
    return table
