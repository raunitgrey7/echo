"""Scripts integrity, dataset builder pieces and the ASR adapters."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from echo.asr.base import Segment, TranscriptionResult, Word
from echo.asr.mock import PLACEHOLDER, MockTranscriber, audio_digest, corrupt_words, spread_words
from echo.asr.registry import resolve_transcriber
from echo.asr.whisper import WhisperTranscriber, model_cached
from echo.corpus.build import (
    CONDITIONS,
    RecordingSpec,
    UtteranceLabel,
    gap_range,
    grid,
    script_turns,
    seed_for,
    speaker_changes,
    utterance_bounds,
)
from echo.corpus.scripts import LECTURES, MEETINGS, lecture_by_id, meeting_by_id
from echo.vad.energy import SpeechRegion


def test_scripts_are_well_formed():
    assert len(LECTURES) == 6 and len(MEETINGS) == 4
    for lecture in LECTURES:
        assert len(lecture.sentences) == 10
        assert all(0 <= i < 10 for i in lecture.key_sentences)
        assert all(0 <= q.answer_index < 10 for q in lecture.queries)
        assert len(lecture.key_terms) == 10
        assert chr(0x2014) not in lecture.text and chr(0x2013) not in lecture.text
    for meeting in MEETINGS:
        assert len(meeting.speakers) == 3
        assert any(turn.action for turn in meeting.turns)
        assert all(0 <= q.answer_index < len(meeting.turns) for q in meeting.queries)
        assert all(turn.owner in (None, *meeting.speakers) for turn in meeting.turns)


def test_lookups():
    assert lecture_by_id("newton").title.startswith("Newton")
    assert meeting_by_id("incident_review").turns[0].speaker == "Grace"
    with pytest.raises(KeyError):
        lecture_by_id("nope")
    with pytest.raises(KeyError):
        meeting_by_id("nope")


def test_grid_sizes_and_unique_ids():
    full, quick = grid("full"), grid("quick")
    assert len(full) == 110 and len(quick) == 7
    assert len({s.id for s in full}) == len(full)
    assert {s.axis for s in full} == {"voice", "noise", "rate", "meeting"}
    assert all(s.seed == seed_for(s.id) for s in full)
    assert sum(1 for s in full if s.condition.noise == "babble") == 12
    assert len(CONDITIONS) == 10


def test_script_turns_and_speaker_changes():
    spec = RecordingSpec("m", "meeting", "course_staff", ("Samantha", "Daniel", "Karen"))
    turns = script_turns(spec)
    assert turns[0][0] == "Priya" and turns[0][1] == "Samantha"
    assert turns[2][3] is True and turns[2][4] == "Tom"
    labels = [UtteranceLabel(i, t[0], t[1], t[2], float(i), float(i) + 0.5) for i, t in enumerate(turns)]
    changes = speaker_changes(labels)
    assert len(changes) == sum(1 for i in range(1, len(turns)) if turns[i][0] != turns[i - 1][0])
    assert gap_range("lecture")[0] > gap_range("meeting")[0]
    lecture_spec = RecordingSpec("l", "lecture", "newton", ("Daniel",))
    assert all(t[0] == "speaker_0" for t in script_turns(lecture_spec))


def test_utterance_bounds_trim_silence():
    sr = 16_000
    samples = np.zeros(sr, dtype=np.float32)
    samples[4000:12_000] = 0.5
    start, end = utterance_bounds(samples, sr)
    assert start == pytest.approx(0.25) and end == pytest.approx(0.75)
    assert utterance_bounds(np.zeros(10, dtype=np.float32), sr) == (0.0, 0.0)


def test_mock_without_oracle_emits_placeholders(samantha):
    result = MockTranscriber().transcribe(samantha)
    assert result.backend == "mock"
    assert all(seg.text == PLACEHOLDER for seg in result.segments)
    assert result.audio_s == pytest.approx(samantha.duration)
    assert result.real_time_factor < 0.5


def test_mock_with_oracle_is_deterministic_and_corrupts_at_rate(two_speakers):
    reference = "the quick brown fox jumps over the lazy dog " * 20
    oracle = {audio_digest(two_speakers): reference.strip()}
    clean = MockTranscriber(oracle=lambda a: oracle.get(audio_digest(a)), error_rate=0.0)
    assert clean.transcribe(two_speakers).text.split() == reference.split()
    noisy_a = MockTranscriber(oracle=lambda a: oracle.get(audio_digest(a)), error_rate=0.2, seed=1).transcribe(
        two_speakers
    )
    noisy_b = MockTranscriber(oracle=lambda a: oracle.get(audio_digest(a)), error_rate=0.2, seed=1).transcribe(
        two_speakers
    )
    other = MockTranscriber(oracle=lambda a: oracle.get(audio_digest(a)), error_rate=0.2, seed=2).transcribe(
        two_speakers
    )
    assert noisy_a.text == noisy_b.text
    assert noisy_a.text != other.text
    assert noisy_a.text != clean.transcribe(two_speakers).text
    assert len(noisy_a.segments) == 4


def test_corrupt_and_spread_helpers():
    words = ["alpha", "beta", "gamma", "delta"]
    assert corrupt_words(words, 0.0, np.random.default_rng(0)) == words
    corrupted = corrupt_words(words * 50, 1.0, np.random.default_rng(0))
    assert corrupted != words * 50
    regions = [SpeechRegion(0, 1), SpeechRegion(2, 3)]
    segments = spread_words(["a", "b", "c", "d"], regions)
    assert [s.text for s in segments] == ["a b", "c d"]
    assert segments[1].words[0].start == pytest.approx(2.0)
    assert spread_words(["a"], []) == []


def test_transcription_result_properties():
    seg = Segment(0, 0.0, 1.0, "hello there", (Word("hello", 0, 0.5), Word("there", 0.5, 1.0)))
    result = TranscriptionResult((seg, Segment(1, 1.0, 2.0, "", ())), "en", "x", "y", 0.5, 2.0)
    assert result.text == "hello there"
    assert result.real_time_factor == 0.25
    assert TranscriptionResult((), "en", "x", "y", 0.5, 0.0).real_time_factor == 0.0


def test_registry_mock_and_errors(monkeypatch):
    monkeypatch.delenv("ECHO_ASR_BACKEND", raising=False)
    assert resolve_transcriber("mock").name == "mock"
    monkeypatch.setenv("ECHO_ASR_BACKEND", "mock")
    assert resolve_transcriber().name == "mock"
    with pytest.raises(ValueError):
        resolve_transcriber("nonsense")
    monkeypatch.setenv("ECHO_ALLOW_DOWNLOAD", "0")
    if not model_cached("medium"):
        with pytest.raises(RuntimeError):
            resolve_transcriber("whisper", "medium")


def test_registry_auto_falls_back_without_model(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setenv("ECHO_ALLOW_DOWNLOAD", "0")
    assert not model_cached("base")
    assert resolve_transcriber("auto").name == "mock"


@pytest.mark.whisper
@pytest.mark.skipif(not WhisperTranscriber.available("base"), reason="faster-whisper base not cached")
def test_whisper_transcribes_fixture_and_is_deterministic(samantha):
    engine = WhisperTranscriber("base", cpu_threads=2)
    first = engine.transcribe(samantha)
    second = engine.transcribe(samantha)
    assert first.text == second.text
    assert "water" in first.text.lower()
    assert first.segments and first.segments[0].words
    assert first.segments[0].words[0].end >= first.segments[0].words[0].start
    assert first.backend == "faster-whisper" and first.model == "base-int8"


def test_manifest_json_round_trip(tmp_path):
    from dataclasses import asdict

    from echo.corpus.build import RecordingManifest, load_manifest

    manifest = RecordingManifest(
        id="x",
        kind="lecture",
        script_id="newton",
        axis="voice",
        voices=("Daniel",),
        rate_wpm=175,
        condition={"name": "clean", "noise": "none", "snr_db": None, "rt60_s": None},
        seed=1,
        path="recordings/x.wav",
        sha256="abc",
        duration_s=1.0,
        sample_rate=16_000,
        reference_text="hi",
        utterances=(UtteranceLabel(0, "speaker_0", "Daniel", "hi", 0.0, 0.5),),
        speaker_changes=(),
        key_terms=("a",),
        key_sentences=(0,),
        queries=({"text": "q", "answer_index": 0},),
    )
    path = tmp_path / "x.json"
    path.write_text(json.dumps(asdict(manifest)))
    loaded = load_manifest(path)
    assert loaded == manifest
    assert loaded.n_speakers == 1
    assert "ECHO_ASR_BACKEND" not in os.environ or isinstance(Path(os.environ.get("ECHO_ASR_BACKEND", "")), Path)
