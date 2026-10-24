"""End-to-end pipeline behaviour, determinism, robustness, and the benchmark harness pieces."""

from __future__ import annotations

import json

import numpy as np
import pytest

from echo.asr.mock import MockTranscriber, audio_digest
from echo.audio.io import Audio
from echo.bench import build_parser, check_thresholds, group_rows, lookup, pool, render_markdown
from echo.pipeline import PipelineOptions, analysis_to_dict, analyze


def oracle_mock(audio: Audio, text: str, error_rate: float = 0.0) -> MockTranscriber:
    table = {audio_digest(audio): text}
    return MockTranscriber(oracle=lambda a: table.get(audio_digest(a)), error_rate=error_rate)


def test_pipeline_end_to_end_on_two_speakers(two_speakers):
    text = (
        "The water cycle describes how water moves between the oceans. Binary search finds an item in a sorted array. "
        "Groundwater moves slowly through aquifers. Tom, please rewrite the rubric by Thursday."
    )
    analysis = analyze(two_speakers, oracle_mock(two_speakers, text))
    assert analysis.speakers.n_speakers == 2
    assert len(analysis.regions) == 4
    assert analysis.transcript.speakers == ("Speaker 1", "Speaker 2")
    assert analysis.keywords
    assert analysis.notes.n_sentences >= 3
    assert any("rubric" in a.text for a in analysis.actions)
    hits = analysis.search_index().search("aquifers")
    assert hits and "aquifers" in hits[0].text
    payload = analysis_to_dict(analysis)
    json.dumps(payload)
    assert payload["n_speakers"] == 2 and payload["transcript"]["segments"]
    assert set(analysis.timings_s) >= {"vad", "asr", "speakers", "keywords", "notes", "actions"}
    assert analysis.real_time_factor > 0


def test_same_input_same_output(two_speakers):
    text = "Repeatable words for the determinism check. Another sentence follows here. And a third one."
    first = analysis_to_dict(analyze(two_speakers, oracle_mock(two_speakers, text, error_rate=0.3)))
    second = analysis_to_dict(analyze(two_speakers, oracle_mock(two_speakers, text, error_rate=0.3)))
    for key in ("transcript", "regions", "speakers", "keywords", "notes", "actions"):
        assert first[key] == second[key], key


def test_pipeline_without_speaker_detection(samantha):
    analysis = analyze(samantha, MockTranscriber(), PipelineOptions(detect_speakers=False))
    assert analysis.speakers.n_speakers == 1
    assert "speakers" not in analysis.timings_s


@pytest.mark.parametrize(
    "samples",
    [
        np.zeros(16_000, dtype=np.float32),
        np.ones(16_000, dtype=np.float32),
        np.full(16_000, np.nan, dtype=np.float32),
        np.zeros(50, dtype=np.float32),
        (np.random.default_rng(0).standard_normal(32_000) * 5).astype(np.float32),
    ],
    ids=["silence", "dc-clipped", "nan", "tiny", "loud-noise"],
)
def test_pipeline_is_robust_to_degenerate_audio(samples):
    analysis = analyze(Audio(samples, 16_000), MockTranscriber())
    payload = analysis_to_dict(analysis)
    json.dumps(payload)
    assert payload["duration"] == pytest.approx(samples.shape[0] / 16_000)


def test_pipeline_resamples_other_rates(samantha):
    from echo.audio.io import resample

    audio_8k = resample(samantha, 8000)
    analysis = analyze(audio_8k, MockTranscriber())
    assert analysis.duration == pytest.approx(samantha.duration, abs=0.01)
    assert len(analysis.regions) >= 1


def _row(**overrides):
    base = {
        "id": "r",
        "kind": "lecture",
        "axis": "voice",
        "script_id": "photosynthesis",
        "voices": ["Samantha"],
        "rate_wpm": 175,
        "condition": "clean",
        "duration_s": 60.0,
        "ref_words": 100,
        "substitutions": 5,
        "deletions": 3,
        "insertions": 2,
        "wer": 0.1,
        "ref_chars": 500,
        "char_errors": 10,
        "cer": 0.02,
        "vad_precision": 0.9,
        "vad_recall": 1.0,
        "vad_f1": 0.95,
        "n_speakers_ref": 1,
        "n_speakers_hyp": 1,
        "speaker_count_correct": True,
        "change_hits": 0,
        "change_n_ref": 0,
        "change_n_hyp": 0,
        "change_f1": 1.0,
        "purity": 1.0,
        "keyword_hits": 4,
        "keyword_total": 10,
        "key_sentence_hits": 2,
        "key_sentence_total": 3,
        "search_ranks": [1, 2, 0],
        "asr_s": 3.0,
        "vad_s": 0.1,
        "speakers_s": 0.2,
        "products_s": 0.05,
        "total_s": 3.4,
    }
    base.update(overrides)
    return base


def test_pool_uses_pooled_rates_and_latency():
    rows = [
        _row(),
        _row(
            id="b",
            ref_words=300,
            substitutions=0,
            deletions=0,
            insertions=0,
            wer=0.0,
            duration_s=120.0,
            asr_s=6.0,
            total_s=6.8,
        ),
    ]
    pooled = pool(rows)
    assert pooled["n_recordings"] == 2
    assert pooled["wer"] == pytest.approx(10 / 400)
    assert pooled["asr_rtf"] == pytest.approx(9.0 / 180.0)
    assert pooled["asr_s_per_audio_min"] == pytest.approx(9.0 / 3.0)
    assert pooled["keyword_recall_at_10"] == pytest.approx(0.4)
    assert pooled["search_hit_at_1"] == pytest.approx(2 / 6) and pooled["search_queries"] == 6
    assert pool([]) == {"n_recordings": 0}


def test_group_rows_and_markdown():
    rows = [
        _row(),
        _row(id="n", axis="noise", condition="white_10db", voices=["Daniel"]),
        _row(id="r220", axis="rate", rate_wpm=220),
        _row(
            id="m",
            kind="meeting",
            axis="meeting",
            script_id="course_staff",
            n_speakers_ref=3,
            n_speakers_hyp=3,
            change_hits=8,
            change_n_ref=10,
            change_n_hyp=9,
            action_tp=5,
            action_fp=1,
            action_fn=2,
            owner_hits=3,
            owner_total=4,
        ),
    ]
    groups = group_rows(rows)
    assert groups["voice_axis"]["Samantha"]["n_recordings"] == 1
    assert groups["condition_axis"]["white_10db"]["n_recordings"] == 1
    assert groups["rate_axis"]["220"]["n_recordings"] == 1
    meetings = groups["meetings"]["overall"]
    assert meetings["action_precision"] == pytest.approx(5 / 6) and meetings["change_recall"] == pytest.approx(0.8)
    results = {
        "meta": {
            "profile": "quick",
            "backend": "mock",
            "model": "m",
            "n_recordings": 4,
            "total_audio_s": 240.0,
            "cpu_threads": 4,
            "platform": "test",
            "timestamp_utc": "now",
            "is_mock": True,
            "model_load_s": 0.0,
            "git_commit": None,
        },
        "groups": groups,
        "recordings": rows,
    }
    markdown = render_markdown(results)
    assert "Mock backend" in markdown and "| Voice |" in markdown and "white_10db" in markdown
    assert lookup(results, "groups.meetings.overall.action_f1") == pytest.approx(meetings["action_f1"])


def test_threshold_checks():
    results = {"meta": {"backend": "mock"}, "groups": {"overall": {"wer": 0.05, "vad_f1": 0.9}}}
    thresholds = {
        "mock": {
            "groups.overall.wer": {"max": 0.1},
            "groups.overall.vad_f1": {"min": 0.95},
            "groups.overall.missing": {"min": 1},
        }
    }
    failures = check_thresholds(results, thresholds)
    assert len(failures) == 2
    assert any("vad_f1" in f for f in failures) and any("missing" in f for f in failures)
    assert check_thresholds(results, {"other-backend": {"groups.overall.wer": {"max": 0.0}}}) == []
    assert build_parser().parse_args(["--profile", "quick"]).profile == "quick"
