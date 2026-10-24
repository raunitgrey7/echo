"""Speaker clustering and its metrics."""

from __future__ import annotations

import numpy as np
from tests.conftest import harmonic_voice

from echo.audio.io import Audio
from echo.diarization.change import (
    SpeakerConfig,
    agglomerate,
    change_points,
    gaussian_stats,
    label_speakers,
    region_frames,
    relabel_by_first_appearance,
    symmetric_kl,
)
from echo.metrics.diarization import LabelledSpan, purity, score_changes
from echo.vad.energy import SpeechRegion, detect_speech


def test_symmetric_kl_is_zero_for_identical_and_grows_with_shift():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((300, 4))
    same = symmetric_kl(gaussian_stats(a), gaussian_stats(a))
    shifted = symmetric_kl(gaussian_stats(a), gaussian_stats(a + 2.0))
    assert same == 0.0
    assert shifted > 10.0


def test_agglomerate_two_synthetic_voices_and_one_voice():
    rng = np.random.default_rng(1)
    low = [rng.standard_normal((200, 3)) for _ in range(3)]
    high = [rng.standard_normal((200, 3)) + 6.0 for _ in range(3)]
    labels, closest = agglomerate(low + high, SpeakerConfig(distance_threshold=3.0))
    assert len(set(labels[:3])) == 1 and len(set(labels[3:])) == 1 and labels[0] != labels[3]
    assert closest >= 3.0
    one, _ = agglomerate(low, SpeakerConfig(distance_threshold=3.0))
    assert len(set(one.tolist())) == 1
    assert agglomerate([], SpeakerConfig()).__len__() == 2


def test_relabel_and_change_points():
    assert relabel_by_first_appearance(np.array([2, 2, 0, 2, 1])) == (0, 0, 1, 0, 2)
    regions = [SpeechRegion(0, 1), SpeechRegion(1.5, 2.5), SpeechRegion(3, 4)]
    assert change_points(regions, (0, 1, 1)) == (1.25,)
    assert change_points(regions, (0, 0, 0)) == ()


def test_label_speakers_on_synthetic_two_voice_recording():
    sr = 16_000
    gap = np.zeros(sr, dtype=np.float32)
    parts = []
    for i in range(4):
        voice = harmonic_voice(1.2, 120.0 if i % 2 == 0 else 230.0, tilt=1.0 if i % 2 == 0 else 1.6, seed=i)
        parts += [0.5 * voice, gap]
    audio = Audio(np.concatenate(parts).astype(np.float32), sr)
    regions = detect_speech(audio)
    assert len(regions) == 4
    result = label_speakers(audio, regions)
    assert result.n_speakers == 2
    assert result.labels == (0, 1, 0, 1)
    assert len(result.changes) == 3


def test_label_speakers_on_real_voices(two_speakers):
    regions = detect_speech(two_speakers)
    assert len(regions) == 4
    result = label_speakers(two_speakers, regions)
    assert result.n_speakers == 2
    assert result.labels == (0, 1, 0, 1)


def test_label_speakers_single_voice_and_edge_cases(samantha):
    regions = detect_speech(samantha)
    result = label_speakers(samantha, regions)
    assert result.n_speakers == 1
    empty = label_speakers(samantha, [])
    assert empty.n_speakers == 0 and empty.labels == ()
    tiny = label_speakers(samantha, [SpeechRegion(0.0, 0.1)])
    assert tiny.labels == (0,)
    frames = region_frames(samantha, regions)
    assert frames[0].shape[1] == 13


def test_change_score_matching_with_tolerance():
    score = score_changes([1.0, 5.0, 9.0], [1.2, 5.9, 12.0], 0.5)
    assert score.n_reference == 3 and score.n_hypothesis == 3
    assert score.precision == 1 / 3 and score.recall == 1 / 3
    assert score_changes([], [], 0.5).f1 == 1.0
    assert score_changes([1.0], [], 0.5).recall == 0.0
    assert score_changes([], [1.0], 0.5).precision == 0.0


def test_purity_is_duration_weighted():
    ref = [LabelledSpan(SpeechRegion(0, 2), "A"), LabelledSpan(SpeechRegion(2, 3), "B")]
    perfect = [LabelledSpan(SpeechRegion(0, 2), "x"), LabelledSpan(SpeechRegion(2, 3), "y")]
    merged = [LabelledSpan(SpeechRegion(0, 3), "x")]
    assert purity(ref, perfect) == 1.0
    assert purity(ref, merged) == 2 / 3
    assert purity(ref, []) == 0.0
