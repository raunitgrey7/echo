"""The calibration sweep helper and the demo audio fallback."""

from __future__ import annotations

import numpy as np

from echo.api import app as app_module
from echo.api.app import demo_audio
from echo.calibrate import evaluate
from echo.corpus.build import RecordingManifest, UtteranceLabel
from echo.vad.energy import SpeechRegion


def _manifest(n_speakers: int) -> RecordingManifest:
    speakers = [f"s{i % n_speakers}" for i in range(4)]
    utterances = tuple(UtteranceLabel(i, speakers[i], "v", "t", 2.0 * i, 2.0 * i + 1.5) for i in range(4))
    return RecordingManifest(
        id="m",
        kind="meeting",
        script_id="x",
        axis="meeting",
        voices=("a",),
        rate_wpm=175,
        condition={"name": "clean"},
        seed=0,
        path="p",
        sha256="s",
        duration_s=8.0,
        sample_rate=16_000,
        reference_text="t",
        utterances=utterances,
        speaker_changes=(),
    )


def test_evaluate_counts_speakers_and_purity():
    rng = np.random.default_rng(0)
    regions = [SpeechRegion(2.0 * i, 2.0 * i + 1.5) for i in range(4)]
    far = [rng.standard_normal((150, 3)) + (0.0 if i % 2 == 0 else 8.0) for i in range(4)]
    two = (_manifest(2), regions, far)
    one = (_manifest(1), regions, [rng.standard_normal((150, 3)) for _ in range(4)])
    accuracy, mean_purity = evaluate([two, one], threshold=3.0)
    assert accuracy == 1.0
    assert mean_purity == 1.0
    accuracy_strict, _ = evaluate([two, one], threshold=0.01)
    assert accuracy_strict < 1.0


def test_demo_audio_falls_back_to_bundled_clip(monkeypatch):
    monkeypatch.setattr(app_module, "say_available", lambda: False)
    audio = demo_audio()
    assert audio.sample_rate == 16_000
    assert 20 < audio.duration < 40
