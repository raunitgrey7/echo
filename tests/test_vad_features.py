"""Framing, VAD, MFCC and pitch."""

from __future__ import annotations

import numpy as np
import pytest

from echo.audio.features import frame_signal, frame_times, next_pow2, power_spectrum
from echo.audio.io import Audio
from echo.diarization.mfcc import hz_to_mel, mel_filterbank, mel_to_hz, mfcc
from echo.diarization.pitch import frame_f0, median_f0
from echo.vad.energy import (
    SpeechRegion,
    VadConfig,
    adaptive_thresholds,
    band_energy_db,
    detect_speech,
    hysteresis_mask,
    mask_to_regions,
    merge_regions,
    regions_to_mask,
    speech_ratio,
)
from tests.conftest import harmonic_voice, speech_like


def test_frame_signal_shapes_and_padding():
    frames = frame_signal(np.arange(100, dtype=float), 40, 20)
    assert frames.shape == (4, 40)
    assert frames[1, 0] == 20
    short = frame_signal(np.arange(10, dtype=float), 40, 20)
    assert short.shape == (1, 40)
    with pytest.raises(ValueError):
        frame_signal(np.zeros(10), 0, 1)


def test_power_spectrum_peak_and_pow2():
    sr, f = 16_000, 1000.0
    t = np.arange(400) / sr
    frames = np.sin(2 * np.pi * f * t)[None, :]
    spectrum = power_spectrum(frames, 512)
    assert spectrum.shape == (1, 257)
    assert abs(np.argmax(spectrum[0]) * sr / 512 - f) < sr / 512
    assert next_pow2(400) == 512 and next_pow2(512) == 512 and next_pow2(1) == 1
    assert frame_times(3, 400, 160, sr)[1] == pytest.approx((160 + 200) / sr)


def test_vad_finds_bursts_with_accurate_boundaries():
    audio = speech_like([(0.8, 0.7), (1.0, 0.9), (0.6, 0.5)])
    regions = detect_speech(audio)
    assert len(regions) == 3
    expected = [(0.0, 0.8), (1.5, 2.5), (3.4, 4.0)]
    for region, (start, end) in zip(regions, expected, strict=True):
        assert abs(region.start - start) < 0.12
        assert abs(region.end - end) < 0.12


def test_vad_on_silence_and_constant_noise_returns_nothing_useful():
    assert detect_speech(Audio(np.zeros(16_000, dtype=np.float32), 16_000)) == []
    assert detect_speech(Audio(np.zeros(0, dtype=np.float32), 16_000)) == []
    noise = Audio((0.1 * np.random.default_rng(0).standard_normal(32_000)).astype(np.float32), 16_000)
    regions = detect_speech(noise)
    assert speech_ratio(regions, noise.duration) < 0.3


def test_vad_survives_noise_at_ten_db():
    audio = speech_like([(1.0, 0.8), (1.0, 0.8)])
    rng = np.random.default_rng(0)
    noisy = audio.samples + 0.03 * rng.standard_normal(audio.samples.shape[0]).astype(np.float32)
    regions = detect_speech(Audio(noisy, 16_000))
    assert len(regions) == 2
    reference = [SpeechRegion(0.0, 1.0), SpeechRegion(1.8, 2.8)]
    mask_ref, mask_hyp = regions_to_mask(reference, audio.duration), regions_to_mask(regions, audio.duration)
    assert np.count_nonzero(mask_ref & mask_hyp) / mask_ref.sum() > 0.9


def test_hysteresis_and_region_helpers():
    energy = np.array([0, 0, 10, 10, 6, 6, 3, 0, 10, 0], dtype=float)
    mask = hysteresis_mask(energy, 0.0, 9.0, 4.0)
    assert mask.tolist() == [False, False, True, True, True, True, False, False, True, False]
    regions = mask_to_regions(mask, 0.01, 0.02)
    assert len(regions) == 2
    assert regions[0].start == pytest.approx(0.03)
    merged = merge_regions([SpeechRegion(0, 1), SpeechRegion(1.2, 2), SpeechRegion(3, 4)], 0.5)
    assert merged == [SpeechRegion(0, 2), SpeechRegion(3, 4)]
    assert SpeechRegion(0, 2).overlap(SpeechRegion(1, 5)) == 1.0
    assert mask_to_regions(np.zeros(0, dtype=bool), 0.01, 0.02) == []


def test_adaptive_thresholds_shrink_with_dynamic_range():
    config = VadConfig()
    wide = np.concatenate([np.full(50, -60.0), np.full(50, -20.0)])
    narrow = np.concatenate([np.full(50, -30.0), np.full(50, -26.0)])
    _, onset_wide, _ = adaptive_thresholds(wide, config)
    _, onset_narrow, _ = adaptive_thresholds(narrow, config)
    assert onset_wide == config.onset_db
    assert config.min_onset_db <= onset_narrow < onset_wide


def test_band_energy_prefers_speech_band():
    sr = 16_000
    t = np.arange(sr) / sr
    in_band = Audio(np.sin(2 * np.pi * 1000 * t).astype(np.float32), sr)
    out_band = Audio(np.sin(2 * np.pi * 6000 * t).astype(np.float32), sr)
    assert band_energy_db(in_band).mean() > band_energy_db(out_band).mean() + 20


def test_mel_scale_round_trip_and_filterbank():
    hz = np.array([100.0, 1000.0, 4000.0])
    assert np.allclose(mel_to_hz(hz_to_mel(hz)), hz)
    bank = mel_filterbank(16_000, 512, 26, 50, 8000)
    assert bank.shape == (26, 257)
    assert np.allclose(bank.max(axis=1), 1.0)
    centres = [np.argmax(row) for row in bank]
    assert centres == sorted(centres)


def test_mfcc_shape_and_sensitivity_to_timbre():
    bright = harmonic_voice(1.0, 150, tilt=0.5)
    dull = harmonic_voice(1.0, 150, tilt=2.0)
    a, b = mfcc(bright, 16_000), mfcc(dull, 16_000)
    assert a.shape[1] == 13
    assert a.shape[0] == b.shape[0]
    assert abs(a[:, 1].mean() - b[:, 1].mean()) > 1.0


@pytest.mark.parametrize("f0", [110.0, 180.0, 260.0])
def test_pitch_tracker_recovers_f0(f0):
    voice = harmonic_voice(1.0, f0)
    assert median_f0(voice, 16_000) == pytest.approx(f0, rel=0.04)
    frames = frame_f0(voice, 16_000)
    assert (frames > 0).mean() > 0.6


def test_pitch_tracker_is_silent_on_noise_and_silence():
    assert median_f0(np.zeros(16_000), 16_000) == 0.0
    noise = 0.3 * np.random.default_rng(0).standard_normal(16_000)
    assert (frame_f0(noise, 16_000) > 0).mean() < 0.2


def test_pitch_on_real_voices(samantha, daniel):
    assert 150 < median_f0(samantha.samples, 16_000) < 220
    assert 90 < median_f0(daniel.samples, 16_000) < 140
