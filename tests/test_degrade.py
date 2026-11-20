"""Noise and reverberation have the parameters they claim."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from echo.audio.degrade import (
    CLEAN,
    Condition,
    active_rms,
    add_noise,
    apply_condition,
    babble_noise,
    peak_normalize,
    pink_noise,
    reverberate,
    synthetic_rir,
    white_noise,
)
from tests.conftest import speech_like


def achieved_snr_db(speech: np.ndarray, mixed: np.ndarray) -> float:
    added = mixed.astype(np.float64) - speech.astype(np.float64)
    return 20 * np.log10(active_rms(speech) / np.sqrt(np.mean(added**2)))


@settings(max_examples=12, deadline=None)
@given(snr=st.floats(min_value=-5.0, max_value=30.0), seed=st.integers(min_value=0, max_value=1000))
def test_add_noise_hits_requested_snr(snr, seed):
    speech = speech_like([(0.6, 0.3), (0.5, 0.2)], seed=seed).samples
    noise = white_noise(speech.shape[0], np.random.default_rng(seed))
    assert achieved_snr_db(speech, add_noise(speech, noise, snr)) == pytest.approx(snr, abs=0.01)


def test_pink_noise_has_falling_spectrum_and_unit_rms():
    noise = pink_noise(65_536, np.random.default_rng(0))
    assert np.sqrt(np.mean(noise**2)) == pytest.approx(1.0, abs=1e-6)
    spectrum = np.abs(np.fft.rfft(noise)) ** 2
    low = spectrum[10:100].mean()
    high = spectrum[10_000:20_000].mean()
    assert low > 20 * high


def test_white_noise_is_flat_on_average():
    noise = white_noise(1 << 16, np.random.default_rng(1))
    spectrum = np.abs(np.fft.rfft(noise)) ** 2
    assert spectrum[100:1000].mean() == pytest.approx(spectrum[20_000:30_000].mean(), rel=0.15)


def test_babble_mixes_sources_and_normalises():
    rng = np.random.default_rng(3)
    sources = [speech_like([(0.4, 0.1)], f0=120, seed=1).samples, speech_like([(0.3, 0.1)], f0=210, seed=2).samples]
    babble = babble_noise(16_000, rng, sources, talkers=3)
    assert babble.shape == (16_000,)
    assert np.sqrt(np.mean(babble**2)) == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError):
        babble_noise(100, rng, [])


def test_rir_decays_sixty_db_at_rt60():
    sr = 16_000
    rir = synthetic_rir(0.5, sr, np.random.default_rng(0))
    assert rir.shape == (sr // 2,)
    assert rir[0] == pytest.approx(1.0)
    envelope = np.abs(rir[1:])
    early = np.sqrt(np.mean(envelope[: sr // 100] ** 2))
    late = np.sqrt(np.mean(envelope[-sr // 100 :] ** 2))
    assert 20 * np.log10(early / late) > 45


def test_reverberate_keeps_length_and_rms():
    speech = speech_like([(1.0, 0.2)]).samples
    wet = reverberate(speech, synthetic_rir(0.4, 16_000, np.random.default_rng(0)))
    assert wet.shape == speech.shape
    assert np.sqrt(np.mean(wet**2)) == pytest.approx(np.sqrt(np.mean(speech**2)), rel=1e-3)
    assert not np.allclose(wet, speech)


def test_peak_normalize():
    assert np.max(np.abs(peak_normalize(np.array([0.1, -0.2])))) == pytest.approx(0.89)
    assert peak_normalize(np.zeros(5)).tolist() == [0.0] * 5


def test_apply_condition_is_deterministic_and_clean_is_identity_up_to_gain():
    speech = speech_like([(0.5, 0.2)]).samples
    cond = Condition("w", "white", 5.0, 0.3)
    a = apply_condition(speech, cond, 16_000, np.random.default_rng(42))
    b = apply_condition(speech, cond, 16_000, np.random.default_rng(42))
    c = apply_condition(speech, cond, 16_000, np.random.default_rng(43))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    clean = apply_condition(speech, CLEAN, 16_000, np.random.default_rng(0))
    assert np.allclose(clean / np.max(np.abs(clean)), speech / np.max(np.abs(speech)), atol=1e-6)
    assert CLEAN.is_clean and not cond.is_clean


def test_condition_without_snr_is_rejected():
    with pytest.raises(ValueError):
        apply_condition(np.zeros(100, dtype=np.float32), Condition("bad", "white"), 16_000, np.random.default_rng(0))
