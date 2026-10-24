"""Fundamental frequency by normalised autocorrelation.

Why: on this corpus pitch is the single strongest speaker cue: a voice keeps its median F0
within a few hertz across sentences while different voices sit tens of hertz apart. The tracker
is deliberately plain: 40 ms frames, autocorrelation via FFT, the strongest peak between 60 and
400 Hz, and a frame counts as voiced only when that peak is clear. The region statistic is the
median over voiced frames, which ignores the occasional octave error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from echo.audio.features import frame_signal, next_pow2


@dataclass(frozen=True, slots=True)
class PitchConfig:
    """Tracker parameters."""

    frame_ms: float = 40.0
    hop_ms: float = 10.0
    fmin_hz: float = 60.0
    fmax_hz: float = 400.0
    voicing_threshold: float = 0.5
    min_rms: float = 0.01


DEFAULT_PITCH = PitchConfig()


def frame_f0(samples: np.ndarray, sample_rate: int, config: PitchConfig = DEFAULT_PITCH) -> np.ndarray:
    """F0 per frame in Hz, zero where the frame is unvoiced or quiet."""
    frame = round(config.frame_ms * sample_rate / 1000.0)
    hop = round(config.hop_ms * sample_rate / 1000.0)
    frames = frame_signal(samples, frame, hop)
    frames = frames - frames.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    n_fft = next_pow2(2 * frame)
    spectrum = np.fft.rfft(frames, n=n_fft, axis=1)
    autocorr = np.fft.irfft(np.abs(spectrum) ** 2, n=n_fft, axis=1)[:, :frame]
    autocorr = autocorr / (autocorr[:, :1] + 1e-12)
    lag_lo = int(sample_rate / config.fmax_hz)
    lag_hi = min(int(sample_rate / config.fmin_hz), frame - 1)
    window = autocorr[:, lag_lo:lag_hi]
    best = np.argmax(window, axis=1)
    peak = window[np.arange(window.shape[0]), best]
    f0 = sample_rate / (best + lag_lo).astype(np.float64)
    voiced = (peak >= config.voicing_threshold) & (rms >= config.min_rms)
    return np.where(voiced, f0, 0.0)


def median_f0(samples: np.ndarray, sample_rate: int, config: PitchConfig = DEFAULT_PITCH) -> float:
    """Median F0 over voiced frames; zero when nothing is voiced."""
    f0 = frame_f0(samples, sample_rate, config)
    voiced = f0[f0 > 0]
    return float(np.median(voiced)) if voiced.size else 0.0
