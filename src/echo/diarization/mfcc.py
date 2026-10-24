"""Mel-frequency cepstral coefficients in numpy.

Why: speaker identity lives in the spectral envelope (vocal tract shape), which MFCCs capture
in a dozen numbers per frame while discarding pitch harmonics and loudness (c0 is dropped by the
caller). A numpy implementation keeps the package free of audio toolkits and makes every step
inspectable: pre-emphasis, Hamming window, power spectrum, mel filterbank, log, DCT-II.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct

from echo.audio.features import frame_signal, next_pow2, power_spectrum


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    """HTK mel scale."""
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    """Inverse of :func:`hz_to_mel`."""
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Triangular filters on a mel grid, shape (n_mels, n_fft // 2 + 1), each peaking at 1."""
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for m in range(1, n_mels + 1):
        left, centre, right = bins[m - 1], bins[m], bins[m + 1]
        centre = max(centre, left + 1)
        right = max(right, centre + 1)
        for k in range(left, min(centre, bank.shape[1])):
            bank[m - 1, k] = (k - left) / (centre - left)
        for k in range(centre, min(right, bank.shape[1])):
            bank[m - 1, k] = (right - k) / (right - centre)
    return bank


def mfcc(
    samples: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 13,
    n_mels: int = 26,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    pre_emphasis: float = 0.97,
    fmin: float = 50.0,
    fmax: float | None = None,
) -> np.ndarray:
    """MFCC matrix of shape (n_frames, n_mfcc); coefficient 0 is the log energy."""
    x = np.asarray(samples, dtype=np.float64)
    if x.size > 1:
        x = np.append(x[0], x[1:] - pre_emphasis * x[:-1])
    frame = round(frame_ms * sample_rate / 1000.0)
    hop = round(hop_ms * sample_rate / 1000.0)
    frames = frame_signal(x, frame, hop)
    n_fft = next_pow2(frame)
    spectrum = power_spectrum(frames, n_fft)
    bank = mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax or sample_rate / 2.0)
    log_mel = np.log(spectrum @ bank.T + 1e-10)
    return dct(log_mel, type=2, norm="ortho", axis=1)[:, :n_mfcc]
