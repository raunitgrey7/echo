"""Shared short-time analysis: framing, windows and power spectra.

Why: the VAD, the MFCC front end and the latency probes all slice audio into overlapping frames.
Keeping one stride-trick implementation here means every stage sees the same frame grid, which
is what lets frame indices be converted to seconds in one place.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def frame_signal(samples: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """Return a (n_frames, frame) view; the tail shorter than a frame is zero-padded into one frame."""
    if frame <= 0 or hop <= 0:
        raise ValueError("frame and hop must be positive")
    x = np.asarray(samples, dtype=np.float64)
    if x.shape[0] < frame:
        x = np.pad(x, (0, frame - x.shape[0]))
    remainder = (x.shape[0] - frame) % hop
    if remainder:
        x = np.pad(x, (0, hop - remainder))
    return sliding_window_view(x, frame)[::hop]


def hamming(frame: int) -> np.ndarray:
    """Periodic Hamming window; the usual choice for speech spectra because of its low leakage."""
    return np.hamming(frame).astype(np.float64)


def power_spectrum(frames: np.ndarray, n_fft: int) -> np.ndarray:
    """Windowed one-sided power spectrum per frame, shape (n_frames, n_fft // 2 + 1)."""
    windowed = frames * hamming(frames.shape[1])
    spectrum = np.fft.rfft(windowed, n=n_fft, axis=1)
    return (np.abs(spectrum) ** 2) / float(n_fft)


def next_pow2(n: int) -> int:
    """Smallest power of two not below ``n``; FFT sizes should be powers of two for speed."""
    return 1 << (max(n, 1) - 1).bit_length()


def frame_times(n_frames: int, frame: int, hop: int, sample_rate: int) -> np.ndarray:
    """Centre time in seconds of each frame."""
    return (np.arange(n_frames) * hop + frame / 2.0) / float(sample_rate)
