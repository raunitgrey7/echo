"""Degradations with a physical story: additive noise at a target SNR and synthetic reverberation.

Why: WER on clean synthetic speech says little about a lecture hall or a laptop microphone in
a meeting room. Each degradation here has a parameter with units (SNR in dB, RT60 in seconds)
so results read as a curve, and every random draw comes from a seeded generator so the
benchmark is reproducible to the sample.

Noise models: white (flat spectrum, the electronic noise floor), pink (1/f, the usual shape of
room and ventilation noise) and babble (a sum of other talkers, the hardest case for ASR because
it shares the speech spectrum). Reverberation uses the Schroeder model: a direct impulse followed
by exponentially decaying Gaussian noise whose decay rate is fixed by RT60, the time for the
energy to fall 60 dB (a factor of 1000, hence ``ln(1000)``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.signal import fftconvolve

NoiseKind = Literal["none", "white", "pink", "babble"]
LN_1000 = 6.907755278982137


@dataclass(frozen=True, slots=True)
class Condition:
    """One acoustic condition of the benchmark grid."""

    name: str
    noise: NoiseKind = "none"
    snr_db: float | None = None
    rt60_s: float | None = None

    @property
    def is_clean(self) -> bool:
        """True for the untouched reference condition."""
        return self.noise == "none" and self.rt60_s is None


CLEAN = Condition("clean")


def white_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance Gaussian noise."""
    return rng.standard_normal(n).astype(np.float64)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise by shaping white noise in the frequency domain, then normalised to unit RMS."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.arange(spectrum.shape[0], dtype=np.float64)
    freqs[0] = 1.0
    shaped = np.fft.irfft(spectrum / np.sqrt(freqs), n=n)
    return shaped / (np.sqrt(np.mean(np.square(shaped))) + 1e-12)


def babble_noise(n: int, rng: np.random.Generator, sources: list[np.ndarray], talkers: int = 4) -> np.ndarray:
    """Sum of ``talkers`` speech signals looped and started at random offsets, unit RMS."""
    if not sources:
        raise ValueError("babble needs at least one source signal")
    mix = np.zeros(n, dtype=np.float64)
    for _ in range(talkers):
        src = sources[int(rng.integers(len(sources)))].astype(np.float64)
        if src.size == 0:
            continue
        offset = int(rng.integers(src.size))
        rolled = np.roll(src, -offset)
        reps = int(np.ceil(n / rolled.size)) if rolled.size else 1
        mix += np.tile(rolled, reps)[:n]
    return mix / (np.sqrt(np.mean(np.square(mix))) + 1e-12)


def active_rms(speech: np.ndarray, frame: int = 320, percentile: float = 40.0) -> float:
    """RMS over the louder frames only, so SNR refers to speech rather than to pauses.

    Frames whose energy sits in the lowest ``percentile`` of the recording are treated as pauses.
    """
    usable = speech.shape[0] - speech.shape[0] % frame
    if usable == 0:
        return float(np.sqrt(np.mean(np.square(speech)))) if speech.size else 0.0
    frames = speech[:usable].reshape(-1, frame).astype(np.float64)
    energy = np.mean(np.square(frames), axis=1)
    threshold = np.percentile(energy, percentile)
    active = energy[energy >= threshold]
    return float(np.sqrt(np.mean(active))) if active.size else float(np.sqrt(np.mean(energy)))


def add_noise(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale ``noise`` so that active-speech RMS over noise RMS equals ``snr_db``, then add."""
    speech64 = speech.astype(np.float64)
    noise64 = noise[: speech64.shape[0]].astype(np.float64)
    speech_rms = active_rms(speech64)
    noise_rms = float(np.sqrt(np.mean(np.square(noise64)))) + 1e-12
    gain = (speech_rms / noise_rms) / (10.0 ** (snr_db / 20.0))
    return (speech64 + gain * noise64).astype(np.float32)


def synthetic_rir(rt60_s: float, sample_rate: int, rng: np.random.Generator, direct_gain: float = 1.0) -> np.ndarray:
    """Schroeder-style room impulse response: direct path plus exponentially decaying noise tail.

    The tail amplitude follows ``exp(-LN_1000 * t / RT60)`` so its energy is 60 dB down at RT60.
    The tail is scaled so the reverberant-to-direct energy ratio is 0 dB at RT60 = 0.5 s, a
    typical seminar room, and grows with longer RT60.
    """
    n = int(rt60_s * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate
    tail = rng.standard_normal(n) * np.exp(-LN_1000 * t / rt60_s)
    tail[0] = 0.0
    tail_energy = float(np.sum(np.square(tail))) + 1e-12
    tail *= np.sqrt((rt60_s / 0.5) * direct_gain**2 / tail_energy)
    tail[0] = direct_gain
    return tail


def reverberate(speech: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve with the impulse response, truncate to the input length and restore the RMS."""
    speech64 = speech.astype(np.float64)
    wet = fftconvolve(speech64, rir)[: speech64.shape[0]]
    dry_rms = float(np.sqrt(np.mean(np.square(speech64)))) + 1e-12
    wet_rms = float(np.sqrt(np.mean(np.square(wet)))) + 1e-12
    return (wet * dry_rms / wet_rms).astype(np.float32)


def peak_normalize(samples: np.ndarray, peak: float = 0.89) -> np.ndarray:
    """Scale so the largest magnitude equals ``peak`` (-1 dBFS by default); silence is unchanged."""
    top = float(np.max(np.abs(samples))) if samples.size else 0.0
    if top <= 0.0:
        return samples.astype(np.float32)
    return (samples.astype(np.float64) * peak / top).astype(np.float32)


def apply_condition(
    speech: np.ndarray,
    condition: Condition,
    sample_rate: int,
    rng: np.random.Generator,
    babble_sources: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Reverberate first (the room acts on the talker), then add noise at the microphone."""
    out = speech.astype(np.float32)
    if condition.rt60_s is not None:
        out = reverberate(out, synthetic_rir(condition.rt60_s, sample_rate, rng))
    if condition.noise != "none":
        if condition.snr_db is None:
            raise ValueError(f"condition {condition.name!r} sets noise without an SNR")
        n = out.shape[0]
        if condition.noise == "white":
            noise = white_noise(n, rng)
        elif condition.noise == "pink":
            noise = pink_noise(n, rng)
        else:
            noise = babble_noise(n, rng, babble_sources or [])
        out = add_noise(out, noise, condition.snr_db)
    return peak_normalize(out)
