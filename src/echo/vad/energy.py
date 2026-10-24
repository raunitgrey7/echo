"""Band-limited energy VAD with a percentile noise floor and hysteresis.

Why: whisper has an internal VAD, but echo needs speech regions before ASR runs (to chunk long
recordings), for the dashboard timeline and as the units for speaker clustering. A frame-energy
detector measured in a speech band (200 Hz to 4 kHz) is dependency-free, runs in milliseconds and
can be scored against the clean-signal boundaries the synthesis harness knows. The noise floor is
the 15th percentile of frame energy, so the thresholds adapt to the recording, and hysteresis
(a higher onset than offset threshold) stops a syllable from flickering on and off.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from echo.audio.features import frame_signal, next_pow2, power_spectrum
from echo.audio.io import Audio


@dataclass(frozen=True, slots=True)
class SpeechRegion:
    """Half-open interval of speech in seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        """Length in seconds."""
        return self.end - self.start

    def overlap(self, other: SpeechRegion) -> float:
        """Seconds shared with another region."""
        return max(0.0, min(self.end, other.end) - max(self.start, other.start))


@dataclass(frozen=True, slots=True)
class VadConfig:
    """Detector parameters; defaults were chosen on the synthetic corpus and are reported as such."""

    frame_ms: float = 25.0
    hop_ms: float = 10.0
    band_hz: tuple[float, float] = (200.0, 4000.0)
    floor_percentile: float = 15.0
    peak_percentile: float = 90.0
    onset_db: float = 9.0
    offset_db: float = 4.0
    onset_fraction: float = 0.6
    min_onset_db: float = 3.0
    min_speech_s: float = 0.20
    min_silence_s: float = 0.50
    pad_s: float = 0.10


DEFAULT_VAD = VadConfig()


def band_energy_db(audio: Audio, config: VadConfig = DEFAULT_VAD) -> np.ndarray:
    """Per-frame energy in dB restricted to the speech band."""
    frame = round(config.frame_ms * audio.sample_rate / 1000.0)
    hop = round(config.hop_ms * audio.sample_rate / 1000.0)
    frames = frame_signal(audio.samples, frame, hop)
    n_fft = next_pow2(frame)
    spectrum = power_spectrum(frames, n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / audio.sample_rate)
    low, high = config.band_hz
    band = (freqs >= low) & (freqs <= high)
    energy = spectrum[:, band].sum(axis=1)
    return 10.0 * np.log10(energy + 1e-12)


def hysteresis_mask(energy_db: np.ndarray, floor_db: float, onset_db: float, offset_db: float) -> np.ndarray:
    """Frame-level speech mask: switch on above floor + onset, off below floor + offset."""
    on_threshold = floor_db + onset_db
    off_threshold = floor_db + offset_db
    mask = np.zeros(energy_db.shape[0], dtype=bool)
    active = False
    for index, value in enumerate(energy_db):
        active = value >= off_threshold if active else value >= on_threshold
        mask[index] = active
    return mask


def mask_to_regions(mask: np.ndarray, hop_s: float, frame_s: float) -> list[SpeechRegion]:
    """Turn a boolean frame mask into intervals, using frame centres as boundaries."""
    if mask.size == 0 or not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = edges[0::2], edges[1::2]
    half = frame_s / 2.0
    return [SpeechRegion(s * hop_s + half, e * hop_s + half) for s, e in zip(starts, ends, strict=True)]


def merge_regions(regions: list[SpeechRegion], min_gap_s: float) -> list[SpeechRegion]:
    """Join regions separated by less than ``min_gap_s`` (a pause inside a sentence, not a boundary)."""
    merged: list[SpeechRegion] = []
    for region in regions:
        if merged and region.start - merged[-1].end < min_gap_s:
            merged[-1] = SpeechRegion(merged[-1].start, region.end)
        else:
            merged.append(region)
    return merged


def adaptive_thresholds(energy_db: np.ndarray, config: VadConfig) -> tuple[float, float, float]:
    """Noise floor plus onset and offset margins scaled to the recording's dynamic range.

    Clean audio has 30 dB or more between pauses and speech, so the fixed margins apply. At low
    SNR the range shrinks and a fixed 9 dB onset would never fire; the margins are then a fraction
    of the observed range, with a 3 dB minimum so pure noise does not trigger on its own jitter.
    """
    floor = float(np.percentile(energy_db, config.floor_percentile))
    peak = float(np.percentile(energy_db, config.peak_percentile))
    span = max(peak - floor, 0.0)
    onset = max(config.min_onset_db, min(config.onset_db, config.onset_fraction * span))
    offset = max(config.min_onset_db / 2.0, min(config.offset_db, onset / 2.0))
    return floor, onset, offset


def detect_speech(audio: Audio, config: VadConfig = DEFAULT_VAD) -> list[SpeechRegion]:
    """Speech regions in seconds, merged, filtered by duration and padded."""
    if audio.samples.size == 0:
        return []
    energy = band_energy_db(audio, config)
    floor, onset, offset = adaptive_thresholds(energy, config)
    mask = hysteresis_mask(energy, floor, onset, offset)
    raw = mask_to_regions(mask, config.hop_ms / 1000.0, config.frame_ms / 1000.0)
    merged = merge_regions(raw, config.min_silence_s)
    kept = [r for r in merged if r.duration >= config.min_speech_s]
    duration = audio.duration
    padded = [SpeechRegion(max(0.0, r.start - config.pad_s), min(duration, r.end + config.pad_s)) for r in kept]
    return merge_regions(padded, 0.0)


def regions_to_mask(regions: list[SpeechRegion], duration_s: float, hop_s: float = 0.01) -> np.ndarray:
    """Boolean mask on a fixed frame grid, used by the VAD metrics."""
    n = int(np.ceil(duration_s / hop_s))
    mask = np.zeros(n, dtype=bool)
    for region in regions:
        lo = max(0, int(np.floor(region.start / hop_s)))
        hi = min(n, int(np.ceil(region.end / hop_s)))
        mask[lo:hi] = True
    return mask


def speech_ratio(regions: list[SpeechRegion], duration_s: float) -> float:
    """Fraction of the recording covered by speech regions."""
    if duration_s <= 0:
        return 0.0
    return min(1.0, sum(r.duration for r in regions) / duration_s)
