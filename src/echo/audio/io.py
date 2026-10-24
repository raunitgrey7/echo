"""WAV input and output on top of the standard library.

Why: Python 3.13 removed ``aifc`` and ``audioop`` but kept ``wave``. echo keeps one internal
format, 16 kHz mono float32 in [-1, 1], and this module owns the only conversion boundary:
16-bit PCM WAV on disk, float32 in memory. Resampling uses scipy's polyphase filter so the API
can accept recordings at other sample rates without any external binary.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

SAMPLE_RATE = 16_000
PCM16_SCALE = 32_768.0


class AudioFormatError(ValueError):
    """Raised when a byte stream is not a PCM WAV file echo can decode."""


@dataclass(frozen=True, slots=True)
class Audio:
    """Mono float32 samples with their sample rate."""

    samples: np.ndarray
    sample_rate: int = SAMPLE_RATE

    @property
    def duration(self) -> float:
        """Length in seconds."""
        return float(self.samples.shape[0]) / float(self.sample_rate)

    def __post_init__(self) -> None:
        if self.samples.ndim != 1:
            raise AudioFormatError(f"expected a 1-D sample array, got shape {self.samples.shape}")
        if self.sample_rate <= 0:
            raise AudioFormatError(f"sample rate must be positive, got {self.sample_rate}")


def to_mono(frames: np.ndarray, channels: int) -> np.ndarray:
    """Average interleaved channels into one, keeping float64 precision until the caller casts."""
    if channels == 1:
        return frames.astype(np.float64)
    usable = frames.shape[0] - frames.shape[0] % channels
    return frames[:usable].reshape(-1, channels).astype(np.float64).mean(axis=1)


def pcm_to_float(raw: bytes, sample_width: int) -> np.ndarray:
    """Decode 8, 16, 24 or 32-bit integer PCM bytes into float64 in [-1, 1]."""
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float64) / PCM16_SCALE
    if sample_width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        value = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        value = np.where(value >= 1 << 23, value - (1 << 24), value)
        return value.astype(np.float64) / float(1 << 23)
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float64) / float(1 << 31)
    raise AudioFormatError(f"unsupported PCM sample width {sample_width}")


def float_to_pcm16(samples: np.ndarray) -> bytes:
    """Clip to [-1, 1] and quantise to little-endian 16-bit PCM with the same 32768 scale as decoding."""
    clipped = np.clip(np.nan_to_num(samples, nan=0.0), -1.0, 1.0)
    quantised = np.clip(np.round(clipped * PCM16_SCALE), -32_768.0, 32_767.0)
    return quantised.astype("<i2").tobytes()


def read_wav(source: Path | str | bytes) -> Audio:
    """Read a PCM WAV file or byte string into mono float32.

    Non-PCM encodings (float, mu-law, ADPCM) raise ``AudioFormatError`` with a clear message
    instead of the generic ``wave.Error`` so the API can answer with a 415.
    """
    handle = io.BytesIO(source) if isinstance(source, bytes) else open(source, "rb")  # noqa: SIM115
    try:
        with wave.open(handle, "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            rate = wav.getframerate()
            raw = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as exc:
        raise AudioFormatError(f"not a PCM WAV stream: {exc}") from exc
    finally:
        handle.close()
    mono = to_mono(pcm_to_float(raw, width), channels)
    return Audio(mono.astype(np.float32), rate)


def write_wav(path: Path | str, audio: Audio) -> Path:
    """Write mono 16-bit PCM WAV; creates parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(audio.sample_rate)
        wav.writeframes(float_to_pcm16(audio.samples))
    return target


def wav_bytes(audio: Audio) -> bytes:
    """Serialise to an in-memory WAV, used by the API to stream audio back to the player."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(audio.sample_rate)
        wav.writeframes(float_to_pcm16(audio.samples))
    return buffer.getvalue()


def resample(audio: Audio, target_rate: int) -> Audio:
    """Polyphase resampling; a no-op when the rate already matches."""
    if audio.sample_rate == target_rate:
        return audio
    divisor = gcd(audio.sample_rate, target_rate)
    up, down = target_rate // divisor, audio.sample_rate // divisor
    out = resample_poly(audio.samples.astype(np.float64), up, down)
    return Audio(out.astype(np.float32), target_rate)


def ensure_16k(audio: Audio) -> Audio:
    """Bring any input to the internal 16 kHz contract."""
    return resample(audio, SAMPLE_RATE)


def rms_db(samples: np.ndarray, floor: float = 1e-10) -> float:
    """RMS level in dBFS; silence returns the floor (-100 dB) rather than -inf."""
    if samples.size == 0:
        return 10.0 * float(np.log10(floor))
    mean_square = float(np.mean(np.square(samples.astype(np.float64))))
    return 10.0 * float(np.log10(max(mean_square, floor)))


def silence(seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Zero samples of the requested length, rounded to whole samples."""
    return np.zeros(round(seconds * sample_rate), dtype=np.float32)
