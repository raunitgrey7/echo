"""Ground truth for free: macOS ``say`` turns a known transcript into speech.

Why: a benchmark needs audio whose transcript is certain. Recording people is expensive and
licensed corpora are large, while ``say`` ships with every Mac, offers dozens of voices and a
speaking rate flag, so each (voice, rate, text) triple is a labelled sample. ``say`` writes
22.05 kHz AIFF; ``afconvert`` turns that into 16 kHz mono PCM. When afconvert is missing the
AIFF is decoded here in numpy and resampled, so the only hard dependency is ``say`` itself.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from echo.audio.io import SAMPLE_RATE, Audio, read_wav, resample, write_wav

DEFAULT_RATE_WPM = 175
_VOICE_LINE = re.compile(r"^(?P<name>.+?)\s{2,}(?P<locale>[a-z]{2,3}_[A-Z0-9]{2,3})\s+#\s*(?P<sample>.*)$")


class SynthesisError(RuntimeError):
    """Raised when ``say`` or the conversion step fails."""


@dataclass(frozen=True, slots=True)
class Voice:
    """One installed system voice."""

    name: str
    locale: str
    sample: str


def say_available() -> bool:
    """True when the ``say`` binary is on PATH (macOS only)."""
    return shutil.which("say") is not None


def list_voices(locale_prefix: str = "en") -> list[Voice]:
    """Parse ``say -v ?``; the locale prefix keeps the benchmark to one language."""
    if not say_available():
        return []
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=True).stdout
    voices: list[Voice] = []
    for line in out.splitlines():
        match = _VOICE_LINE.match(line.rstrip())
        if match and match.group("locale").startswith(locale_prefix):
            voices.append(Voice(match.group("name"), match.group("locale"), match.group("sample")))
    return voices


def cache_key(text: str, voice: str, rate_wpm: int) -> str:
    """Stable file name for one utterance; the version tag invalidates the cache on format changes."""
    digest = hashlib.sha256(f"v1|{voice}|{rate_wpm}|{text}".encode()).hexdigest()[:20]
    return f"{voice.lower().replace(' ', '_')}_{rate_wpm}_{digest}"


def synthesize(text: str, voice: str, rate_wpm: int = DEFAULT_RATE_WPM, cache_dir: Path | None = None) -> Audio:
    """Return 16 kHz mono speech for ``text``; cached on disk when ``cache_dir`` is given."""
    if cache_dir is not None:
        target = cache_dir / f"{cache_key(text, voice, rate_wpm)}.wav"
        if target.exists():
            return read_wav(target)
    audio = _synthesize_uncached(text, voice, rate_wpm)
    if cache_dir is not None:
        write_wav(target, audio)
    return audio


def _synthesize_uncached(text: str, voice: str, rate_wpm: int) -> Audio:
    if not say_available():
        raise SynthesisError("macOS `say` is not available on this machine")
    with tempfile.TemporaryDirectory(prefix="echo-say-") as tmp:
        tmp_dir = Path(tmp)
        script = tmp_dir / "text.txt"
        script.write_text(text, encoding="utf-8")
        aiff = tmp_dir / "out.aiff"
        cmd = ["say", "-v", voice, "-r", str(rate_wpm), "-f", str(script), "-o", str(aiff)]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not aiff.exists():
            raise SynthesisError(f"say failed for voice {voice!r}: {proc.stderr.strip()}")
        return _convert(aiff, tmp_dir / "out.wav")


def _convert(aiff: Path, wav: Path) -> Audio:
    """Prefer afconvert (CoreAudio resampler); fall back to the numpy AIFF decoder."""
    if shutil.which("afconvert") is not None:
        cmd = ["afconvert", "-f", "WAVE", "-d", f"LEI16@{SAMPLE_RATE}", "-c", "1", str(aiff), str(wav)]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0 and wav.exists():
            return read_wav(wav)
    return resample(read_aiff(aiff), SAMPLE_RATE)


def _parse_extended(raw: bytes) -> float:
    """Decode the 80-bit IEEE extended float AIFF uses for the sample rate."""
    sign_exp, mantissa = struct.unpack(">HQ", raw)
    exponent = (sign_exp & 0x7FFF) - 16_383
    if exponent == -16_383 and mantissa == 0:
        return 0.0
    value = mantissa / float(1 << 63) * (2.0**exponent)
    return -value if sign_exp & 0x8000 else value


def read_aiff(path: Path) -> Audio:
    """Minimal AIFF/AIFC reader for the big-endian PCM ``say`` produces.

    Supports the ``NONE`` and ``twos`` (big-endian) and ``sowt`` (little-endian) codecs, which
    cover every PCM variant CoreAudio writes.
    """
    data = path.read_bytes()
    if data[:4] != b"FORM" or data[8:12] not in (b"AIFF", b"AIFC"):
        raise SynthesisError(f"{path} is not an AIFF file")
    pos, channels, bits, rate, codec, frames_raw = 12, 1, 16, 0.0, b"NONE", b""
    while pos + 8 <= len(data):
        chunk_id, size = data[pos : pos + 4], struct.unpack(">I", data[pos + 4 : pos + 8])[0]
        body = data[pos + 8 : pos + 8 + size]
        if chunk_id == b"COMM":
            channels, _n_frames, bits = struct.unpack(">hIh", body[:8])
            rate = _parse_extended(body[8:18])
            codec = body[18:22] if len(body) >= 22 else b"NONE"
        elif chunk_id == b"SSND":
            offset = struct.unpack(">I", body[:4])[0]
            frames_raw = body[8 + offset :]
        pos += 8 + size + (size & 1)
    if not frames_raw or rate <= 0:
        raise SynthesisError(f"{path} has no sample data")
    return Audio(_decode_frames(frames_raw, channels, bits, codec), round(rate))


def _decode_frames(raw: bytes, channels: int, bits: int, codec: bytes) -> np.ndarray:
    if bits != 16:
        raise SynthesisError(f"unsupported AIFF bit depth {bits}")
    dtype = "<i2" if codec == b"sowt" else ">i2"
    usable = len(raw) - len(raw) % (2 * channels)
    frames = np.frombuffer(raw[:usable], dtype=dtype).astype(np.float64) / 32_768.0
    if channels > 1:
        frames = frames.reshape(-1, channels).mean(axis=1)
    return frames.astype(np.float32)
