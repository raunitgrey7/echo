"""The say harness and the numpy AIFF decoder."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from echo.audio.synth import (
    SynthesisError,
    _parse_extended,
    cache_key,
    list_voices,
    read_aiff,
    say_available,
    synthesize,
)

needs_say = pytest.mark.skipif(not say_available(), reason="macOS say not available")


def _extended(value: float) -> bytes:
    exponent = int(np.floor(np.log2(value)))
    mantissa = int(value / 2.0**exponent * (1 << 63))
    return struct.pack(">HQ", exponent + 16_383, mantissa)


def write_aiff(path, samples: np.ndarray, rate: int, channels: int = 1, codec: bytes = b"NONE") -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype(">i2" if codec != b"sowt" else "<i2").tobytes()
    comm = struct.pack(">hIh", channels, len(samples) // channels, 16) + _extended(float(rate))
    if codec != b"NONE":
        comm += codec + b"\x00"
    ssnd = struct.pack(">II", 0, 0) + pcm
    form_type = b"AIFC" if codec != b"NONE" else b"AIFF"
    body = form_type + b"COMM" + struct.pack(">I", len(comm)) + comm + (b"\x00" if len(comm) % 2 else b"")
    body += b"SSND" + struct.pack(">I", len(ssnd)) + ssnd
    path.write_bytes(b"FORM" + struct.pack(">I", len(body)) + body)


def test_parse_extended_known_rates():
    assert _parse_extended(_extended(22_050.0)) == pytest.approx(22_050.0)
    assert _parse_extended(_extended(44_100.0)) == pytest.approx(44_100.0)
    assert _parse_extended(b"\x00" * 10) == 0.0


def test_read_aiff_big_endian(tmp_path):
    samples = np.linspace(-0.5, 0.5, 2205).astype(np.float32)
    path = tmp_path / "a.aiff"
    write_aiff(path, samples, 22_050)
    audio = read_aiff(path)
    assert audio.sample_rate == 22_050
    assert np.max(np.abs(audio.samples - samples)) < 1e-3


def test_read_aiff_sowt_little_endian_stereo(tmp_path):
    left = np.full(100, 0.5, dtype=np.float32)
    right = np.full(100, -0.5, dtype=np.float32)
    interleaved = np.column_stack([left, right]).reshape(-1)
    path = tmp_path / "b.aifc"
    write_aiff(path, interleaved, 16_000, channels=2, codec=b"sowt")
    audio = read_aiff(path)
    assert audio.samples.shape == (100,)
    assert np.allclose(audio.samples, 0.0, atol=1e-3)


def test_read_aiff_rejects_other_files(tmp_path):
    path = tmp_path / "x.aiff"
    path.write_bytes(b"RIFF....WAVEfmt ")
    with pytest.raises(SynthesisError):
        read_aiff(path)


def test_cache_key_depends_on_all_inputs():
    base = cache_key("hello", "Samantha", 175)
    assert base != cache_key("hello", "Daniel", 175)
    assert base != cache_key("hello", "Samantha", 200)
    assert base != cache_key("hello there", "Samantha", 175)
    assert base == cache_key("hello", "Samantha", 175)
    assert base.startswith("samantha_175_")


@needs_say
def test_list_voices_includes_english_defaults():
    names = {v.name for v in list_voices("en")}
    assert "Samantha" in names
    assert all(v.locale.startswith("en") for v in list_voices("en"))


@needs_say
def test_synthesize_is_16k_mono_and_cached(tmp_path):
    first = synthesize("Cache test.", "Samantha", 175, tmp_path)
    assert first.sample_rate == 16_000
    assert 0.3 < first.duration < 3.0
    files = list(tmp_path.glob("*.wav"))
    assert len(files) == 1
    second = synthesize("Cache test.", "Samantha", 175, tmp_path)
    assert np.array_equal(first.samples, second.samples)
    assert len(list(tmp_path.glob("*.wav"))) == 1


def test_missing_say_raises(monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert not say_available()
    with pytest.raises(SynthesisError):
        synthesize("x", "Samantha", 175)
