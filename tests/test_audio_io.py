"""WAV I/O, resampling and level utilities."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from echo.audio.io import (
    Audio,
    AudioFormatError,
    ensure_16k,
    float_to_pcm16,
    pcm_to_float,
    read_wav,
    resample,
    rms_db,
    silence,
    to_mono,
    wav_bytes,
    write_wav,
)


def test_write_then_read_round_trip(tmp_path):
    samples = (0.5 * np.sin(np.linspace(0, 200, 16_000))).astype(np.float32)
    path = write_wav(tmp_path / "nested" / "tone.wav", Audio(samples, 16_000))
    back = read_wav(path)
    assert back.sample_rate == 16_000
    assert back.samples.dtype == np.float32
    assert np.max(np.abs(back.samples - samples)) < 1.0 / 32_000


def test_wav_bytes_matches_file(tmp_path):
    audio = Audio(np.linspace(-0.9, 0.9, 1000).astype(np.float32), 8000)
    assert wav_bytes(audio) == write_wav(tmp_path / "a.wav", audio).read_bytes()


def test_read_bytes_source():
    audio = Audio(np.zeros(100, dtype=np.float32), 16_000)
    assert read_wav(wav_bytes(audio)).duration == pytest.approx(100 / 16_000)


def test_stereo_is_averaged_to_mono():
    frames = np.array([[0.5, -0.5], [1.0, 0.0]], dtype=np.float32)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(float_to_pcm16(frames.reshape(-1)))
    mono = read_wav(buffer.getvalue())
    assert mono.samples.shape == (2,)
    assert mono.samples[0] == pytest.approx(0.0, abs=1e-4)
    assert mono.samples[1] == pytest.approx(0.5, abs=1e-4)


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_pcm_widths_decode_full_scale(width):
    if width == 1:
        raw = bytes([255, 128, 0])
        expected = [127 / 128, 0.0, -1.0]
    elif width == 2:
        raw = np.array([32767, 0, -32768], dtype="<i2").tobytes()
        expected = [32767 / 32768, 0.0, -1.0]
    elif width == 3:
        raw = b"\xff\xff\x7f" + b"\x00\x00\x00" + b"\x00\x00\x80"
        expected = [(2**23 - 1) / 2**23, 0.0, -1.0]
    else:
        raw = np.array([2**31 - 1, 0, -(2**31)], dtype="<i4").tobytes()
        expected = [(2**31 - 1) / 2**31, 0.0, -1.0]
    assert pcm_to_float(raw, width).tolist() == pytest.approx(expected, abs=1e-6)


def test_unsupported_width_raises():
    with pytest.raises(AudioFormatError):
        pcm_to_float(b"\x00" * 10, 5)


def test_garbage_bytes_raise_format_error():
    with pytest.raises(AudioFormatError):
        read_wav(b"RIFF not really a wav file at all")


def test_float_to_pcm16_clips_and_handles_nan():
    out = np.frombuffer(float_to_pcm16(np.array([2.0, -2.0, np.nan, 0.5])), dtype="<i2")
    assert out.tolist() == [32767, -32768, 0, 16384]


def test_resample_preserves_duration_and_tone():
    sr = 44_100
    t = np.arange(sr) / sr
    audio = Audio(np.sin(2 * np.pi * 440 * t).astype(np.float32), sr)
    out = ensure_16k(audio)
    assert out.sample_rate == 16_000
    assert out.duration == pytest.approx(1.0, abs=1e-3)
    spectrum = np.abs(np.fft.rfft(out.samples))
    assert abs(np.argmax(spectrum) * 16_000 / out.samples.shape[0] - 440) < 2


def test_resample_same_rate_is_identity():
    audio = Audio(np.ones(10, dtype=np.float32), 16_000)
    assert resample(audio, 16_000) is audio


def test_rms_db_values():
    assert rms_db(np.zeros(10)) == pytest.approx(-100.0)
    assert rms_db(np.ones(10)) == pytest.approx(0.0)
    assert rms_db(np.array([])) == pytest.approx(-100.0)


def test_silence_and_to_mono():
    assert silence(0.5, 16_000).shape == (8000,)
    assert to_mono(np.array([1.0, 3.0, 5.0]), 1).tolist() == [1.0, 3.0, 5.0]
    assert to_mono(np.array([1.0, 3.0, 5.0, 7.0, 9.0]), 2).tolist() == [2.0, 6.0]


def test_audio_rejects_bad_shapes():
    with pytest.raises(AudioFormatError):
        Audio(np.zeros((2, 10), dtype=np.float32), 16_000)
    with pytest.raises(AudioFormatError):
        Audio(np.zeros(10, dtype=np.float32), 0)


@settings(max_examples=30, deadline=None)
@given(st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=1, max_size=500))
def test_pcm16_round_trip_within_one_lsb(values):
    samples = np.array(values, dtype=np.float32)
    back = pcm_to_float(float_to_pcm16(samples), 2)
    assert np.max(np.abs(back - samples)) <= 1.0 / 32_768 + 1e-6
