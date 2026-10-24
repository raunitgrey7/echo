"""Choose a backend from configuration, with a documented fallback.

Why: the same code path must serve a laptop with whisper cached, a CI runner without it and a
test that wants the mock on purpose. ``auto`` prefers whisper and falls back to the mock, and
the choice is logged so a benchmark can never silently report mock numbers as recognition.
"""

from __future__ import annotations

import logging
import os

from echo.asr.base import Transcriber
from echo.asr.mock import MockTranscriber, Oracle
from echo.asr.whisper import WhisperTranscriber

log = logging.getLogger("echo.asr")


def env_backend() -> str:
    """Backend name from the environment: ``auto``, ``whisper`` or ``mock``."""
    return os.environ.get("ECHO_ASR_BACKEND", "auto").lower()


def env_model() -> str:
    """Whisper model size from the environment."""
    return os.environ.get("ECHO_WHISPER_MODEL", "base")


def env_threads() -> int:
    """CPU threads for whisper; modest by default because other work shares the machine."""
    return int(os.environ.get("ECHO_CPU_THREADS", "4"))


def resolve_transcriber(
    backend: str | None = None,
    model_size: str | None = None,
    oracle: Oracle | None = None,
    mock_error_rate: float = 0.0,
) -> Transcriber:
    """Build the configured transcriber; raises for an unknown name or an unavailable whisper."""
    name = (backend or env_backend()).lower()
    size = model_size or env_model()
    allow_download = os.environ.get("ECHO_ALLOW_DOWNLOAD", "0") == "1"
    if name == "mock":
        return MockTranscriber(oracle=oracle, error_rate=mock_error_rate)
    if name == "whisper":
        if not WhisperTranscriber.available(size, allow_download):
            raise RuntimeError(
                f"faster-whisper model {size!r} is not cached; set ECHO_ALLOW_DOWNLOAD=1 or use ECHO_ASR_BACKEND=mock"
            )
        return WhisperTranscriber(model_size=size, cpu_threads=env_threads(), allow_download=allow_download)
    if name == "auto":
        if WhisperTranscriber.available(size, allow_download):
            log.info("asr backend: faster-whisper %s", size)
            return WhisperTranscriber(model_size=size, cpu_threads=env_threads(), allow_download=allow_download)
        log.warning("asr backend: faster-whisper %s not available, using the deterministic mock", size)
        return MockTranscriber(oracle=oracle, error_rate=mock_error_rate)
    raise ValueError(f"unknown ASR backend {name!r}; expected auto, whisper or mock")
