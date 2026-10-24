"""faster-whisper adapter: CPU int8 inference with word timestamps.

Why: whisper is the strongest open model that runs on a laptop CPU, and the CTranslate2 build
is several times faster than the reference implementation. The adapter passes numpy audio
directly, so no ffmpeg is needed, decodes greedily with beam search at temperature zero so a
given input always yields the same output, and loads the model lazily so importing echo stays
cheap. Models are read from the Hugging Face cache; downloads happen only when explicitly allowed.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from echo.asr.base import Segment, TranscriptionResult, Word
from echo.audio.io import Audio, ensure_16k

MODEL_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
}


def hf_cache_dir() -> Path:
    """Hugging Face hub cache location, honouring HF_HOME and HF_HUB_CACHE."""
    if explicit := os.environ.get("HF_HUB_CACHE"):
        return Path(explicit)
    home = os.environ.get("HF_HOME")
    return Path(home) / "hub" if home else Path.home() / ".cache" / "huggingface" / "hub"


def model_cached(model_size: str) -> bool:
    """True when the converted model directory exists locally with its weights."""
    repo = MODEL_REPOS.get(model_size)
    if repo is None:
        return False
    folder = hf_cache_dir() / f"models--{repo.replace('/', '--')}" / "snapshots"
    return folder.exists() and any(folder.glob("*/model.bin"))


def whisper_importable() -> bool:
    """True when faster-whisper and its native runtime import cleanly."""
    try:
        import faster_whisper  # noqa: F401
    except Exception:  # pragma: no cover - depends on the environment
        return False
    return True


class WhisperTranscriber:
    """Lazy-loading faster-whisper backend."""

    name = "faster-whisper"

    def __init__(
        self,
        model_size: str = "base",
        compute_type: str = "int8",
        cpu_threads: int = 4,
        beam_size: int = 5,
        vad_filter: bool = False,
        allow_download: bool = False,
    ) -> None:
        self.model_size = model_size
        self.model = f"{model_size}-{compute_type}"
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.allow_download = allow_download
        self._engine: Any | None = None

    @classmethod
    def available(cls, model_size: str = "base", allow_download: bool = False) -> bool:
        """True when this backend can run here without surprises."""
        return whisper_importable() and (allow_download or model_cached(model_size))

    def _load(self) -> Any:
        if self._engine is None:
            from faster_whisper import WhisperModel

            self._engine = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                local_files_only=not self.allow_download,
            )
        return self._engine

    def warm_up(self) -> None:
        """Load the weights now so the first request is not charged for it."""
        self._load()

    def transcribe(self, audio: Audio, language: str = "en") -> TranscriptionResult:
        """Recognise the audio with word timestamps; deterministic for a given input."""
        engine = self._load()
        audio16 = ensure_16k(audio)
        samples = np.ascontiguousarray(audio16.samples, dtype=np.float32)
        started = time.perf_counter()
        raw_segments, info = engine.transcribe(
            samples,
            language=language,
            beam_size=self.beam_size,
            word_timestamps=True,
            vad_filter=self.vad_filter,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        segments = tuple(_convert(index, seg) for index, seg in enumerate(raw_segments))
        return TranscriptionResult(
            segments=segments,
            language=getattr(info, "language", language),
            backend=self.name,
            model=self.model,
            elapsed_s=time.perf_counter() - started,
            audio_s=audio16.duration,
        )


def _convert(index: int, seg: Any) -> Segment:
    words = tuple(
        Word(w.word.strip(), float(w.start), float(w.end), float(w.probability), w.word.startswith(" "))
        for w in (seg.words or [])
    )
    return Segment(index, float(seg.start), float(seg.end), seg.text.strip(), words)
