"""The FastAPI application.

Why: the API is how the product is used: upload a WAV, receive the transcript with speakers,
keywords with timestamps, notes, action items, and search within the recording. Jobs are kept in
memory and mirrored to disk so the dashboard can stream the audio back for the synced player and
a restart does not lose finished work. Authentication is a shared API key header, enabled only when
``APP_API_KEY`` is set, so local use needs no configuration.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from echo import __version__
from echo.api.middleware import Metrics, RequestContextMiddleware, configure_logging
from echo.api.schemas import HealthOut, JobOut, JobSummary, SearchOut
from echo.asr.base import Transcriber
from echo.asr.registry import resolve_transcriber
from echo.audio.io import Audio, AudioFormatError, read_wav, wav_bytes
from echo.audio.synth import say_available
from echo.corpus.build import MEETING_VOICE_SETS, RecordingSpec, compose
from echo.pipeline import Analysis, analysis_to_dict, analyze
from echo.products.transcript import to_srt, to_text, to_vtt

STATIC_DIR = Path(__file__).parent / "static"
ASSETS_DIR = Path(__file__).parents[1] / "assets"
DEMO_WAV = ASSETS_DIR / "demo_meeting.wav"
TRANSCRIPT_RENDERERS = {
    "srt": (to_srt, "application/x-subrip"),
    "vtt": (to_vtt, "text/vtt"),
    "txt": (to_text, "text/plain"),
}


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration read from the environment."""

    api_key: str | None = None
    backend: str = "auto"
    model_size: str = "base"
    job_dir: Path = Path("data") / "jobs"
    max_upload_mb: float = 50.0
    max_audio_s: float = 3600.0
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        """Read ``APP_API_KEY`` and the ``ECHO_*`` variables."""
        return cls(
            api_key=os.environ.get("APP_API_KEY") or None,
            backend=os.environ.get("ECHO_ASR_BACKEND", "auto"),
            model_size=os.environ.get("ECHO_WHISPER_MODEL", "base"),
            job_dir=Path(os.environ.get("ECHO_JOB_DIR", "data/jobs")),
            max_upload_mb=float(os.environ.get("ECHO_MAX_UPLOAD_MB", "50")),
            max_audio_s=float(os.environ.get("ECHO_MAX_AUDIO_S", "3600")),
            log_level=os.environ.get("ECHO_LOG_LEVEL", "INFO"),
        )


@dataclass(slots=True)
class Job:
    """A finished analysis and where its audio lives."""

    id: str
    title: str
    created_at: str
    audio_path: Path
    analysis: Analysis
    payload: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        """The list view."""
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "duration": self.analysis.duration,
            "backend": self.analysis.asr.backend,
            "model": self.analysis.asr.model,
            "n_speakers": self.analysis.speakers.n_speakers,
            "n_actions": len(self.analysis.actions),
        }


class JobStore:
    """In-memory jobs mirrored to ``job_dir`` as WAV plus JSON."""

    def __init__(self, job_dir: Path) -> None:
        self.job_dir = job_dir
        self.jobs: dict[str, Job] = {}

    def add(self, title: str, audio: Audio, analysis: Analysis) -> Job:
        """Persist and register a finished analysis."""
        job_id = uuid.uuid4().hex[:12]
        self.job_dir.mkdir(parents=True, exist_ok=True)
        audio_path = self.job_dir / f"{job_id}.wav"
        audio_path.write_bytes(wav_bytes(audio))
        payload = {"id": job_id, "title": title, "created_at": datetime.now(UTC).isoformat(timespec="seconds")}
        payload.update(analysis_to_dict(analysis))
        (self.job_dir / f"{job_id}.json").write_text(json.dumps(payload), encoding="utf-8")
        job = Job(job_id, title, payload["created_at"], audio_path, analysis, payload)
        self.jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job:
        """Fetch or raise 404."""
        job = self.jobs.get(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"job {job_id!r} not found")
        return job


def create_app(settings: Settings | None = None, transcriber: Transcriber | None = None) -> FastAPI:
    """Build the application; tests pass their own settings and transcriber."""
    cfg = settings or Settings.from_env()
    logger = configure_logging(cfg.log_level)
    metrics = Metrics()
    store = JobStore(cfg.job_dir)
    engine: dict[str, Transcriber] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        engine["asr"] = transcriber or resolve_transcriber(cfg.backend, cfg.model_size)
        logger.info("asr ready", extra={"extra_fields": {"backend": engine["asr"].name, "model": engine["asr"].model}})
        yield

    app = FastAPI(
        title="echo",
        version=__version__,
        description="Speech intelligence for lectures and meetings.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestContextMiddleware, metrics=metrics, logger=logger)
    app.state.settings, app.state.store, app.state.metrics, app.state.engine = cfg, store, metrics, engine
    _register_routes(app, cfg, store, metrics, engine)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _require_key(settings: Settings) -> Any:
    async def dependency(request: Request) -> None:
        if settings.api_key and request.headers.get("x-api-key") != settings.api_key:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing X-API-Key header")

    return dependency


def _register_routes(
    app: FastAPI, cfg: Settings, store: JobStore, metrics: Metrics, engine: dict[str, Transcriber]
) -> None:
    _register_core(app, cfg, store, metrics, engine)
    _register_jobs(app, cfg, store, metrics, engine)


def _register_core(
    app: FastAPI, cfg: Settings, store: JobStore, metrics: Metrics, engine: dict[str, Transcriber]
) -> None:
    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse({"detail": exc.detail, "request_id": request_id}, status_code=exc.status_code)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> str:
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/health", response_model=HealthOut)
    async def health() -> dict[str, Any]:
        asr = engine.get("asr")
        return {
            "status": "ok" if asr else "starting",
            "version": __version__,
            "backend": asr.name if asr else "none",
            "model": asr.model if asr else "none",
            "auth_required": cfg.api_key is not None,
            "jobs": len(store.jobs),
            "demo_available": say_available() or DEMO_WAV.exists(),
        }

    @app.get("/metrics", include_in_schema=False)
    async def prometheus() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)


def _register_jobs(
    app: FastAPI, cfg: Settings, store: JobStore, metrics: Metrics, engine: dict[str, Transcriber]
) -> None:
    auth = Depends(_require_key(cfg))
    created = status.HTTP_201_CREATED

    @app.post("/v1/jobs", response_model=JobOut, status_code=created, dependencies=[auth])
    async def create_job(
        file: Annotated[UploadFile, File()], title: Annotated[str | None, Form()] = None
    ) -> dict[str, Any]:
        raw = await file.read()
        if len(raw) > cfg.max_upload_mb * 1024 * 1024:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"upload exceeds {cfg.max_upload_mb:g} MB")
        try:
            audio = read_wav(raw)
        except AudioFormatError as exc:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"expected a PCM WAV file: {exc}") from exc
        return _run_job(cfg, store, metrics, engine, audio, title or (file.filename or "recording"))

    @app.post("/v1/demo", response_model=JobOut, status_code=created, dependencies=[auth])
    async def demo() -> dict[str, Any]:
        return _run_job(cfg, store, metrics, engine, demo_audio(), "Demo meeting")

    @app.get("/v1/jobs", response_model=list[JobSummary], dependencies=[auth])
    async def list_jobs() -> list[dict[str, Any]]:
        return [job.summary() for job in sorted(store.jobs.values(), key=lambda j: j.created_at, reverse=True)]

    @app.get("/v1/jobs/{job_id}", response_model=JobOut, dependencies=[auth])
    async def get_job(job_id: str) -> dict[str, Any]:
        return store.get(job_id).payload

    @app.get("/v1/jobs/{job_id}/audio", dependencies=[auth])
    async def get_audio(job_id: str) -> Response:
        return Response(store.get(job_id).audio_path.read_bytes(), media_type="audio/wav")

    @app.get("/v1/jobs/{job_id}/transcript", response_class=PlainTextResponse, dependencies=[auth])
    async def get_transcript(
        job_id: str, format: Annotated[str, Query(pattern="^(srt|vtt|txt|json)$")] = "txt"
    ) -> Response:
        job = store.get(job_id)
        if format == "json":
            return JSONResponse(job.payload["transcript"])
        render, media = TRANSCRIPT_RENDERERS[format]
        return PlainTextResponse(render(job.analysis.transcript), media_type=media)

    @app.get("/v1/jobs/{job_id}/notes", response_class=PlainTextResponse, dependencies=[auth])
    async def get_notes(job_id: str) -> str:
        job = store.get(job_id)
        return job.analysis.notes_markdown(job.title)

    @app.get("/v1/jobs/{job_id}/search", response_model=SearchOut, dependencies=[auth])
    async def search(
        job_id: str,
        q: Annotated[str, Query(min_length=1, max_length=200)],
        k: Annotated[int, Query(ge=1, le=50)] = 5,
    ) -> dict[str, Any]:
        hits = store.get(job_id).analysis.search_index().search(q, k)
        return {"query": q, "hits": [asdict(hit) for hit in hits]}


def _run_job(
    settings: Settings, store: JobStore, metrics: Metrics, engine: dict[str, Transcriber], audio: Audio, title: str
) -> dict[str, Any]:
    asr = engine.get("asr")
    if asr is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "ASR engine is still loading")
    if audio.duration > settings.max_audio_s:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"audio longer than {settings.max_audio_s:g} s")
    if audio.duration < 0.1:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "audio shorter than 0.1 s")
    try:
        analysis = analyze(audio, asr)
    except Exception:
        metrics.jobs.labels("error").inc()
        raise
    metrics.jobs.labels("ok").inc()
    metrics.audio_seconds.inc(analysis.duration)
    metrics.pipeline_seconds.observe(analysis.total_elapsed_s)
    return store.add(title, audio, analysis).payload


def demo_audio() -> Audio:
    """A short three-speaker meeting: synthesised with ``say`` when present, else the bundled WAV."""
    if say_available():
        spec = RecordingSpec("demo_product_sync", "meeting", "product_sync", MEETING_VOICE_SETS[0], seed=7)
        samples, _ = compose(spec, Path(os.environ.get("ECHO_DATA_DIR", "data")) / "cache", np.random.default_rng(7))
        return Audio(samples, 16_000)
    if DEMO_WAV.exists():
        return read_wav(DEMO_WAV)
    raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "no demo audio available on this machine")


app = create_app()
