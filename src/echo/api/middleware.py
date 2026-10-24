"""Request IDs, structured JSON logs and Prometheus metrics in one ASGI middleware.

Why: an API that will sit behind a reverse proxy needs three things an operator can rely on: a
request id that travels through logs, one JSON line per request with method, path, status and
latency, and counters a Prometheus scraper can read. Doing all three in one place keeps the
handlers free of plumbing.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Histogram

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

REQUEST_ID_HEADER = b"x-request-id"


class JsonFormatter(logging.Formatter):
    """One JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise one record, merging ``extra_fields`` when a caller attached them."""
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Route the ``echo`` logger to stderr as JSON lines; idempotent."""
    logger = logging.getLogger("echo")
    if not any(isinstance(h.formatter, JsonFormatter) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level.upper())
    return logger


class Metrics:
    """Prometheus collectors for the HTTP layer and the audio pipeline."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "echo_http_requests_total", "HTTP requests", ["method", "path", "status"], registry=self.registry
        )
        self.latency = Histogram(
            "echo_http_request_duration_seconds",
            "HTTP request latency",
            ["method", "path"],
            registry=self.registry,
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
        )
        self.audio_seconds = Counter("echo_audio_seconds_total", "Audio seconds analysed", registry=self.registry)
        self.jobs = Counter("echo_jobs_total", "Analysis jobs by outcome", ["outcome"], registry=self.registry)
        self.pipeline_seconds = Histogram(
            "echo_pipeline_seconds",
            "Wall seconds per analysis",
            registry=self.registry,
            buckets=(0.5, 1, 2, 5, 10, 20, 40, 80, 160, 320),
        )


def route_template(scope: Scope) -> str:
    """Path with the job id collapsed so metric labels stay bounded."""
    parts = scope.get("path", "/").split("/")
    if len(parts) > 3 and parts[1] == "v1" and parts[2] == "jobs":
        parts[3] = "{job_id}"
    return "/".join(parts)


class RequestContextMiddleware:
    """Pure ASGI middleware: request id in, request id out, one log line, metrics updated."""

    def __init__(self, app: ASGIApp, metrics: Metrics, logger: logging.Logger) -> None:
        self.app = app
        self.metrics = metrics
        self.logger = logger

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        request_id = headers.get(REQUEST_ID_HEADER, b"").decode() or uuid.uuid4().hex[:16]
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.perf_counter()
        status_holder = {"status": 500}

        async def send_with_id(message: MutableMapping[str, Any]) -> None:
            """Stamp the response with the request id."""
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                message["headers"] = [*message.get("headers", []), (REQUEST_ID_HEADER, request_id.encode())]
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            self._record(scope, request_id, status_holder["status"], time.perf_counter() - started)

    def _record(self, scope: Scope, request_id: str, status: int, elapsed: float) -> None:
        method, template = scope.get("method", "GET"), route_template(scope)
        self.metrics.requests.labels(method, template, str(status)).inc()
        self.metrics.latency.labels(method, template).observe(elapsed)
        fields = {
            "request_id": request_id,
            "method": method,
            "path": scope.get("path"),
            "status": status,
            "duration_ms": round(elapsed * 1000, 2),
        }
        self.logger.info("request", extra={"extra_fields": fields})
