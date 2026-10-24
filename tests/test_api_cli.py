"""HTTP API contract and CLI entry points, all on the mock backend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from echo.api.app import DEMO_WAV, Settings, create_app
from echo.api.middleware import JsonFormatter, route_template
from echo.asr.mock import MockTranscriber
from echo.audio.io import Audio, wav_bytes
from echo.cli import main

FIXTURE = Path(__file__).parent / "fixtures" / "samantha_water_cycle.wav"


@pytest.fixture
def client(tmp_path):
    app = create_app(
        Settings(api_key=None, job_dir=tmp_path / "jobs", max_upload_mb=1.0, max_audio_s=100.0), MockTranscriber()
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def secured(tmp_path):
    app = create_app(Settings(api_key="s3cret", job_dir=tmp_path / "jobs"), MockTranscriber())
    with TestClient(app) as test_client:
        yield test_client


def upload(client: TestClient, payload: bytes, name: str = "clip.wav", **headers):
    return client.post(
        "/v1/jobs", files={"file": (name, payload, "audio/wav")}, data={"title": "Clip"}, headers=headers
    )


def test_health_and_index(client):
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["backend"] == "mock" and health["auth_required"] is False
    assert client.get("/").status_code == 200 and "echo" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


def test_job_lifecycle(client):
    response = upload(client, FIXTURE.read_bytes())
    assert response.status_code == 201
    assert response.headers["x-request-id"]
    job = response.json()
    assert job["title"] == "Clip" and job["transcript"]["segments"] and job["n_speakers"] >= 1
    job_id = job["id"]
    assert client.get("/v1/jobs").json()[0]["id"] == job_id
    assert client.get(f"/v1/jobs/{job_id}").json()["id"] == job_id
    audio = client.get(f"/v1/jobs/{job_id}/audio")
    assert (
        audio.status_code == 200
        and audio.headers["content-type"].startswith("audio/wav")
        and audio.content[:4] == b"RIFF"
    )
    for fmt, marker in (("srt", "-->"), ("vtt", "WEBVTT"), ("txt", "Speaker 1")):
        body = client.get(f"/v1/jobs/{job_id}/transcript", params={"format": fmt})
        assert body.status_code == 200 and marker in body.text
    assert client.get(f"/v1/jobs/{job_id}/transcript", params={"format": "json"}).json()["segments"]
    assert client.get(f"/v1/jobs/{job_id}/transcript", params={"format": "doc"}).status_code == 422
    assert client.get(f"/v1/jobs/{job_id}/notes").text.startswith("# Clip")
    search = client.get(f"/v1/jobs/{job_id}/search", params={"q": "speech", "k": 3}).json()
    assert search["query"] == "speech" and isinstance(search["hits"], list)
    assert client.get(f"/v1/jobs/{job_id}/search", params={"q": ""}).status_code == 422


def test_not_found_carries_request_id(client):
    response = client.get("/v1/jobs/missing", headers={"X-Request-ID": "trace-123"})
    assert response.status_code == 404
    assert response.json() == {"detail": "job 'missing' not found", "request_id": "trace-123"}
    assert response.headers["x-request-id"] == "trace-123"


def test_input_validation(client):
    assert upload(client, b"definitely not audio").status_code == 415
    too_big = wav_bytes(Audio(np.zeros(16_000 * 40, dtype=np.float32), 16_000))
    assert upload(client, too_big).status_code == 413
    too_long = wav_bytes(Audio(np.zeros(4000 * 101, dtype=np.float32), 4000))
    assert len(too_long) < 1024 * 1024
    assert upload(client, too_long).status_code == 422
    too_short = wav_bytes(Audio(np.zeros(100, dtype=np.float32), 16_000))
    assert upload(client, too_short).status_code == 422


def test_metrics_endpoint_counts_requests(client):
    client.get("/health")
    body = client.get("/metrics").text
    assert 'echo_http_requests_total{method="GET",path="/health",status="200"}' in body
    assert "echo_http_request_duration_seconds_bucket" in body
    upload(client, FIXTURE.read_bytes())
    body = client.get("/metrics").text
    assert 'echo_jobs_total{outcome="ok"} 1.0' in body
    assert "echo_audio_seconds_total" in body


def test_api_key_protection(secured):
    assert secured.get("/health").json()["auth_required"] is True
    assert secured.get("/v1/jobs").status_code == 401
    assert secured.get("/v1/jobs", headers={"X-API-Key": "wrong"}).status_code == 401
    assert secured.get("/v1/jobs", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert secured.get("/metrics").status_code == 200
    response = upload(secured, FIXTURE.read_bytes(), **{"X-API-Key": "s3cret"})
    assert response.status_code == 201


def test_demo_endpoint(client):
    response = client.post("/v1/demo")
    assert response.status_code == 201
    assert response.json()["title"] == "Demo meeting"
    assert DEMO_WAV.exists()


def test_route_template_and_json_logs():
    assert route_template({"path": "/v1/jobs/abc123/audio"}) == "/v1/jobs/{job_id}/audio"
    assert route_template({"path": "/health"}) == "/health"
    import logging

    record = logging.LogRecord("echo", logging.INFO, __file__, 1, "request", None, None)
    record.extra_fields = {"request_id": "r1", "status": 200}
    line = json.loads(JsonFormatter().format(record))
    assert line["message"] == "request" and line["request_id"] == "r1" and line["level"] == "INFO"


def test_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_API_KEY", "k")
    monkeypatch.setenv("ECHO_MAX_UPLOAD_MB", "7")
    monkeypatch.setenv("ECHO_JOB_DIR", str(tmp_path))
    settings = Settings.from_env()
    assert settings.api_key == "k" and settings.max_upload_mb == 7.0 and settings.job_dir == tmp_path


def test_cli_transcribe_and_analyze(tmp_path, capsys):
    assert main(["transcribe", str(FIXTURE), "--backend", "mock", "--format", "srt"]) == 0
    assert "-->" in capsys.readouterr().out
    assert main(["transcribe", str(FIXTURE), "--backend", "mock", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["segments"]
    out_dir = tmp_path / "out"
    assert main(["analyze", str(FIXTURE), "--backend", "mock", "--out", str(out_dir)]) == 0
    assert {p.name for p in out_dir.iterdir()} >= {
        "analysis.json",
        "transcript.srt",
        "transcript.vtt",
        "transcript.txt",
        "notes.md",
        "actions.json",
        "keywords.json",
    }


def test_cli_version_and_bad_command(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "echo 0.1.0" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["nonsense"])
