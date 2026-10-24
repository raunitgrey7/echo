# Multi-stage build: dependencies resolved with uv in a builder, copied into a slim runtime
# that runs as an unprivileged user. The image serves the API and dashboard; whisper weights are
# downloaded on first use when ECHO_ALLOW_DOWNLOAD=1 (or mounted into /home/echo/.cache), and the
# deterministic mock is used otherwise. macOS `say` is not available in Linux, so the demo endpoint
# falls back to the bundled recording.

FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.13-slim AS runtime
RUN groupadd --system echo && useradd --system --gid echo --create-home --home-dir /home/echo echo
WORKDIR /app
COPY --from=builder --chown=echo:echo /app /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    ECHO_JOB_DIR=/data/jobs \
    ECHO_DATA_DIR=/data \
    ECHO_ASR_BACKEND=auto \
    HF_HOME=/home/echo/.cache/huggingface
RUN mkdir -p /data/jobs && chown -R echo:echo /data
USER echo
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)"
CMD ["uvicorn", "echo.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
