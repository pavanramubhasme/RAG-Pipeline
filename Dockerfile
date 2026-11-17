FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    RAG_MODEL_CACHE_DIR=/models

WORKDIR /app

# Dependencies first: exact versions from uv.lock, runtime only (no eval/dev/ui groups).
# This layer stays cached until pyproject.toml or uv.lock changes.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-default-groups --no-install-project

COPY ragapp ./ragapp
COPY config ./config

# Bake the open-source embedding/BM25/reranker models into the image:
# fast cold starts, no runtime downloads, works in air-gapped environments.
RUN python -m ragapp.cli warmup

COPY data ./data
COPY examples ./examples

RUN useradd --create-home --uid 1000 app && chown -R app /app /models
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/live', timeout=4)"

# Each worker loads its own copy of the models (~300 MB). Scale with replicas, not workers.
CMD ["uvicorn", "ragapp.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
