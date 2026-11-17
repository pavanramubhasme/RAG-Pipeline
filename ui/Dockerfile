# Build from the repo root:  docker build -f ui/Dockerfile .
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /ui
COPY pyproject.toml uv.lock .python-version ./
# Only the "ui" dependency group (streamlit + httpx), nothing else
RUN uv sync --frozen --only-group ui --no-install-project
COPY ui/app.py .
COPY config ./config

RUN useradd --create-home --uid 1000 app
USER app
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
