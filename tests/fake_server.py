"""Run the real API with test doubles (no models, no Ollama). For smoke tests and UI work:

python -m tests.fake_server --port 8000
python -m tests.fake_server --docs tests/data/secure     # labelled corpus: try the demo users
"""

import argparse
from pathlib import Path

import uvicorn

from ragapp.api import create_app
from ragapp.config import Settings
from tests.conftest import SAMPLE_DIR, make_service


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--docs", type=Path, default=SAMPLE_DIR)
    args = parser.parse_args()
    settings = Settings(qdrant_url=None, qdrant_path=":memory:", collection="fake", min_relevance=0.3)

    def factory():
        service = make_service(settings)
        service.ingest([args.docs])
        return service

    uvicorn.run(create_app(settings=settings, service_factory=factory), port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
