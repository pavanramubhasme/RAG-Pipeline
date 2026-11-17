import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ragapp.api import create_app
from ragapp.config import Settings, get_settings
from ragapp.service import RAGService
from ragapp.store import KnowledgeStore
from tests.fakes import FakeChatModel, FakeEmbeddings, FakeReranker, FakeSparseEmbeddings

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"


def make_service(settings: Settings, llm: FakeChatModel | None = None, guard_llm=None) -> RAGService:
    return RAGService(
        store=KnowledgeStore.from_settings(settings, FakeEmbeddings(), FakeSparseEmbeddings()),
        reranker=FakeReranker(),
        llm=llm or FakeChatModel(calls=[], prompts=[]),
        settings=settings,
        llm_health=lambda: True,
        guard_llm=guard_llm,
    )


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """Tests never see the developer's configuration: uv loads .env into the environment
    (UV_ENV_FILE), and pydantic-settings would read .env too. Seen live: RAG_API_KEY in .env
    made every API test fail with 401."""
    for name in list(os.environ):
        if name.upper().startswith("RAG_"):
            monkeypatch.delenv(name)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    # Access control off: these tests are about retrieval and the agent; tests/test_access.py turns it on.
    return Settings(
        qdrant_url=None,
        qdrant_path=":memory:",
        collection="test",
        min_relevance=0.3,
        tracing_enabled=False,
        access_control=False,
    )


@pytest.fixture
def fake_llm() -> FakeChatModel:
    return FakeChatModel(calls=[], prompts=[])


@pytest.fixture
def service(settings, fake_llm) -> RAGService:
    svc = make_service(settings, fake_llm)
    yield svc
    svc.close()


@pytest.fixture
def indexed(service) -> RAGService:
    service.ingest([SAMPLE_DIR])
    return service


@pytest.fixture
def client(indexed, settings):
    app = create_app(settings=settings, service_factory=lambda: indexed)
    with TestClient(app) as test_client:
        yield test_client