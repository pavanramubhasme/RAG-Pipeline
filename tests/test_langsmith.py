"""LangSmith tracing (RAG_TRACING_BACKEND=langsmith): every LangGraph node and LLM call is
reported as a run, and by default questions, chunks and answers are hidden before sending.

The client is the real ``langsmith.Client``; only its two send methods are replaced, so the
test sees exactly what would go over the network (after hiding) and needs no account."""

import pytest
from langsmith import Client
from langsmith.run_helpers import get_tracing_context

from ragapp import observability as obs
from ragapp.config import Settings


class RecordingClient(Client):
    """A LangSmith client that records runs instead of sending them."""

    def __init__(self, **kwargs):
        super().__init__(
            api_key="test-key", api_url="http://langsmith.invalid", auto_batch_tracing=False, **kwargs
        )
        self.created: list[dict] = []
        self.updated: list[dict] = []

    def _create_run(self, run_create, **kwargs):
        self.created.append(run_create)

    def _update_run(self, run_update, **kwargs):
        self.updated.append(run_update)


@pytest.fixture
def langsmith_runs(settings):
    def start(hide: bool) -> RecordingClient:
        client = RecordingClient(hide_inputs=True if hide else None, hide_outputs=True if hide else None)
        obs.setup_tracing(
            settings.model_copy(update={"langsmith_project": "rag-test"}), langsmith_client=client
        )
        return client

    yield start
    obs.shutdown_tracing()


def _text(runs: list[dict]) -> str:
    return " ".join(str(run.get("inputs", "")) + str(run.get("outputs", "")) for run in runs)


def test_every_node_and_llm_call_becomes_a_run(indexed, langsmith_runs):
    client = langsmith_runs(hide=False)
    indexed.ask("What's the WFH quota?")
    names = {run["name"] for run in client.created}
    for node in ("retrieve", "rewrite", "generate", "check_grounding"):
        assert node in names, node
    assert "FakeChatModel" in names and "HybridRerankRetriever" in names
    assert {run["session_name"] for run in client.created} == {"rag-test"}
    assert obs.tracing_backend() == "langsmith"


def test_content_is_hidden_by_default(indexed, langsmith_runs):
    client = langsmith_runs(hide=True)
    indexed.ask("How many vacation days per calendar year?")
    assert {run["name"] for run in client.created} >= {"retrieve", "generate"}  # structure is there
    sent = _text(client.created + client.updated)
    assert "vacation" not in sent.lower()  # the question
    assert "30 days" not in sent  # the retrieved chunk and the answer
    everything = str(client.created + client.updated).lower()  # names, metadata, events, prompts
    assert "vacation" not in everything and "30 days" not in everything


def test_content_is_sent_when_hiding_is_off(indexed, langsmith_runs):
    client = langsmith_runs(hide=False)
    indexed.ask("How many vacation days per calendar year?")
    assert "vacation" in _text(client.created + client.updated).lower()


def test_shutdown_switches_tracing_off_again(settings):
    obs.setup_tracing(settings, langsmith_client=RecordingClient())
    assert get_tracing_context()["enabled"] is True
    obs.shutdown_tracing()
    assert obs.tracing_backend() is None
    assert get_tracing_context()["enabled"] is None  # back to the default: not traced


def test_without_an_api_key_tracing_stays_off(settings):
    assert (
        obs.setup_tracing(
            settings.model_copy(update={"tracing_enabled": True, "tracing_backend": "langsmith"})
        )
        is None
    )
    assert obs.tracing_backend() is None


def test_settings_hide_content_by_default():
    s = Settings(tracing_backend="langsmith", langsmith_api_key="lsv2-secret")
    assert s.langsmith_hide_content is True
    assert "lsv2-secret" not in repr(s)  # the key never shows up in logs or tracebacks


def test_traces_link_follows_the_backend(settings):
    assert obs.traces_url(settings) is None  # nothing running: the UI uses its Phoenix default
    obs.setup_tracing(settings.model_copy(update={"langsmith_endpoint": "https://eu.api.smith.langchain.com"}),
                      langsmith_client=RecordingClient())  # fmt: skip
    try:
        eu = settings.model_copy(update={"langsmith_endpoint": "https://eu.api.smith.langchain.com"})
        assert obs.traces_url(eu) == "https://eu.smith.langchain.com"
        assert obs.traces_url(settings) == "https://smith.langchain.com"
    finally:
        obs.shutdown_tracing()


def test_health_reports_the_backend(service):
    assert service.health()["tracing"] is None
    obs.setup_tracing(service.settings, langsmith_client=RecordingClient())
    try:
        assert service.health()["tracing"] == "langsmith"
    finally:
        obs.shutdown_tracing()