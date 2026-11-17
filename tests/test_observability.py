"""Tracing: every LangGraph node, chain, LLM call and the retriever must produce spans."""

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ragapp import observability as obs


@pytest.fixture
def spans(settings):
    exporter = InMemorySpanExporter()
    obs.setup_tracing(settings, exporter=exporter)
    yield exporter
    obs.shutdown_tracing()


def test_query_produces_full_trace(indexed, spans):
    indexed.ask("What's the WFH quota?", request_id="req-123")
    finished = spans.get_finished_spans()
    names = [s.name for s in finished]

    assert "rag_query" in names  # root span
    for node in ("retrieve", "rewrite", "generate", "check_grounding"):
        assert node in names, node
    assert "HybridRerankRetriever" in names
    assert "FakeChatModel" in names  # every LLM call

    kinds = {s.attributes.get("openinference.span.kind") for s in finished}
    assert {"CHAIN", "RETRIEVER", "LLM"} <= kinds

    root = next(s for s in finished if s.name == "rag_query")
    assert "req-123" in root.attributes.get("metadata", "")  # API request ID links logs and traces
    assert {s.context.trace_id for s in finished} == {root.context.trace_id}  # one trace


def test_retriever_span_contains_documents(indexed, spans):
    indexed.ask("How many vacation days per calendar year?")
    retriever_span = next(s for s in spans.get_finished_spans() if s.name == "HybridRerankRetriever")
    assert any(k.startswith("retrieval.documents.0.document.content") for k in retriever_span.attributes)


def test_tracing_disabled_by_default(settings):
    assert obs.setup_tracing(settings) is None


def test_otlp_header_parsing():
    assert obs._parse_headers("Authorization=Basic abc==,x-team=rag") == {
        "Authorization": "Basic abc==",
        "x-team": "rag",
    }
