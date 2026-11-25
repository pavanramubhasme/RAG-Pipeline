"""Observability: traces (Phoenix via OpenTelemetry, or LangSmith), Prometheus metrics, JSON logs.

Two tracing backends, chosen with RAG_TRACING_BACKEND:

* ``phoenix`` (default): the OpenInference instrumentation for LangChain turns every LangGraph
  node, prompt, LLM call (with token counts) and retriever result into an OpenTelemetry span,
  sent over standard OTLP to a local, open-source Arize Phoenix. Traces never leave the
  machine, which matters for CONFIDENTIAL chunks. Langfuse, Jaeger or Tempo work by URL.
* ``langsmith``: LangChain's own hosted tracing. LangChain and LangGraph report every run to
  LangSmith natively; this module only turns it on in code with a client of our own. By
  default the client hides inputs and outputs, so questions, chunks and answers stay here and
  LangSmith shows the structure, timings and errors (RAG_LANGSMITH_HIDE_CONTENT=false for demo
  data).
"""

import json
import logging

from prometheus_client import Counter, Gauge, Histogram

from ragapp.config import Settings

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------- metrics
HTTP_REQUESTS = Counter("rag_http_requests_total", "HTTP requests", ["method", "path", "status"])
HTTP_LATENCY = Histogram("rag_http_request_duration_seconds", "HTTP request latency", ["path"])
STAGE_LATENCY = Histogram(
    "rag_stage_duration_seconds",
    "Latency of each LangGraph node",
    ["stage"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
REFUSALS = Counter("rag_refusals_total", "Questions answered with 'I don't know'")
REWRITES = Counter("rag_query_rewrites_total", "Agent retries with a rewritten query")
UNGROUNDED = Counter("rag_ungrounded_answers_total", "Answers failing the grounding check")
CHUNKS_INDEXED = Gauge("rag_chunks_indexed", "Chunks in the vector store")
GUARDRAIL_EVENTS = Counter(
    "rag_guardrail_events_total",
    "Guardrail actions: blocked requests, redactions, quarantined chunks",
    ["event"],
)
ANSWER_CACHE = Counter(
    "rag_answer_cache_total", "Answer cache lookups", ["result"]
)  # hit | similar_hit | miss

_active: dict = {}  # running backend: {"backend": "phoenix", "provider": ...} or "langsmith" + "client"


# ----------------------------------------------------------------- tracing
def _parse_headers(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    return dict(item.split("=", 1) for item in raw.split(",") if "=" in item)


def setup_tracing(settings: Settings, exporter=None, langsmith_client=None):
    """Start tracing once per process; returns the tracer provider (Phoenix), the LangSmith
    client, or None if tracing is off. ``exporter``/``langsmith_client`` are for tests."""
    if _active:
        return _active.get("provider") or _active.get("client")
    if exporter is not None:
        return _setup_phoenix(settings, exporter)
    if langsmith_client is not None:
        return _setup_langsmith(settings, langsmith_client)
    if not settings.tracing_enabled:
        return None
    if settings.tracing_backend == "langsmith":
        return _setup_langsmith(settings)
    return _setup_phoenix(settings)


def _setup_phoenix(settings: Settings, exporter=None):
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.semconv.resource import ResourceAttributes
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

    provider = TracerProvider(
        resource=Resource.create(
            {"service.name": settings.service_name, ResourceAttributes.PROJECT_NAME: settings.service_name}
        )
    )
    if exporter is None:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(
            endpoint=settings.otlp_endpoint, headers=_parse_headers(settings.otlp_headers)
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))  # async, batched: no request slowdown
    else:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    LangChainInstrumentor().instrument(tracer_provider=provider)
    _active.update(backend="phoenix", provider=provider)
    logger.info("tracing_enabled", extra={"backend": "phoenix", "endpoint": settings.otlp_endpoint})
    return provider


def _setup_langsmith(settings: Settings, client=None):
    from langsmith import Client
    from langsmith.run_trees import configure

    if client is None:
        if settings.langsmith_api_key is None:
            logger.warning("tracing_disabled", extra={"reason": "RAG_LANGSMITH_API_KEY is not set"})
            return None
        hide = True if settings.langsmith_hide_content else None
        client = Client(
            api_key=settings.langsmith_api_key.get_secret_value(),
            api_url=settings.langsmith_endpoint,
            hide_inputs=hide,
            hide_outputs=hide,
        )
    # In code, not via LANGSMITH_* variables: the key stays in our settings, and it can't
    # collide with a LANGSMITH_TRACING left in someone's shell.
    configure(client=client, enabled=True, project_name=settings.langsmith_project)
    _active.update(backend="langsmith", client=client)
    logger.info(
        "tracing_enabled",
        extra={
            "backend": "langsmith",
            "project": settings.langsmith_project,
            "content_hidden": settings.langsmith_hide_content,
        },
    )
    return client


def tracing_backend() -> str | None:
    """The running backend ("phoenix", "langsmith") or None."""
    return _active.get("backend")


def traces_url(settings: Settings) -> str | None:
    """Where to look at the traces in a browser, for the UI's link (None: the UI's own default)."""
    if tracing_backend() != "langsmith":
        return None
    endpoint = settings.langsmith_endpoint or ""
    return "https://eu.smith.langchain.com" if "eu." in endpoint else "https://smith.langchain.com"


def shutdown_tracing() -> None:
    """Send what is still buffered and stop (end of the CLI, API shutdown, tests)."""
    backend = _active.get("backend")
    if backend == "phoenix":
        from openinference.instrumentation.langchain import LangChainInstrumentor

        _active["provider"].shutdown()  # flush pending spans
        LangChainInstrumentor().uninstrument()
    elif backend == "langsmith":
        from langsmith.run_trees import configure

        _active["client"].flush()
        configure(client=None, enabled=None, project_name=None)
    _active.clear()


# -------------------------------------------------------------------- logs
_STANDARD = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        data.update({k: v for k, v in vars(record).items() if k not in _STANDARD})
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "httpx2", "httpcore2", "opentelemetry", "langsmith"):
        logging.getLogger(noisy).setLevel(logging.WARNING)