"""Observability: traces (OpenTelemetry), metrics (Prometheus) and JSON logs.

Tracing uses the OpenInference instrumentation for LangChain. Every LangGraph
node, prompt, LLM call (with token counts) and retriever result becomes a span.
Spans go out over standard OTLP, so the backend is a config choice:
Arize Phoenix (default, local), Langfuse, Jaeger or Grafana Tempo.
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

_tracer_provider = None


# ----------------------------------------------------------------- tracing
def _parse_headers(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    return dict(item.split("=", 1) for item in raw.split(",") if "=" in item)


def setup_tracing(settings: Settings, exporter=None):
    """Instrument LangChain/LangGraph once per process. ``exporter`` is injectable for tests."""
    global _tracer_provider
    if _tracer_provider is not None:
        return _tracer_provider
    if not settings.tracing_enabled and exporter is None:
        return None

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
    _tracer_provider = provider
    logger.info("tracing_enabled", extra={"endpoint": settings.otlp_endpoint})
    return provider


def shutdown_tracing() -> None:
    global _tracer_provider
    if _tracer_provider is not None:
        from openinference.instrumentation.langchain import LangChainInstrumentor

        _tracer_provider.shutdown()  # flush pending spans
        LangChainInstrumentor().uninstrument()
        _tracer_provider = None


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
    for noisy in ("httpx", "httpcore", "httpx2", "httpcore2", "opentelemetry"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
