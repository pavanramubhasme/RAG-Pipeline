"""RAGService: the one object the API, CLI, MCP server and evaluation all use."""

import logging
import re
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from ragapp import observability as obs
from ragapp.access import UNRESTRICTED, AccessPolicy, Identity
from ragapp.cache import AnswerCache
from ragapp.config import Settings
from ragapp.graph import build_graph
from ragapp.guardrails import Guardrails
from ragapp.ingestion import Ingestor
from ragapp.prompts import REFUSAL
from ragapp.reranker import FastEmbedReranker
from ragapp.retriever import HybridRerankRetriever
from ragapp.schemas import AccessInfo, Citation, IngestResponse, Passage, QueryResponse
from ragapp.store import KnowledgeStore

logger = logging.getLogger(__name__)
CITATION = re.compile(r"\[(\d+)\]")


class LLMError(RuntimeError):
    """The LLM backend failed (down, timeout, model missing)."""


class AccessDenied(PermissionError):
    """No identity may be assumed for this request (access control on, no anonymous role)."""


def to_passage(doc: Document) -> dict:
    m = doc.metadata
    return {
        "source": m.get("source", "unknown"),
        "section": m.get("section"),
        "page": m.get("page"),
        "classification": m.get("classification"),
        "score": m.get("relevance_score", m.get("search_score")),
        "text": doc.page_content,
    }


class RAGService:
    def __init__(
        self,
        *,
        store: KnowledgeStore,
        reranker: FastEmbedReranker,
        llm: BaseChatModel,
        settings: Settings,
        llm_health: Callable[[], bool] | None = None,  # injectable for tests
        guard_llm: BaseChatModel | None = None,  # optional guard model (Granite Guardian)
    ):
        self.store = store
        self.reranker = reranker
        self.llm = llm
        self.guard_llm = guard_llm
        self._llm_health = llm_health
        self.ingestor = Ingestor(store, settings)
        self.configure(settings)

    def configure(self, settings: Settings) -> None:
        """(Re)build retriever and graph from settings. Cheap: models and index are reused,
        which is how the evaluation compares variants (dense vs hybrid, linear vs agentic)."""
        self.settings = settings
        self.reranker.min_score = settings.min_relevance
        self.reranker.top_n = settings.candidates_k  # the retriever applies top_k itself
        self.retriever = HybridRerankRetriever(
            store=self.store,
            reranker=self.reranker if settings.rerank else None,
            mode=settings.retrieval_mode,
            candidates_k=settings.candidates_k,
            top_k=settings.top_k,
        )
        self.guardrails = Guardrails(settings, self.guard_llm) if settings.guardrails_enabled else None
        self.access = (
            AccessPolicy.from_file(settings.identities_file, settings.anonymous_role or None)
            if settings.access_control
            else None
        )
        # A fresh cache: answers produced under the previous settings must not be served.
        self.cache = (
            AnswerCache(settings.cache_max_entries, settings.cache_ttl_s) if settings.cache_enabled else None
        )
        self.ingestor.settings = settings
        self.graph = build_graph(
            self.retriever, self.llm, settings, guardrails=self.guardrails, cache=self.cache
        )

    # ------------------------------------------------------------------ ingest
    def ingest(self, paths: Iterable[Path | str]) -> IngestResponse:
        result = IngestResponse(**self.ingestor.ingest(paths))
        obs.CHUNKS_INDEXED.set(result.total_chunks)
        # Documents changed: answers given before may be stale.
        if result.chunks_added and self.cache is not None and (removed := self.cache.clear()):
            logger.info("answer_cache_cleared", extra={"entries": removed})
        return result

    # ------------------------------------------------------------------ search
    def search(self, query: str, top_k: int = 4, identity: Identity | None = None) -> list[Passage]:
        who = self.effective_identity(identity)
        docs = self.retriever.invoke(query, top_k=top_k, scopes=who.scope_list())
        return [Passage(**to_passage(d)) for d in docs if who.allows(d.metadata.get("classification"))]

    # ------------------------------------------------------------------ access
    def effective_identity(self, identity: Identity | None) -> Identity:
        """Who the request runs as. Access control off: unrestricted. On: the given identity,
        else the anonymous role (deny by default), else AccessDenied."""
        if self.access is None:
            return UNRESTRICTED
        if identity is not None:
            return identity
        if self.access.anonymous is None:
            raise AccessDenied("an identity is required")
        return self.access.anonymous

    # --------------------------------------------------------------------- ask
    def ask(
        self,
        question: str,
        top_k: int | None = None,
        request_id: str | None = None,
        history: list[dict] | None = None,
        identity: Identity | None = None,
    ) -> QueryResponse:
        """Answer one question as ``identity``. ``history`` = earlier turns [{"role", "content"}]."""
        start = time.perf_counter()
        who = self.effective_identity(identity)
        try:
            state = self.graph.invoke(
                {
                    "question": question,
                    "history": history or [],
                    "top_k": top_k,
                    "scopes": who.scope_list(),
                    "rewrites": 0,
                },
                config={
                    "run_name": "rag_query",
                    "metadata": {"request_id": request_id, "user": who.user, "role": who.role},
                },
            )
        except Exception as exc:  # connection refused, timeout, model not pulled, ...
            if _is_llm_failure(exc):
                raise LLMError(str(exc)) from exc
            raise

        docs: list[Document] = state.get("documents", [])
        standalone: str | None = state.get("standalone")
        guardrail_events: list[str] = state.get("guardrails", [])
        answer: str = state["answer"]
        refused: bool = state["refused"]
        if any(not who.allows(d.metadata.get("classification")) for d in docs):
            # Defense in depth: the search filter makes this impossible. If it ever happens, fail closed.
            logger.error("acl_violation_blocked", extra={"request_id": request_id, "user": who.user})
            obs.GUARDRAIL_EVENTS.labels(event="acl_violation_blocked").inc()
            docs, answer, refused = [], REFUSAL, True
        cited = sorted({int(n) for n in CITATION.findall(answer) if 1 <= int(n) <= len(docs)})
        used = [] if refused else (cited or list(range(1, len(docs) + 1)))
        timings = {**state.get("timings_ms", {}), "total": round((time.perf_counter() - start) * 1000, 1)}

        self._record_metrics(state, timings)
        logger.info(  # sizes and timings only: question text may contain personal data
            "query_answered",
            extra={
                "request_id": request_id,
                "user": who.user,
                "role": who.role,
                "question_chars": len(question),
                "history_turns": len(history or []),
                "steps": state.get("steps"),
                "cached": bool(state.get("cached")),
                "refused": refused,
                "grounded": state.get("grounded"),
                "guardrails": guardrail_events,
                **{f"{k}_ms": v for k, v in timings.items()},
            },
        )
        return QueryResponse(
            answer=answer,
            citations=[Citation(ref=n, **to_passage(docs[n - 1])) for n in used],
            refused=refused,
            access=AccessInfo(**who.describe()) if self.access is not None else None,
            cached=bool(state.get("cached")),
            blocked=bool(state.get("blocked")),
            guardrails=guardrail_events,
            grounded=state.get("grounded"),
            standalone_question=standalone
            if standalone not in (None, question, state.get("question"))
            else None,
            search_query=state.get("search_query"),
            steps=state.get("steps", []),
            timings_ms=timings,
        )

    @staticmethod
    def _record_metrics(state: dict, timings: dict[str, float]) -> None:
        for stage, ms in timings.items():
            if stage != "total":
                obs.STAGE_LATENCY.labels(stage=stage).observe(ms / 1000)
        if state["refused"] and not state.get("blocked"):  # blocked requests are counted by the guardrails
            obs.REFUSALS.inc()
        if state.get("rewrites"):
            obs.REWRITES.inc(state["rewrites"])
        if state.get("grounded") is False:
            obs.UNGROUNDED.inc()

    # ------------------------------------------------------------------ health
    def sources(self, identity: Identity | None = None) -> list[str]:
        return self.store.list_sources(self.effective_identity(identity).scope_list())

    def llm_ready(self) -> bool:
        """Ollama is reachable and the configured model has been pulled."""
        if self._llm_health is not None:
            return self._llm_health()
        try:
            import httpx

            url = self.settings.llm_base_url.rstrip("/") + "/api/tags"
            models = httpx.get(url, timeout=3).json().get("models", [])
            names = {m.get("name") for m in models} | {m.get("model") for m in models}
            wanted = self.settings.llm_model
            return wanted in names or f"{wanted}:latest" in names
        except Exception:
            return False

    def health(self) -> dict:
        vector_db = self.store.ping()
        llm = self.llm_ready()
        return {
            "status": "ok" if vector_db and llm else "degraded",
            "vector_db": vector_db,
            "llm": llm,
            "llm_model": self.settings.llm_model,
            "guardrails": self.settings.guardrails_enabled,
            "guard_model": self.settings.guard_model or None,
            "access_control": self.access is not None,
            "cached_answers": len(self.cache) if self.cache is not None else None,
            "chunks_indexed": self.store.count() if vector_db else None,
        }

    def close(self) -> None:
        self.store.close()


def _is_llm_failure(exc: BaseException) -> bool:
    names = {type(e).__name__ for e in _chain(exc)}
    markers = {
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "ResponseError",
        "LLMError",
        "TimeoutException",
    }
    return bool(names & markers) or "connection refused" in str(exc).lower()


def _chain(exc: BaseException | None):
    while exc is not None:
        yield exc
        exc = exc.__cause__ or exc.__context__
