"""Wires the real open-source components together."""

import logging
import time

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_ollama import ChatOllama
from langchain_qdrant import FastEmbedSparse

from ragapp.config import Settings, get_settings
from ragapp.reranker import FastEmbedReranker
from ragapp.service import RAGService
from ragapp.store import KnowledgeStore

logger = logging.getLogger(__name__)


def build_llm(s: Settings):
    llm = ChatOllama(
        model=s.llm_model,
        base_url=s.llm_base_url,
        temperature=s.llm_temperature,
        seed=s.llm_seed,  # with temperature 0: reproducible answers (demo, evaluation, debugging)
        reasoning=s.llm_reasoning,  # False: no "thinking" phase for qwen3 / deepseek-r1
        num_predict=s.llm_max_tokens,
        client_kwargs={"timeout": s.llm_timeout_s},
    )
    # Runnable-level retries with exponential backoff and jitter.
    return llm.with_retry(stop_after_attempt=s.llm_max_retries + 1, wait_exponential_jitter=True)


def build_guard_llm(s: Settings):
    """Optional guard model (e.g. IBM Granite Guardian) served by Ollama; None when not configured."""
    if not s.guard_model:
        return None
    return ChatOllama(
        model=s.guard_model,
        base_url=s.llm_base_url,
        temperature=0,
        seed=s.llm_seed,
        num_predict=20,  # the verdict is a single word or a <score> tag
        client_kwargs={"timeout": s.llm_timeout_s},
    )


def build_dense(s: Settings) -> FastEmbedEmbeddings:
    """The semantic embedding model: used by the search and by the answer cache."""
    return FastEmbedEmbeddings(
        model_name=s.dense_model, cache_dir=s.model_cache_dir, doc_embed_type="passage"
    )


def build_encoders(s: Settings):
    """Downloads ~150 MB of ONNX models on first use, then loads them from cache."""
    dense = build_dense(s)
    sparse = FastEmbedSparse(model_name=s.sparse_model, cache_dir=s.model_cache_dir)
    reranker = FastEmbedReranker(model_name=s.rerank_model, cache_dir=s.model_cache_dir)
    reranker.score("warm up", ["load the model now, not on the first request"])
    return dense, sparse, reranker


def build_service(settings: Settings | None = None) -> RAGService:
    s = settings or get_settings()
    dense, sparse, reranker = build_encoders(s)
    for attempt in range(1, 31):  # Qdrant may still be starting (Docker)
        try:
            store = KnowledgeStore.from_settings(s, dense, sparse)
            break
        except Exception as exc:
            if attempt == 30 or not s.qdrant_url:
                raise
            logger.info("waiting_for_qdrant", extra={"attempt": attempt, "error": str(exc)[:100]})
            time.sleep(1)
    return RAGService(
        store=store, reranker=reranker, llm=build_llm(s), settings=s, guard_llm=build_guard_llm(s)
    )
