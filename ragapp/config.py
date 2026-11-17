"""Central configuration.

Every field can be overridden with an environment variable prefixed ``RAG_``
(e.g. ``RAG_LLM_MODEL=qwen3:8b``) or a ``.env`` file. See ``.env.example``.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # --- Vector database (Qdrant) ---------------------------------------------
    qdrant_url: str | None = None  # http://localhost:6333 for a server; empty = embedded
    qdrant_path: str = ".qdrant"  # embedded mode only; ":memory:" for tests/eval
    qdrant_api_key: str | None = None
    collection: str = "knowledge_base"

    # --- Open-source retrieval models (FastEmbed / ONNX, CPU) ------------------
    dense_model: str = "BAAI/bge-small-en-v1.5"  # MIT, 384-dim
    sparse_model: str = "Qdrant/bm25"  # BM25 keyword weights
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"  # Apache-2.0 cross-encoder
    model_cache_dir: str | None = None

    # --- LLM served by Ollama ----------------------------------------------------
    # 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first and can add ~2 s per call
    llm_base_url: str = "http://127.0.0.1:11434"
    llm_model: str = "granite4:3b"  # IBM Granite 4, Apache-2.0
    llm_temperature: float = 0.0  # 0 + fixed seed = the same question gives the same answer
    llm_seed: int | None = 42
    # Thinking models (qwen3, deepseek-r1) reason before answering. False = answer directly:
    # several times faster on CPU and no risk of the reasoning eating the token budget.
    # None = the model's default (leave it for models without thinking, such as granite4).
    llm_reasoning: bool | None = None
    llm_max_tokens: int = 512
    llm_timeout_s: float = 120.0
    llm_max_retries: int = 2
    judge_model: str | None = None  # RAGAS judge; defaults to llm_model. Bigger is better.

    # --- Chunking (characters) ---------------------------------------------------
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- PDF parsing (ragapp/pdf_docling.py) -------------------------------------------------
    pdf_parser: Literal["auto", "docling", "pypdf"] = "auto"  # auto: Docling if installed, else pypdf
    docling_table_mode: Literal["accurate", "fast"] = "accurate"  # TableFormer: accurate or ~2x faster
    docling_ocr: bool = False  # only needed for scanned PDFs (adds an OCR model and time)
    docling_threads: int = 4

    # --- Retrieval -----------------------------------------------------------------
    retrieval_mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    rerank: bool = True  # False = skip the cross-encoder (ablation)
    candidates_k: int = 20  # hybrid search returns this many
    top_k: int = 4  # chunks passed to the LLM after reranking
    min_relevance: float = 0.05  # reranker probability; below -> chunk dropped
    query_expansion: bool = True  # "channel 1" also searches "CH1" and back (deterministic, no LLM)

    # --- Agentic workflow (LangGraph) ----------------------------------------------
    max_rewrites: int = 1  # retries with an LLM-rewritten query when nothing relevant is found
    grounding_check: bool = True  # LLM verifies the answer is supported by the context
    # Numbers in an answer must appear in the context or the question (deterministic, no LLM).
    # refuse = answer "I don't know" instead; flag = keep the answer, grounded = false; off.
    number_check: Literal["refuse", "flag", "off"] = "refuse"
    max_history_turns: int = 6  # earlier messages used to understand follow-ups; 0 = single-turn

    # --- Answer cache ----------------------------------------------------------------------
    cache_enabled: bool = True  # same question again -> answered from memory: no retrieval, no LLM
    cache_max_entries: int = 256
    cache_ttl_s: float = 3600.0  # also cleared whenever documents are ingested

    # --- Access control: who may see what (ragapp/access.py) --------------------------------
    access_control: bool = True  # filter retrieval by the caller's role; False = everyone sees everything
    identities_file: str = "config/identities.json"  # demo users -> roles -> labels they may see
    anonymous_role: str | None = "visitor"  # role for requests without a key; empty = reject (401)
    default_classification: str = "INTERNAL"  # label for content without one: deny by default

    # --- Guardrails ----------------------------------------------------------------------
    guardrails_enabled: bool = True
    block_prompt_injection: bool = True  # rule-based, instant, no LLM call
    redact_input_pii: list[str] = ["secret", "email", "phone", "iban", "card"]  # before LLM, logs, traces
    redact_output_pii: list[str] = ["secret", "iban", "card"]  # in answers (public contact data stays)
    quarantine_suspicious_chunks: bool = True  # drop document chunks containing injection phrases
    mask_secrets: bool = True  # API keys, tokens, passwords, connection strings: masked before indexing
    confidential_pii_kinds: list[str] = ["email", "phone"]  # masked at ingestion in CONFIDENTIAL/RESTRICTED
    guard_model: str | None = None  # e.g. granite3-guardian:2b (Apache-2.0). Empty = off (saves LLM calls)
    guard_risks: list[str] = ["jailbreak", "harm"]  # risks the guard model checks in each question
    guard_fail_closed: bool = False  # guard model down: False = answer anyway (logged), True = block

    # --- API / MCP ---------------------------------------------------------------------
    api_key: str | None = (
        None  # credential the UI/MCP send; checked by the server only if access control is off
    )
    api_url: str = "http://127.0.0.1:8000"  # used by the MCP server and the UI
    max_upload_mb: int = 20
    log_level: str = "INFO"

    # --- Observability (OpenTelemetry) -----------------------------------------------
    tracing_enabled: bool = False
    otlp_endpoint: str = "http://127.0.0.1:6006/v1/traces"  # Phoenix; Langfuse/Jaeger work too
    otlp_headers: str | None = None  # "key1=value1,key2=value2" e.g. Langfuse auth
    service_name: str = "rag-showcase"


@lru_cache
def get_settings() -> Settings:
    return Settings()
