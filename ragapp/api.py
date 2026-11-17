"""REST API.  Run:  uv run uvicorn ragapp.api:app --port 8000   then open /docs

Every request runs as an identity resolved from its X-API-Key on the server
(ragapp/access.py). With access control on, an unknown key is rejected (401), no key means
the anonymous role, and /query, /search and /sources only see what that role may see.
"""

import hmac
import logging
import tempfile
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ragapp import __version__
from ragapp import observability as obs
from ragapp.access import UNRESTRICTED, Identity
from ragapp.config import Settings, get_settings
from ragapp.ingestion import SUPPORTED_EXTENSIONS
from ragapp.schemas import IngestResponse, QueryRequest, QueryResponse, SearchRequest, SearchResponse
from ragapp.service import AccessDenied, LLMError, RAGService

logger = logging.getLogger("ragapp.api")


def create_app(
    settings: Settings | None = None,
    service_factory: Callable[[], RAGService] | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        obs.configure_logging(settings.log_level)
        obs.setup_tracing(settings)  # no-op unless RAG_TRACING_ENABLED=true
        if service_factory is not None:
            app.state.service = service_factory()
        else:
            from ragapp.factory import build_service  # loads the real models

            app.state.service = build_service(settings)
        obs.CHUNKS_INDEXED.set(app.state.service.store.count())
        logger.info("startup_complete", extra={"llm_model": settings.llm_model})
        yield
        app.state.service.close()
        obs.shutdown_tracing()  # flush spans

    app = FastAPI(title="RAG Showcase v6", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")  # template, not raw URL: bounded label cardinality
        obs.HTTP_REQUESTS.labels(request.method, path, str(response.status_code)).inc()
        obs.HTTP_LATENCY.labels(path).observe(elapsed)
        response.headers["X-Request-ID"] = request_id
        if path != "/metrics":
            logger.info(
                "http_request",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": path,
                    "status": response.status_code,
                    "latency_ms": round(elapsed * 1000, 1),
                },
            )
        return response

    def get_service(request: Request) -> RAGService:
        return request.app.state.service

    def get_identity(
        request: Request,
        x_api_key: str | None = Header(default=None),
        service: RAGService = Depends(get_service),
    ) -> Identity:
        """Who is calling. Only the credential decides, never the question text."""
        if service.access is None:  # access control off: optional single shared key, full access
            if settings.api_key and not hmac.compare_digest(
                (x_api_key or "").encode(), settings.api_key.encode()
            ):
                raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
            return UNRESTRICTED
        identity = service.access.resolve(x_api_key)
        if identity is None:
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
        request.state.user = identity.user
        return identity

    @app.get("/live", tags=["ops"])
    def live() -> dict:
        """Liveness: the process is up (container restarts)."""
        return {"status": "alive"}

    @app.get("/health", tags=["ops"])
    def health(service: RAGService = Depends(get_service)):
        """Readiness: Qdrant and the LLM are reachable. 503 = stop routing traffic here."""
        report = service.health()
        return JSONResponse(report, status_code=200 if report["status"] == "ok" else 503)

    @app.get("/metrics", tags=["ops"], include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/whoami", tags=["rag"])
    def whoami(identity: Identity = Depends(get_identity)) -> dict:
        """The caller's user, role and the classification labels it may see."""
        return {**identity.describe(), "can_ingest": identity.can_ingest}

    @app.post("/query", response_model=QueryResponse, tags=["rag"])
    def query(
        body: QueryRequest,
        request: Request,
        service: RAGService = Depends(get_service),
        identity: Identity = Depends(get_identity),
    ):
        """Agentic RAG: understand follow-ups, retrieve, retry with a rewrite if needed, answer, verify.
        Only chunks the caller's role may see are searched."""
        try:
            return service.ask(
                body.question,
                top_k=body.top_k,
                request_id=request.state.request_id,
                history=[turn.model_dump() for turn in body.history],
                identity=identity,
            )
        except AccessDenied as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except LLMError as exc:
            logger.error("llm_failure", extra={"error": str(exc)[:300]})
            raise HTTPException(status_code=503, detail="LLM backend unavailable") from exc

    @app.post("/search", response_model=SearchResponse, tags=["rag"])
    def search(
        body: SearchRequest,
        service: RAGService = Depends(get_service),
        identity: Identity = Depends(get_identity),
    ):
        """Retrieval only (hybrid + rerank), no LLM. Used by the MCP server."""
        return SearchResponse(results=service.search(body.query, top_k=body.top_k, identity=identity))

    @app.get("/sources", tags=["rag"])
    def sources(
        service: RAGService = Depends(get_service), identity: Identity = Depends(get_identity)
    ) -> dict:
        """Documents with at least one chunk the caller may see."""
        return {"sources": service.sources(identity)}

    @app.post("/ingest", response_model=IngestResponse, tags=["rag"])
    async def ingest(
        files: list[UploadFile] = File(...),
        service: RAGService = Depends(get_service),
        identity: Identity = Depends(get_identity),
    ):
        """Index uploaded files. With access control on, only roles with can_ingest may upload:
        an upload decides what other users will read."""
        if not identity.can_ingest:
            raise HTTPException(status_code=403, detail=f"Role {identity.role!r} may not add documents")
        max_bytes = settings.max_upload_mb * 1024 * 1024
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for upload in files:
                name = Path(upload.filename or "").name  # strips "../" path traversal
                if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                    raise HTTPException(415, f"Unsupported file type: {name or '<unnamed>'}")
                data = await upload.read()
                if len(data) > max_bytes:
                    raise HTTPException(413, f"{name} exceeds {settings.max_upload_mb} MB")
                path = Path(tmp) / name
                path.write_bytes(data)
                paths.append(path)
            return await run_in_threadpool(service.ingest, paths)  # CPU-bound embedding

    return app


app = create_app()
