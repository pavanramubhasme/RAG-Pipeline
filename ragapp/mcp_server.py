"""MCP server: exposes the knowledge base as tools for any MCP client
(Claude Desktop, Cursor, VS Code, or the LangGraph agent in examples/).

It is a thin adapter over the REST API, so there is one source of truth,
one place for auth, and every tool call shows up in the API's traces and metrics.

    uv run python -m ragapp.mcp_server                                   # stdio (desktop clients)
    uv run python -m ragapp.mcp_server --transport streamable-http --port 8001   # network
"""

import argparse
import logging

import httpx
from mcp.server.fastmcp import FastMCP

from ragapp.config import get_settings

mcp = FastMCP(
    "rag-knowledge-base",
    instructions=(
        "Company knowledge base for Nordwind Mobility: HR policies, IT security rules, the "
        "VoltPoint wallbox manual and firmware release notes. Use search_documents to look up "
        "facts, ask_knowledge_base for a complete cited answer."
    ),
    log_level="WARNING",
)
_client: httpx.Client | None = None


def api() -> httpx.Client:
    global _client
    if _client is None:
        s = get_settings()
        _client = httpx.Client(
            base_url=s.api_url, headers={"X-API-Key": s.api_key or ""}, timeout=s.llm_timeout_s + 60
        )
    return _client


def set_api_client(client: httpx.Client) -> None:
    """Point the tools at a different API client (used by tests)."""
    global _client
    _client = client


def _call(method: str, path: str, **kwargs) -> dict:
    try:
        response = api().request(method, path, **kwargs)
    except httpx.TransportError as exc:
        raise RuntimeError(f"Knowledge-base API unreachable at {get_settings().api_url}") from exc
    if response.status_code >= 400:
        raise RuntimeError(f"Knowledge-base API error {response.status_code}: {response.text[:200]}")
    return response.json()


@mcp.tool()
def search_documents(query: str, top_k: int = 4) -> list[dict]:
    """Search the company knowledge base with hybrid keyword + semantic search and reranking.

    Returns the most relevant passages with their source file, section and relevance score
    (0-1). An empty list means nothing relevant exists. No LLM is involved, so it is fast.
    """
    return _call("POST", "/search", json={"query": query, "top_k": top_k})["results"]


@mcp.tool()
def ask_knowledge_base(question: str) -> dict:
    """Answer a question from the company knowledge base, with numbered citations.

    'refused': true means the documents do not contain the answer; do not guess then.
    'blocked': true means the guardrails rejected the request (e.g. a prompt-injection attempt).
    'grounded' reports an automatic check that the answer is supported by the sources.
    """
    result = _call("POST", "/query", json={"question": question})
    return {
        "answer": result["answer"],
        "refused": result["refused"],
        "blocked": result.get("blocked", False),
        "grounded": result["grounded"],
        "sources": [f"[{c['ref']}] {c['source']} ({c['section'] or 'n/a'})" for c in result["citations"]],
    }


@mcp.tool()
def list_sources() -> list[str]:
    """List the documents currently indexed in the knowledge base."""
    return _call("GET", "/sources")["sources"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    logging.getLogger("httpx").setLevel(logging.WARNING)  # stdout belongs to the MCP protocol
    mcp.settings.host, mcp.settings.port = args.host, args.port
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
