"""Agentic RAG on LangChain + LangGraph, with MCP, RAGAS and OpenTelemetry. Open-source models only."""

import os

__version__ = "6.1.0"

# Local services (Ollama, this API, Phoenix) must never be reached through a proxy. Company laptops
# often set HTTP(S)_PROXY with NO_PROXY=localhost only, which sends http://127.0.0.1:... to the
# proxy; it answers with an HTML error page instead of our JSON. Downloads from the internet
# (Hugging Face models) still use the proxy.
LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")


def bypass_proxy_for_loopback() -> None:
    current = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    hosts = [host.strip() for host in current.split(",") if host.strip()]
    if "*" in hosts:
        return
    missing = [host for host in LOOPBACK_HOSTS if host not in hosts]
    if missing:
        value = ",".join(hosts + missing)
        os.environ["NO_PROXY"] = value  # same variable on Windows; both spellings elsewhere
        os.environ["no_proxy"] = value


bypass_proxy_for_loopback()  # before any HTTP client (Ollama, OTLP exporter, MCP) is created
