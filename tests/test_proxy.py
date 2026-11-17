"""Local services must bypass a company proxy (HTTP_PROXY with NO_PROXY=localhost only).

Bug seen on a company laptop: after switching to http://127.0.0.1, the proxy intercepted the
UI's call to the API and answered with an HTML page -> JSONDecodeError in the UI.
"""

import httpx
import pytest
import requests

from ragapp import bypass_proxy_for_loopback


@pytest.fixture
def company_proxy(monkeypatch):
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.corp.example:3128")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.corp.example:3128")
    monkeypatch.setenv("NO_PROXY", "localhost,.corp.example")


def _uses_proxy(client: httpx.Client, url: str) -> bool:
    return client._transport_for_url(httpx.URL(url)) is not client._transport


def test_without_the_fix_127_0_0_1_goes_through_the_proxy(company_proxy):
    assert _uses_proxy(httpx.Client(), "http://127.0.0.1:8000/health")  # the bug


def test_loopback_bypasses_the_proxy(company_proxy):
    import os

    bypass_proxy_for_loopback()
    assert os.environ["NO_PROXY"].split(",") == ["localhost", ".corp.example", "127.0.0.1", "::1"]
    client = httpx.Client()  # Ollama and MCP clients are httpx
    assert not _uses_proxy(client, "http://127.0.0.1:11434/api/tags")
    assert not _uses_proxy(client, "http://localhost:8000/health")
    assert _uses_proxy(client, "https://huggingface.co/BAAI/bge-small-en-v1.5")  # downloads still proxied
    # The OTLP trace exporter uses requests
    assert requests.utils.should_bypass_proxies("http://127.0.0.1:6006/v1/traces", no_proxy=None)


def test_no_proxy_star_is_left_alone(monkeypatch):
    import os

    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setenv("NO_PROXY", "*")
    bypass_proxy_for_loopback()
    assert os.environ["NO_PROXY"] == "*"
