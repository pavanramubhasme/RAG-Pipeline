"""The MCP server, tested at two levels: tool logic and the real MCP protocol."""

import json

import anyio
import httpx
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from ragapp import mcp_server


@pytest.fixture
def mcp_api(client):
    mcp_server.set_api_client(client)  # tools now call our app (with fakes) in-process
    yield
    mcp_server.set_api_client(None)


def test_search_tool(mcp_api):
    results = mcp_server.search_documents("minimum password length", top_k=2)
    assert results[0]["source"] == "it_security_policy.md"


def test_ask_tool(mcp_api):
    result = mcp_server.ask_knowledge_base("How many vacation days per calendar year?")
    assert result["refused"] is False
    assert result["sources"][0].startswith("[1] employee_handbook.md")


def test_ask_tool_reports_refusal(mcp_api):
    assert mcp_server.ask_knowledge_base("Who won the 2022 FIFA World Cup?")["refused"] is True


def test_list_sources_tool(mcp_api):
    assert len(mcp_server.list_sources()) == 4


def test_unreachable_api_gives_clear_error():
    def refuse(request):
        raise httpx.ConnectError("refused")

    mcp_server.set_api_client(httpx.Client(base_url="http://kb", transport=httpx.MockTransport(refuse)))
    with pytest.raises(RuntimeError, match="unreachable"):
        mcp_server.list_sources()
    mcp_server.set_api_client(None)


def test_over_the_mcp_protocol(mcp_api):
    async def scenario():
        async with create_connected_server_and_client_session(mcp_server.mcp._mcp_server) as session:
            tools = {t.name: t for t in (await session.list_tools()).tools}
            assert set(tools) == {"search_documents", "ask_knowledge_base", "list_sources"}
            assert "hybrid" in tools["search_documents"].description
            result = await session.call_tool(
                "ask_knowledge_base", {"question": "vacation days per calendar year?"}
            )
            assert not result.isError
            return json.loads(result.content[0].text)

    payload = anyio.run(scenario)
    assert payload["refused"] is False
