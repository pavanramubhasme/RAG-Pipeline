"""An MCP *client*: a LangChain agent (running on LangGraph) that discovers and calls
our knowledge-base tools over MCP, exactly as Claude Desktop or Cursor would.

Needs the API running (uv run uvicorn ragapp.api:app) and Ollama with a tool-calling model.

    uv run python examples/mcp_agent.py "Can I take 3 weeks of vacation in summer, and how early must I ask?"
"""

import asyncio
import os
import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragapp.config import get_settings  # noqa: E402

SYSTEM_PROMPT = (
    "You help Nordwind Mobility employees. Always look facts up with the knowledge-base tools "
    "before answering; never answer from memory. If the tools find nothing, say so. "
    "Mention the source documents you used."
)


async def main(question: str) -> None:
    settings = get_settings()
    client = MultiServerMCPClient(
        {
            "knowledge_base": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "ragapp.mcp_server"],
                "cwd": str(ROOT),
                "env": dict(os.environ),  # pass RAG_* settings through to the server process
            }
        }
    )
    tools = await client.get_tools()
    print("MCP tools discovered:", ", ".join(t.name for t in tools))

    llm = ChatOllama(model=settings.llm_model, base_url=settings.llm_base_url, temperature=0)
    agent = create_agent(llm, tools=tools, system_prompt=SYSTEM_PROMPT)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})

    for message in result["messages"]:
        for call in getattr(message, "tool_calls", None) or []:
            print(f"  -> tool call: {call['name']}({call['args']})")
    print("\n" + result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "How many vacation days do I get?"))
