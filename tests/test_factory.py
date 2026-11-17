"""Model wiring: settings reach the Ollama client (no network needed)."""

from ragapp.config import Settings
from ragapp.factory import build_llm


def test_thinking_can_be_switched_off_for_qwen3():
    llm = build_llm(Settings(llm_model="qwen3:8b", llm_reasoning=False))
    assert llm.bound.model == "qwen3:8b" and llm.bound.reasoning is False


def test_by_default_the_model_decides():
    assert build_llm(Settings()).bound.reasoning is None
