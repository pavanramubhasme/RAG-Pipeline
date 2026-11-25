"""An independent model checks the answer, so the answering model doesn't grade itself.

Seen live: asked for the purpose and scope of "Human Resources - Probation", qwen2.5:7b added
"maintaining confidentiality and compliance with HR policies", which is not on the page, and the
grounding check, done by the same model, said "grounded". With RAG_GROUNDING_MODEL set, another
model checks; a Granite Guardian model uses its trained "groundedness" risk.
"""

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from ragapp.config import Settings
from ragapp.factory import build_grounding_llm
from ragapp.guardrails import groundedness_messages, is_guardian
from tests.conftest import SAMPLE_DIR, make_service
from tests.fakes import FakeChatModel


class FakeGuardian(BaseChatModel):
    """Answers like Granite Guardian's groundedness check: "Yes" = risk (not grounded)."""

    invented: str = "confidentiality"  # an answer containing this word is not grounded
    fail: bool = False
    seen: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "fake-guardian"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        if self.fail:
            raise ConnectionError("guardian down")
        self.seen.append(list(messages))
        answer = str(messages[-1].content).lower()
        verdict = "Yes" if self.invented in answer else "No"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=verdict))])


def _service(settings, answer: str, checker, model: str = "granite3-guardian:2b"):
    llm = FakeChatModel(calls=[], prompts=[], answer=answer)
    service = make_service(settings.model_copy(update={"grounding_model": model}), llm)
    service.grounding_llm = checker
    service.configure(service.settings)
    service.ingest([SAMPLE_DIR])
    return service, llm


def test_guardian_flags_an_invented_phrase(settings):
    guardian = FakeGuardian(seen=[])
    service, llm = _service(
        settings, "You get 30 days of vacation, maintaining confidentiality and compliance. [1]", guardian
    )
    try:
        response = service.ask("How many vacation days do I get?")
        assert response.grounded is False
        assert "grounding" not in llm.calls  # the answering model did not grade itself
    finally:
        service.close()


def test_guardian_accepts_a_supported_answer(settings):
    guardian = FakeGuardian(seen=[])
    service, _ = _service(settings, "Full-time employees receive 30 days of paid vacation. [1]", guardian)
    try:
        assert service.ask("How many vacation days do I get?").grounded is True
    finally:
        service.close()


def test_guardian_gets_its_message_format(settings):
    guardian = FakeGuardian(seen=[])
    service, _ = _service(settings, "Full-time employees receive 30 days of paid vacation. [1]", guardian)
    try:
        service.ask("How many vacation days do I get?")
        system, context, answer = guardian.seen[-1]
        assert system.content == "groundedness"
        assert context.type == "chat" and context.role == "context" and "30 days" in context.content
        assert answer.type == "ai" and answer.content.startswith("Full-time employees")
    finally:
        service.close()


def test_another_ordinary_model_uses_the_grounding_prompt(settings):
    checker = FakeChatModel(calls=[], prompts=[], grounded="no")
    service, llm = _service(settings, "You get 30 days. [1]", checker, model="granite4:3b")
    try:
        assert service.ask("How many vacation days do I get?").grounded is False
        assert checker.calls == ["grounding"] and "grounding" not in llm.calls
    finally:
        service.close()


def test_a_failing_checker_keeps_the_answer(settings):
    service, _ = _service(settings, "You get 30 days. [1]", FakeGuardian(seen=[], fail=True))
    try:
        response = service.ask("How many vacation days do I get?")
        assert response.answer == "You get 30 days. [1]" and response.grounded is None
    finally:
        service.close()


def test_without_a_grounding_model_the_answering_model_checks(service, fake_llm):
    service.ingest([SAMPLE_DIR])
    service.ask("How many vacation days do I get?")
    assert "grounding" in fake_llm.calls


@pytest.mark.parametrize(
    ("model", "expected"),
    [("granite3-guardian:2b", True), ("granite3-guardian:8b", True), ("granite4:3b", False), (None, False)],
)
def test_is_guardian(model, expected):
    assert is_guardian(model) is expected


def test_groundedness_messages():
    system, context, answer = groundedness_messages("[1] 30 days", "30 days [1]")
    assert (system.content, context.role, context.content, answer.content) == (
        "groundedness",
        "context",
        "[1] 30 days",
        "30 days [1]",
    )


def test_factory_builds_a_checker_only_for_a_different_model():
    assert build_grounding_llm(Settings()) is None
    assert build_grounding_llm(Settings(llm_model="qwen2.5:7b", grounding_model="qwen2.5:7b")) is None
    checker = build_grounding_llm(Settings(grounding_model="granite3-guardian:2b"))
    assert checker is not None and checker.model == "granite3-guardian:2b"


def test_health_reports_who_checks(settings):
    service = make_service(settings.model_copy(update={"grounding_model": "granite3-guardian:2b"}))
    try:
        assert service.health()["grounding_model"] == "granite3-guardian:2b"
    finally:
        service.close()