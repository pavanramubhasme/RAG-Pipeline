"""Deterministic test doubles implementing the real LangChain / RAGAS interfaces.

The whole system (graph, API, evaluation, tracing) runs offline in
seconds, so the tests check *our* logic rather than a model's mood.
"""

import enum
import math
import re
import types
import typing
import zlib
from collections import Counter
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_qdrant import SparseEmbeddings, SparseVector
from pydantic import BaseModel
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM

from ragapp.reranker import FastEmbedReranker

STOPWORDS = set(
    [
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "do",
        "does",
        "i",
        "my",
        "me",
        "what",
        "which",
        "how",
        "when",
        "who",
        "can",
        "it",
        "be",
        "with",
        "by",
        "at",
        "from",
        "long",
        "many",
        "much",
        "there",
        "this",
        "that",
        "s",
    ]
)
SYNONYMS = {"wfh": "remote work days per week", "pto": "vacation days per calendar year"}


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class FakeEmbeddings(Embeddings):
    """Hashing bag-of-words: texts sharing words get similar vectors."""

    dim = 256

    def _vector(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in tokens(text):
            v[zlib.crc32(tok.encode()) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v))
        if norm == 0:
            v[0], norm = 1.0, 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeSparseEmbeddings(SparseEmbeddings):
    def _vector(self, text: str) -> SparseVector:
        counts = Counter(zlib.crc32(t.encode()) % (2**31) for t in tokens(text))
        return SparseVector(indices=list(counts), values=[float(c) for c in counts.values()])

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> SparseVector:
        return self._vector(text)


class FakeReranker(FastEmbedReranker):
    """Score = share of query words present in the passage. Keeps the real filtering logic."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        q = set(tokens(query))
        return [len(q & set(tokens(t))) / len(q) if q else 0.0 for t in texts]


class FakeChatModel(BaseChatModel):
    """Plays the four LLM roles in the graph, chosen by the system prompt it receives."""

    answer: str | None = None  # fixed answer override
    standalone: str | None = None  # fixed contextualize output (simulates a bad rewrite)
    grounded: str = "yes"
    fail: bool = False  # every call fails (LLM down)
    fail_on: list[str] = []  # only these roles fail: "contextualize", "rewrite", "grounding", "answer"
    calls: list[str] = []
    prompts: list[str] = []  # every human message received, to assert what reached the LLM

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    @staticmethod
    def _role(system: str) -> str:
        if "standalone question" in system:
            return "contextualize"
        if "search queries" in system:
            return "rewrite"
        if "fully supported" in system:
            return "grounding"
        return "answer"

    def _respond(self, messages: list[BaseMessage]) -> str:
        if self.fail:
            raise ConnectionError("connection refused (fake outage)")
        system, human = str(messages[0].content), str(messages[-1].content)
        role = self._role(system)
        self.prompts.append(human)
        if role in self.fail_on:
            raise ConnectionError(f"connection refused (fake {role} outage)")
        self.calls.append(role)
        if role == "contextualize":
            if self.standalone is not None:
                return self.standalone
            # "Standalone" = the last user question from the history + the follow-up.
            earlier = re.findall(r"^User: (.+)$", human, re.MULTILINE)
            latest = human.rsplit("Latest message: ", 1)[-1].strip()
            return f"{earlier[-1]} {latest}" if earlier else latest
        if role == "rewrite":
            question = human.split("\n")[0].removeprefix("Question: ")
            return " ".join(SYNONYMS.get(w.lower().strip("?"), w) for w in question.split())
        if role == "grounding":
            return self.grounded
        if self.answer is not None:
            return self.answer
        match = re.search(r"\[1\] \(.*?\)\n(?:\[.*?\]\n)?(.+?)(?:\.\s|\n|$)", human, re.DOTALL)
        return f"{match.group(1).strip() if match else 'See context'}. [1]"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self._respond(messages)))])


class FakeGuardModel(BaseChatModel):
    """Stands in for Granite Guardian: system prompt = risk name, answers Yes/No."""

    risky_words: dict[str, list[str]] = {"harm": ["weapon", "explosive"], "jailbreak": ["no limits"]}
    fail: bool = False
    style: str = "plain"  # "plain" -> "Yes"/"No" (granite3-guardian), "score" -> "<score> yes </score>"
    calls: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "fake-guard"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        if self.fail:
            raise ConnectionError("connection refused (fake guard outage)")
        risk, text = str(messages[0].content), str(messages[-1].content).lower()
        self.calls.append(risk)
        risky = any(word in text for word in self.risky_words.get(risk, []))
        verdict = (
            ("<score> yes </score>" if risky else "<score> no </score>")
            if self.style == "score"
            else ("Yes" if risky else "No")
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=verdict))])


# ------------------------------------------------------------- RAGAS doubles
def _fake_value(tp: Any) -> Any:
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin in (list, typing.List):  # noqa: UP006
        return [_fake_value(args[0])] if args else []
    if origin in (typing.Union, types.UnionType):
        return _fake_value(next(a for a in args if a is not type(None)))
    if origin is typing.Literal:
        return args[0]
    if origin is dict:
        return {}
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        return schema_valid_instance(tp)
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return next(iter(tp))
    return {str: "The statement is supported by the context.", int: 1, float: 1.0, bool: True}.get(tp)


def schema_valid_instance(model: type[BaseModel]) -> BaseModel:
    """Build a valid instance of any Pydantic response model a RAGAS metric asks for."""
    return model(**{name: _fake_value(f.annotation) for name, f in model.model_fields.items()})


class FakeRagasLLM(InstructorBaseRagasLLM):
    """A judge that always returns well-formed structured output (verdicts = 1)."""

    def generate(self, prompt, response_model):
        return schema_valid_instance(response_model)

    async def agenerate(self, prompt, response_model):
        return schema_valid_instance(response_model)


class FakeRagasEmbedding(BaseRagasEmbedding):
    def __init__(self):
        super().__init__()
        self._inner = FakeEmbeddings()

    def embed_text(self, text: str, **kwargs) -> list[float]:
        return self._inner.embed_query(text)

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return self.embed_text(text)
