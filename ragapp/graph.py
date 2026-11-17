"""The agentic RAG workflow as a LangGraph state machine.

    START --> input_guard --(blocked)--> END
    input_guard --(follow-up + history)--> contextualize --> cache_lookup
    input_guard --(anything else)-------------------------> cache_lookup
    cache_lookup --(same question answered before)--> END
    cache_lookup --(new question)--> retrieve
    retrieve --(relevant chunks)--> generate --> check_grounding --> output_guard --> cache_store --> END
                                    (generate refuses if the answer states a number found in no source)
    retrieve --(nothing relevant)--> rewrite --> retrieve      (at most max_rewrites times)
    retrieve --(still nothing, or the rewrite failed)--> refuse --> cache_store --> END

* input_guard / output_guard are the guardrails (see ragapp/guardrails.py). They are
  traced like every node but only listed in ``steps`` when they block a request.
* contextualize turns a follow-up ("what are their addresses?") into a standalone
  question using the recent conversation, so retrieval knows what "their" means. It only
  runs for messages that point back (ragapp/conversation.py): a new topic is searched as typed.
* cache_lookup / cache_store: the same question again, or one of your own earlier questions
  with the same meaning and the same numbers, is answered from memory, with no retrieval and
  no LLM call (ragapp/cache.py). Listed in ``steps`` only on a hit. The caller's access scope
  is part of the key.
* retrieve only sees chunks whose classification the caller may see (``scopes``, enforced
  by Qdrant), so every later node, the cache and the citations are limited to them too.
* Relevance is graded by the cross-encoder's calibrated score, not by asking the
  LLM about each chunk: same decision quality for a fraction of the cost and latency.
"""

import logging
import operator
import time
from collections.abc import Callable
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, StateGraph

from ragapp import observability as obs
from ragapp.cache import AnswerCache
from ragapp.config import Settings
from ragapp.conversation import is_follow_up, keeps_user_words
from ragapp.guardrails import BLOCKED_MESSAGE, Guardrails, unsupported_numbers
from ragapp.prompts import (
    ANSWER_PROMPT,
    CONTEXTUALIZE_PROMPT,
    GROUNDING_PROMPT,
    REFUSAL,
    REWRITE_PROMPT,
    clean_output,
    format_context,
    format_history,
)

logger = logging.getLogger(__name__)


def _sum_timings(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    merged = dict(left)
    for key, value in right.items():
        merged[key] = round(merged.get(key, 0.0) + value, 1)
    return merged


class RAGState(TypedDict, total=False):
    question: str  # the user's message (personal data masked by the input guard)
    history: list[dict]  # earlier turns: [{"role": "user"|"assistant", "content": "..."}]
    standalone: str  # question made self-contained from history (== question if no history)
    context_resolved: bool  # False: a follow-up could not be resolved (LLM failed or rewrite rejected)
    cached: bool  # answered from the answer cache
    user: str | None  # who asks: same-meaning cache hits only come from your own questions
    question_vector: list[float] | None  # embedding of the question, computed once by cache_lookup
    cache_similar_to: str | None  # same-meaning hit: the earlier question it matched
    cache_similarity: float | None
    top_k: int | None
    scopes: list[str] | None  # classification labels the caller may see; None = all
    search_query: str  # what we actually search for (rewritten on retry)
    rewrites: int
    rewrite_failed: bool
    documents: list[Document]
    answer: str
    refused: bool
    unsupported_numbers: list[str]  # numbers in the answer found in no source (flag mode)
    grounded: bool | None
    blocked: bool  # stopped by the input guardrail
    guardrails: Annotated[list[str], operator.add]  # e.g. "redacted:email", "blocked:prompt_injection"
    steps: Annotated[list[str], operator.add]  # path taken through the graph
    timings_ms: Annotated[dict[str, float], _sum_timings]


def _timed(name: str, fn: Callable[[RAGState], dict], record_step: bool = True) -> Callable[[RAGState], dict]:
    def node(state: RAGState) -> dict:
        start = time.perf_counter()
        update = fn(state)
        if record_step:
            update["steps"] = [name]
        update["timings_ms"] = {name: (time.perf_counter() - start) * 1000}
        return update

    node.__name__ = name
    return node


def is_refusal(answer: str) -> bool:
    return REFUSAL.lower().rstrip(".") in answer.lower()


def _question(state: RAGState) -> str:
    return state.get("standalone") or state["question"]


CACHED_FIELDS = ("answer", "documents", "refused", "grounded")


def build_graph(
    retriever: BaseRetriever,
    llm: BaseChatModel,
    settings: Settings,
    guardrails: Guardrails | None = None,
    cache: AnswerCache | None = None,
    embed: Callable[[str], list[float]] | None = None,
):
    entry = "cache_lookup" if cache is not None else "retrieve"  # first node after the guards
    finish = "cache_store" if cache is not None else END  # last node of answers and refusals

    answer_chain = (ANSWER_PROMPT | llm | StrOutputParser()).with_config(run_name="answer_chain")
    contextualize_chain = (CONTEXTUALIZE_PROMPT | llm | StrOutputParser()).with_config(
        run_name="contextualize_chain"
    )
    rewrite_chain = (REWRITE_PROMPT | llm | StrOutputParser()).with_config(run_name="rewrite_chain")
    grounding_chain = (GROUNDING_PROMPT | llm | StrOutputParser()).with_config(run_name="grounding_chain")

    def input_guard(state: RAGState) -> dict:
        check = guardrails.check_input(state["question"], state.get("history"))
        update: dict = {"question": check.question, "history": check.history, "guardrails": check.findings}
        if check.blocked:  # stop here: no retrieval, no LLM call
            update.update(
                answer=BLOCKED_MESSAGE,
                refused=True,
                blocked=True,
                grounded=None,
                documents=[],
                steps=["input_guard"],
            )
        return update

    def route_start(state: RAGState) -> str:
        if state.get("blocked"):
            return END
        use_history = state.get("history") and settings.max_history_turns > 0
        return "contextualize" if use_history and is_follow_up(state["question"]) else entry

    def output_guard(state: RAGState) -> dict:
        answer, findings = guardrails.check_output(state["answer"])
        return {"answer": answer, "guardrails": findings}

    def contextualize(state: RAGState) -> dict:
        question = state["question"]
        unresolved = {"standalone": question, "context_resolved": False}  # use the message as typed
        history = format_history(state["history"], max_turns=settings.max_history_turns)
        try:
            standalone = clean_output(contextualize_chain.invoke({"history": history, "question": question}))
        except Exception as exc:  # an optimisation: fall back to the message as typed
            logger.warning("contextualize_failed", extra={"error": str(exc)[:200]})
            return unresolved
        standalone = standalone.splitlines()[0][:500] if standalone else question
        if not keeps_user_words(question, standalone):  # the LLM replaced the question
            logger.warning("contextualize_rejected", extra={"question_chars": len(question)})
            return unresolved
        return {"standalone": standalone, "context_resolved": True}

    def cache_key(state: RAGState) -> str:
        return AnswerCache.key(_question(state), state.get("top_k") or settings.top_k, state.get("scopes"))

    def cache_lookup(state: RAGState) -> dict:
        if state.get("context_resolved") is False:  # "those" still unresolved: not a safe key
            return {"cached": False}
        found = cache.lookup(
            cache_key(state),
            user=state.get("user"),
            question=_question(state),
            embed=embed if settings.cache_semantic else None,
            threshold=settings.cache_similarity,
        )
        if found.value is None:
            return {"cached": False, "question_vector": found.vector}
        return {
            **found.value,
            "cached": True,
            "cache_similar_to": found.similar_to,
            "cache_similarity": found.similarity,
            "steps": ["cache_hit"],
        }

    def route_after_cache(state: RAGState) -> str:
        return END if state.get("cached") else "retrieve"

    def cacheable(state: RAGState) -> bool:
        if state.get("context_resolved") is False:
            return False
        if state.get("refused"):  # genuine "I don't know", not one caused by an LLM failure
            return not state.get("rewrite_failed")
        return state.get("grounded") is True or not settings.grounding_check

    def cache_store(state: RAGState) -> dict:
        if cacheable(state):
            cache.put(
                cache_key(state),
                {field: state.get(field) for field in CACHED_FIELDS},
                user=state.get("user"),
                question=_question(state),
                vector=state.get("question_vector"),
            )
        return {}

    def retrieve(state: RAGState) -> dict:
        docs = retriever.invoke(
            _question(state),
            search_query=state.get("search_query"),
            top_k=state.get("top_k"),
            scopes=state.get("scopes"),  # enforced inside Qdrant: forbidden chunks never come back
        )
        return {"documents": docs}

    def route_after_retrieve(state: RAGState) -> Literal["generate", "rewrite", "refuse"]:
        if state["documents"]:
            return "generate"
        if state.get("rewrites", 0) < settings.max_rewrites:
            return "rewrite"
        return "refuse"

    def rewrite(state: RAGState) -> dict:
        previous = state.get("search_query") or _question(state)
        try:
            query = clean_output(rewrite_chain.invoke({"question": _question(state), "previous": previous}))
        except Exception as exc:  # rewriting is an optimisation: degrade to a refusal, don't fail
            logger.warning("rewrite_failed", extra={"error": str(exc)[:200]})
            return {"rewrite_failed": True}
        query = query.splitlines()[0][:300] if query else _question(state)
        return {"search_query": query, "rewrites": state.get("rewrites", 0) + 1}

    def route_after_rewrite(state: RAGState) -> Literal["retrieve", "refuse"]:
        return "refuse" if state.get("rewrite_failed") else "retrieve"

    def generate(state: RAGState) -> dict:
        answer = clean_output(
            answer_chain.invoke({"context": format_context(state["documents"]), "question": _question(state)})
        )
        if is_refusal(answer) or settings.number_check == "off":
            return {"answer": answer, "refused": is_refusal(answer)}
        sources = [d.page_content for d in state["documents"]] + [state["question"], _question(state)]
        missing = unsupported_numbers(answer, sources)
        if not missing:
            return {"answer": answer, "refused": False}
        # An invented amount or date is the most damaging hallucination: never show it as fact.
        logger.warning("unsupported_numbers", extra={"count": len(missing)})
        obs.GUARDRAIL_EVENTS.labels(event="unsupported_numbers").inc()
        if settings.number_check == "refuse":
            return {"answer": REFUSAL, "refused": True, "guardrails": ["refused:unsupported_numbers"]}
        return {
            "answer": answer,
            "refused": False,
            "unsupported_numbers": missing,
            "guardrails": ["flag:unsupported_numbers"],
        }

    def check_grounding(state: RAGState) -> dict:
        if state["refused"]:
            return {"grounded": None}
        if state.get("unsupported_numbers"):  # already known to be unsupported: no LLM call needed
            return {"grounded": False}
        try:
            verdict = clean_output(
                grounding_chain.invoke(
                    {"context": format_context(state["documents"]), "answer": state["answer"]}
                )
            )
        except Exception as exc:  # keep the answer, report the check as "not performed"
            logger.warning("grounding_check_failed", extra={"error": str(exc)[:200]})
            return {"grounded": None}
        return {"grounded": verdict.lower().startswith("yes")}

    def refuse(state: RAGState) -> dict:
        return {"answer": REFUSAL, "refused": True, "grounded": None, "documents": []}

    graph = StateGraph(RAGState)
    for name, fn in [
        ("contextualize", contextualize),
        ("retrieve", retrieve),
        ("rewrite", rewrite),
        ("generate", generate),
        ("check_grounding", check_grounding),
        ("refuse", refuse),
    ]:
        graph.add_node(name, _timed(name, fn))

    if cache is not None:
        graph.add_node("cache_lookup", _timed("cache_lookup", cache_lookup, record_step=False))
        graph.add_node("cache_store", _timed("cache_store", cache_store, record_step=False))
        graph.add_conditional_edges("cache_lookup", route_after_cache, [END, "retrieve"])
        graph.add_edge("cache_store", END)

    start_targets = [END, "contextualize", entry]
    if guardrails is not None:
        graph.add_node("input_guard", _timed("input_guard", input_guard, record_step=False))
        graph.add_node("output_guard", _timed("output_guard", output_guard, record_step=False))
        graph.add_edge(START, "input_guard")
        graph.add_conditional_edges("input_guard", route_start, start_targets)
        after_answer = "output_guard"
        graph.add_edge("output_guard", finish)
    else:
        graph.add_conditional_edges(START, route_start, start_targets)
        after_answer = finish

    graph.add_edge("contextualize", entry)
    graph.add_conditional_edges("retrieve", route_after_retrieve)
    graph.add_conditional_edges("rewrite", route_after_rewrite)
    if settings.grounding_check:
        graph.add_edge("generate", "check_grounding")
        graph.add_edge("check_grounding", after_answer)
    else:
        graph.add_edge("generate", after_answer)
    graph.add_edge("refuse", finish)
    return graph.compile(name="agentic_rag")
