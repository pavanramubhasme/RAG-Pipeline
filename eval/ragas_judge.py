"""RAGAS wiring for a fully local judge: Ollama for the LLM, FastEmbed for embeddings."""

import asyncio
import math
from typing import Any

from langchain_core.embeddings import Embeddings
from ragas.embeddings.base import BaseRagasEmbedding

from ragapp.config import Settings

METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "factual_correctness",
]
DEFAULT_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


class LangChainRagasEmbedding(BaseRagasEmbedding):
    """Lets RAGAS reuse the pipeline's own (LangChain) embedding model."""

    def __init__(self, embeddings: Embeddings):
        super().__init__()
        self.embeddings = embeddings

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        return self.embeddings.embed_query(text)

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return await asyncio.to_thread(self.embed_text, text)


def build_judge_llm(settings: Settings):
    """Judge LLM over Ollama's OpenAI-compatible endpoint, with Instructor JSON mode."""
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    client = AsyncOpenAI(base_url=settings.llm_base_url.rstrip("/") + "/v1", api_key="ollama")
    model = settings.judge_model or settings.llm_model
    return llm_factory(model, provider="openai", client=client, temperature=0.0, max_tokens=2048)


def build_metrics(llm, embeddings: BaseRagasEmbedding, names: list[str]) -> dict:
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        FactualCorrectness,
        Faithfulness,
    )

    factories = {
        # Is every claim in the answer supported by the retrieved context? (hallucination)
        "faithfulness": lambda: Faithfulness(llm=llm),
        # Does the answer address the question? (strictness=1 keeps judge calls low on CPU)
        "answer_relevancy": lambda: AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=1),
        # Are the relevant chunks ranked at the top? (retrieval precision)
        "context_precision": lambda: ContextPrecisionWithReference(llm=llm),
        # Does the retrieved context contain everything the reference answer needs?
        "context_recall": lambda: ContextRecall(llm=llm),
        # Does the answer state the same facts as the reference answer?
        "factual_correctness": lambda: FactualCorrectness(llm=llm),
    }
    return {name: factories[name]() for name in names}


async def score_row(metrics: dict, row: dict, semaphore: asyncio.Semaphore) -> dict[str, float]:
    """Run every metric on one answered question. A judge failure becomes NaN, not a crash."""
    q, answer, contexts, ref = row["question"], row["answer"], row["contexts"], row["reference"]
    inputs = {
        "faithfulness": {"user_input": q, "response": answer, "retrieved_contexts": contexts},
        "answer_relevancy": {"user_input": q, "response": answer},
        "context_precision": {"user_input": q, "reference": ref, "retrieved_contexts": contexts},
        "context_recall": {"user_input": q, "retrieved_contexts": contexts, "reference": ref},
        "factual_correctness": {"response": answer, "reference": ref},
    }
    scores: dict[str, float] = {}
    for name, metric in metrics.items():
        async with semaphore:
            try:
                result = await metric.ascore(**inputs[name])
                scores[name] = float(result.value)
            except Exception:  # small local judges sometimes return malformed JSON
                scores[name] = math.nan
    return scores


async def score_rows(metrics: dict, rows: list[dict], concurrency: int = 2) -> list[dict[str, float]]:
    semaphore = asyncio.Semaphore(concurrency)
    return await asyncio.gather(*(score_row(metrics, row, semaphore) for row in rows))
