"""Evaluation: deterministic retrieval metrics + RAGAS with a local open-source judge.

    uv run python -m eval.run_eval --retrieval-only     # seconds, no LLM: hit rate, MRR, precision/recall@k
    uv run python -m eval.run_eval --ablation           # dense vs sparse vs hybrid vs hybrid+rerank
    uv run python -m eval.run_eval                      # answers + RAGAS (needs Ollama; slow on CPU)
    uv run python -m eval.run_eval --limit 5            # quick RAGAS check on 5 questions
    uv run python -m eval.run_eval --compare-agentic    # linear pipeline vs LangGraph agent, side by side
    uv run python -m eval.run_eval --fake-models        # offline smoke test with test doubles

RAGAS needs about 10 judge-LLM calls per question. A bigger judge than the answering
model gives more trustworthy scores:  RAG_JUDGE_MODEL=qwen3:8b
"""

import argparse
import asyncio
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")  # no telemetry from the eval library

from eval.ragas_judge import (  # noqa: E402  # noqa: E402
    DEFAULT_METRICS,
    METRIC_NAMES,
    LangChainRagasEmbedding,
    build_judge_llm,
    build_metrics,
    score_rows,
)
from ragapp.config import get_settings  # noqa: E402
from ragapp.service import RAGService  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"


# ---------------------------------------------------------------- setup
def load_golden(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build(fake: bool) -> RAGService:
    settings = get_settings().model_copy(
        update={
            "qdrant_url": None,
            "qdrant_path": ":memory:",
            "collection": "eval",
            "tracing_enabled": False,
            "cache_enabled": False,  # measure real latency; variants must not reuse each other's answers
            "access_control": False,  # quality of retrieval and answers; access is tested by security_eval
        }
    )
    if fake:
        sys.path.insert(0, str(ROOT))
        from tests.conftest import make_service

        return make_service(settings.model_copy(update={"min_relevance": 0.3}))
    from ragapp.factory import build_service

    return build_service(settings)


def build_judge(service: RAGService, fake: bool):
    if fake:
        from tests.fakes import FakeRagasEmbedding, FakeRagasLLM

        return FakeRagasLLM(), FakeRagasEmbedding()
    return build_judge_llm(service.settings), LangChainRagasEmbedding(service.store.embeddings)


def mean(values) -> float | None:
    clean = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return round(sum(clean) / len(clean), 3) if clean else None


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)] if ordered else 0.0


# ---------------------------------------------------- retrieval (no LLM)
def _mentions(text: str, term: str) -> bool:
    # "3" must not match "30" or "13", but the prefix "approv" matches "approval".
    tail = r"(?!\d)" if term[-1].isdigit() else ""
    return re.search(rf"(?<![0-9A-Za-z]){re.escape(term)}{tail}", text, re.IGNORECASE) is not None


def contains_answer(text: str, must_include: list) -> bool:
    """The chunk states every fact of the reference answer (any of the listed spellings)."""
    groups = [[g] if isinstance(g, str) else g for g in must_include]
    return bool(groups) and all(any(_mentions(text, term) for term in group) for group in groups)


def is_relevant(text: str, item: dict) -> bool:
    """Relevance label for precision/recall: the chunk contains the answer's facts and, if the
    golden item names one, comes from the reference section. Every chunk starts with its header
    "[Document > Section]", so the section check needs no extra metadata. Facts alone are not
    enough: "30" is also in "submit expenses within 30 days", not only in the vacation rule."""
    section = item.get("reference_section")
    header = text.split("]", 1)[0].lower() if text.startswith("[") else ""
    in_section = not section or section.lower() in header
    return in_section and contains_answer(text, item["must_include"])


def evaluate_retrieval(service: RAGService, golden: list[dict]) -> dict:
    """Deterministic retrieval metrics, no LLM involved.

    hit_rate@k   share of questions where the right document is among the k chunks
    mrr          1 / rank of the first chunk from the right document, averaged
    precision@k  share of the returned chunks that contain the answer
                 (divided by the chunks actually returned: the reranker threshold may return
                 fewer than k, and those are exactly what the LLM sees)
    recall@k     share of all answer-bearing chunks in the index that were returned
    A chunk is relevant if it comes from the reference document (and section, if given) and
    contains the facts of the reference answer (``must_include``): no hand-labelled chunk ids,
    so the labels survive re-chunking.
    """
    k = service.settings.top_k
    relevant_counts: dict[tuple, int] = {}
    rows = []
    for item in golden:
        docs = service.retriever.invoke(item["question"])
        sources = [d.metadata["source"] for d in docs]
        expected = item["reference_source"]
        row = {
            "id": item["id"],
            "answerable": expected is not None,
            "hit": (expected in sources) if expected else None,
            "rr": next((1 / (i + 1) for i, s in enumerate(sources) if s == expected), 0.0)
            if expected
            else None,
            "precision": None,
            "recall": None,
            "refused": not docs,
        }
        if expected and item.get("must_include"):
            key = (expected, item.get("reference_section"), json.dumps(item["must_include"]))
            if key not in relevant_counts:
                relevant_counts[key] = sum(
                    is_relevant(text, item) for text in service.store.chunk_texts(expected)
                )
            total = relevant_counts[key]
            found = sum(d.metadata["source"] == expected and is_relevant(d.page_content, item) for d in docs)
            row["precision"] = found / len(docs) if docs else 0.0
            row["recall"] = min(1.0, found / total) if total else None  # None: labels match no chunk
            row["relevant_in_index"] = total
        rows.append(row)
    answerable = [r for r in rows if r["answerable"]]
    negatives = [r for r in rows if not r["answerable"]]
    unlabeled = [r["id"] for r in answerable if r.get("relevant_in_index") == 0]
    if unlabeled:
        print(f"Note: no chunk contains the must_include facts for {unlabeled}; recall skipped for them")
    return {
        "rows": rows,
        "summary": {
            f"hit_rate@{k}": mean(r["hit"] for r in answerable),
            "mrr": mean(r["rr"] for r in answerable),
            f"precision@{k}": mean(r["precision"] for r in answerable),
            f"recall@{k}": mean(r["recall"] for r in answerable),
            "refusal_accuracy": mean(r["refused"] for r in negatives) if service.settings.rerank else None,
        },
    }


ABLATIONS = [
    ("dense only", {"retrieval_mode": "dense", "rerank": False}),
    ("BM25 only", {"retrieval_mode": "sparse", "rerank": False}),
    ("hybrid (RRF)", {"retrieval_mode": "hybrid", "rerank": False}),
    ("hybrid + rerank (default)", {"retrieval_mode": "hybrid", "rerank": True}),
]


def run_ablation(service: RAGService, golden: list[dict]) -> list[dict]:
    base = service.settings
    table = []
    for name, overrides in ABLATIONS:
        service.configure(base.model_copy(update=overrides))
        table.append({"config": name, **evaluate_retrieval(service, golden)["summary"]})
    service.configure(base)
    return table


# --------------------------------------------- end to end + RAGAS (LLM)
def collect_answers(service: RAGService, golden: list[dict]) -> list[dict]:
    rows = []
    for item in golden:
        r = service.ask(item["question"])
        docs = service.retriever.invoke(item["question"], search_query=r.search_query)
        text = r.answer.lower()
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "reference": item["reference"],
                "answerable": item["reference_source"] is not None,
                "answer": r.answer,
                "contexts": [d.page_content for d in docs],
                "refused": r.refused,
                "grounded": r.grounded,
                "rewrites": int(r.search_query is not None),
                "keyword_correct": all(any(v.lower() in text for v in alts) for alts in item["must_include"])
                if item["reference_source"]
                else None,
                "cited_right_doc": item["reference_source"] in {c.source for c in r.citations}
                if item["reference_source"]
                else None,
                "latency_ms": r.timings_ms["total"],
            }
        )
    return rows


def evaluate_end_to_end(service, golden, judge, metric_names, concurrency) -> dict:
    rows = collect_answers(service, golden)
    to_score = [r for r in rows if r["answerable"] and not r["refused"]]
    if judge is not None and to_score:
        metrics = build_metrics(*judge, metric_names)
        for row, scores in zip(
            to_score, asyncio.run(score_rows(metrics, to_score, concurrency)), strict=True
        ):
            row.update(scores)

    answerable = [r for r in rows if r["answerable"]]
    negatives = [r for r in rows if not r["answerable"]]
    latencies = [r["latency_ms"] for r in rows]
    summary = {
        "answer_rate": mean(not r["refused"] for r in answerable),
        "refusal_accuracy": mean(r["refused"] for r in negatives),
        "keyword_accuracy": mean(r["keyword_correct"] for r in answerable),
        "citation_accuracy": mean(r["cited_right_doc"] for r in answerable),
        "grounded_rate": mean(r["grounded"] for r in answerable if r["grounded"] is not None),
        "rewrite_rate": mean(r["rewrites"] for r in rows),
    }
    if judge is not None:
        for name in metric_names:
            summary[f"ragas_{name}"] = mean(r.get(name) for r in to_score)
        summary["judge_failures"] = sum(
            1 for r in to_score for n in metric_names if math.isnan(r.get(n, math.nan))
        )
    summary["latency_p50_ms"] = round(statistics.median(latencies), 1)
    summary["latency_p95_ms"] = round(percentile(latencies, 0.95), 1)
    return {"rows": rows, "summary": summary}


# --------------------------------------------------------------- output
def print_table(rows: list[dict], key: str = "config") -> None:
    columns = list(rows[0])
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in columns}
    print("  ".join(c.ljust(widths[c]) for c in columns))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in columns))


def write_markdown(path: Path, title: str, rows: list[dict]) -> None:
    columns = list(rows[0])
    lines = [f"## {title}", "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(str(r[c]) for c in columns) + " |" for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=ROOT / "eval" / "golden_set.jsonl")
    parser.add_argument("--docs", type=Path, default=ROOT / "data" / "sample")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--compare-agentic", action="store_true")
    parser.add_argument("--no-ragas", action="store_true", help="skip the LLM judge")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS), help=f"subset of {METRIC_NAMES}")
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N questions")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--fake-models", action="store_true")
    parser.add_argument("--min-hit-rate", type=float, default=0.0, help="CI gate")
    parser.add_argument("--min-faithfulness", type=float, default=0.0, help="CI gate")
    args = parser.parse_args()

    golden = load_golden(args.golden)
    if args.limit:
        answerable = [g for g in golden if g["reference_source"]][: args.limit]
        golden = answerable + [g for g in golden if not g["reference_source"]][:1]
    service = build(args.fake_models)
    service.ingest([args.docs])
    RESULTS.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    if args.ablation:
        table = run_ablation(service, golden)
        print_table(table)
        write_markdown(RESULTS / "ablation.md", "Retrieval ablation", table)
        (RESULTS / "ablation.json").write_text(json.dumps(table, indent=2))
    elif args.retrieval_only:
        report = evaluate_retrieval(service, golden)
        print(json.dumps(report["summary"], indent=2))
        (RESULTS / "retrieval.json").write_text(json.dumps(report, indent=2))
        hit = next(v for k, v in report["summary"].items() if k.startswith("hit_rate")) or 0.0
        if hit < args.min_hit_rate:
            failures.append(f"hit rate {hit} < {args.min_hit_rate}")
    else:
        metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
        judge = None if args.no_ragas else build_judge(service, args.fake_models)
        variants = [("agentic (default)", {})]
        if args.compare_agentic:
            variants.insert(
                0, ("linear (no rewrite, no grounding)", {"max_rewrites": 0, "grounding_check": False})
            )
        base, table = service.settings, []
        for name, overrides in variants:
            service.configure(base.model_copy(update=overrides))
            report = evaluate_end_to_end(service, golden, judge, metric_names, args.concurrency)
            table.append({"config": name, **report["summary"]})
            slug = "linear" if overrides else "agentic"
            (RESULTS / f"end_to_end_{slug}.json").write_text(json.dumps(report, indent=2))
        service.configure(base)
        print_table(table)
        write_markdown(RESULTS / "end_to_end.md", "End-to-end evaluation (RAGAS)", table)
        faithfulness = table[-1].get("ragas_faithfulness")
        if faithfulness is not None and faithfulness < args.min_faithfulness:
            failures.append(f"faithfulness {faithfulness} < {args.min_faithfulness}")

    service.close()
    print(f"\nReports written to {RESULTS}")
    if failures:
        print("QUALITY GATE FAILED: " + "; ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
