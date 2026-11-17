import numpy as np
import pytest
from langchain_core.documents import Document

from ragapp.reranker import FastEmbedReranker


@pytest.mark.parametrize(
    ("question", "expected_source"),
    [
        ("How many vacation days per calendar year?", "employee_handbook.md"),
        ("minimum password length characters", "it_security_policy.md"),
        ("What does error code E03 overheating mean?", "voltpoint_charger_faq.md"),
        ("Which version added dynamic load balancing?", "firmware_release_notes.md"),
    ],
)
@pytest.mark.parametrize("mode", ["hybrid", "dense", "sparse"])
def test_finds_right_document_in_every_mode(indexed, question, expected_source, mode):
    indexed.configure(indexed.settings.model_copy(update={"retrieval_mode": mode}))
    hits = indexed.search(question)
    assert hits and hits[0].source == expected_source


def test_results_sorted_limited_and_scored(indexed):
    hits = indexed.search("vacation days carried over", top_k=3)
    assert len(hits) <= 3
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= s <= 1 for s in scores)


def test_threshold_drops_irrelevant_chunks(indexed):
    assert indexed.search("Who won the 2022 FIFA World Cup?") == []


def test_without_reranker_nothing_is_filtered(indexed):
    indexed.configure(indexed.settings.model_copy(update={"rerank": False}))
    assert len(indexed.search("Who won the 2022 FIFA World Cup?", top_k=4)) == 4


def test_retriever_is_a_langchain_runnable(indexed):
    docs = indexed.retriever.invoke("vacation days per calendar year")
    assert isinstance(docs[0], Document)
    assert "relevance_score" in docs[0].metadata


# ---------------------------------------------------------------- reranker adapter
class StubCrossEncoder:
    def rerank(self, query, documents):
        # logit +10 when the query word appears in the text, -10 otherwise
        return iter([10.0 if query in d else -10.0 for d in documents])


def test_reranker_maps_logits_and_takes_best_query():
    reranker = FastEmbedReranker(top_n=5, min_score=0.5)
    reranker._model = StubCrossEncoder()
    docs = [Document(page_content=t) for t in ["about vacation", "about remote", "about parking"]]

    only_original = reranker.rank(docs, ["vacation"])
    assert [d.page_content for d in only_original] == ["about vacation"]
    assert only_original[0].metadata["relevance_score"] == pytest.approx(1.0, abs=1e-4)

    with_rewrite = reranker.rank(docs, ["vacation", "remote"])
    assert {d.page_content for d in with_rewrite} == {"about vacation", "about remote"}


def test_reranker_sigmoid_is_calibrated():
    reranker = FastEmbedReranker()
    reranker._model = type("M", (), {"rerank": lambda self, q, d: iter(np.array([0.0]))})()
    assert reranker.score("q", ["t"])[0] == pytest.approx(0.5)
