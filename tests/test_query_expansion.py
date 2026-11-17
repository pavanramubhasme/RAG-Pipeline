"""Query expansion: "channel 1" also searches "CH1" (and back), so BM25 and the cross-encoder
see the words a datasheet actually uses. Seen live on the TAS6424 datasheet: "To mute channel 1,
what value do I set?" found the volume register instead of the CH1 STATE CONTROL table."""

import pytest
from langchain_core.documents import Document

from ragapp.query_expansion import expand_query, expansions
from ragapp.retriever import HybridRerankRetriever
from tests.conftest import make_service


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("To mute channel 1, what value do I set?", ["CH1"]),
        ("What does CH2 STATE CONTROL do?", ["channel 2"]),
        ("Is CH 3 muted?", ["channel 3"]),
        ("Gain of channel-4", ["CH4"]),
        ("Channels 2 and 3", ["CH2"]),
        ("Enable input 3 and the microphone 1 bias", ["IN3", "MIC1"]),
        ("Route OUT2 to IN1", ["output 2", "input 1"]),
        ("channel 1 and CH1 again, channel 1", []),  # both spellings already there
    ],
)
def test_adds_the_other_spelling(query, expected):
    assert expansions(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        "How many vacation days do I get?",
        "Can I work from home 2 days in 2 weeks?",  # "in 2" is not "IN2"
        "I logged out 3 times",  # "out" is not "output"
        "What changed IN 2026?",  # a year is not a numbered term
        "Which version added dynamic load balancing?",
    ],
)
def test_ordinary_questions_are_unchanged(query):
    assert expansions(query) == []
    assert expand_query(query) == query


def test_expanded_query_keeps_the_question_and_adds_terms_in_brackets():
    assert expand_query("To mute channel 1, what value do I set?") == (
        "To mute channel 1, what value do I set? (CH1)"
    )


# ---------------------------------------------------------------- retriever wiring
class RecordingStore:
    def __init__(self):
        self.searched: list[str] = []

    def search(self, text, k, mode, scopes):
        self.searched.append(text)
        return [Document(page_content="| 7–6 | CH1 STATE CONTROL | 10: MUTE |")]


class RecordingReranker:
    def __init__(self):
        self.queries: list[str] = []

    def rank(self, documents, queries):
        self.queries = queries
        return list(documents)


def test_retriever_searches_and_reranks_with_the_expanded_query():
    store, reranker = RecordingStore(), RecordingReranker()
    retriever = HybridRerankRetriever.model_construct(
        store=store, reranker=reranker, mode="hybrid", candidates_k=20, top_k=4, expand_queries=True
    )
    retriever.invoke("To mute channel 1, what value do I set?")
    assert store.searched == ["To mute channel 1, what value do I set? (CH1)"]
    # original, rewrite (none here) and the expansion: the reranker keeps the best score per chunk
    assert reranker.queries[0] == "To mute channel 1, what value do I set?"
    assert reranker.queries[-1] == "To mute channel 1, what value do I set? (CH1)"


def test_expansion_also_applies_to_an_llm_rewrite():
    store, reranker = RecordingStore(), RecordingReranker()
    retriever = HybridRerankRetriever.model_construct(
        store=store, reranker=reranker, mode="hybrid", candidates_k=20, top_k=4, expand_queries=True
    )
    retriever.invoke("mute it", search_query="How do I mute channel 2?")
    assert store.searched == ["How do I mute channel 2? (CH2)"]
    assert reranker.queries == ["mute it", "How do I mute channel 2?", "How do I mute channel 2? (CH2)"]


def test_expansion_can_be_switched_off():
    store, reranker = RecordingStore(), RecordingReranker()
    retriever = HybridRerankRetriever.model_construct(
        store=store, reranker=reranker, mode="hybrid", candidates_k=20, top_k=4, expand_queries=False
    )
    retriever.invoke("To mute channel 1, what value do I set?")
    assert store.searched == ["To mute channel 1, what value do I set?"]
    assert set(reranker.queries) == {"To mute channel 1, what value do I set?"}


# ---------------------------------------------------------------- end to end (fake models)
DATASHEET = """# Channel State Control Register (address = 0x04)

| Bits | Field | Type | Reset | Description |
|---|---|---|---|---|
| 7–6 | CH1 STATE CONTROL | R/W | 01 | 00: PLAY 01: Hi-Z 10: MUTE 11: DC load diagnostics |
"""
OTHER = """# Clock Configuration

The PLL locks to the serial audio clock. The sample rate is 48 kHz.
"""


def test_the_ch1_table_is_found_for_channel_1_only_with_expansion(settings, tmp_path):
    (tmp_path / "datasheet.md").write_text(DATASHEET, encoding="utf-8")
    (tmp_path / "clock.md").write_text(OTHER, encoding="utf-8")
    question = "Which value mutes channel 1?"

    with_expansion = make_service(settings)
    without = make_service(settings.model_copy(update={"query_expansion": False}))
    try:
        for svc in (with_expansion, without):
            svc.ingest([tmp_path])
        found = with_expansion.search(question)
        assert found and "CH1 STATE CONTROL" in found[0].text
        assert without.search(question) == []  # "channel 1" alone matches no chunk well enough
    finally:
        with_expansion.close()
        without.close()


def test_service_passes_the_setting_to_the_retriever(service):
    assert service.retriever.expand_queries is True
    service.configure(service.settings.model_copy(update={"query_expansion": False}))
    assert service.retriever.expand_queries is False
