"""Register-address questions find the register *at* that address, not one whose default value
looks the same. Seen live: "What is the register for the register address 0x04?" was answered
"SAP Control Register", whose heading is "(address = 0x03) [default = 0x04]"."""

import pytest
from langchain_core.documents import Document

from ragapp.register_lookup import asked_addresses, heading_addresses, keep_matching
from ragapp.retriever import HybridRerankRetriever
from tests.conftest import make_service

SAP = "9.6.4 SAP Control (Serial Audio-Port Control) Register (address = 0x03) [default = 0x04]"
STATE = "9.6.5 Channel State Control Register (address = 0x04) [default = 0x55]"
VOLUME = "9.6.6 Channel 1 Through 4 Volume Control Registers (address = 0x05-0x08) [default = 0xCF]"


@pytest.mark.parametrize(
    ("question", "addresses"),
    [
        ("What is the register for the register address 0x04?", {0x04}),
        ("Which register is at address = 0x74?", {0x74}),
        ("What does register 0x05 do?", {0x05}),
        ("0x04? Is the address right?", set()),  # no address named next to it: unchanged
        ("Explain the 0x06 register", {0x06}),
        ("Which register has default 0x04?", set()),  # a default value, not an address
        ("To mute channel 1, what value do I set?", set()),
    ],
)
def test_asked_addresses(question, addresses):
    assert asked_addresses(question) == addresses


def test_heading_addresses_include_ranges_and_pages():
    assert heading_addresses(SAP) == {0x03}  # the default 0x04 is not an address
    assert heading_addresses(STATE) == {0x04}
    assert heading_addresses(VOLUME) == {0x05, 0x06, 0x07, 0x08}
    assert heading_addresses("8.6.1.3.84 ASI_OUT_CH_EN Register (page = 0x00, address = 0x74)") == {0x74}
    assert heading_addresses("Employee Handbook > Vacation") == set()


def _doc(section: str) -> Document:
    return Document(page_content=f"{section}\nbits and fields", metadata={"section": section})


def test_keeps_only_the_register_at_the_asked_address():
    documents = [_doc(SAP), _doc(VOLUME), _doc(STATE)]
    kept = keep_matching("What is the register for the register address 0x04?", documents)
    assert [d.metadata["section"] for d in kept] == [STATE]
    assert [d.metadata["section"] for d in keep_matching("Register 0x07?", documents)] == [VOLUME]


def test_other_questions_and_unknown_addresses_change_nothing():
    documents = [_doc(SAP), _doc(STATE)]
    assert keep_matching("Which register has default 0x04?", documents) == documents
    assert keep_matching("What is the register at address 0x99?", documents) == documents


class RecordingStore:
    def __init__(self, documents):
        self.documents, self.k = documents, None

    def search(self, text, k, mode, scopes):
        self.k = k
        return list(self.documents)


class PassThroughReranker:
    def rank(self, documents, queries):
        return list(documents)


def test_retriever_filters_and_widens_the_search_for_address_questions():
    store = RecordingStore([_doc(SAP), _doc(STATE)])
    retriever = HybridRerankRetriever.model_construct(
        store=store,
        reranker=PassThroughReranker(),
        mode="hybrid",
        candidates_k=20,
        top_k=4,
        expand_queries=True,
    )
    found = retriever.invoke("What is the register for the register address 0x04?")
    assert [d.metadata["section"] for d in found] == [STATE]
    assert store.k == 60
    retriever.invoke("How do I mute channel 1?")
    assert store.k == 20


DATASHEET = f"""# TAS6424 register map

## {SAP}

Bits 7-6 select the serial audio format. The default value 0x04 selects I2S.

## {STATE}

| Bits | Field | Description |
|---|---|---|
| 7-6 | CH1 STATE CONTROL | 00: PLAY 01: Hi-Z 10: MUTE |
"""


def test_end_to_end_the_register_at_0x04_is_channel_state_control(settings, tmp_path):
    (tmp_path / "datasheet.md").write_text(DATASHEET, encoding="utf-8")
    service = make_service(settings)
    try:
        service.ingest([tmp_path])
        found = service.search("What is the register for the register address 0x04?")
        assert found and all("Channel State Control" in p.section for p in found)
    finally:
        service.close()