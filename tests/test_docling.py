"""Layout-aware PDF parsing: Docling's Markdown pages through the normal chunking pipeline.

Docling itself (models, PyTorch) is not needed here: its output is simulated as Markdown pages,
and the one test that builds a real DoclingDocument is skipped when docling-core is missing.
"""

import pytest
from langchain_core.documents import Document

from ragapp import ingestion, pdf_docling
from ragapp.access import find_label
from ragapp.config import Settings
from tests.conftest import SAMPLE_DIR

CH1_ROW = "| 7–6 | CH1 STATE CONTROL | R/W | 01 | 00: PLAY 01: Hi-Z 10: MUTE 11: DC load diagnostics |"
TABLE_HEAD = "| Bit | Field | Type | Reset | Description |\n| - | - | - | - | - |"


def docling_pages() -> list[Document]:
    """What Docling returns for three register pages of the TAS6424 datasheet (shortened)."""
    pages = [
        "### 9.6.5 Channel State Control Register (address = 0x04) [default = 0x55]\n\n"
        "The Channel State Control register is shown in Figure 36 and described in Table 14.\n\n"
        f"{TABLE_HEAD}\n{CH1_ROW}\n"
        "| 5–4 | CH2 STATE CONTROL | R/W | 01 | 00: PLAY 01: Hi-Z 10: MUTE 11: DC load diagnostics |",
        "### 9.6.6 Channel 1 Through 4 Volume Control Registers (address = 0x05–0x08) [default = 0xCF]\n\n"
        f"{TABLE_HEAD}\n| 7–0 | CH x VOLUME | R/W | CF | 0xFF: 24 dB 0xCF: 0 dB 0x07: –100 dB < 0x07: MUTE |",
        "### 9.6.7 DC Load Diagnostic Control 1 Register (address = 0x09) [default = 0x00]\n\n"
        f"{TABLE_HEAD}\n| 7 | DC LDG ABORT | R/W | 0 | 1: Aborts the load diagnostics in progress |",
    ]
    return [
        Document(page_content=text, metadata={"source": "TAS6424LS-Q1.pdf", "page": 32 + i})
        for i, text in enumerate(pages)
    ]


def test_markdown_headings_become_sections_and_table_rows_stay_whole(service):
    chunks = service.ingestor.split(docling_pages())
    state = next(c for c in chunks if c.metadata["section"].startswith("9.6.5 Channel State Control"))
    assert "#" not in state.metadata["section"]  # "### " removed from the title
    assert CH1_ROW in state.page_content  # the field, its reset value and MUTE = 10 in one line
    assert state.page_content.startswith("[TAS6424LS-Q1 > 9.6.5 Channel State Control Register")
    volume = next(c for c in chunks if c.metadata["section"].startswith("9.6.6"))
    assert "< 0x07: MUTE" in volume.page_content and volume.metadata["page"] == 33


def test_a_long_table_repeats_its_header_in_every_piece(service):
    rows = "\n".join(
        f"| {i} | FIELD_{i} | R/W | 0 | Description of bit field number {i} |" for i in range(80)
    )
    page = Document(
        page_content=f"### 9.6.9 Big Register Map (address = 0x10)\n\n{TABLE_HEAD}\n{rows}",
        metadata={"source": "map.pdf", "page": 1},
    )
    chunks = service.ingestor.split(docling_pages() + [page])
    table_chunks = [c for c in chunks if c.metadata["section"].startswith("9.6.9")]
    assert len(table_chunks) > 2
    assert all("| Bit | Field | Type | Reset | Description |" in c.page_content for c in table_chunks)
    assert "FIELD_79" in table_chunks[-1].page_content  # no row lost


def test_repeat_table_headers_stops_at_the_end_of_a_table():
    pieces = [
        f"{TABLE_HEAD}\n| 1 | A | R/W | 0 | x |",
        "| 2 | B | R/W | 0 | y |\nText after the table.",
        "| new | table |",
    ]
    out = ingestion.repeat_table_headers(pieces)
    assert out[1].startswith("| Bit | Field")  # continuation of the first table
    assert out[2] == "| new | table |"  # a different table: no wrong header


def test_table_rows_are_never_removed_as_boilerplate():
    pages = [f"Register page {i}\n{TABLE_HEAD}\n| {i} | F | R/W | 0 | d |" for i in range(6)]
    cleaned = ingestion.strip_repeated_lines(pages)
    assert all("| Bit | Field | Type | Reset | Description |" in page for page in cleaned)


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("## CONFIDENTIAL:HR — Synthetic compensation register", "CONFIDENTIAL:HR"),
        ("**Classification:** INTERNAL | Record: NST-IT-07", "INTERNAL"),
        ("### RESTRICTED:Security — Secret canaries", "RESTRICTED:SECURITY"),
        ("| CONFIDENTIAL:HR | Sensitive HR information |", None),  # a table cell is not a label
    ],
)
def test_labels_are_found_in_docling_markdown(text, label):
    assert find_label(text) == label


# ---------------------------------------------------------------- choosing the parser
def test_auto_uses_docling_only_when_installed(monkeypatch):
    monkeypatch.setattr(pdf_docling, "docling_available", lambda: False)
    assert ingestion.pdf_parser(Settings(pdf_parser="auto")) == "pypdf"
    with pytest.raises(RuntimeError, match="uv sync --group docling"):
        ingestion.pdf_parser(Settings(pdf_parser="docling"))
    monkeypatch.setattr(pdf_docling, "docling_available", lambda: True)
    assert ingestion.pdf_parser(Settings(pdf_parser="auto")) == "docling"
    assert ingestion.pdf_parser(Settings(pdf_parser="pypdf")) == "pypdf"


def test_auto_falls_back_to_pypdf_when_docling_fails(monkeypatch, tmp_path, caplog):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    fallback = [Document(page_content="pypdf text", metadata={"source": "doc.pdf", "page": 1})]

    def broken(*args, **kwargs):
        raise OSError("model download blocked")

    monkeypatch.setattr(pdf_docling, "docling_available", lambda: True)
    monkeypatch.setattr(pdf_docling, "load_pdf", broken)
    monkeypatch.setattr(ingestion, "_load_pdf_pypdf", lambda path: fallback)
    assert ingestion.load_file(pdf, Settings(pdf_parser="auto")) == fallback
    assert "docling_failed_using_pypdf" in caplog.text
    with pytest.raises(OSError):
        ingestion.load_file(pdf, Settings(pdf_parser="docling"))  # explicit choice: no silent fallback


def test_docling_is_used_with_the_configured_options(monkeypatch, tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    calls = []
    monkeypatch.setattr(pdf_docling, "docling_available", lambda: True)
    monkeypatch.setattr(pdf_docling, "load_pdf", lambda *args: calls.append(args) or [])
    ingestion.load_file(pdf, Settings(pdf_parser="auto", docling_table_mode="fast", docling_threads=2))
    assert calls == [(pdf, "fast", False, 2)]


def test_unchanged_files_are_skipped_before_parsing(indexed, monkeypatch):
    def must_not_parse(*args, **kwargs):
        raise AssertionError("an unchanged file was parsed again")

    monkeypatch.setattr(ingestion, "load_file", must_not_parse)
    report = indexed.ingest([SAMPLE_DIR])
    assert report.skipped_unchanged == 4 and report.chunks_added == 0


def test_switching_the_parser_reindexes(indexed, monkeypatch, tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 same bytes")
    monkeypatch.setattr(pdf_docling, "docling_available", lambda: False)
    before = indexed.ingestor._hash(pdf)
    monkeypatch.setattr(pdf_docling, "docling_available", lambda: True)
    assert indexed.ingestor._hash(pdf) != before  # pypdf chunks are replaced by Docling chunks


# ------------------------------------------------------------- real Docling document
def test_pages_from_a_docling_document():
    pytest.importorskip("docling_core")
    from docling_core.types.doc import (
        BoundingBox,
        DocItemLabel,
        DoclingDocument,
        ProvenanceItem,
        Size,
        TableCell,
        TableData,
    )
    from docling_core.types.doc.document import ContentLayer

    def prov(page):
        return ProvenanceItem(page_no=page, bbox=BoundingBox(l=0, t=10, r=100, b=0), charspan=(0, 1))

    doc = DoclingDocument(name="tas")
    for page in (1, 2):
        doc.add_page(page_no=page, size=Size(width=595, height=842))
    doc.add_text(
        label=DocItemLabel.PAGE_HEADER,
        text="TAS6424LS-Q1 SLOS996",
        prov=prov(1),
        content_layer=ContentLayer.FURNITURE,
    )
    doc.add_heading(text="9.6.5 Channel State Control Register (address = 0x04)", level=2, prov=prov(1))
    rows = [["Bit", "Field", "Description"], ["7–6", "CH1_STATE", "10: MUTE"]]
    cells = [
        TableCell(
            text=value,
            start_row_offset_idx=r,
            end_row_offset_idx=r + 1,
            start_col_offset_idx=c,
            end_col_offset_idx=c + 1,
            column_header=r == 0,
        )
        for r, row in enumerate(rows)
        for c, value in enumerate(row)
    ]
    doc.add_table(data=TableData(num_rows=2, num_cols=3, table_cells=cells), prov=prov(1))
    doc.add_picture(prov=prov(1))
    doc.add_text(label=DocItemLabel.TEXT, text="0x07: –100 dB, < 0x07: MUTE", prov=prov(2))

    pages = pdf_docling.pages_from_document(doc, "TAS6424LS-Q1.pdf")
    assert [p.metadata["page"] for p in pages] == [1, 2]
    first = pages[0].page_content
    assert first.lstrip("#").startswith(" 9.6.5 Channel State Control Register")  # a Markdown heading
    assert "| 7–6 | CH1_STATE | 10: MUTE |" in first  # compact table, underscores not escaped
    assert "SLOS996" not in first and "<!-- image -->" not in first  # page header and image dropped
    assert pages[1].page_content == "0x07: –100 dB, < 0x07: MUTE"  # "<" not turned into "&lt;"
