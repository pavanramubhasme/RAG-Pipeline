import shutil

from ragapp.ingestion import load_file
from tests.conftest import SAMPLE_DIR


def test_ingest_sample_folder(service):
    result = service.ingest([SAMPLE_DIR])
    assert result.files == 4
    assert result.chunks_added > 10
    assert result.total_chunks == result.chunks_added
    assert len(service.sources()) == 4


def test_reingest_is_idempotent(service):
    first = service.ingest([SAMPLE_DIR])
    second = service.ingest([SAMPLE_DIR])
    assert second.chunks_added == 0
    assert second.skipped_unchanged == 4
    assert second.total_chunks == first.total_chunks


def test_changed_file_replaces_old_chunks(service, tmp_path):
    doc = tmp_path / "office.md"
    doc.write_text("# Office\n\n## Parking\n\nThere are 12 parking spaces.")
    service.ingest([doc])
    doc.write_text("# Office\n\n## Parking\n\nThere are 40 parking spaces.")
    result = service.ingest([doc])
    assert result.chunks_added == 1
    assert result.total_chunks == 1  # old version removed, not duplicated
    assert "40" in service.search("parking spaces")[0].text


def test_unsupported_and_empty_files_are_ignored(service, tmp_path):
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "empty.md").write_text("   ")
    shutil.copy(SAMPLE_DIR / "it_security_policy.md", tmp_path)
    result = service.ingest([tmp_path])
    assert result.files == 2  # png filtered out; empty.md discovered but skipped
    assert result.total_chunks > 0


def test_chunks_have_section_path_and_size_limit(service):
    chunks = service.ingestor.split(load_file(SAMPLE_DIR / "employee_handbook.md"))
    sections = {c.metadata["section"] for c in chunks}
    assert "Employee Handbook — Nordwind Mobility GmbH > Vacation" in sections
    assert all(c.page_content.startswith("[") for c in chunks)  # contextual header
    body_lengths = [len(c.page_content.split("\n", 1)[1]) for c in chunks]
    assert max(body_lengths) <= service.settings.chunk_size


def test_plain_text_files_work(service, tmp_path):
    (tmp_path / "notes.txt").write_text("The canteen opens at 11:30 and closes at 14:00.")
    service.ingest([tmp_path])
    hit = service.search("canteen opening time")[0]
    assert hit.source == "notes.txt"
    assert hit.section is None


# ------------------------------------------------------- PDF structure-aware chunking
from langchain_core.documents import Document  # noqa: E402

from ragapp import ingestion  # noqa: E402
from ragapp.ingestion import split_pdf_sections, strip_repeated_lines  # noqa: E402

PDF_BODIES = {
    1: "Table of Contents\n1.1 Overview ........ 2\n1.2 Features ........ 2\n2.1 CTRL Register ........ 3\n"
    "2.2 GAIN Register ........ 4\n2.3 STATUS Register ........ 5",
    2: "1.1 Overview\nThe ACME-Q1 is a four-channel audio converter for automotive use.\n"
    "1.2 Features\nLow power, high dynamic range and an integrated microphone bias.\n-6\n-12",
    3: "2.1 CTRL Register (page = 0x00, address = 0x10) [reset = 0h]\n"
    "This register controls the power state of the device.\n7 PWR_EN RW 0h Enables power.",
    4: "6 MUTE RW 0h Mutes all channels, continued from the previous page.\n-6\n-12\n"
    "2.2 GAIN Register (page = 0x01, address = 0x10) [reset = 0h]\n"
    "This register sets the channel gain in 0.5 dB steps.",
    5: "2.3 STATUS Register (page = 0x00, address = 0x20) [reset = 0h]\n"
    "This read-only register reports device faults.",
    6: "Revision history: first public release of this datasheet document.",
}


def pdf_pages() -> list[Document]:
    pages = []
    for n, body in PDF_BODIES.items():
        # odd and even pages put the header words in a different order and spacing, like real datasheets
        header = "ACME-Q1 Datasheet www.acme.example" if n % 2 else "www.acme.example ACME-Q1 Datasheet"
        footer = (
            "Submit Feedback Copyright © 2024, ACME Inc"
            if n % 2
            else "Submit FeedbackCopyright © 2024, ACME Inc"
        )
        text = f"{n}\n{header}\n{body}\n{footer}"
        pages.append(Document(page_content=text, metadata={"source": "acme.pdf", "page": n}))
    return pages


def test_repeated_headers_footers_and_page_numbers_are_removed():
    cleaned = "\n".join(strip_repeated_lines([p.page_content for p in pdf_pages()]))
    assert "Copyright" not in cleaned and "www.acme.example" not in cleaned
    assert not any(line.strip().isdigit() for line in cleaned.splitlines())
    assert "-6" in cleaned and "-12" in cleaned  # repeated numbers in tables are content, not boilerplate


def test_pdf_is_split_at_numbered_headings_across_pages():
    sections = {title: (page, text) for title, page, text, _ in split_pdf_sections(pdf_pages())}
    ctrl = sections["2.1 CTRL Register (page = 0x00, address = 0x10) [reset = 0h]"]
    assert ctrl[0] == 3
    assert "PWR_EN" in ctrl[1] and "MUTE" in ctrl[1]  # the section continues on page 4
    gain = sections["2.2 GAIN Register (page = 0x01, address = 0x10) [reset = 0h]"]
    assert "MUTE" not in gain[1]  # same address, other page: never mixed into one chunk
    assert not any("...." in (title or "") for title in sections)  # the table of contents is not a heading


def test_every_pdf_chunk_carries_its_section_title(service):
    chunks = service.ingestor.split(pdf_pages())
    gain_chunks = [c for c in chunks if c.metadata["section"] and "GAIN Register" in c.metadata["section"]]
    assert gain_chunks and all(c.page_content.startswith("[acme > 2.2 GAIN Register") for c in gain_chunks)
    assert all(c.metadata["page"] for c in chunks)


def test_pdf_without_numbered_headings_falls_back_to_pages(service):
    pages = [
        Document(
            page_content=f"Plain page {n} text about topic {n}.", metadata={"source": "plain.pdf", "page": n}
        )
        for n in (1, 2)
    ]
    chunks = service.ingestor.split(pages)
    assert [c.metadata["page"] for c in chunks] == [1, 2]
    assert all(c.metadata["section"] is None for c in chunks)


def test_changing_the_chunking_version_reindexes(service, monkeypatch):
    first = service.ingest([SAMPLE_DIR])
    monkeypatch.setattr(ingestion, "CHUNKING_VERSION", "test-bump")
    second = service.ingest([SAMPLE_DIR])
    assert second.skipped_unchanged == 0 and second.chunks_added == first.chunks_added
    assert second.total_chunks == first.total_chunks  # replaced, not duplicated
