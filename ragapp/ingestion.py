"""Load -> clean -> split -> index with LangChain, idempotently.

Chunking is structure-aware, because documents are not flat text:

* **Markdown**: split by headings (``MarkdownHeaderTextSplitter``) so each chunk
  knows its section ("Employee Handbook > Vacation").
* **PDF**: repeated page headers/footers ("Copyright ...", page numbers) are
  removed, then the text is split at numbered headings ("8.6.1.3.84 ASI_OUT_CH_EN
  Register (page = 0x00, address = 0x74)"). Pages are joined first, so a section
  that continues on the next page stays together. PDFs without numbered
  headings fall back to page-by-page splitting.
* Every section is then cut to size (``RecursiveCharacterTextSplitter``) and
  each piece is prefixed with its section title: a contextual header that tells
  BM25, the embedding model and the reranker which register/section a piece of
  a long table belongs to.

A content hash per file (including the chunking version) makes re-ingestion skip
unchanged files and re-index changed ones; deterministic chunk IDs prevent duplicates.

Security at ingestion (nothing below can be undone by a prompt later):

* every chunk gets a **classification label** from its page ("Classification:
  CONFIDENTIAL:HR"), used to filter retrieval by the caller's role (ragapp/access.py).
  A chunk never mixes two labels: a section that crosses into a page with another label
  is cut there;
* **secrets** (API keys, tokens, passwords, connection strings) are masked in every chunk,
  and **contact data** (e-mail, phone) in CONFIDENTIAL/RESTRICTED chunks: never embedded,
  never shown, even to authorized roles (data minimization);
* chunks containing prompt-injection text ("ignore all previous instructions ...") are
  quarantined (not indexed), so a malicious upload cannot steer the LLM.
"""

import hashlib
import logging
import re
import uuid
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from ragapp import observability as obs
from ragapp.access import find_label, is_sensitive, normalize_label
from ragapp.config import Settings
from ragapp.guardrails import find_injection, redact
from ragapp.store import KnowledgeStore

logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}
HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
CHUNKING_VERSION = "3"  # bump when the chunking logic changes: forces a re-index of every file

# Numbered headings, e.g. "7.1 Absolute Maximum Ratings" or
# "8.6.1.3.84 ASI_OUT_CH_EN Register (page = 0x00, address = 0x74) [reset = 0h]"
PDF_HEADING = re.compile(r"^\d{1,2}(?:\.\d{1,3}){1,6}\s+[A-Z][A-Za-z0-9_(].{2,200}$")
TOC_LEADER = re.compile(r"\.{3,}")  # "7.1 Absolute Maximum Ratings ........ 12"
PAGE_NUMBER = re.compile(r"^(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?$", re.IGNORECASE)  # "7", "Page 7 of 9"


# ------------------------------------------------------------------- loading
def discover_files(paths: Iterable[Path | str]) -> list[Path]:
    found: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found.update(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)
        elif path.suffix.lower() in SUPPORTED_EXTENSIONS and path.exists():
            found.add(path)
    return sorted(found)


def load_file(path: Path) -> list[Document]:
    """One Document per file (text) or per page (PDF), with source/page metadata."""
    if path.suffix.lower() == ".pdf":
        pages = PyPDFLoader(str(path)).load()
        return [
            Document(page_content=p.page_content, metadata={"source": path.name, "page": i})
            for i, p in enumerate(pages, start=1)
            if p.page_content.strip()
        ]
    text = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()[0].page_content
    return [Document(page_content=text, metadata={"source": path.name})] if text.strip() else []


# ------------------------------------------------------------ PDF cleaning
def _line_key(line: str) -> str:
    """Comparable form of a line: digits masked ("Page 73" == "Page 74"), case, spacing and
    word order ignored (headers alternate between left and right on odd and even pages)."""
    masked = re.sub(r"\d+", "#", line.lower())
    return "".join(sorted(re.sub(r"\s+", "", masked)))


def strip_repeated_lines(pages: list[str], min_share: float = 0.5, min_pages: int = 4) -> list[str]:
    """Remove lines that appear on at least ``min_share`` of the pages (headers, footers,
    copyright notices) and bare page numbers at the top or bottom of a page.

    Only lines containing letters can be boilerplate, so repeated numbers in tables survive.
    """
    boilerplate: set[str] = set()
    if len(pages) >= min_pages:
        counts = Counter(
            key for page in pages for key in {_line_key(x) for x in page.splitlines() if x.strip()}
        )
        threshold = max(2, int(len(pages) * min_share))
        boilerplate = {
            key for key, n in counts.items() if n >= threshold and len(key) >= 8 and re.search("[a-z]", key)
        }
    cleaned_pages = []
    for page in pages:
        lines = [x for x in page.splitlines() if x.strip() and _line_key(x) not in boilerplate]
        while lines and PAGE_NUMBER.match(lines[0].strip()):  # page number on top
            lines.pop(0)
        while lines and PAGE_NUMBER.match(lines[-1].strip()):  # page number at the bottom
            lines.pop()
        cleaned_pages.append("\n".join(lines))
    return cleaned_pages


def _is_toc_page(text: str) -> bool:
    return sum(1 for line in text.splitlines() if TOC_LEADER.search(line)) >= 5


def split_pdf_sections(
    pages: list[Document], labels: dict[int, str] | None = None
) -> list[tuple[str | None, int, str, str | None]]:
    """Return (section_title, start_page, text, label) for a PDF, split at numbered headings.

    ``labels`` maps page number -> classification. A section that continues onto a page
    with a different label is cut there, so no chunk ever mixes two classifications.
    Returns an empty list when the PDF has fewer than 3 numbered headings, so the
    caller can fall back to page-by-page splitting.
    """
    labels = labels or {}
    texts = strip_repeated_lines([p.page_content for p in pages])
    sections: list[tuple[str | None, int, str, str | None]] = []
    title: str | None = None
    start_page = pages[0].metadata.get("page", 1) if pages else 1
    label = labels.get(start_page)
    buffer: list[str] = []
    headings = 0

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if len(body) >= 20:  # heading-only sections (parents of sub-sections) carry no content
            sections.append((title, start_page, body, label))

    for doc, text in zip(pages, texts, strict=True):
        page_no = doc.metadata.get("page", 1)
        page_label = labels.get(page_no)
        if page_label != label:  # classification changes with this page: never mix the two
            flush()
            start_page, label, buffer = page_no, page_label, []
        toc_page = _is_toc_page(text)
        for line in text.splitlines():
            stripped = line.strip()
            if not toc_page and PDF_HEADING.match(stripped) and not TOC_LEADER.search(stripped):
                flush()
                title, start_page, buffer = stripped, page_no, []
                headings += 1
            else:
                if not buffer and title is None:
                    start_page = page_no
                buffer.append(line)
    flush()
    return sections if headings >= 3 else []


def page_title(text: str) -> tuple[str | None, str]:
    """(title, rest) if the page starts with a heading-like line ("03. IT Operations — Access
    provisioning"), else (None, text). Used as the contextual header of every chunk of the page,
    which matters when many pages share the same template text."""
    lines = text.splitlines()
    if len(lines) < 2:
        return None, text
    first = lines[0].strip()
    heading_like = (
        5 <= len(first) <= 100
        and len(first.split()) >= 2
        and (first[0].isupper() or first[0].isdigit())
        and first[-1] not in ".,;:"
    )
    return (first, "\n".join(lines[1:])) if heading_like else (None, text)


# ---------------------------------------------------------------- ingestor
class Ingestor:
    def __init__(self, store: KnowledgeStore, settings: Settings):
        self.store = store
        self.settings = settings
        self.header_splitter = MarkdownHeaderTextSplitter(HEADERS, strip_headers=True)
        self.size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _page_labels(self, docs: list[Document]) -> dict[int, str]:
        """Classification per page: its own label, else the cover page's, else the default."""
        default = normalize_label(self.settings.default_classification)
        baseline = find_label(docs[0].page_content) or default
        return {d.metadata.get("page", 1): find_label(d.page_content) or baseline for d in docs}

    def _sections(self, docs: list[Document]) -> list[tuple[str | None, int | None, str, str]]:
        """(section_title, page, text, label) for any supported document type."""
        source = docs[0].metadata["source"].lower()
        default = normalize_label(self.settings.default_classification)
        if source.endswith(".pdf"):
            labels = self._page_labels(docs)  # from the raw pages: cleaning may drop repeated label lines
            sections = split_pdf_sections(docs, labels)
            if sections:
                return [(title, page, text, label or default) for title, page, text, label in sections]
            cleaned = strip_repeated_lines([d.page_content for d in docs])  # no headings: page by page
            result = []
            for d, text in zip(docs, cleaned, strict=True):
                title, body = page_title(text)
                page = d.metadata.get("page")
                result.append((title, page, body, labels.get(page, default)))
            return result
        label = find_label("\n".join(d.page_content for d in docs)) or default  # one label per text file
        if source.endswith(".md"):
            result = []
            for section in self.header_splitter.split_text(docs[0].page_content):
                path = " > ".join(section.metadata[key] for _, key in HEADERS if key in section.metadata)
                result.append((path or None, None, section.page_content, label))
            return result
        return [(None, None, d.page_content, label) for d in docs]

    def _protect(self, text: str, label: str) -> tuple[str, list[str]]:
        """Mask secrets everywhere and contact data in CONFIDENTIAL/RESTRICTED text."""
        kinds = ["secret"] if self.settings.mask_secrets else []
        if is_sensitive(label):
            kinds += self.settings.confidential_pii_kinds
        return redact(text, kinds) if kinds else (text, [])

    def split(self, docs: list[Document]) -> list[Document]:
        if not docs:
            return []
        source = docs[0].metadata["source"]
        stem = Path(source).stem.replace("_", " ")
        is_markdown = source.lower().endswith(".md")
        chunks: list[Document] = []
        for title, page, text, label in self._sections(docs):
            # The Markdown path already starts with the document title; PDF sections get the file name.
            header = (title or stem) if is_markdown else (f"{stem} > {title}" if title else stem)
            for piece in self.size_splitter.split_text(text):
                content, masked = self._protect(f"[{header}]\n{piece}", label)
                metadata = {"source": source, "section": title, "classification": label}
                if page is not None:
                    metadata["page"] = page
                if masked:
                    metadata["masked"] = masked
                chunks.append(Document(page_content=content, metadata=metadata))
        return chunks

    def _hash(self, docs: list[Document]) -> str:
        s = self.settings
        signature = (
            f"v{CHUNKING_VERSION}|{s.chunk_size}|{s.chunk_overlap}|{s.default_classification}|"
            f"{s.mask_secrets}|{sorted(s.confidential_pii_kinds)}|"
        )
        return hashlib.sha256((signature + "".join(d.page_content for d in docs)).encode()).hexdigest()[:16]

    def ingest(self, paths: Iterable[Path | str]) -> dict:
        files = discover_files(paths)
        added = skipped = quarantined = 0
        labels: Counter[str] = Counter()
        masked: Counter[str] = Counter()
        for path in files:
            docs = load_file(path)
            if not docs:
                logger.warning("empty_file", extra={"source": path.name})
                continue
            source = path.name
            doc_hash = self._hash(docs)
            if self.store.get_doc_hash(source) == doc_hash:
                skipped += 1
                continue
            chunks = self.split(docs)
            if self.settings.guardrails_enabled and self.settings.quarantine_suspicious_chunks:
                safe = []
                for chunk in chunks:
                    if rule := find_injection(chunk.page_content):
                        quarantined += 1
                        obs.GUARDRAIL_EVENTS.labels(event="quarantined_chunk").inc()
                        logger.warning("chunk_quarantined", extra={"source": source, "rule": rule})
                    else:
                        safe.append(chunk)
                chunks = safe
            for chunk in chunks:
                labels[chunk.metadata["classification"]] += 1
                masked.update(chunk.metadata.get("masked", []))
            ids = []
            for i, chunk in enumerate(chunks):
                chunk.metadata.update(doc_hash=doc_hash, chunk_index=i)
                ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{doc_hash}:{i}")))
            self.store.delete_source(source)  # drop chunks of the previous version
            self.store.add(chunks, ids)
            added += len(chunks)
            logger.info("file_ingested", extra={"source": source, "chunks": len(chunks)})
        return {
            "files": len(files),
            "chunks_added": added,
            "skipped_unchanged": skipped,
            "quarantined_chunks": quarantined,
            "masked": dict(masked),
            "classifications": dict(sorted(labels.items())),
            "total_chunks": self.store.count(),
        }
