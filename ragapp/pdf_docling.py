"""Layout-aware PDF parsing with IBM Docling (MIT licence, runs locally).

Why: pypdf returns a page as a stream of text lines, so a register table such as

    Bit | Field             | Reset | Description
    7–6 | CH1 STATE CONTROL | 01    | 00: PLAY 01: Hi-Z 10: MUTE 11: DC load diagnostics

arrives as loose lines, and a small LLM has to guess which value belongs to which field
(seen live: "< 0x07: MUTE" was read as "0x07"). Docling runs a layout model and a table
model (TableFormer) on every page and returns Markdown: headings as "## ...", tables as
Markdown tables with one row per line, page headers and footers removed.

Each page is exported separately, so page numbers (citations) and per-page classification
labels (access control) keep working exactly as with pypdf. The models (~0.5 GB) are
downloaded from Hugging Face on the first run; conversion takes a few seconds per page on
a CPU, once per file (unchanged files are skipped before parsing).
"""

import logging
import re
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)
_BLANK_LINES = re.compile(r"\n{3,}")


def docling_available() -> bool:
    try:
        import docling.document_converter  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=4)
def _converter(table_mode: str, ocr: bool, threads: int):
    """Build once per process: loading the models takes seconds."""
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(do_ocr=ocr, do_table_structure=True)  # OCR only for scanned PDFs
    options.table_structure_options.mode = TableFormerMode(table_mode)  # "accurate" or "fast"
    options.table_structure_options.do_cell_matching = True  # cell text from the PDF, not re-read
    options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.AUTO, num_threads=threads)
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})


def pages_from_document(document, source: str) -> list[Document]:
    """One LangChain Document per page, as Markdown, from a converted DoclingDocument."""
    pages = []
    for page_no in sorted(document.pages):
        text = document.export_to_markdown(
            page_no=page_no,
            escape_underscores=False,  # register names: ASI_OUT_CH_EN, not ASI\\_OUT\\_CH\\_EN
            escape_html=False,  # "< 0x07: MUTE", not "&lt; 0x07: MUTE"
            image_placeholder="",
            compact_tables=True,  # no padding spaces: more rows fit in a chunk
        )
        text = _BLANK_LINES.sub("\n\n", text).strip()
        if text:
            pages.append(Document(page_content=text, metadata={"source": source, "page": page_no}))
    return pages


def load_pdf(path: Path, table_mode: str = "accurate", ocr: bool = False, threads: int = 4) -> list[Document]:
    logger.info("docling_converting", extra={"source": path.name})
    result = _converter(table_mode, ocr, threads).convert(str(path))
    return pages_from_document(result.document, path.name)
