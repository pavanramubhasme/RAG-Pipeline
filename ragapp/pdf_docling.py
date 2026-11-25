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

Code blocks: Docling joins every line of a block with a space, so
"def f(x):\n    if not x:\n        return False" arrives as "def f(x): if not x: return False"
(seen live: an LLM rebuilt the function and dropped its last line). pypdf's layout mode keeps
line breaks and indentation, so each code block is looked up there by its words and replaced
with that slice. If the words can't be found, Docling's text is kept. Docling doesn't label
every snippet as code (seen live: a Python function inside an ordinary paragraph after a JSON
fragment), so paragraphs with at least two code signs (def, return, print(, braces, "key": ...)
are restored the same way. Docling's layout model even labelled one snippet a section heading,
and headings are exported on one line, so every restored piece is exported as a fenced code
block in its place, whatever Docling called it. Docling also cuts one snippet into several
items (seen live: JSON, the "def" line, the body and the example call as four blocks), so code
blocks that follow each other with nothing but blank lines between them are joined into one.
"""

import logging
import re
import textwrap
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)
_BLANK_LINES = re.compile(r"\n{3,}")
# Same length on both sides, so positions found in the normalised text are valid in it.
_PLAIN = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "\u00a0": " "})
_TRAILING_SPACES = re.compile(r"[ \t]+\n")  # Docling marks kept line breaks with "  \n"
_ADJACENT_FENCES = re.compile(r"^```[ \t]*\n(?:[ \t]*\n)*```[^\n]*\n", re.MULTILINE)  # end, blanks, start
_FENCED_BLOCK = re.compile(r"(^```[^\n]*\n)(.*?)(\n```[ \t]*$)", re.MULTILINE | re.DOTALL)
# Signs of code in a paragraph. Two different ones are needed, so prose such as
# "the return policy" or "print the form" is left alone.
_CODE_SIGNS = (
    re.compile(r"\bdef \w+\("),
    re.compile(r"\bclass \w+[(:]"),
    re.compile(r"\breturn (?:True|False|None|\w+\(|[\w.]+$)", re.MULTILINE),
    re.compile(r"\bprint\("),
    re.compile(r"\bimport \w+"),
    re.compile(r"[{}]"),
    re.compile(r'"\w+"\s*:\s*["\[{\d]'),
    re.compile(r"\bif .+:\s"),
    re.compile(r"[=!<>]=|\w\(\w*\)"),
)


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


def restore_code_lines(flat: str, page_text: str, dedent: bool = True) -> str | None:
    """The code block ``flat`` (lines joined by Docling) with its line breaks and indentation,
    taken from ``page_text`` (pypdf layout mode). None if its words aren't there in order.
    ``dedent=False`` keeps the page's columns, so pieces of one snippet can be joined first."""
    tokens = flat.translate(_PLAIN).split()
    if len(tokens) < 2:
        return None
    text = page_text.translate(_PLAIN)
    match = re.search(r"\s+".join(re.escape(token) for token in tokens), text)
    if match is None:
        return None
    line_start = text.rfind("\n", 0, match.start()) + 1
    block = " " * (match.start() - line_start) + match.group(0)  # first line keeps its indent
    lines = [line.rstrip() for line in block.split("\n")]
    if len(lines) > 2 and all(not line for line in lines[1::2]):
        lines = lines[::2]  # loose line spacing: layout mode put a blank line after every line
    joined = "\n".join(lines)
    code = (textwrap.dedent(joined) if dedent else joined).strip("\n")
    return re.sub(r"\n{3,}", "\n\n", code)


def looks_like_code(text: str) -> bool:
    """At least two different signs of code, e.g. "def f(x):" and "return False"."""
    return sum(1 for sign in _CODE_SIGNS if sign.search(text)) >= 2


def _code_items(document):
    """Code blocks, plus ordinary paragraphs that look like code."""
    from docling_core.types.doc import CodeItem, TextItem

    for item, _level in document.iterate_items():
        if not getattr(item, "prov", None):
            continue
        if isinstance(item, CodeItem) or (isinstance(item, TextItem) and looks_like_code(item.text)):
            yield item


def restore_code_blocks(document, page_texts: dict[int, str]) -> dict[str, str]:
    """Find the code with its line breaks. Each restored item's text becomes a placeholder that
    ``pages_from_document`` swaps for a fenced code block; returns {placeholder: code}."""
    codes: dict[str, str] = {}
    for number, item in enumerate(_code_items(document)):
        page_text = page_texts.get(item.prov[0].page_no)
        code = restore_code_lines(item.text, page_text, dedent=False) if page_text else None
        if code and "\n" in code:
            placeholder = f"RAGCODEBLOCK{number:05d}END"  # letters and digits: survives any export
            item.text = placeholder
            codes[placeholder] = code
    return codes


def _insert_code(markdown: str, codes: dict[str, str]) -> str:
    """Replace each placeholder line (a paragraph, a "## heading" or a fenced block) with the code."""
    for placeholder, code in codes.items():
        if placeholder in markdown:
            fenced = re.compile(rf"```[^\n]*\n{placeholder}\n```")
            block = f"```\n{code}\n```"
            if fenced.search(markdown):
                markdown = fenced.sub(lambda _, block=block: block, markdown)
            else:
                line = re.compile(rf"^.*{placeholder}.*$", re.MULTILINE)
                markdown = line.sub(lambda _, block=block: block, markdown)
    return markdown


def _code_pages(document) -> set[int]:
    return {item.prov[0].page_no for item in _code_items(document)}


def layout_texts(path: Path, pages: set[int]) -> dict[int, str]:
    """pypdf's layout-mode text of the given pages (1-based); empty if pypdf can't read them."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return {n: reader.pages[n - 1].extract_text(extraction_mode="layout") or "" for n in sorted(pages)}
    except Exception as exc:  # a nicety: without it, code blocks keep Docling's text
        logger.warning("code_layout_failed", extra={"source": path.name, "error": str(exc)[:200]})
        return {}


def pages_from_document(document, source: str, codes: dict[str, str] | None = None) -> list[Document]:
    """One LangChain Document per page, as Markdown, from a converted DoclingDocument.
    ``codes``: restored code from ``restore_code_blocks``, put back as fenced blocks."""
    pages = []
    for page_no in sorted(document.pages):
        text = document.export_to_markdown(
            page_no=page_no,
            escape_underscores=False,  # register names: ASI_OUT_CH_EN, not ASI\\_OUT\\_CH\\_EN
            escape_html=False,  # "< 0x07: MUTE", not "&lt; 0x07: MUTE"
            image_placeholder="",
            compact_tables=True,  # no padding spaces: more rows fit in a chunk
        )
        text = _insert_code(_TRAILING_SPACES.sub("\n", text), codes or {})
        text = _ADJACENT_FENCES.sub("", text)  # one snippet that Docling cut into pieces
        # the pieces kept the page's columns: remove the common margin only now, so the body of a
        # function stays indented under its "def" line
        text = _FENCED_BLOCK.sub(lambda m: m.group(1) + textwrap.dedent(m.group(2)) + m.group(3), text)
        text = _BLANK_LINES.sub("\n\n", text).strip()
        if text:
            pages.append(Document(page_content=text, metadata={"source": source, "page": page_no}))
    return pages


def load_pdf(path: Path, table_mode: str = "accurate", ocr: bool = False, threads: int = 4) -> list[Document]:
    logger.info("docling_converting", extra={"source": path.name})
    result = _converter(table_mode, ocr, threads).convert(str(path))
    codes: dict[str, str] = {}
    if code_pages := _code_pages(result.document):
        codes = restore_code_blocks(result.document, layout_texts(path, code_pages))
        logger.info("code_blocks_restored", extra={"source": path.name, "restored": len(codes)})
    return pages_from_document(result.document, path.name, codes)