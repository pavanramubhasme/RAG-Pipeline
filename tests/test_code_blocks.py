"""Code blocks keep their lines, and the LLM is told to copy code completely.

Seen live on the Northstar PDF: "Python — request validation full code?" was answered with the
function but without its last line (the example print call). Docling had joined the block into
one line: "def validate_access_request(user, resource, reason): if not user or not resource: ..."
"""

import pytest

from ragapp import pdf_docling
from ragapp.prompts import ANSWER_PROMPT

FLAT = (
    "Python - request validation def validate_access_request(user, resource, reason): "
    "if not user or not resource: return False if len(reason.strip()) < 12: return False "
    'return True print(validate_access_request( "EMP-1042", "Analytics Portal", '
    '"Read access for monthly reporting" ))'
)
CODE = """Python - request validation
def validate_access_request(user, resource, reason):
    if not user or not resource:
        return False
    if len(reason.strip()) < 12:
        return False
    return True
print(validate_access_request(
    "EMP-1042",
    "Analytics Portal",
    "Read access for monthly reporting"
))"""
LAYOUT_PAGE = (
    "   Northstar Engineering Handbook                               page 44\n\n"
    "   05. Engineering - Code review\n\n"
    "   Reviewers check the example below before merging.\n\n"
    + "\n\n".join("   " + line for line in CODE.splitlines())
    + "\n\n   The reason must be at least 12 characters long.\n"
)


def test_restores_line_breaks_and_indentation():
    assert pdf_docling.restore_code_lines(FLAT, LAYOUT_PAGE) == CODE  # loose spacing: blank lines dropped
    tight = LAYOUT_PAGE.replace("\n\n   ", "\n   ")
    assert pdf_docling.restore_code_lines(FLAT, tight) == CODE


def test_typographic_quotes_and_dashes_still_match():
    curly = FLAT.replace('"EMP-1042"', "“EMP-1042”").replace("Python - request", "Python – request")
    assert "print(validate_access_request(" in pdf_docling.restore_code_lines(curly, LAYOUT_PAGE)


def test_returns_none_when_the_words_are_not_on_the_page():
    assert pdf_docling.restore_code_lines(FLAT, "a completely different page") is None
    assert pdf_docling.restore_code_lines("x", LAYOUT_PAGE) is None  # too short to locate safely


def test_code_blocks_in_a_docling_document_get_their_lines_back():
    pytest.importorskip("docling_core")
    from docling_core.types.doc import BoundingBox, DoclingDocument, ProvenanceItem, Size

    def prov(page):
        return ProvenanceItem(page_no=page, bbox=BoundingBox(l=0, t=10, r=100, b=0), charspan=(0, 1))

    doc = DoclingDocument(name="northstar")
    doc.add_page(page_no=44, size=Size(width=595, height=842))
    doc.add_heading(text="05. Engineering - Code review", level=2, prov=prov(44))
    doc.add_code(text=FLAT, prov=prov(44))

    assert pdf_docling._code_pages(doc) == {44}
    codes = pdf_docling.restore_code_blocks(doc, {44: LAYOUT_PAGE})
    assert len(codes) == 1
    markdown = pdf_docling.pages_from_document(doc, "northstar.pdf", codes)[0].page_content
    assert markdown.count("```") == 2  # Docling's own fence, not a second one
    assert "    if not user or not resource:\n        return False" in markdown
    assert '"Read access for monthly reporting"' in markdown


def test_unmatched_code_keeps_doclings_text():
    pytest.importorskip("docling_core")
    from docling_core.types.doc import BoundingBox, DoclingDocument, ProvenanceItem, Size

    doc = DoclingDocument(name="x")
    doc.add_page(page_no=1, size=Size(width=595, height=842))
    doc.add_code(
        text=FLAT, prov=ProvenanceItem(page_no=1, bbox=BoundingBox(l=0, t=10, r=100, b=0), charspan=(0, 1))
    )
    assert pdf_docling.restore_code_blocks(doc, {1: "nothing like it"}) == {}
    assert FLAT in pdf_docling.pages_from_document(doc, "x.pdf")[0].page_content


def _pdf_with_code(lines: list[str]) -> bytes:
    """A one-page PDF in Courier, one line per row: enough for pypdf's layout mode."""
    ops = ["BT", "/F1 10 Tf", "12 TL", "72 720 Td"]
    for line in lines:
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        ops.append(f"({escaped}) Tj T*")
    ops.append("ET")
    stream = "\n".join(ops).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
    ]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def test_pypdf_layout_mode_keeps_the_indentation_of_a_real_pdf(tmp_path):
    pdf = tmp_path / "code.pdf"
    pdf.write_bytes(_pdf_with_code(["Some text before the code.", *CODE.splitlines()]))
    page = pdf_docling.layout_texts(pdf, {1})[1]
    restored = pdf_docling.restore_code_lines(FLAT, page)
    assert restored is not None
    assert (
        "def validate_access_request(user, resource, reason):\n    if not user or not resource:" in restored
    )
    assert restored.rstrip().endswith("))")


def test_an_unreadable_pdf_just_means_no_restoring(tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")
    assert pdf_docling.layout_texts(broken, {1}) == {}


def test_the_answer_prompt_asks_for_complete_code():
    system = ANSWER_PROMPT.messages[0].prompt.template
    assert "copy it from the context completely and exactly" in system
    assert "including example calls" in system


# ---------------------------------------------- code inside an ordinary paragraph (v6.3.4)
@pytest.mark.parametrize(
    ("text", "is_code"),
    [
        (FLAT, True),
        ('"owner": "IT Operations process owner" } ' + FLAT, True),  # seen live on page 26
        ('{ "id": "CHG-2041", "approval_required": true }', True),
        ("import json\nprint(json.dumps(record))", True),
        ("Employees may return equipment to the IT desk. The return policy is in section 4.", False),
        ("Print the form and sign it.", False),
        ("Use the formula (a + b) to compute the total. See section 3 (Appendix).", False),
        ("If approved, the request is forwarded: the owner validates it.", False),
    ],
)
def test_looks_like_code(text, is_code):
    assert pdf_docling.looks_like_code(text) is is_code


def test_code_in_an_ordinary_paragraph_is_restored_too():
    pytest.importorskip("docling_core")
    from docling_core.types.doc import BoundingBox, DocItemLabel, DoclingDocument, ProvenanceItem, Size

    prov = ProvenanceItem(page_no=26, bbox=BoundingBox(l=0, t=10, r=100, b=0), charspan=(0, 1))
    doc = DoclingDocument(name="northstar")
    doc.add_page(page_no=26, size=Size(width=595, height=842))
    doc.add_text(label=DocItemLabel.TEXT, text="Changes need a ticket and an approver.", prov=prov)
    doc.add_text(label=DocItemLabel.TEXT, text=FLAT, prov=prov)  # Docling didn't call it code

    assert pdf_docling._code_pages(doc) == {26}
    codes = pdf_docling.restore_code_blocks(doc, {26: LAYOUT_PAGE})
    assert len(codes) == 1
    markdown = pdf_docling.pages_from_document(doc, "northstar.pdf", codes)[0].page_content
    assert markdown.startswith("Changes need a ticket and an approver.")  # prose untouched
    assert (
        "def validate_access_request(user, resource, reason):\n    if not user or not resource:" in markdown
    )
    assert '    "Read access for monthly reporting"\n))' in markdown
    assert "  \n" not in markdown  # Docling's Markdown line-break spaces are removed


def test_code_that_docling_called_a_heading_becomes_a_code_block():
    """Seen live: the citation began with "## ... def validate_access_request(...)": a heading,
    which Docling exports on one line, so the restored line breaks were lost again."""
    pytest.importorskip("docling_core")
    from docling_core.types.doc import BoundingBox, DoclingDocument, ProvenanceItem, Size

    prov = ProvenanceItem(page_no=26, bbox=BoundingBox(l=0, t=10, r=100, b=0), charspan=(0, 1))
    doc = DoclingDocument(name="northstar")
    doc.add_page(page_no=26, size=Size(width=595, height=842))
    doc.add_heading(text="03. IT Operations - Change management", level=1, prov=prov)
    doc.add_heading(text=FLAT, level=2, prov=prov)  # the layout model's mistake

    codes = pdf_docling.restore_code_blocks(doc, {26: LAYOUT_PAGE})
    markdown = pdf_docling.pages_from_document(doc, "northstar.pdf", codes)[0].page_content
    assert "## 03. IT Operations - Change management" in markdown  # the real heading stays
    assert "```\nPython - request validation\ndef validate_access_request(" in markdown
    assert "    if not user or not resource:\n        return False" in markdown
    assert not any(line.startswith("#") and "def " in line for line in markdown.splitlines())


# ------------------------------------------------------- answers copy code, not retype it
from ragapp.code_answers import complete_code  # noqa: E402

PASSAGE = (
    f"[Northstar > 03. IT Operations - Change management]\n```\n{CODE}\n```\nThe reason needs 12 characters."
)
SHORTENED = """The source code is:

```python
def validate_access_request(user, resource, reason):
    if not user or not resource:
        return False
    if len(reason.strip()) < 12:
        return False
    return True
```

Cited from [1]."""


def test_a_shortened_code_answer_is_completed_from_the_passage():
    completed = complete_code(SHORTENED, "Python - request validation full code?", [PASSAGE])
    assert completed.startswith("The source code is:\n\n```python\ndef validate_access_request(")
    assert '    "Read access for monthly reporting"\n))\n```' in completed
    assert "Python - request validation\ndef" not in completed  # starts where the model started
    assert completed.endswith("Cited from [1].")


def test_complete_answers_and_other_questions_are_unchanged():
    full = f"```python\n{CODE.split(chr(10), 1)[1]}\n```"
    assert complete_code(full, "full code?", [PASSAGE]) == full  # nothing missing
    assert complete_code(SHORTENED, "How many vacation days do I get?", [PASSAGE]) == SHORTENED


def test_code_that_matches_no_passage_is_left_alone():
    invented = "```python\ndef something_else(x):\n    return x * 2\n```"
    assert complete_code(invented, "Show the code", [PASSAGE]) == invented
    assert complete_code(SHORTENED, "Show the code", ["no code in this passage"]) == SHORTENED


def test_lines_out_of_order_are_not_completed():
    reordered = "```\nreturn True\ndef validate_access_request(user, resource, reason):\n```"
    assert complete_code(reordered, "the code?", [PASSAGE]) == reordered


def test_end_to_end_the_answer_contains_the_whole_code(settings, tmp_path):
    from tests.conftest import make_service
    from tests.fakes import FakeChatModel

    (tmp_path / "engineering.md").write_text(
        f"# Engineering\n\n## Code review\n\nUse this check before merging:\n\n```\n{CODE}\n```\n",
        encoding="utf-8",
    )
    llm = FakeChatModel(calls=[], prompts=[], answer=SHORTENED)  # a model that stops at return True
    service = make_service(settings, llm)
    try:
        service.ingest([tmp_path])
        answer = service.ask("Python - request validation full code?").answer
        assert answer.startswith("The source code is:")
        assert '"Read access for monthly reporting"\n))' in answer  # the line the model left out
    finally:
        service.close()


# ----------------------------------------- one snippet cut into four pieces (seen live, v6.3.6)
JSON_TAIL = (
    '{ "document_id": "NST-ENG-02", "department": "Engineering", "topic": "Code review", '
    '"classification": "INTERNAL", "owner": "Engineering process owner"'
)
PAGE_44 = (
    "   05. Engineering - Code review\n"
    "   {\n"
    '     "document_id": "NST-ENG-02",\n'
    '     "department": "Engineering",\n'
    '     "topic": "Code review",\n'
    '     "classification": "INTERNAL",\n'
    '     "owner": "Engineering process owner"\n'
    "   }\n" + "\n".join("   " + line for line in CODE.splitlines()) + "\n"
)


def test_a_snippet_docling_cut_into_pieces_becomes_one_code_block():
    pytest.importorskip("docling_core")
    from docling_core.types.doc import BoundingBox, DocItemLabel, DoclingDocument, ProvenanceItem, Size

    prov = ProvenanceItem(page_no=44, bbox=BoundingBox(l=0, t=10, r=100, b=0), charspan=(0, 1))
    doc = DoclingDocument(name="northstar")
    doc.add_page(page_no=44, size=Size(width=595, height=842))
    doc.add_heading(text="05. Engineering - Code review", level=1, prov=prov)
    doc.add_code(text=JSON_TAIL, prov=prov)
    doc.add_heading(  # the layout model's mistake
        text="} Python - request validation def validate_access_request(user, resource, reason):",
        level=2,
        prov=prov,
    )
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="if not user or not resource: return False "
        "if len(reason.strip()) < 12: return False return True",
        prov=prov,
    )
    doc.add_code(
        text='print(validate_access_request( "EMP-1042", "Analytics Portal", '
        '"Read access for monthly reporting" ))',
        prov=prov,
    )

    codes = pdf_docling.restore_code_blocks(doc, {44: PAGE_44})
    assert len(codes) == 4
    markdown = pdf_docling.pages_from_document(doc, "northstar.pdf", codes)[0].page_content
    assert markdown.count("```") == 2  # one block, not four
    block = markdown.split("```")[1]
    assert (
        '"owner": "Engineering process owner"\n}\nPython - request validation\ndef validate_access_request('
        in block
    )
    assert "    return True\nprint(validate_access_request(" in block
    assert block.rstrip().endswith("))")


def test_completion_works_on_the_joined_block():
    passage = '```\n{\n  "owner": "Engineering process owner"\n}\n' + CODE + "\n```"
    completed = complete_code(SHORTENED, 'What is the code for "Python - request validation"', [passage])
    assert completed.split("```python\n")[1].startswith("def validate_access_request(")
    assert '"Read access for monthly reporting"\n))' in completed


def test_a_fence_that_shares_its_line_with_citations_is_split():
    glued = "The code is:\n\n```python\ndef f(x):\n    return x\n``` [1][2][3][4]"
    fixed = complete_code(glued, "What is the code?", [])
    assert fixed.endswith("```\n[1][2][3][4]")
    inline = "The code is: ```python\ndef f(x):\n    return x```"
    assert complete_code(inline, "the code?", []).endswith("    return x\n```")