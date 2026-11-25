"""Code in answers is copied from the source, not retyped by the model.

Seen live, twice: asked for the full "Python - request validation" code, the 3B model wrote the
function and stopped at "return True", leaving out the example call that follows it in the
document, even with a prompt rule to copy code completely. A small model summarises; code must
not be summarised.

So after generation, each code block in the answer is compared with the code blocks of the
retrieved passages. If the answer's lines appear, in order, in a passage's block that goes on
after them, the answer's block is replaced with that passage's block, from the answer's first
line to the end. The model chooses *which* code; the document supplies *all* of it. Answers to
other questions, and code that matches no passage, are left exactly as they are.
"""

import re
import textwrap

# "return True\n``` [1][2]": the closing fence must be alone on its line, or Markdown keeps the
# code block open and shows the fence and the citations inside it (seen live).
_FENCE_THEN_TEXT = re.compile(r"^```[ \t]*(\S[^\n]*\[\d+\][^\n]*)$", re.MULTILINE)
_TEXT_THEN_FENCE = re.compile(r"^([^\n`]*\S)[ \t]*```[ \t]*$", re.MULTILINE)

ASKS_FOR_CODE = re.compile(r"\b(?:code|source|snippet|script|function|implementation|program)\b", re.I)
_FENCED = re.compile(r"```([^\n`]*)\n(.*?)\n?```", re.DOTALL)


def _key(line: str) -> str:
    return " ".join(line.split())


def _complete(lines: list[str], source_lines: list[str]) -> list[str] | None:
    """The source lines from the answer's first line to the end of the block, if every answer
    line appears there in order and the source has more; else None."""
    wanted = [_key(line) for line in lines if line.strip()]
    keys = [_key(line) for line in source_lines]
    if len(wanted) < 2:
        return None
    for start, key in enumerate(keys):
        if key != wanted[0]:
            continue
        position, matched = start, 0
        for want in wanted:
            while position < len(keys) and keys[position] != want:
                position += 1
            if position == len(keys):
                break
            matched, position = matched + 1, position + 1
        tail = source_lines[start:]
        if matched == len(wanted) and len([line for line in tail if line.strip()]) > len(wanted):
            return textwrap.dedent("\n".join(tail)).strip("\n").splitlines()
    return None


def fix_fences(answer: str) -> str:
    """Put a code fence that shares its line with text on a line of its own."""
    answer = _FENCE_THEN_TEXT.sub(lambda m: "```\n" + m.group(1), answer)
    return _TEXT_THEN_FENCE.sub(lambda m: m.group(1) + "\n```", answer)


def complete_code(answer: str, question: str, passages: list[str]) -> str:
    """``answer`` with its code blocks completed from ``passages`` (see the module docstring)."""
    answer = fix_fences(answer)
    if "```" not in answer or not ASKS_FOR_CODE.search(question):
        return answer
    blocks = [body.splitlines() for passage in passages for _lang, body in _FENCED.findall(passage)]
    if not blocks:
        return answer

    def replace(match: re.Match) -> str:
        language, body = match.group(1), match.group(2).splitlines()
        for block in blocks:
            if full := _complete(body, block):
                return f"```{language}\n" + "\n".join(full) + "\n```"
        return match.group(0)

    return _FENCED.sub(replace, answer)