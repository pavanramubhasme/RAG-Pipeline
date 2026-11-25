"""Register-address questions: "Which register is at address 0x04?"

Seen live on the TAS6424 datasheet: "What is the register for the register address 0x04?" was
answered "SAP Control Register" (relevance 1.00), because its heading reads
"SAP Control Register (address = 0x03) [default = 0x04]". BM25 and the cross-encoder match the
word "address" and the token 0x04, and can't tell an address from a default value. The right
register's heading says "(address = 0x04)".

The fix is a rule: when a question asks for a register address, keep only the chunks whose
section heading has that address (ranges such as "0x05-0x08" included). If no heading matches,
nothing changes. "Which register has default 0x04?" is not an address question and is left alone.
"""

import re

from langchain_core.documents import Document

# "address 0x04", "address = 0x04", "register 0x04", "register at 0x04", "0x04 register"
_ASKED = re.compile(
    r"\b(?:address(?:es)?|addr|register|reg)\s*(?:=|:|is|at|of)?\s*(0x[0-9a-f]+)\b"
    r"|\b(0x[0-9a-f]+)\s+(?:address|register)\b",
    re.IGNORECASE,
)
# "(address = 0x04)", "(page = 0x00, address = 0x74)", "(address = 0x05-0x08)"
_HEADING = re.compile(r"address\s*=\s*(0x[0-9a-f]+)(?:\s*[-–]\s*(0x[0-9a-f]+))?", re.IGNORECASE)
_MAX_RANGE = 256


def asked_addresses(question: str) -> set[int]:
    """Register addresses the question asks about (empty if it asks about none)."""
    return {int(first or second, 16) for first, second in _ASKED.findall(question)}


def heading_addresses(heading: str) -> set[int]:
    """Addresses a section heading declares, ranges expanded."""
    found: set[int] = set()
    for start, end in _HEADING.findall(heading):
        low = int(start, 16)
        high = int(end, 16) if end else low
        if 0 <= high - low < _MAX_RANGE:
            found.update(range(low, high + 1))
    return found


def keep_matching(question: str, documents: list[Document]) -> list[Document]:
    """For an address question, only the chunks of registers at that address; else unchanged."""
    asked = asked_addresses(question)
    if not asked:
        return documents
    matching = [d for d in documents if asked & heading_addresses(d.metadata.get("section") or "")]
    return matching or documents