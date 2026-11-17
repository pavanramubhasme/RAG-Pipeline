"""Deterministic query expansion: "channel 1" <-> "CH1".

Datasheets name numbered things with short forms (CH1, IN2, OUT3, MIC1) while people spell
them out ("channel 1"), and the other way round. BM25 matches tokens, so "channel 1" never
matches "CH1", and the cross-encoder doesn't know they are the same thing either. Seen live:
"To mute channel 1, what value do I set?" retrieved the volume register (relevance 0.10)
instead of the Channel State Control register, whose table only says "CH1".

The fix adds the other spelling to the search text. Rules, not an LLM: no extra call,
no latency, and every expansion is predictable and tested.
"""

import re

# long form -> short form. Short forms are only recognised in upper case ("IN2", not "in 2"),
# so ordinary words ("in 2 weeks", "out 3 days") are never expanded.
NUMBERED_TERMS = {
    "channel": "CH",
    "input": "IN",
    "output": "OUT",
    "microphone": "MIC",
}
_NUMBER = r"(\d{1,2})"
_LONG = re.compile(rf"\b({'|'.join(NUMBERED_TERMS)})s?[ _-]?{_NUMBER}\b", re.IGNORECASE)
_SHORT = re.compile(rf"\b({'|'.join(NUMBERED_TERMS.values())})[ _-]?{_NUMBER}\b")
_LONG_FOR = {short: long for long, short in NUMBERED_TERMS.items()}


def expansions(query: str) -> list[str]:
    """The other spellings of every numbered term in ``query``, without duplicates."""
    found: list[str] = []
    for match in _LONG.finditer(query):
        found.append(f"{NUMBERED_TERMS[match.group(1).lower()]}{match.group(2)}")
    for match in _SHORT.finditer(query):
        found.append(f"{_LONG_FOR[match.group(1)]} {match.group(2)}")
    present = query.lower()
    return [term for term in dict.fromkeys(found) if term.lower() not in present]


def expand_query(query: str) -> str:
    """``query`` plus the other spellings in brackets, or ``query`` unchanged."""
    extra = expansions(query)
    return f"{query} ({', '.join(extra)})" if extra else query
