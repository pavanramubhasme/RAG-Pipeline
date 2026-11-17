"""Answer cache: a question asked again is answered from memory, with no retrieval and no LLM call.

Key = the caller's access scope + top_k + normalised question. The question is the standalone
version (follow-ups are resolved first) after the input guard, so personal data is already
masked in the key. The scope is part of the key, so an answer built from HR-only chunks is
never served to someone without HR access.

Design choices:
* Exact match after normalisation (case, spaces, final "?"). A semantic cache ("similar
  question -> same answer") would also serve the 2025 answer for "vacation days in 2026".
* Only trustworthy results are stored: grounded answers and genuine refusals. Not stored:
  failed grounding checks, LLM failures, blocked requests, unresolved follow-ups.
* Never older than the index: cleared whenever documents are ingested, plus a TTL.
* In-process LRU (one API worker). Several workers or pods would share it via Redis.
"""

import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable

from ragapp.observability import ANSWER_CACHE

_SPACES = re.compile(r"\s+")
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def normalize(question: str) -> str:
    """'  What’s the WFH quota? ' and 'what's the wfh quota' get the same key."""
    text = _SPACES.sub(" ", question.translate(_QUOTES).lower()).strip()
    return text.rstrip("?!. ").strip()


class AnswerCache:
    def __init__(
        self, max_entries: int = 256, ttl_s: float = 3600.0, clock: Callable[[], float] = time.monotonic
    ):
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self._clock = clock  # injectable for tests
        self._entries: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self._lock = threading.Lock()  # the API answers requests in a thread pool

    @staticmethod
    def key(question: str, top_k: int, scopes: list[str] | None = None) -> str:
        scope = "*" if scopes is None else ",".join(sorted(scopes))
        return f"{scope}|{top_k}|{normalize(question)}"

    def get(self, key: str) -> dict | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and self._clock() - entry[0] > self.ttl_s:
                del self._entries[key]
                entry = None
            if entry is not None:
                self._entries.move_to_end(key)  # least recently used is evicted first
        ANSWER_CACHE.labels(result="hit" if entry else "miss").inc()
        return dict(entry[1]) if entry else None

    def put(self, key: str, value: dict) -> None:
        if self.max_entries <= 0:
            return
        with self._lock:
            self._entries[key] = (self._clock(), dict(value))
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> int:
        with self._lock:
            removed = len(self._entries)
            self._entries.clear()
        return removed

    def list_keys(self) -> list[str]:
        with self._lock:
            return list(self._entries)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
