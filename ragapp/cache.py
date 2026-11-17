"""Answer cache: a question asked again is answered from memory, with no retrieval and no LLM call.

Key = the caller's access scope + top_k + normalised question. The question is the standalone
version (follow-ups are resolved first) after the input guard, so personal data is already
masked in the key. The scope is part of the key, so an answer built from HR-only chunks is
never served to someone without HR access.

Two ways to hit:
* **Exact** (any user with the same scope): the same question after normalisation
  (case, spaces, final "?").
* **Same meaning** (v6.3, only your own earlier questions): "What is the full time employee
  leave policy?" after "What is the leave policy of the full-time employee?". A semantic cache
  is risky: "vacation days in 2025" and "in 2026" are near-identical embeddings with different
  answers. So a similar question only counts when
    1. its numbers and codes are the same (2025, CH1, E03, EMP-7001), checked by code, and
    2. it has the same content words, in the same order or with one phrase moved ("leave policy
       of the full-time employee" / "full time employee leave policy"; similarity >= 0.90), or
       different words but almost the same meaning (similarity >= ``threshold``, default 0.98).
       Same words in any other order never match: "Can HR see employee data?" and "Can
       employees see HR data?" swap who does what, and embeddings barely notice word order.
  Small embedding models put "Fridays" and "Mondays" close together, so the threshold is set
  high and checked with real models: ``uv run python -m eval.cache_calibration``.
  Matching only your own questions means another user's wording never decides your answer.

Only trustworthy results are stored: grounded answers and genuine refusals. Never older than
the index: cleared whenever documents are ingested, plus a TTL. In-process LRU (one API worker);
several workers or pods would share it via Redis (with a vector index for the semantic part).
"""

import logging
import math
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from ragapp.observability import ANSWER_CACHE

logger = logging.getLogger(__name__)

_SPACES = re.compile(r"\s+")
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})
_TOKEN = re.compile(r"[a-z0-9]+(?:[.:][0-9]+)*")  # keeps 2.4.0, 02:00, 0x04 whole
SAME_WORDS_SIMILARITY = (
    0.90  # same words, other order: the embedding must still agree ("HR sees" vs "sees HR")
)

# Words that carry no content in a question. Kept short on purpose: a word missing here only
# makes a hit less likely (safe), a wrong word here could make two different questions equal.
# "for", "to", "from", "by", "with" stay content words: they say who does what to whom
# ("leave for managers" is not "managers for leave"). "of" goes: "policy of X" = "X policy".
STOPWORDS = frozenset(
    """a an the is are was were be been am do does did of in on at about and or
    what whats which who whom whose how when where why can could may might shall should will would
    i me my mine we our us you your it its this that these those there here please tell explain
    give show get got have has had any some s""".split()  # noqa: SIM905 (readable as a block)
)


def normalize(question: str) -> str:
    """'  What’s the WFH quota? ' and 'what's the wfh quota' get the same key."""
    text = _SPACES.sub(" ", question.translate(_QUOTES).lower()).strip()
    return text.rstrip("?!. ").strip()


def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"  # policies -> policy
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]  # employees -> employee, days -> day
    return word


def content_words(question: str) -> tuple[str, ...]:
    """Words that carry the meaning, in order: "full-time" = "full time", "employees" = "employee"."""
    text = normalize(question).replace("-", " ").replace("_", " ").replace("/", " ")
    return tuple(_stem(w) for w in _TOKEN.findall(text) if w not in STOPWORDS)


def word_relation(a: tuple[str, ...], b: tuple[str, ...]) -> str:
    """How two questions' content words relate:
    "same"      identical sequence;
    "moved"     same words, one phrase moved ("leave policy of full-time employee" ->
                "full time employee leave policy"): the two blocks swap places;
    "reordered" same words in another order ("HR sees employee data" -> "employees see HR
                data"): who does what may have changed, so this never matches;
    "different" other words."""
    if a == b:
        return "same"
    if sorted(a) != sorted(b):
        return "different"
    start = 0
    while a[start] == b[start]:
        start += 1
    end = 0
    while a[-1 - end] == b[-1 - end]:
        end += 1
    middle_a, middle_b = a[start : len(a) - end], b[start : len(b) - end]
    rotations = (middle_a[k:] + middle_a[:k] for k in range(1, len(middle_a)))
    return "moved" if middle_b in rotations else "reordered"


def key_terms(question: str) -> frozenset[str]:
    """Numbers and codes (2025, ch1, e03, 7001, 0x04): two questions must agree on all of them."""
    return frozenset(w for w in _TOKEN.findall(normalize(question)) if any(c.isdigit() for c in w))


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


@dataclass
class _Entry:
    stored_at: float
    value: dict
    user: str | None = None
    question: str | None = None  # as asked (standalone, personal data masked), shown on a hit
    words: tuple[str, ...] = ()
    terms: frozenset[str] = frozenset()
    vector: list[float] | None = None


@dataclass
class Lookup:
    value: dict | None  # None = miss
    similar_to: str | None = None  # same-meaning hit: your earlier question
    similarity: float | None = None
    vector: list[float] | None = None  # the question's embedding, reused when the answer is stored


def _bucket(key: str) -> str:
    """'scope|top_k|question' -> 'scope|top_k': only answers built for the same scope match."""
    parts = key.split("|", 2)
    return "|".join(parts[:2]) if len(parts) == 3 else key


class AnswerCache:
    def __init__(
        self, max_entries: int = 256, ttl_s: float = 3600.0, clock: Callable[[], float] = time.monotonic
    ):
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self._clock = clock  # injectable for tests
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.Lock()  # the API answers requests in a thread pool

    @staticmethod
    def key(question: str, top_k: int, scopes: list[str] | None = None) -> str:
        scope = "*" if scopes is None else ",".join(sorted(scopes))
        return f"{scope}|{top_k}|{normalize(question)}"

    # ------------------------------------------------------------------ exact
    def _get_entry(self, key: str) -> _Entry | None:
        entry = self._entries.get(key)
        if entry is not None and self._clock() - entry.stored_at > self.ttl_s:
            del self._entries[key]
            entry = None
        if entry is not None:
            self._entries.move_to_end(key)  # least recently used is evicted first
        return entry

    def get(self, key: str) -> dict | None:
        with self._lock:
            entry = self._get_entry(key)
        ANSWER_CACHE.labels(result="hit" if entry else "miss").inc()
        return dict(entry.value) if entry else None

    # ---------------------------------------------------------- exact, then meaning
    def lookup(
        self,
        key: str,
        *,
        user: str | None = None,
        question: str | None = None,
        embed: Callable[[str], list[float]] | None = None,
        threshold: float = 0.98,
    ) -> Lookup:
        """Exact match first (no embedding needed). Then, with ``embed``, the closest of
        ``user``'s earlier questions in the same scope that passes the checks above."""
        with self._lock:
            entry = self._get_entry(key)
        if entry is not None:
            ANSWER_CACHE.labels(result="hit").inc()
            return Lookup(dict(entry.value))
        if embed is None or question is None:
            ANSWER_CACHE.labels(result="miss").inc()
            return Lookup(None)
        try:
            vector = list(embed(normalize(question)))
        except Exception as exc:  # an optimisation: a failed embedding is just a miss
            logger.warning("cache_embedding_failed", extra={"error": str(exc)[:200]})
            ANSWER_CACHE.labels(result="miss").inc()
            return Lookup(None)

        words, terms, bucket = content_words(question), key_terms(question), _bucket(key)
        best: tuple[float, str] | None = None
        with self._lock:
            now = self._clock()
            for other_key, other in self._entries.items():
                if (
                    other.vector is None
                    or other.user != user
                    or _bucket(other_key) != bucket
                    or other.terms != terms
                    or now - other.stored_at > self.ttl_s
                ):
                    continue
                relation = word_relation(other.words, words)
                if relation == "reordered":
                    continue
                similarity = _cosine(vector, other.vector)
                needed = threshold if relation == "different" else SAME_WORDS_SIMILARITY
                if similarity >= needed and (best is None or similarity > best[0]):
                    best = (similarity, other_key)
            if best is None:
                ANSWER_CACHE.labels(result="miss").inc()
                return Lookup(None, vector=vector)
            match = self._entries[best[1]]
            self._entries.move_to_end(best[1])
        ANSWER_CACHE.labels(result="similar_hit").inc()
        return Lookup(dict(match.value), match.question, round(best[0], 3), vector)

    def put(
        self,
        key: str,
        value: dict,
        *,
        user: str | None = None,
        question: str | None = None,
        vector: list[float] | None = None,
    ) -> None:
        if self.max_entries <= 0:
            return
        entry = _Entry(self._clock(), dict(value), user, question)
        if question is not None and vector is not None:  # findable by meaning, too
            entry.words, entry.terms, entry.vector = content_words(question), key_terms(question), vector
        with self._lock:
            self._entries[key] = entry
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
