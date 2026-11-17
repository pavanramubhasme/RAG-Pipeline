"""Same meaning, other words: answered from the cache, but only when it is safe.

Seen live: "What is the leave policy of the full-time employee?" and "What is the full time
employee leave policy?" each ran the whole pipeline. The risk of a semantic cache is a confident
wrong answer ("vacation days in 2025" vs "in 2026"), so a match also needs the same numbers and
codes, and only your own earlier questions count."""

import pytest

from ragapp.cache import AnswerCache, content_words, key_terms, normalize, word_relation
from ragapp.config import Settings
from tests.conftest import SAMPLE_DIR, make_service


# ---------------------------------------------------------------- the deterministic checks
@pytest.mark.parametrize(
    ("a", "b", "relation"),
    [
        ("How many vacation days do I get?", "how many vacation days?", "same"),
        ("Which policies apply to employees?", "Which policy applies to an employee", "same"),
        (
            "What is the leave policy of the full-time employee?",
            "What is the full time employee leave policy?",
            "moved",
        ),
        ("How many vacation days per calendar year?", "Per calendar year, how many vacation days?", "moved"),
        ("How many vacation days do I get?", "How many days of vacation do I get?", "moved"),
        ("Can HR see employee data?", "Can employees see HR data?", "reordered"),  # who sees whom
        ("Who approves leave for managers?", "Who approves managers for leave?", "reordered"),
        ("Leave policy for full-time employees?", "Leave policy for part-time employees?", "different"),
        ("Can I carry over vacation days?", "Can't I carry over vacation days?", "different"),
        ("What is the leave policy?", "What is the leave policy of the full-time employee?", "different"),
    ],
)
def test_word_relation(a, b, relation):
    assert word_relation(content_words(a), content_words(b)) == relation


@pytest.mark.parametrize(
    ("question", "terms"),
    [
        ("How many vacation days in 2026?", {"2026"}),
        ("To mute CH1, which value in register 0x04?", {"ch1", "0x04"}),
        ("What does error E03 mean in firmware 2.4.0?", {"e03", "2.4.0"}),
        ("What salary does EMP-7001 earn?", {"7001"}),
        ("How many vacation days do I get?", set()),
    ],
)
def test_numbers_and_codes(question, terms):
    assert key_terms(question) == terms


# ---------------------------------------------------------------- the cache itself
class Embedder:
    """Fixed vectors per question, so every similarity in these tests is chosen on purpose."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = {normalize(k): v for k, v in vectors.items()}
        self.calls = 0

    def __call__(self, text: str) -> list[float]:
        self.calls += 1
        return self.vectors[text]


def _store(cache, question, answer, embed, user="employee_demo", scopes=("INTERNAL",)):
    key = AnswerCache.key(question, 4, list(scopes))
    cache.put(key, {"answer": answer}, user=user, question=question, vector=embed(normalize(question)))


def _lookup(cache, question, embed, user="employee_demo", scopes=("INTERNAL",), threshold=0.98):
    key = AnswerCache.key(question, 4, list(scopes))
    return cache.lookup(key, user=user, question=question, embed=embed, threshold=threshold)


FULL_TIME = "What is the leave policy of the full-time employee?"
REORDERED = "What is the full time employee leave policy?"
PARAPHRASE = "How much paid leave does a full-time employee get?"
PART_TIME = "What is the leave policy of the part-time employee?"


@pytest.fixture
def embed():
    return Embedder(
        {
            FULL_TIME: [1.0, 0.0, 0.0],
            REORDERED: [0.95, 0.312, 0.0],  # similarity 0.95: same words, other order
            PARAPHRASE: [0.99, 0.141, 0.0],  # 0.99: other words, same meaning
            PART_TIME: [0.975, 0.2223, 0.0],  # 0.975: one word changes the answer
            "Can HR see employee data?": [0.0, 1.0, 0.0],
            "Can employees see HR data?": [0.0, 1.0, 0.0],  # identical embedding, reversed meaning
            "Vacation days in 2025?": [0.0, 0.0, 1.0],
            "Vacation days in 2026?": [0.0, 0.01, 1.0],  # 0.99995, but another year
        }
    )


def test_same_words_in_another_order_is_a_hit(embed):
    cache = AnswerCache()
    _store(cache, FULL_TIME, "30 days", embed)
    found = _lookup(cache, REORDERED, embed)
    assert found.value == {"answer": "30 days"}
    assert found.similar_to == FULL_TIME and found.similarity == pytest.approx(0.95, abs=0.01)


def test_other_words_need_almost_the_same_meaning(embed):
    cache = AnswerCache()
    _store(cache, FULL_TIME, "30 days", embed)
    assert _lookup(cache, PARAPHRASE, embed).value == {"answer": "30 days"}  # 0.99 >= 0.98
    assert _lookup(cache, PART_TIME, embed).value is None  # 0.975: part-time is not full-time


def test_same_words_with_swapped_roles_never_match(embed):
    cache = AnswerCache()
    _store(cache, "Can HR see employee data?", "Yes, for HR purposes", embed)
    # even at similarity 1.0: same words, but who sees whom changed
    assert _lookup(cache, "Can employees see HR data?", embed, threshold=0.5).value is None


def test_numbers_must_match_even_at_similarity_one(embed):
    cache = AnswerCache()
    _store(cache, "Vacation days in 2025?", "28 days", embed)
    assert _lookup(cache, "Vacation days in 2026?", embed, threshold=0.5).value is None


def test_only_your_own_earlier_questions(embed):
    cache = AnswerCache()
    _store(cache, FULL_TIME, "30 days", embed, user="employee_demo")
    assert _lookup(cache, REORDERED, embed, user="support_demo").value is None
    # the exact same question is still shared within the same scope, as before
    assert _lookup(cache, FULL_TIME, embed, user="support_demo").value == {"answer": "30 days"}


def test_only_the_same_access_scope(embed):
    cache = AnswerCache()
    _store(cache, FULL_TIME, "built from HR pages", embed, scopes=("CONFIDENTIAL:HR", "INTERNAL"))
    assert _lookup(cache, REORDERED, embed, scopes=("INTERNAL",)).value is None


def test_exact_hit_needs_no_embedding(embed):
    cache = AnswerCache()
    _store(cache, FULL_TIME, "30 days", embed)
    calls = embed.calls
    found = _lookup(cache, "  what is the leave policy of the full-time employee ", embed)
    assert found.value == {"answer": "30 days"} and found.similar_to is None
    assert embed.calls == calls


def test_expired_entries_never_match(embed):
    now = [0.0]
    cache = AnswerCache(ttl_s=60, clock=lambda: now[0])
    _store(cache, FULL_TIME, "30 days", embed)
    now[0] = 61
    assert _lookup(cache, REORDERED, embed).value is None


def test_a_failing_embedding_is_just_a_miss():
    cache = AnswerCache()

    def broken(text):
        raise RuntimeError("model not loaded")

    assert _lookup(cache, FULL_TIME, broken).value is None


def test_the_miss_returns_the_vector_so_it_is_computed_once(embed):
    found = _lookup(AnswerCache(), FULL_TIME, embed)
    assert found.value is None and found.vector == [1.0, 0.0, 0.0]


# ---------------------------------------------------------------- end to end (fake models)
def test_reworded_question_skips_retrieval_and_llm(indexed, fake_llm, monkeypatch):
    searches = []
    original = indexed.store.search
    monkeypatch.setattr(indexed.store, "search", lambda *a, **k: searches.append(a) or original(*a, **k))

    first = indexed.ask("How many vacation days per calendar year?")
    llm_calls, search_calls = len(fake_llm.calls), len(searches)
    second = indexed.ask("Per calendar year, how many vacation days?")

    assert not first.cached and second.cached
    assert second.cache_similar_to == "How many vacation days per calendar year?"
    assert second.cache_similarity is not None and second.cache_similarity >= 0.9
    assert second.steps == ["cache_hit"]
    assert len(fake_llm.calls) == llm_calls and len(searches) == search_calls
    assert second.answer == first.answer


def test_another_year_is_answered_fresh(indexed):
    indexed.ask("How many vacation days in 2025?")
    assert not indexed.ask("How many vacation days in 2026?").cached


def test_semantic_matching_can_be_switched_off(settings):
    service = make_service(settings.model_copy(update={"cache_semantic": False}))
    try:
        service.ingest([SAMPLE_DIR])
        service.ask("How many vacation days per calendar year?")
        assert not service.ask("Per calendar year, how many vacation days?").cached
    finally:
        service.close()


def test_defaults_are_conservative():
    s = Settings()
    assert s.cache_semantic is True and s.cache_similarity >= 0.95
