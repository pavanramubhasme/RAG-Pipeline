"""Calibrate the same-meaning answer cache with the real embedding model: measure, don't guess.

    uv run python -m eval.cache_calibration                    # current RAG_CACHE_SIMILARITY
    uv run python -m eval.cache_calibration --threshold 0.95   # try another value
    uv run python -m eval.cache_calibration --fake-models      # offline smoke test

Two kinds of question pairs go through the real cache code (ragapp/cache.py):
* "same": the same question in other words: a hit saves a search and an LLM call;
* "different": close wording, different answer (full-time / part-time, 2025 / 2026,
  minimum / maximum): a hit here would be a confident wrong answer.

A good threshold hits as many "same" pairs as possible and no "different" pair. The report
shows each similarity, so you can see how much room there is. Exit code 1 if any "different"
pair would be served from the cache.
"""

import argparse
import math
import sys
from pathlib import Path

from ragapp.cache import (
    SAME_WORDS_SIMILARITY,
    AnswerCache,
    content_words,
    key_terms,
    normalize,
    word_relation,
)
from ragapp.config import get_settings

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"

SAME = [
    ("What is the leave policy of the full-time employee?", "What is the full time employee leave policy?"),
    ("How many vacation days do I get?", "How many days of vacation do I get?"),
    ("How many vacation days do I get?", "How many days of holiday do I get per year?"),
    ("What is the minimum password length?", "How long must a password be at least?"),
    ("What does error code E03 mean?", "What is the meaning of error E03?"),
    (
        "Which version added dynamic load balancing?",
        "In which firmware version was dynamic load balancing added?",
    ),
]
DIFFERENT = [
    (
        "What is the leave policy of the full-time employee?",
        "What is the leave policy of the part-time employee?",
    ),
    ("How many vacation days do I get in 2025?", "How many vacation days do I get in 2026?"),
    ("How do I mute channel 1?", "How do I mute channel 2?"),
    ("Can I carry over unused vacation days?", "Can I cash out unused vacation days?"),
    ("What is the leave policy?", "What is the sick leave policy?"),
    ("What is the minimum password length?", "What is the maximum password length?"),
    ("Who approves an access request?", "Who submits an access request?"),
    ("Is remote work allowed on Fridays?", "Is remote work allowed on Mondays?"),
    ("Can HR see employee data?", "Can employees see HR data?"),  # same words, roles swapped
    ("Who approves leave for managers?", "Who approves managers for leave?"),  # same words, roles swapped
]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def would_hit(first: str, second: str, embed, threshold: float) -> bool:
    """Ask ``first``, then ``second`` as the same user: is ``second`` served from the cache?"""
    cache = AnswerCache()
    key = AnswerCache.key(first, 4, ["INTERNAL"])
    cache.put(key, {"answer": "x"}, user="u", question=first, vector=embed(normalize(first)))
    found = cache.lookup(
        AnswerCache.key(second, 4, ["INTERNAL"]), user="u", question=second, embed=embed, threshold=threshold
    )
    return found.value is not None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--threshold", type=float, default=None, help="default: RAG_CACHE_SIMILARITY")
    parser.add_argument("--fake-models", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    threshold = args.threshold if args.threshold is not None else settings.cache_similarity
    if args.fake_models:
        sys.path.insert(0, str(ROOT))
        from tests.fakes import FakeEmbeddings

        embed = FakeEmbeddings().embed_query
    else:
        from ragapp.factory import build_dense

        embed = build_dense(settings).embed_query

    rows, wrong_hits, same_hits, closest_wrong = [], 0, 0, 0.0
    for kind, pairs in (("same", SAME), ("different", DIFFERENT)):
        for first, second in pairs:
            similarity = _cosine(embed(normalize(first)), embed(normalize(second)))
            hit = would_hit(first, second, embed, threshold)
            relation = word_relation(content_words(first), content_words(second))
            same_terms = key_terms(first) == key_terms(second)
            if kind == "same":
                same_hits += hit
            else:
                wrong_hits += hit
                if same_terms and relation != "reordered":  # only the similarity stands in the way
                    closest_wrong = max(closest_wrong, similarity)
            verdict = ("ok" if hit else "miss (safe)") if kind == "same" else ("WRONG HIT" if hit else "ok")
            checks = (relation, "yes" if same_terms else "no")
            rows.append(
                (kind, f"{similarity:.3f}", *checks, "hit" if hit else "-", verdict, f"{first} / {second}")
            )

    header = ("pair", "similarity", "words", "same numbers", "cache", "verdict", "questions")
    lines = [
        f"Threshold {threshold} (same content words: {SAME_WORDS_SIMILARITY})\n",
        "| " + " | ".join(header) + " |",
    ]
    lines += ["|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]
    lines.append(
        f"\nSame-meaning pairs served from the cache: {same_hits}/{len(SAME)}. "
        f"Wrong hits: {wrong_hits}/{len(DIFFERENT)}. "
        f"Most similar different pair with the same numbers: {closest_wrong:.3f}. "
        "Keep RAG_CACHE_SIMILARITY above it, with some margin."
    )
    report = "\n".join(lines)
    print(report)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "cache_calibration.md").write_text(report + "\n", encoding="utf-8")
    print(f"\nReport written to {RESULTS / 'cache_calibration.md'}")
    return 1 if wrong_hits else 0


if __name__ == "__main__":
    sys.exit(main())
