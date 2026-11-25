"""When does a message need the conversation to be understood?

Bug this fixes (seen in a live test): after "What's the WFH quota?" the user asked
"Python - parse a CSV safely?". The LLM merged both into one question, retrieval found
the remote-work policy, and the agent answered the WFH question again. Faithful to its
source, grounded = yes, and still wrong: it answered a different question.

Two deterministic checks, no extra LLM call:

* ``is_follow_up``: only messages that point back ("those registers", "and for ch2?",
  "is that allowed?", "can you get me the values in hex?") are rewritten with the
  conversation. Every other message is searched exactly as typed, which also saves one LLM
  call per question.
* ``keeps_user_words``: the rewrite must keep the user's own words. If the LLM replaced
  the question with an earlier one, the message is used as typed.

The detector is deliberately conservative: a missed follow-up costs an "I don't know"
(safe), a false one can give a confident answer to the wrong question (unsafe).
"""

import re

_TOKEN = re.compile(r"[a-z0-9_]+")


def _words(text: str) -> frozenset[str]:
    return frozenset(text.split())


# Words that point back to something said earlier.
STRONG_REFERENCES = _words(
    "they them their theirs those these its same above former latter aforementioned previous"
)
# Ambiguous: "What does that mean?" (follow-up) vs "My wallbox shows E03. What does that mean?"
# (self-contained). They only count in short messages.
WEAK_REFERENCES = _words("it this that he she him her his hers one ones such")
OPENERS = _words("and also but so or plus then")
# Asking to reshape the last answer ("Can you get me the values in hex?", "as a table?"): a short
# message with one of these words is about what was just said. Seen live: the hex question was
# searched on its own and rewritten into "How do I convert decimal values to hexadecimal?".
TRANSFORMS = _words(
    "hex hexadecimal binary decimal convert table summarize summarise summary shorter shorten "
    "simpler simplify translate bullet bullets"
)
TRANSFORM_MAX_CONTENT_WORDS = 3
TWO_WORD_OPENERS = frozenset(["what about", "how about", "what else", "and what", "same for"])

# Words without a topic of their own; everything else counts as a content word.
_FILLER = _words(
    """
    a an the is are was were be been am do does did of to in on for and or but so with by at from as
    into about i me my we us our you your what which who whom whose how when where why can could
    should would will may might must there here s t please tell give show explain list describe know
    want need like get more detail details example examples elaborate exactly also then else just only
    very much many long ok okay yes no
    """
)
WEAK_MAX_CONTENT_WORDS = 3


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if len(t) > 1 or t.isdigit()]


def content_words(text: str) -> list[str]:
    references = STRONG_REFERENCES | WEAK_REFERENCES
    return [t for t in _tokens(text) if t not in _FILLER and t not in references]


def is_follow_up(message: str) -> bool:
    """True if the message probably cannot be understood without the conversation."""
    words = _tokens(message)
    if not words:
        return True
    if words[0] in OPENERS or " ".join(words[:2]) in TWO_WORD_OPENERS:
        return True
    if any(w in STRONG_REFERENCES for w in words):
        return True
    topic = content_words(message)
    if len(topic) <= 1:  # "Why?", "Is that allowed?", "Encoding?"
        return True
    if any(w in TRANSFORMS for w in words) and len(topic) <= TRANSFORM_MAX_CONTENT_WORDS:
        return True  # "the values in hex", "as a table, please"
    return any(w in WEAK_REFERENCES for w in words) and len(topic) <= WEAK_MAX_CONTENT_WORDS


def _same_word(a: str, b: str) -> bool:
    # Lenient on purpose ("address" ~ "addresses", "write" ~ "writing"): the check only has to
    # catch a rewrite that dropped the user's question, not judge its grammar.
    return a == b or (len(a) >= 4 and len(b) >= 4 and a[:4] == b[:4])


def keeps_user_words(message: str, standalone: str, min_share: float = 0.5) -> bool:
    """True if the rewritten question still contains most of the user's own content words."""
    mine = content_words(message)
    if not mine:  # "Why?" has nothing to keep
        return True
    theirs = set(content_words(standalone))
    kept = sum(1 for word in mine if any(_same_word(word, other) for other in theirs))
    return kept / len(mine) >= min_share