"""Guardrails: layered checks before and after the RAG workflow, plus at ingestion.

Layer 1, rules (always on, instant, free, deterministic):
  * input:  prompt-injection / jailbreak phrases are blocked before any LLM call;
            secrets and personal data (e-mail, international phone, IBAN, card numbers)
            in the question and history are masked, so they never reach the LLM, logs or
            traces. Claims of identity ("As hr_demo, ...") are noted and ignored: the role
            comes from the credential (ragapp/access.py).
  * output: secrets, card numbers and IBANs are masked in answers.
  * ingestion: document chunks that contain injection phrases are quarantined (defends
            against *indirect* prompt injection through uploaded files); secrets are masked
            in every chunk, contact data in CONFIDENTIAL/RESTRICTED chunks.

Layer 2, an open guard model (optional, off by default because it costs an LLM call):
  IBM Granite Guardian (Apache-2.0) via Ollama checks the question for each risk in
  ``guard_risks`` ("jailbreak", "harm", ...). It answers yes/no.

Rules catch the obvious cheaply; the guard model catches paraphrases the rules miss.
Neither is perfect, which is why the RAG prompt also treats context as data and the
grounding check verifies answers.
"""

import logging
import re
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ragapp import observability as obs
from ragapp.config import Settings

logger = logging.getLogger(__name__)

BLOCKED_MESSAGE = (
    "I can't help with that request. I answer questions about the documents in this knowledge base."
)

# Phrases typical of prompt-injection and jailbreak attempts (case-insensitive). Kept specific
# on purpose: "ignore the E04 error and follow the instructions" must NOT match.
_INJECTION_RULES = {
    "ignore_instructions": (
        r"\b(ignore|disregard|forget|override)\b.{0,20}"
        r"\b(previous|prior|above|earlier|all|any|your)\b.{0,20}"
        r"\b(instructions?|rules|prompts?|guidelines|directives)\b"
    ),
    "reveal_prompt": (
        r"\b(reveal|show|print|repeat|output|tell me)\b.{0,30}"
        r"\b(system|hidden|initial|original)\s+(prompt|instructions?|message)"
    ),
    "role_override": (
        r"\b(you are now|from now on you|act as|pretend (to be|you are))\b.{0,40}"
        r"\b(unfiltered|uncensored|jailbroken|dan|without (any )?(rules|restrictions|limits))\b"
    ),
    "jailbreak_mode": r"\b(dan|jailbreak|god)\s+mode\b",
    "no_restrictions": (
        r"\b(answer|respond|reply)\b.{0,30}\b(without|with no)\s+(any\s+)?"
        r"(rules|restrictions|limits|filters|censorship)\b"
    ),
    "fake_delimiter": r"</?\s*(system|context|instructions?)\s*>",
}
INJECTION_PATTERNS = {name: re.compile(rule, re.IGNORECASE) for name, rule in _INJECTION_RULES.items()}

# Credentials and secret-like markers. Specific shapes only, so policy prose such as
# "Passwords must be at least 14 characters" or a variable name like DB_PASSWORD is not touched.
SECRET_PATTERN = re.compile(
    r"\bsk-[A-Za-z0-9_-]{8,}"  # OpenAI-style API keys
    r"|(?i:\bbearer)\s+[A-Za-z0-9._~+/-]{8,}=*"  # bearer tokens
    r"|\b[a-z][a-z0-9+.-]*://[^\s:/@]+:[^\s@/]+@[^\s/]+"  # user:password@host in connection strings
    r"|\bAKIA[0-9A-Z]{16}\b"  # AWS access key IDs
    r"|\bgh[pousr]_[A-Za-z0-9]{20,}\b"  # GitHub tokens
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    # password=... / api_key: ... with a value that contains a digit (not "Password: must be ...")
    r"|(?i:\b(?:password|passwd|pwd|secret|api[_-]?key|token))\s*[:=]\s*[\"']?(?=[^\s\"']*\d)[^\s\"']{6,}"
    # upper-case markers such as DUMMY-PASSWORD-DO-NOT-USE or TEST_TOKEN_INVALID (case-sensitive)
    r"|\b[A-Z0-9]+(?:[-_][A-Z0-9]+)*[-_](?:PASSWORD|SECRET|TOKEN|APIKEY)(?:[-_][A-Z0-9]+)+\b"
)

# Secrets and personal data, applied in this order. Phone numbers only in international format
# (+49 30 1234 5678, 0049 ...), so register values, versions and measurements are not touched.
PII_PATTERNS: dict[str, re.Pattern] = {
    "secret": SECRET_PATTERN,
    "email": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}(?:[ ]?[A-Z0-9]{1,3})?\b"),
    "card": re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
    "phone": re.compile(
        r"(?<![\w+])(?:\+|00)\d{1,3}[ /-]?(?:\(\d{1,5}\)[ /-]?)?\d{2,5}(?:[ /-]?\d{2,5}){1,3}\b"
    ),
}
_SCORE = re.compile(r"<score>\s*(yes|no)\s*</score>", re.IGNORECASE)
# "As hr_demo, ...", "I am the CEO": text cannot change who the caller is. Noted, not blocked
# (it can be harmless), and the answer is still limited to the caller's real role.
IDENTITY_CLAIM = re.compile(
    r"\b(?:as|i am|i'm|i\u2019m)\s+(?:the\s+|an?\s+)?"
    r"(?:\w+_demo|ceo|cfo|admin(?:istrator)?|root|superuser)\b",
    re.IGNORECASE,
)


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_CITATION_MARK = re.compile(r"\[\d+\]")


def _number_forms(text: str) -> tuple[set[tuple[str, ...]], set[str]]:
    """Numbers as digit groups ("2.4.0" -> ("2", "4", "0"); "02" -> "2") and as joined digits."""
    groups, joined = set(), set()
    for raw in _NUMBER.findall(text):
        parts = tuple(str(int(p)) for p in re.split(r"[.,]", raw))
        groups.add(parts)
        joined.add(re.sub(r"\D", "", raw).lstrip("0") or "0")
    return groups, joined


def unsupported_numbers(answer: str, sources: list[str]) -> list[str]:
    """Numbers in the answer that appear in none of the sources (the context and the question).

    A deterministic hallucination check for the facts that matter most (amounts, dates, limits):
    a small LLM judge can wave "$1.2 billion" through, string matching can't. Lenient on format:
    "1,500" = "1.500", "02:00" = "2:00", "2.4" matches "2.4.0", "18,00,000" = "1800000".
    """
    known_groups, known_joined = set(), set()
    for source in sources:
        groups, joined = _number_forms(source)
        known_groups |= groups
        known_joined |= joined
    missing = []
    for raw in _NUMBER.findall(_CITATION_MARK.sub(" ", answer)):
        parts = tuple(str(int(p)) for p in re.split(r"[.,]", raw))
        digits = re.sub(r"\D", "", raw).lstrip("0") or "0"
        prefix_match = any(known[: len(parts)] == parts for known in known_groups)
        if not prefix_match and digits not in known_joined and raw not in missing:
            missing.append(raw)
    return missing


def find_injection(text: str) -> str | None:
    """Name of the first prompt-injection pattern found in ``text``, or None."""
    for name, pattern in INJECTION_PATTERNS.items():
        if pattern.search(text):
            return name
    return None


def _luhn_valid(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)][::-1]
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(digits))
    return 13 <= len(digits) <= 19 and total % 10 == 0


def redact(text: str, kinds: list[str]) -> tuple[str, list[str]]:
    """Mask personal data of the given kinds. Returns (masked_text, kinds_found)."""
    found: list[str] = []
    # Fixed order, most specific first: an IBAN contains digit groups a phone pattern would also match.
    for kind, pattern in PII_PATTERNS.items():
        if kind not in kinds:
            continue

        def mask(match: re.Match, kind: str = kind) -> str:
            if kind == "card" and not _luhn_valid(match.group(0)):
                return match.group(0)  # a long number, not a payment card
            if kind not in found:
                found.append(kind)
            return f"[{kind.upper()}]"

        text = pattern.sub(mask, text)
    return text, found


def parse_verdict(text: str) -> bool:
    """True if a guard model said "yes" (risk). Handles 'Yes'/'No' and '<score> yes </score>'."""
    match = _SCORE.search(text)
    word = match.group(1) if match else (text.strip().split() or [""])[0]
    return word.strip(".:,").lower() == "yes"


@dataclass
class InputCheck:
    question: str
    history: list[dict]
    blocked: bool = False
    findings: list[str] = field(default_factory=list)


class Guardrails:
    def __init__(self, settings: Settings, guard_llm: BaseChatModel | None = None):
        self.settings = settings
        self.guard_llm = guard_llm if settings.guard_model else None

    # ------------------------------------------------------------- input rail
    def check_input(self, question: str, history: list[dict] | None = None) -> InputCheck:
        s = self.settings
        history = [dict(turn) for turn in history or []]
        check = InputCheck(question=question, history=history)

        if s.block_prompt_injection:
            # The user's own earlier messages count too; assistant turns are our own output.
            user_text = " ".join([question] + [t["content"] for t in history if t["role"] == "user"])
            if rule := find_injection(user_text):
                return self._block(check, f"prompt_injection:{rule}")

        if s.access_control and IDENTITY_CLAIM.search(question):
            check.findings.append("identity_claim_ignored")
            obs.GUARDRAIL_EVENTS.labels(event="identity_claim_ignored").inc()

        if s.redact_input_pii:
            check.question, found = redact(question, s.redact_input_pii)
            for turn in check.history:
                turn["content"], more = redact(turn["content"], s.redact_input_pii)
                found += [k for k in more if k not in found]
            for kind in found:
                check.findings.append(f"redacted:{kind}")
                obs.GUARDRAIL_EVENTS.labels(event=f"redacted:{kind}").inc()

        if self.guard_llm is not None:
            for risk in s.guard_risks:
                verdict = self._ask_guard(risk, check.question)
                if verdict is None:  # guard model unavailable
                    check.findings.append("guard_error")
                    obs.GUARDRAIL_EVENTS.labels(event="guard_error").inc()
                    if s.guard_fail_closed:
                        return self._block(check, "guard_unavailable")
                    break
                if verdict:
                    return self._block(check, f"guard:{risk}")
        return check

    def _ask_guard(self, risk: str, text: str) -> bool | None:
        try:
            reply = self.guard_llm.invoke(
                [SystemMessage(content=risk), HumanMessage(content=text)],
                config={"run_name": f"guard_{risk}"},
            )
        except Exception as exc:
            logger.warning("guard_model_failed", extra={"risk": risk, "error": str(exc)[:200]})
            return None
        return parse_verdict(str(reply.content))

    @staticmethod
    def _block(check: InputCheck, reason: str) -> InputCheck:
        check.blocked = True
        check.findings.append(f"blocked:{reason}")
        obs.GUARDRAIL_EVENTS.labels(event=f"blocked:{reason.split(':')[0]}").inc()
        logger.warning("request_blocked", extra={"reason": reason})
        return check

    # ------------------------------------------------------------ output rail
    def check_output(self, answer: str) -> tuple[str, list[str]]:
        masked, found = redact(answer, self.settings.redact_output_pii)
        for kind in found:
            obs.GUARDRAIL_EVENTS.labels(event=f"redacted_output:{kind}").inc()
        return masked, [f"redacted_output:{kind}" for kind in found]