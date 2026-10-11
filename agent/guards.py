"""Runtime safeguards (proposal section 8). Each one is demonstrable in code
and recorded in the turn trace as a ``guard_flags`` entry.

1. Prompt-injection screen on user input            check_user_input
2. Instruction stripping on retrieved (untrusted) text sanitize_evidence_text
3. Citation validation: every [n] must exist         validate_citations
4. Quotation limit: <= 25 consecutive source words   trim_long_quotes
5. High-stakes disclaimer / professional referral    disclaimer_for
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

INJECTION = re.compile(
    r"(ignore (?:all |any |the )?(?:previous|prior|above|earlier) (?:instructions|rules|prompts?)|"
    r"disregard (?:the |your )?(?:system|previous|above)|reveal (?:your |the )?(?:system )?prompt|"
    r"print (?:your |the )?(?:system )?prompt|you are now\b|developer mode|jailbreak|do anything now|"
    r"<\s*/?\s*system\s*>|BEGIN SYSTEM|new instructions:)",
    re.I,
)

HIGH_STAKES = re.compile(
    r"\b(accident|crashed|injur\w*|warning light|airbag light|check engine|smoke|smoking|fire|burning smell|"
    r"brakes? (?:failed|failing|not working|feel)|is it safe to drive|stalled|stalling|leak\w*|"
    r"won'?t stop|pulls? to|grinding)\b",
    re.I,
)

DISCLAIMER = (
    "AutoGuide AI gives general information from NHTSA sources. It does not replace your "
    "owner's manual or a qualified technician. To check open recalls for your exact vehicle, "
    "enter your VIN at https://www.nhtsa.gov/recalls."
)

WORD = re.compile(r"[a-z0-9]+")


@dataclass
class GuardResult:
    blocked: bool = False
    reason: str | None = None
    flags: list[str] = field(default_factory=list)


def check_user_input(text: str) -> GuardResult:
    if INJECTION.search(text):
        return GuardResult(blocked=True, reason="prompt_injection", flags=["input_injection_blocked"])
    return GuardResult()


def sanitize_evidence_text(text: str) -> tuple[str, bool]:
    """Drop instruction-like lines from retrieved documents."""
    removed = False
    kept = []
    for line in text.splitlines():
        if INJECTION.search(line):
            kept.append("[removed: instruction-like text in source]")
            removed = True
        else:
            kept.append(line)
    return "\n".join(kept), removed


CITATION = re.compile(r"\[(\d+)\]")


def validate_citations(answer: str, n_evidence: int) -> tuple[bool, set[int], set[int]]:
    cited = {int(n) for n in CITATION.findall(answer)}
    invalid = {n for n in cited if n < 1 or n > n_evidence}
    return bool(cited) and not invalid, cited, invalid


def _words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def longest_shared_run(answer: str, sources: list[str]) -> int:
    answer_words = _words(answer)
    source_strings = [" " + " ".join(_words(s)) + " " for s in sources]
    best = 0
    for start in range(len(answer_words)):
        length = best + 1
        while start + length <= len(answer_words):
            phrase = " " + " ".join(answer_words[start:start + length]) + " "
            if any(phrase in s for s in source_strings):
                best = length
                length += 1
            else:
                break
    return best


def trim_long_quotes(answer: str, sources: list[str], max_words: int = 25) -> tuple[str, bool]:
    """If the answer copies > max_words consecutive source words, cut the run.

    Works on whitespace tokens of the answer so formatting and citations
    survive. Returns (answer, trimmed?).
    """
    if longest_shared_run(answer, sources) <= max_words:
        return answer, False
    source_strings = [" " + " ".join(_words(s)) + " " for s in sources]
    tokens = answer.split()
    out: list[str] = []
    run: list[str] = []

    def in_sources(seq: list[str]) -> bool:
        phrase = " " + " ".join(w for t in seq for w in _words(t)) + " "
        return phrase.strip() != "" and any(phrase in s for s in source_strings)

    for token in tokens:
        if in_sources(run + [token]):
            run.append(token)
            if len(run) == max_words + 1:
                out.append("...")
            if len(run) > max_words:
                if CITATION.search(token):
                    out.append(CITATION.search(token).group(0))
                continue
            out.append(token)
        else:
            run = [token] if in_sources([token]) else []
            out.append(token)
    return " ".join(out), True


def disclaimer_for(question: str, intent: str | None) -> str | None:
    if HIGH_STAKES.search(question) or intent == "recall_lookup":
        return DISCLAIMER
    return None
