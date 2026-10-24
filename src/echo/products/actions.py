"""Rule-based action item detection with owner and deadline extraction.

Why: in a meeting the sentences that matter afterwards are commitments and requests. They are
marked by a small set of surface cues: a person or pronoun followed by ``will``, modal obligation
(``need to``, ``must``, ``should``), polite requests (``please``, ``can you``) and a deadline phrase.
A weighted sum of cues against a threshold is transparent, deterministic and easy to audit; the
weights were chosen by hand and the detector is scored against labelled turns in the benchmark.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from echo.products.notes import Sentence, split_sentences
from echo.products.transcript import Transcript

DEADLINE = re.compile(
    r"\b(by|before|until|due|on|at)\s+"
    r"((the\s+)?(end\s+of\s+(the\s+)?(day|week|month|sprint)|next\s+(week|month|monday|tuesday|wednesday|thursday|"
    r"friday|review|meeting)|monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today|tonight|"
    r"noon|the\s+next\s+\w+|the\s+release|the\s+deadline|\d{1,2}(st|nd|rd|th)?\s+(of\s+)?\w+))\b",
    re.IGNORECASE,
)
RELATIVE_DUE = re.compile(
    r"\b(this\s+week|this\s+afternoon|this\s+morning|tomorrow\s+morning|tomorrow|today)\b", re.IGNORECASE
)
NAME = r"[A-Z][a-z]+"
CUES: tuple[tuple[str, float, re.Pattern[str]], ...] = (
    ("will", 2.0, re.compile(rf"\b(I|we|you|he|she|they|{NAME})\s+(will|'ll)\b")),
    ("going_to", 1.5, re.compile(r"\b(I|we|you|he|she|they)\s+(am|are|is)\s+going\s+to\b", re.IGNORECASE)),
    ("need_to", 1.5, re.compile(r"\b(need|needs)\s+to\b", re.IGNORECASE)),
    ("must", 1.5, re.compile(r"\b(must|has\s+to|have\s+to)\b", re.IGNORECASE)),
    ("we_should", 1.5, re.compile(r"\b(we|you)\s+should\b", re.IGNORECASE)),
    ("should", 0.5, re.compile(r"\bshould\b", re.IGNORECASE)),
    ("please", 1.5, re.compile(r"\bplease\b", re.IGNORECASE)),
    ("request", 1.5, re.compile(r"\b(can|could|would)\s+you\b", re.IGNORECASE)),
    ("make_sure", 1.5, re.compile(r"\b(make\s+sure|remember\s+to|don't\s+forget)\b", re.IGNORECASE)),
    ("action_item", 3.0, re.compile(r"\b(action\s+item|to\s*do|follow\s+up)\b", re.IGNORECASE)),
    ("let_us", 1.0, re.compile(r"\blet's\s+(do|add|move|schedule|write|fix|ship)\b", re.IGNORECASE)),
    ("deadline", 1.0, DEADLINE),
    (
        "task_verb",
        0.5,
        re.compile(
            r"\b(send|share|schedule|draft|review|update|rewrite|record|raise|confirm|post|rerun|fix|remove|grant|"
            r"add|write|book|prepare|set\s+up|talk\s+to|ask|check|release|ship|deploy|merge|publish|validate)\b",
            re.IGNORECASE,
        ),
    ),
)
NEGATIVE: tuple[tuple[str, float, re.Pattern[str]], ...] = (
    ("past", 1.0, re.compile(r"\b(was|were|did|had|fired|flagged|reported|cost)\b", re.IGNORECASE)),
    ("thanks", 2.0, re.compile(r"\b(thanks|thank\s+you|welcome|covers\s+the\s+agenda|wrap\s+up)\b", re.IGNORECASE)),
)
THRESHOLD = 2.0
INTERJECTIONS = frozenset(
    (
        "yes no okay ok sure great thanks agreed right well noted also so and but good fine alright anyway "
        "actually basically honestly next last first second finally"
    ).split()
)


@dataclass(frozen=True, slots=True)
class ActionItem:
    """One detected commitment or request."""

    text: str
    start: float
    end: float
    speaker: str
    owner: str
    due: str | None
    score: float
    cues: tuple[str, ...] = field(default_factory=tuple)


def score_sentence(text: str) -> tuple[float, tuple[str, ...]]:
    """Weighted cue sum minus negative cues, with the cue names that fired."""
    total = 0.0
    fired: list[str] = []
    for name, weight, pattern in CUES:
        if pattern.search(text):
            total += weight
            fired.append(name)
    for name, weight, pattern in NEGATIVE:
        if pattern.search(text):
            total -= weight
            fired.append(f"-{name}")
    return total, tuple(fired)


def extract_owner(text: str, speaker: str) -> str:
    """Owner from a vocative name, a named subject of ``will``, or the speaker for first-person commitments."""
    vocative = re.match(rf"^\s*({NAME}),", text)
    if vocative and vocative.group(1).lower() not in INTERJECTIONS:
        return vocative.group(1)
    named = re.search(rf"\b({NAME})\s+(will|must|should|can|has\s+to|needs\s+to|is\s+going\s+to)\b", text)
    if named and named.group(1).lower() not in {"i", "we", "the", "this", "that", "next", "please"}:
        return named.group(1)
    if re.search(r"\bI\s+(will|'ll|am\s+going\s+to|must|need\s+to|should)\b", text):
        return speaker
    if re.search(r"\b(we|let's)\b", text, re.IGNORECASE):
        return "team"
    if re.search(r"\b(can|could|would)\s+you\b|\bplease\b", text, re.IGNORECASE):
        return "unassigned"
    return speaker


def extract_due(text: str) -> str | None:
    """The deadline phrase if any."""
    match = DEADLINE.search(text)
    if match:
        return match.group(0)
    relative = RELATIVE_DUE.search(text)
    return relative.group(0) if relative else None


def detect_actions(transcript: Transcript, threshold: float = THRESHOLD) -> list[ActionItem]:
    """Action items in time order."""
    return [item for item in (score_action(s, threshold) for s in split_sentences(transcript)) if item]


def score_action(sentence: Sentence, threshold: float = THRESHOLD) -> ActionItem | None:
    """Build an action item when the sentence clears the threshold."""
    score, cues = score_sentence(sentence.text)
    if score < threshold:
        return None
    return ActionItem(
        sentence.text,
        sentence.start,
        sentence.end,
        sentence.speaker,
        extract_owner(sentence.text, sentence.speaker),
        extract_due(sentence.text),
        score,
        cues,
    )
