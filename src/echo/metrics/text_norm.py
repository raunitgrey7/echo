"""Text normalisation applied to both reference and hypothesis before scoring.

Why: WER should count recognition errors, not formatting choices. Case, punctuation, hyphenation,
thousands separators and the spelling of numbers are all folded identically on both sides: a
hypothesis "40 submissions" against a reference "forty submissions" is not a recognition error.
The number folding is a small deterministic state machine over number words (units, teens, tens,
hundred, thousand, ordinals), the same idea as whisper's English number normaliser but far
smaller. There is no spelling map, so a British spelling against an American one does count as
an error, and the corpus is written in American spelling to keep that fair.
"""

from __future__ import annotations

import re
import unicodedata

_APOSTROPHES = re.compile("[" + "".join(chr(c) for c in (0x2018, 0x2019, 0x60, 0xB4)) + "]")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_NON_ALNUM = re.compile(r"[^a-z0-9' ]+")
_LOOSE_QUOTE = re.compile(r"(?<![a-z])'|'(?![a-z])")

UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "thirtieth": 30,
    "fortieth": 40,
    "fiftieth": 50,
    "sixtieth": 60,
    "seventieth": 70,
    "eightieth": 80,
    "ninetieth": 90,
    "hundredth": 100,
    "thousandth": 1000,
}
SCALES = {"hundred": 100, "thousand": 1000}


def ordinal_suffix(value: int) -> str:
    """English ordinal suffix for a number: 1st, 2nd, 3rd, 4th, 11th, 21st."""
    if 10 <= value % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")


class _NumberRun:
    """Accumulates one spoken number such as ``one hundred and thirty five``."""

    def __init__(self) -> None:
        self.total = 0
        self.current = 0
        self.active = False

    def feed(self, word: str) -> bool:
        """Consume a number word; returns False when the word is not part of the number."""
        if word in UNITS:
            if self.active and self.current and self.current % 10 == 0 and UNITS[word] < 10:
                self.current += UNITS[word]
            elif self.active and self.current and self.current < 100 and self.current % 10 != 0 and UNITS[word] < 10:
                return False
            else:
                self.current += UNITS[word]
        elif word in TENS:
            if self.active and self.current and self.current % 100 != 0:
                return False
            self.current += TENS[word]
        elif word in SCALES:
            if not self.active:
                return False
            self.current = max(self.current, 1) * SCALES[word]
            if word == "thousand":
                self.total += self.current
                self.current = 0
        else:
            return False
        self.active = True
        return True

    def flush(self) -> list[str]:
        """Emit the digits and reset."""
        if not self.active:
            return []
        value = self.total + self.current
        self.__init__()
        return [str(value)]


def fold_numbers(words: list[str]) -> list[str]:
    """Replace runs of number words with digits; ordinals get their suffix; ``and`` is kept inside runs."""
    out: list[str] = []
    run = _NumberRun()
    pending_and = False
    for word in words:
        if word == "and" and run.active and (run.total + run.current) >= 100 and run.current % 100 == 0:
            pending_and = True
            continue
        if word in ORDINALS:
            value = run.total + run.current + ORDINALS[word] if run.active else ORDINALS[word]
            run.__init__()
            pending_and = False
            out.append(f"{value}{ordinal_suffix(value)}")
            continue
        if run.feed(word):
            pending_and = False
            continue
        out.extend(run.flush())
        if pending_and:
            out.append("and")
            pending_and = False
        out.append(word)
    out.extend(run.flush())
    return out


def normalize_text(text: str) -> str:
    """Lowercase, fold unicode, drop punctuation, keep intra-word apostrophes, fold numbers."""
    folded = unicodedata.normalize("NFKC", text).lower()
    folded = _APOSTROPHES.sub("'", folded)
    folded = folded.replace("%", " percent ").replace("&", " and ")
    folded = _THOUSANDS.sub("", folded)
    folded = folded.replace("-", " ").replace("/", " ")
    folded = _NON_ALNUM.sub(" ", folded)
    folded = _LOOSE_QUOTE.sub(" ", folded)
    return " ".join(fold_numbers(folded.split()))


def tokens(text: str) -> list[str]:
    """Normalised word list."""
    normalized = normalize_text(text)
    return normalized.split() if normalized else []
