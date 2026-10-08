"""How transcripts write fillers and phrase corrections, to compare generated inputs with real dictation.

Four real Parakeet dictations were measured this way: fillers per 100 words, the share with a
comma after them, the share written as a capitalized bare "Uh"/"Um", and fillers between a repeated
word. Corrections are counted by the stock cue phrases the generator leaned on, stacks of them, and
the re-decision forms real speakers used instead ("let's go with", "instead").
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_FILLER = re.compile(r"\b(um+|uh+|uhm|erm?|hmm+)\b([,.?!]?)", re.IGNORECASE)
_REPEAT_AROUND_FILLER = re.compile(r"\b(\w+),? (?:(?:um+|uh+|uhm|erm?|hmm+),? )+\1\b", re.IGNORECASE)
_STOCK_CUE = re.compile(
    r"\b(?:no wait|wait no|scratch that|i mean|sorry|make that|or rather|correction|no no)\b", re.IGNORECASE
)
_STACK = re.compile(
    r"\b(?:no wait|wait|scratch that|i mean|sorry|actually|no)\b[, ]+(?:\w+[, ]+){0,2}?"
    r"(?:no wait|wait|scratch that|i mean|sorry|actually)\b",
    re.IGNORECASE,
)
_REDECISION = re.compile(r"\blet's (?:just )?(?:go with|make (?:it|that)|do|get|pick)\b", re.IGNORECASE)
_INSTEAD = re.compile(r"\binstead\b", re.IGNORECASE)
_DOUBLED = re.compile(r"\b(\w+(?:'\w+)?) \1\b", re.IGNORECASE)


def profile(texts: Iterable[str]) -> dict:
    """Filler and correction rates over `texts`; shares are of all fillers."""
    words = fillers = comma_after = capitalized_bare = around_repeat = 0
    stock = stacked = redecision = instead = doubled = rows = 0
    for text in texts:
        rows += 1
        words += len(text.split())
        for match in _FILLER.finditer(text):
            fillers += 1
            comma_after += match.group(2) == ","
            capitalized_bare += match.group(1)[0].isupper() and match.group(2) == ""
        around_repeat += len(_REPEAT_AROUND_FILLER.findall(text))
        stock += len(_STOCK_CUE.findall(text))
        stacked += len(_STACK.findall(text))
        redecision += len(_REDECISION.findall(text))
        instead += len(_INSTEAD.findall(text))
        doubled += len(_DOUBLED.findall(text))

    def per_100(count: int) -> float:
        return round(100 * count / max(1, words), 2)

    def share(count: int) -> float:
        return round(100 * count / max(1, fillers), 1)

    return {
        "rows": rows,
        "words": words,
        "fillers_per_100": per_100(fillers),
        "comma_after_pct": share(comma_after),
        "capitalized_bare_pct": share(capitalized_bare),
        "filler_in_repeat_per_100": per_100(around_repeat),
        "doubled_word_per_100": per_100(doubled),
        "stock_cues_per_100": per_100(stock),
        "stacked_cues_per_100": per_100(stacked),
        "redecisions_per_100": per_100(redecision),
        "instead_per_100": per_100(instead),
    }


COLUMNS = (
    ("rows", "rows"),
    ("fillers /100w", "fillers_per_100"),
    ("comma after %", "comma_after_pct"),
    ("capitalized bare %", "capitalized_bare_pct"),
    ("the uh the /100w", "filler_in_repeat_per_100"),
    ("doubled /100w", "doubled_word_per_100"),
    ("stock cues /100w", "stock_cues_per_100"),
    ("stacked /100w", "stacked_cues_per_100"),
    ("let's go with /100w", "redecisions_per_100"),
    ("instead /100w", "instead_per_100"),
)


def table(profiles: dict[str, dict]) -> str:
    lines = ["| | " + " | ".join(name for name, _ in COLUMNS) + " |", "|---" * (len(COLUMNS) + 1) + "|"]
    for label, p in profiles.items():
        lines.append(f"| {label} | " + " | ".join(str(p[key]) for _, key in COLUMNS) + " |")
    return "\n".join(lines)
