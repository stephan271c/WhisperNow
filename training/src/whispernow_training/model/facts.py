"""Where a reference carries its facts, found in code (todo 114).

Two uses:
- **Fact-weighted SFT** weights the loss on the reference tokens inside `fact_spans`: values
  (numbers, amounts, times, dates, code and other protected tokens), names, hedges, reasons,
  negations, and the first word of each sentence and list item, where a model that drops a whole unit
  makes its choice.
- **Code-made pairs** (`code_pairs`) break one of these: drop a unit, a reason or a hedge, or change
  a value.

Character offsets index the reference text, so the caller maps them onto tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .metrics import _PROTECTED, _TRAILING_PUNCTUATION

_WEEKDAYS = r"Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
_MONTHS = r"January|February|March|April|June|July|August|September|October|November|December"
# Units are case-sensitive, so "A" (amps) never takes the article after a number.
_NUMBER = re.compile(r"\$?\d[\d,.:]*(?:\s?(?:%|am|pm|a\.m\.|p\.m\.|k|K|ms|GB|MB|kWh|kW|A|V)\b)?")
_WORD_VALUE = re.compile(
    rf"\b(?:{_WEEKDAYS}|{_MONTHS})\b"
    r"|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|hundred|thousand|"
    r"half|quarter|dozen|first|second|third|fourth|fifth|noon|midnight|tomorrow|tonight|yesterday|"
    r"red|blue|green|yellow|black|white|purple|pink|grey|gray|brown|silver|orange)\b",
    re.IGNORECASE,
)
# Hedges the cleanup contract keeps; dropping one changes how sure the speaker sounds.
HEDGES = (
    "I think", "I guess", "I believe", "I suppose", "I'm not sure", "not sure", "maybe", "probably", "possibly",
    "perhaps", "might", "roughly", "about", "around", "approximately", "kind of", "sort of",
)
_HEDGE = re.compile(r"\b(?:" + "|".join(re.escape(h) for h in HEDGES) + r")\b", re.IGNORECASE)
# A dropped negation reverses the meaning: "no", "not", "never" and the like, and whole "n't" words.
_NEGATION = re.compile(
    r"\b(?:not|no|never|nothing|none|nobody|nowhere|neither|nor|without|cannot)\b|\b\w+n['\u2019]t\b",
    re.IGNORECASE,
)
# A reason runs from its connective to the end of the clause.
_REASON = re.compile(r"\b(?:because|since|so that|in case|otherwise)\b[^.;:!?\n]*", re.IGNORECASE)
_CAPITALIZED = re.compile(r"\b[A-Z][\w'&.-]*[A-Za-z0-9]\b|\b[A-Z]\b")
_LIST_ITEM = re.compile(r"^(\s*(?:-|\d+\.)\s+)(.*)$")
_SENTENCE = re.compile(r"[^.!?]+(?:[.!?]+[\"')\]]*|$)")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    kind: str  # value, name, hedge, negation, reason, unit_start

    def text(self, source: str) -> str:
        return source[self.start : self.end]


@dataclass(frozen=True)
class Unit:
    """A sentence, or a list item's whole line, with the text before it on its line."""

    start: int
    end: int
    kind: str  # sentence, bullet, numbered
    line_start: int
    line_end: int
    first_in_line: bool
    last_in_line: bool


def units(text: str) -> list[Unit]:
    """Sentences of each line, and list items as one unit each."""
    found = []
    offset = 0
    for line in text.split("\n"):
        line_end = offset + len(line)
        if match := _LIST_ITEM.match(line):
            kind = "bullet" if match.group(1).strip() == "-" else "numbered"
            found.append(Unit(offset, line_end, kind, offset, line_end, True, True))
        elif line.strip():
            sentences = [m for m in _SENTENCE.finditer(line) if m.group(0).strip()]
            for index, match in enumerate(sentences):
                lead = len(match.group(0)) - len(match.group(0).lstrip())
                found.append(Unit(offset + match.start() + lead, offset + match.end(), "sentence", offset, line_end,
                                  index == 0, index == len(sentences) - 1))
        offset = line_end + 1
    return found


def _sentence_starts(text: str) -> set[int]:
    starts = set()
    for unit in units(text):
        position = unit.start
        if unit.kind != "sentence":
            position += len(_LIST_ITEM.match(text[unit.start : unit.end]).group(1))
        starts.add(position)
    return starts


def fact_spans(text: str) -> list[Span]:
    """Fact-bearing spans of a reference, sorted and possibly overlapping."""
    spans: list[Span] = []
    for match in _PROTECTED.finditer(text):
        end = match.start() + len(match.group(0).rstrip(_TRAILING_PUNCTUATION))
        if end > match.start():
            spans.append(Span(match.start(), end, "value"))
    for pattern in (_NUMBER, _WORD_VALUE):
        spans += [Span(m.start(), m.start() + len(m.group(0).rstrip(".,:")), "value") for m in pattern.finditer(text)]
    starts = _sentence_starts(text)
    for match in _CAPITALIZED.finditer(text):
        if match.start() not in starts and match.group(0) not in ("I", "I'm", "I'll", "I've", "I'd", "OK"):
            spans.append(Span(match.start(), match.end(), "name"))
    spans += [Span(m.start(), m.end(), "hedge") for m in _HEDGE.finditer(text)]
    spans += [Span(m.start(), m.end(), "negation") for m in _NEGATION.finditer(text)]
    spans += [Span(m.start(), m.start() + len(m.group(0).rstrip()), "reason") for m in _REASON.finditer(text)]
    for start in starts:
        match = re.compile(r"\S+").match(text, start)
        if match:
            spans.append(Span(match.start(), match.end(), "unit_start"))
    return sorted(set(spans), key=lambda s: (s.start, s.end, s.kind))


def char_mask(text: str, spans: list[Span] | None = None) -> list[bool]:
    """True for each character of `text` inside a fact span."""
    mask = [False] * len(text)
    for span in fact_spans(text) if spans is None else spans:
        for index in range(span.start, span.end):
            mask[index] = True
    return mask
