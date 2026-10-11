"""Transcript and cleanup rules shared by dataset stages and evaluation."""

from __future__ import annotations

import re

from .datagen.schema import LIST_ITEM

FILLER_FORMS = r"(?:um+|uh+m?|erm?|hmm+|mm+|mhm|ah+)"
FILLER = re.compile(rf"\b{FILLER_FORMS}\b", re.IGNORECASE)
FILLER_TOKEN = re.compile(rf"^({FILLER_FORMS})([,.?!;:]?)$", re.IGNORECASE)
# References use mm and ah as measurement units; uppercase forms may be acronyms.
REFERENCE_FILLER_EXCLUSIONS = {"mm", "ah"}

# Discourse phrases may carry meaning; only evaluation counts excess occurrences as leftovers.
DISCOURSE_FILLERS = ("you know", "i mean", "basically", "like")
DISCOURSE_FILLER_PATTERNS = {
    phrase: re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE) for phrase in DISCOURSE_FILLERS
}

CORRECTION = re.compile(
    r"\b(?:no wait|wait no|scratch|actually|sorry|i mean|make (?:it|that)|never ?mind|or rather|"
    r"no no|rather|correction|let me rephrase|no|wait)\b",
    re.IGNORECASE,
)
# A possible correction excuses a missing protected token pending the completeness judge.
POSSIBLE_CORRECTION = re.compile(
    rf"{CORRECTION.pattern}|\b(?:nope|change|instead|except|go with|switch|forget|skip|drop|remove|"
    r"cancel|take (?:off|out)|don't need|already have|not|let me|hold on|or|i guess|i meant|"
    r"whoops|oops|my bad|er|really|say|maybe|probably)\b",
    re.IGNORECASE,
)
# Only sentence-opening, retained correction instructions count as cleanup style violations.
SENTENCE_CORRECTION = re.compile(
    r"(?:(?:oh|no),?\s+)?wait(?=\s*(?:[,.!?…—–-]|$))|okay,?\s+so\b|scratch that\b|oh wait\b",
    re.IGNORECASE,
)

MARKDOWN = re.compile(
    r"(?<![\w*])\*\*(?=\S).+?(?<=\S)\*\*(?![\w*])"
    r"|(?<![\w_])__(?![A-Za-z_]\w*__)(?=\S)[^_\n]+?(?<=\S)__(?![\w_.])"
    r"|`|^\s*#{1,6}\s|\[[^\]]+\]\([^)]+\)|^\s*\|.*\|\s*$",
    re.MULTILINE,
)
ASSISTANT_OPENING = re.compile(
    r"^(?:sure|certainly|of course|absolutely|here is|here's|here are|i can|i'd be happy|as an ai|"
    r"i'm sorry|i am sorry|i cannot|i can't help|great question)\b",
    re.IGNORECASE,
)
# Keep this evaluation rule fixed so scores compare across runs.
EVAL_ASSISTANT_OPENING = re.compile(
    r"^(?:sure|certainly|of course|absolutely|here(?:'s| is| are)|i can(?:'t| not|not)|i'm sorry|i am sorry|as an ai)\b",
    re.IGNORECASE,
)
_SPOKEN_OPENERS = re.compile(
    rf"^(?:(?:{FILLER_FORMS}|so|yeah|oh|okay|ok|alright|well|hey|like|and)\b[\s,.]*)+",
    re.IGNORECASE,
)
_OTHER_BULLET = re.compile(r"^\s*(?:[*•+·]\s|\d+[)\]]\s|[a-z][.)]\s)", re.MULTILINE)
_NUMBERED = re.compile(r"^\s*(\d+)\.\s")

# Paths, files, emails, URLs and flags keep their case; a word-like token may be recased.
CASE_SENSITIVE = re.compile(r"[/@_\\]|\.[A-Za-z]{2,}|--")
CODE_TOKEN = re.compile(rf"{CASE_SENSITIVE.pattern}|::|[a-z][A-Z]|[A-Za-z]\d|\d[A-Za-z]")
PROTECTED = re.compile(
    r"https?://\S+|www\.\S+"
    r"|(?<!\w)[\w.:-]*\\\S+"
    r"|\b[A-Za-z_]\w*(?:::\w+)+(?:\(\))?"
    r"|[\w.+-]+@[\w-]+\.[\w.-]+"
    r"|(?<![\w/])(?:~|\.{1,2})?/[\w.\-/~]+"
    r"|\b[\w.-]+(?:/[\w.-]+)+/?"
    r"|(?<![\w-])--[a-zA-Z][\w-]*(?:=\S+)?|(?<![\w-])-[a-zA-Z]\b(?:=\S+)?"
    r"|\b[A-Za-z_]\w*(?:[._:]\w+)+(?:\(\))?"
    r"|\b[A-Za-z_]\w*_\w*\b"
    r"|\b[A-Za-z]*[a-z][A-Z]\w*\b"
    r"|\b[A-Za-z]*\d[\w.,:/%$-]*"
)
TRAILING_PUNCTUATION = ".,;:!?)]}'\"—–"


def protected_tokens(text: str) -> list[str]:
    return [match.group(0).rstrip(TRAILING_PUNCTUATION) for match in PROTECTED.finditer(text)]


def has_markdown(text: str) -> bool:
    return bool(MARKDOWN.search(text))


def assistant_opener(text: str, spoken: str) -> bool:
    spoken = _SPOKEN_OPENERS.sub("", spoken.strip().lstrip(",. "))
    return bool(ASSISTANT_OPENING.match(text.strip())) and not ASSISTANT_OPENING.match(spoken)


def malformed_list(text: str) -> str | None:
    if _OTHER_BULLET.search(text):
        return "bullet other than '- '"
    expected = 1
    separated = False
    for line in text.split("\n"):
        if LIST_ITEM.match(line) is None and re.match(r"^\s*(?:-|\d+\.)\s*$", line):
            return "empty list item"
        number = _NUMBERED.match(line)
        if number:
            value = int(number.group(1))
            if value != expected and not (separated and value == 1):
                return "numbering skips"
            expected = value + 1
            separated = False
        elif not LIST_ITEM.match(line):
            separated = True
    return None
