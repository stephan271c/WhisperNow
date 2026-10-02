"""Deterministic evaluation checks for one output against its row's reference.

These live only in evaluation; the app gets no output-policing heuristics. The LLM judge
(`prompts/judge/eval.md`) covers answered transcripts and dropped or invented facts.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from ..datagen.schema import EMPTY, LIST_ITEM, Example, layout_of

# Fillers counted as leftovers when the output has more of them than the reference.
FILLERS = ("um", "umm", "uh", "uhh", "uhm", "erm", "er", "hmm", "mhm", "you know", "i mean", "basically", "like")
_FILLER_PATTERNS = {filler: re.compile(rf"\b{re.escape(filler)}\b", re.IGNORECASE) for filler in FILLERS}

# Code, paths, URLs, emails, flags, identifiers and numbers: things that must come through exactly.
_PROTECTED = re.compile(
    r"https?://\S+|www\.\S+"
    r"|[\w.+-]+@[\w-]+\.[\w.-]+"
    r"|(?<![\w/])(?:~|\.{1,2})?/[\w.\-/~]+"
    r"|\b[\w.-]+(?:/[\w.-]+)+/?"
    r"|(?<![\w-])--[a-zA-Z][\w-]*|(?<![\w-])-[a-zA-Z]\b"
    r"|\b[A-Za-z_]\w*(?:[._:]\w+)+(?:\(\))?"
    r"|\b[a-z]+[A-Z]\w*\b"
    r"|\b[A-Za-z]*\d[\w.,:/%$-]*"
)
_TRAILING_PUNCTUATION = ".,;:!?)]}'\""
_BAD_LIST_MARKER = re.compile(r"^\s*(?:[*•+·]\s|\d+[)\]]\s|[a-z][.)]\s)")
_MARKDOWN = re.compile(r"^\s*#{1,6}\s|\*\*|__\w|```|`[^`\n]+`", re.MULTILINE)
_ASSISTANT_OPENERS = re.compile(
    r"^(?:sure|certainly|of course|absolutely|here(?:'s| is| are)|i can(?:'t| not|not)|i'm sorry|i am sorry|as an ai)\b",
    re.IGNORECASE,
)
_LOOP_NGRAM = 5
_LOOP_REPEATS = 3


def protected_tokens(text: str) -> list[str]:
    return [match.group(0).rstrip(_TRAILING_PUNCTUATION) for match in _PROTECTED.finditer(text)]


def _words(text: str) -> list[str]:
    return re.findall(r"[\w'’-]+", text.lower())


def _filler_counts(text: str) -> Counter:
    return Counter({filler: len(pattern.findall(text)) for filler, pattern in _FILLER_PATTERNS.items()})


def _max_ngram_repeats(text: str, n: int = _LOOP_NGRAM) -> int:
    words = _words(text)
    grams = Counter(tuple(words[index : index + n]) for index in range(len(words) - n + 1))
    return max(grams.values(), default=0)


def _numbering_ok(output: str) -> bool:
    """Each run of numbered items counts 1, 2, 3 without skips."""
    expected = 1
    for line in output.split("\n"):
        match = re.match(r"^\s*(\d+)\. ", line)
        if match:
            if int(match.group(1)) != expected:
                return False
            expected += 1
        elif line.strip() and not LIST_ITEM.match(line):
            expected = 1
    return True


def chrf(hypothesis: str, reference: str, n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (chrF), 0–100, ignoring whitespace."""
    hypothesis, reference = re.sub(r"\s+", " ", hypothesis.strip()), re.sub(r"\s+", " ", reference.strip())
    if not hypothesis or not reference:
        return 100.0 if hypothesis == reference else 0.0
    precisions, recalls = [], []
    for order in range(1, n + 1):
        hyp = Counter(hypothesis[i : i + order] for i in range(len(hypothesis) - order + 1))
        ref = Counter(reference[i : i + order] for i in range(len(reference) - order + 1))
        if not hyp or not ref:
            continue
        overlap = sum((hyp & ref).values())
        precisions.append(overlap / sum(hyp.values()))
        recalls.append(overlap / sum(ref.values()))
    precision, recall = sum(precisions) / len(precisions), sum(recalls) / len(recalls)
    if precision + recall == 0:
        return 0.0
    return 100 * (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def input_style(example: Example) -> str:
    """`unpunctuated` for lowercase inputs with no sentence punctuation, else `punctuated`."""
    text = example.input
    return "unpunctuated" if text == text.lower() and not re.search(r"[.,?!;:]", text) else "punctuated"


@dataclass
class RowChecks:
    """Per-row outcomes. `None` means the check does not apply to this row."""

    exact: bool
    chrf: float
    layout_match: bool
    formatting_added: bool | None
    formatting_missed: bool | None
    bad_list_marker: bool
    markdown: bool
    protected_kept: bool | None
    protected_tokens_total: int
    protected_tokens_kept: int
    leftover_filler: bool
    passthrough_kept: bool | None
    empty_correct: bool | None
    false_empty: bool
    loop: bool
    hit_limit: bool
    assistant_opener: bool
    deleted_over_30pct: bool | None
    output_words: int
    reference_words: int


def check(example: Example, output: str, *, hit_limit: bool = False) -> RowChecks:
    reference = example.output.strip()
    output = output.strip()
    expected_layout, got_layout = example.layout, layout_of(output)
    wanted = protected_tokens(reference) if reference != EMPTY else []
    kept = [token for token in wanted if token in output]
    reference_fillers, output_fillers = _filler_counts(reference), _filler_counts(output)
    input_repeats = _max_ngram_repeats(example.input)
    output_words, reference_words = len(_words(output)), len(_words(reference))
    return RowChecks(
        exact=output == reference,
        chrf=chrf(output, reference),
        layout_match=got_layout == expected_layout,
        formatting_added=(got_layout != "plain") if expected_layout == "plain" and reference != EMPTY else None,
        formatting_missed=(got_layout == "plain") if expected_layout != "plain" else None,
        bad_list_marker=any(_BAD_LIST_MARKER.match(line) for line in output.split("\n")) or not _numbering_ok(output),
        markdown=bool(_MARKDOWN.search(output)),
        protected_kept=(len(kept) == len(wanted)) if wanted else None,
        protected_tokens_total=len(wanted),
        protected_tokens_kept=len(kept),
        leftover_filler=any(output_fillers[f] > reference_fillers[f] for f in FILLERS),
        passthrough_kept=(output == example.input.strip()) if reference == example.input.strip() else None,
        empty_correct=(output == EMPTY) if reference == EMPTY else None,
        false_empty=output == EMPTY and reference != EMPTY,
        loop=hit_limit or _max_ngram_repeats(output) >= max(_LOOP_REPEATS, input_repeats + 1),
        hit_limit=hit_limit,
        assistant_opener=bool(_ASSISTANT_OPENERS.match(output)) and not _ASSISTANT_OPENERS.match(reference),
        deleted_over_30pct=(output_words < 0.7 * reference_words) if reference != EMPTY and reference_words >= 10 else None,
        output_words=output_words,
        reference_words=reference_words,
    )
