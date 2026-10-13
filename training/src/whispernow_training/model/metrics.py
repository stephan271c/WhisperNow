"""Deterministic evaluation checks for one output against its row's reference.

These live only in evaluation; the app gets no output-policing heuristics. The LLM judge
(`prompts/judge/eval.md`) covers answered transcripts and dropped or invented facts.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from ..datagen.schema import EMPTY, Example, layout_of
from ..text_rules import (
    CORRECTION, DISCOURSE_FILLER_PATTERNS, EVAL_ASSISTANT_OPENING, FILLER, SENTENCE_CORRECTION,
    has_markdown, malformed_list, protected_tokens,
)

_LOOP_NGRAM = 5
_LOOP_REPEATS = 3


# The cleanup contract's style: checked in references by the gates, reported for outputs.
EM_DASH = "—"
_CARDINAL = (
    r"(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
    r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)"
)
# A spelled-out cardinal from 2 up and the word after it; a hyphenated compound ("two-factor") is not one.
_SPELLED_COUNT = re.compile(rf"\b({_CARDINAL}(?:[ -](?:and )?{_CARDINAL})*) ([A-Za-z][\w']*)(?!-)", re.IGNORECASE)
# Words after a number that are not the thing counted, so the number is not a count of a noun.
_NOT_COUNTED = {
    "of", "or", "and", "to", "the", "a", "an", "in", "on", "at", "for", "by", "with", "from", "is", "are", "was",
    "were", "be", "more", "less", "than", "then", "so", "but", "if", "as", "that", "this", "it", "we", "you",
    "they", "i", "he", "she", "me", "us", "them", "who", "which", "would", "will", "can", "could", "should",
    "o'clock",
}
# Idioms whose number stays a word.
NUMBER_IDIOMS = ("two cents", "two minds", "two birds", "two peas", "two sides", "two feet", "two hands")
HEDGES = ("maybe", "probably", "i think", "i guess", "i believe")
_REASONS = ("because",)
# The hedges and reasons a cleanup keeps, counted against the transcript.
KEPT_WORDS = (*HEDGES, *_REASONS)
_KEPT_PATTERNS = {word: re.compile(rf"\b{word.replace(' ', r'\s+')}\b", re.IGNORECASE) for word in KEPT_WORDS}
_SENTENCE_START = re.compile(r"(?:^|(?<=[.!?…:])\s+|\n)(?:\s*(?:- |\d+\. ))?\s*", re.MULTILINE)


# Casual contractions the cleanup writes out ("wanna" as "want to").
_CONTRACTIONS = re.compile(r"\b(?:wanna|gonna|gotta|kinda|sorta)\b", re.IGNORECASE)


def casual_contractions(text: str) -> list[str]:
    return _CONTRACTIONS.findall(text)


def spelled_counts(text: str) -> list[str]:
    """Spelled-out cardinals from 2 up next to the noun or unit they count ("two bags"), for the judge to
    resolve: a title, a name or an idiom can hold one."""
    found = []
    for match in _SPELLED_COUNT.finditer(text):
        number, word = match.group(1), match.group(2)
        phrase = f"{number} {word}"
        if word.casefold() in _NOT_COUNTED or phrase.casefold().startswith(NUMBER_IDIOMS):
            continue
        # "Three Rivers" inside a sentence is a name.
        before = text[: match.start()].rstrip(" ")
        if number[0].isupper() and word[0].isupper() and before and not before.endswith((".", "!", "?", ":", "\n", "-")):
            continue
        found.append(phrase)
    return found


def correction_cues(text: str, transcript: str) -> list[str]:
    """Correction cues ("wait", "oh wait", "okay so", "scratch that") opening a sentence of `text` that
    the transcript said, so they were kept rather than written."""
    spoken = " ".join(transcript.casefold().replace(",", " ").split())
    cues = []
    for start in _SENTENCE_START.finditer(text):
        match = SENTENCE_CORRECTION.match(text, start.end())
        if match:
            cue = " ".join(match.group(0).casefold().replace(",", " ").split())
            if cue in spoken:
                cues.append(match.group(0))
    return cues


_CLAUSE_WORDS = 3
_SKIPPED = {"like", "you", "know", "i", "mean", "so", "the", "a", "an"}
_DIGITS = {word: str(value) for value, word in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split()
)}


def _clause_words(text: str) -> list[str]:
    words = (_DIGITS.get(word, word) for word in re.findall(r"[\w']+", text.casefold()))
    # A stutter ("I don't I don't remember") says a word once.
    kept: list[str] = []
    for word in words:
        if not FILLER.fullmatch(word) and word not in _SKIPPED and word not in kept[-2:]:
            kept.append(word)
    return kept


# Occurrences a reference may drop: a hedge the speaker says again straight away ("I think that's um I
# think that's"), a closing aside ("I think that's everything", "that's it, I think") and the abandoned
# side of a correction ("Marcus I think or maybe Mateo, no, Marcus").
_REPEAT_WORDS = 8
_CORRECTION_WORDS = 10
_CLOSING = re.compile(r"^[\W_]*(?:that'?s|that is)\s+(?:\w+\s+)?(?:it|everything|all)\b", re.IGNORECASE)
_CLOSED = re.compile(r"\b(?:that'?s|that is)\s+(?:\w+\s+)?(?:it|everything|all)[\W_]*$", re.IGNORECASE)


def _excused(pattern: re.Pattern[str], transcript: str, match: re.Match[str]) -> bool:
    after = transcript[match.end() :]
    following = " ".join(after.split()[:_REPEAT_WORDS])
    if pattern.search(following) or CORRECTION.search(" ".join(after.split()[:_CORRECTION_WORDS])):
        return True
    return bool(_CLOSING.match(after) or _CLOSED.search(transcript[max(0, match.start() - 40) : match.start()]))


def _reasons_kept(matches: list[re.Match[str]], transcript: str, text: str) -> int:
    """Reason clauses of the transcript whose first words `text` keeps, with or without the word that
    opened them: a list item's reason goes in parentheses ("(Joaquin is bringing his kids)")."""
    kept_words = " " + " ".join(_clause_words(text)) + " "
    kept = 0
    for match in matches:
        clause = _clause_words(transcript[match.end() :])[:_CLAUSE_WORDS]
        kept += bool(clause) and f" {' '.join(clause)} " in kept_words
    return kept


def kept_counts(transcript: str, text: str, excuse: bool = False) -> dict[str, tuple[int, int]]:
    """For each hedge and reason word the transcript says: how often it says it, and how many of those
    `text` keeps. A reason counts as kept when its clause's first words are there, as a reason in
    parentheses drops "because". A drop can be right, when the word sat in the abandoned side of a
    correction; `excuse` leaves out the occurrences a reference may drop (`_excused`)."""
    counts = {}
    for word, pattern in _KEPT_PATTERNS.items():
        matches = list(pattern.finditer(transcript))
        if excuse:
            matches = [match for match in matches if not _excused(pattern, transcript, match)]
        if matches:
            kept = len(pattern.findall(text))
            if word in _REASONS:
                kept = max(kept, _reasons_kept(matches, transcript, text))
            counts[word] = (len(matches), min(len(matches), kept))
    return counts


def dropped_kept_words(transcript: str, text: str) -> list[str]:
    return [word for word, (said, kept) in kept_counts(transcript, text).items() if kept < said]


def _words(text: str) -> list[str]:
    return re.findall(r"[\w'’-]+", text.lower())


def _filler_counts(text: str) -> Counter:
    counts = Counter(match.group(0).casefold() for match in FILLER.finditer(text))
    counts.update({phrase: len(pattern.findall(text)) for phrase, pattern in DISCOURSE_FILLER_PATTERNS.items()})
    return counts


def _max_ngram_repeats(text: str, n: int = _LOOP_NGRAM) -> int:
    words = _words(text)
    grams = Counter(tuple(words[index : index + n]) for index in range(len(words) - n + 1))
    return max(grams.values(), default=0)


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
    em_dash: bool
    spelled_count: bool
    correction_cue: bool
    # Hedge and reason words the transcript says: (times said, times the output keeps them).
    kept_words: dict[str, tuple[int, int]]


def check(example: Example, output: str, *, hit_limit: bool = False) -> RowChecks:
    reference = example.output.strip()
    output = output.strip()
    expected_layout, got_layout = example.layout, layout_of(output)
    wanted = protected_tokens(reference) if reference != EMPTY else []
    kept = sum((Counter(wanted) & Counter(protected_tokens(output))).values())
    reference_fillers, output_fillers = _filler_counts(reference), _filler_counts(output)
    input_repeats = _max_ngram_repeats(example.input)
    output_words, reference_words = len(_words(output)), len(_words(reference))
    return RowChecks(
        exact=output == reference,
        chrf=chrf(output, reference),
        layout_match=got_layout == expected_layout,
        formatting_added=(got_layout != "plain") if expected_layout == "plain" and reference != EMPTY else None,
        formatting_missed=(got_layout == "plain") if expected_layout != "plain" else None,
        bad_list_marker=malformed_list(output) is not None,
        markdown=has_markdown(output),
        protected_kept=(kept == len(wanted)) if wanted else None,
        protected_tokens_total=len(wanted),
        protected_tokens_kept=kept,
        leftover_filler=any(count > reference_fillers[filler] for filler, count in output_fillers.items()),
        passthrough_kept=(output == example.input.strip()) if reference == example.input.strip() else None,
        empty_correct=(output == EMPTY) if reference == EMPTY else None,
        false_empty=output == EMPTY and reference != EMPTY,
        loop=hit_limit or _max_ngram_repeats(output) >= max(_LOOP_REPEATS, input_repeats + 1),
        hit_limit=hit_limit,
        assistant_opener=bool(EVAL_ASSISTANT_OPENING.match(output)) and not EVAL_ASSISTANT_OPENING.match(reference),
        deleted_over_30pct=(output_words < 0.7 * reference_words) if reference != EMPTY and reference_words >= 10 else None,
        output_words=output_words,
        reference_words=reference_words,
        em_dash=EM_DASH in output,
        spelled_count=bool(spelled_counts(output)),
        correction_cue=bool(correction_cues(output, example.input)),
        kept_words=kept_counts(example.input, output),
    )
