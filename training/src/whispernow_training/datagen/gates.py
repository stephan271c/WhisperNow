"""The gates every new reference passes (todo 110, step 6).

1. The teacher writes the reference, exactly as the app would.
2. The completeness judge (`prompts/judge/completeness.md`) checks it against its own transcript:
   every fact the speaker kept, the right side of each self-correction, nothing invented.
3. A code check confirms that every protected token the transcript wrote verbatim (a URL, email,
   path, identifier or number already in digits) is in the reference.

A reference that fails is relabelled once, as a new sample of the same teacher request, and checked
again; one that fails twice is dropped. The label audit (`audit.py`) uses the same check.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..model.metrics import protected_tokens
from .judge import Verdict, judge_batch, size_batches
from .llm import LlmClient, Role
from .paths import PROMPTS
from .teacher import relabel

COMPLETENESS = PROMPTS / "judge" / "completeness.md"
COMPLETENESS_CRITERIA = ("facts_complete", "corrections_right", "nothing_invented")
# New generated rows: the completeness criteria, plus the realism and format checks stage 3's judge made.
COMPOSE = PROMPTS / "judge" / "compose.md"
COMPOSE_CRITERIA = (*COMPLETENESS_CRITERIA, "input_realistic", "not_answered", "layout_ok", "clean")
# A failure a new reference cannot fix: the row is dropped without a relabel.
INPUT_CRITERIA = ("input_realistic",)
# A request holds up to 10 rows, or about 14,000 characters: three 300-word rows with their
# references, or one 1,000-word row.
JUDGE_MAX_ITEMS = 10
JUDGE_MAX_CHARS = 14_000

# A token said before one of these, close by, may be the discarded side of a correction or an item
# the speaker took back, so the reference may lack it; the completeness judge still checks the row.
_CORRECTION_CUE = re.compile(
    r"\b(?:no|nope|wait|sorry|actually|i mean|scratch|make (?:it|that)|change|instead|rather|except|"
    r"go with|switch|never ?mind|forget|skip|drop|remove|cancel|take (?:off|out)|don't need|already have|"
    r"not|correction|let me|hold on|or|i guess|i meant|whoops|oops|my bad|er|really|say|maybe|probably)\b",
    re.IGNORECASE,
)
_CUE_WINDOW_WORDS = 25
# "a.m."/"p.m." match the identifier pattern but are written "AM"/"PM" just as well.
_TIME_OF_DAY = re.compile(r"^[ap]\.m$", re.IGNORECASE)
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
# "23rd", "2am", "28s" and "20%" hold the numbers 23, 2, 28 and 20.
_NUMBER_SUFFIX = re.compile(r"^(\d+(?:\.\d+)?)(?:st|nd|rd|th|am|pm|s|%)$", re.IGNORECASE)
# Paths, files, emails, URLs and flags keep their case; a word-like token may be recased ("g2" as "G2").
_CASE_SENSITIVE = re.compile(r"[/@_\\]|\.[A-Za-z]{2,}|--")


def _normalize(text: str) -> str:
    return _THOUSANDS.sub("", text)


def _present(token: str, reference: str) -> bool:
    """Whether the reference has the token: a number as the same number ("23rd" as "23", "250" as the
    "250" of "250000000", "0" in "10.0.0.14", but "20" not in "2026"), a word-like token in any case,
    others as written."""
    token = _NUMBER_SUFFIX.sub(r"\1", token)
    if re.fullmatch(r"\d+(?:\.\d+)?", token):
        return re.search(rf"(?<!\d){re.escape(token)}0*(?!\d)", reference) is not None
    if token in reference:
        return True
    return not _CASE_SENSITIVE.search(token) and token.casefold() in reference.casefold()


def missing_protected(transcript: str, reference: str) -> list[str]:
    """Protected tokens the transcript wrote that the reference lacks, except ones near a correction cue."""
    transcript, reference = _normalize(transcript), _normalize(reference)
    missing = []
    words = transcript.split()
    starts = [m.start() for m in re.finditer(r"\S+", transcript)]
    for token in protected_tokens(transcript):
        # A single digit is often part of something spoken digit by digit ("2 1 4" as "214").
        if not token or _TIME_OF_DAY.match(token) or re.fullmatch(r"\d", token) or _present(token, reference):
            continue
        # Any mention may be the one the speaker took back ("100? ... I said 100, make it 120").
        tentative = False
        for match in re.finditer(re.escape(token), transcript):
            index = max(0, sum(start <= match.start() for start in starts) - 1)
            window = " ".join(words[max(0, index - _CUE_WINDOW_WORDS) : index + _CUE_WINDOW_WORDS])
            # A value asked as a question ("15 of them?") is tentative too.
            if _CORRECTION_CUE.search(window) or "?" in " ".join(words[index : index + 3]):
                tentative = True
                break
        if not tentative:
            missing.append(token)
    return missing


@dataclass(frozen=True)
class Judge:
    """The judge prompt and criteria a gate uses, and the style each row's transcript is in."""

    instructions: Path = COMPLETENESS
    criteria: tuple[str, ...] = COMPLETENESS_CRITERIA
    styles: dict[str, str] | None = None
    # Appended to the instructions, for a check the prompt file does not cover.
    addendum: str = ""
    # Rows per request: 1 gives each long row the judge's full attention.
    max_items: int = JUDGE_MAX_ITEMS
    # The judge model's reasoning for this judge's checks, and for references the gate has the judge's
    # model write; None keeps `LLM_JUDGE_REASONING`.
    reasoning: str | None = None


COMPLETENESS_JUDGE = Judge()


async def check_references(
    llm: LlmClient, rows: Sequence[tuple[str, str, str]], judge: Judge = COMPLETENESS_JUDGE
) -> dict[str, Verdict]:
    """(id, transcript, reference) rows, checked in code and then by the judge."""
    verdicts: dict[str, Verdict] = {}
    judged = []
    for id_, transcript, reference in rows:
        missing = missing_protected(transcript, reference)
        if missing:
            note = "missing protected tokens: " + ", ".join(repr(token) for token in missing[:5])
            verdicts[id_] = Verdict(id_, False, ["protected_tokens"], note, "code")
        else:
            style = {"style": judge.styles[id_]} if judge.styles else {}
            judged.append((id_, {**style, "transcript": transcript, "reference": reference}))
    instructions = judge.instructions.read_text(encoding="utf-8") + (f"\n{judge.addendum}\n" if judge.addendum else "")
    judged.sort(key=lambda item: len(item[1]["transcript"]))
    batches = size_batches(judged, judge.max_items, JUDGE_MAX_CHARS)
    for batch_verdicts in await asyncio.gather(
        *(judge_batch(llm, instructions, judge.criteria, batch, judge.reasoning) for batch in batches)
    ):
        verdicts.update((verdict.id, verdict) for verdict in batch_verdicts)
    return verdicts


@dataclass
class Gated:
    """A reference and how it passed or failed the gates."""

    id: str
    output: str
    passed: bool
    attempt: int
    failed: list[str]
    note: str
    first_failed: list[str]
    first_note: str


async def gate(
    llm: LlmClient,
    rows: Sequence[tuple[str, str]],
    references: dict[str, str] | None = None,
    first_attempt: int = 0,
    judge: Judge = COMPLETENESS_JUDGE,
    teacher_role: Role = "teacher",
    relabel_role: Role | None = None,
    teacher_reasoning: str | None = None,
) -> dict[str, Gated]:
    """Labels (id, transcript) rows with the teacher, checks them, and relabels each failure once.

    `references` supplies a first reference per id instead of the teacher's attempt `first_attempt`
    (the audit passes the existing ones). A relabel is teacher attempt `first_attempt + 1`.
    `teacher_role="judge"` has the judge's model write the references; `relabel_role="judge"` only
    the relabel, so a failure gets its second reference from the stronger model, at `judge.reasoning`.
    `teacher_reasoning` overrides `LLM_REASONING` for the references the teacher role writes.
    """
    first_reasoning = judge.reasoning if teacher_role == "judge" else teacher_reasoning
    relabel_role = relabel_role or teacher_role
    relabel_reasoning = judge.reasoning if relabel_role == "judge" else teacher_reasoning
    transcripts = dict(rows)
    ids = list(transcripts)
    first = dict(references or {})
    missing = [id_ for id_ in ids if id_ not in first]
    for id_, output in zip(
        missing,
        await asyncio.gather(
            *(relabel(llm, transcripts[id_], first_attempt, teacher_role, first_reasoning) for id_ in missing)
        ),
        strict=True,
    ):
        first[id_] = output
    verdicts = await check_references(llm, [(id_, transcripts[id_], first[id_]) for id_ in ids], judge)
    results = {
        id_: Gated(id_, first[id_], True, first_attempt, [], "", [], "") for id_ in ids if verdicts[id_].passed
    }
    for id_ in ids:
        verdict = verdicts[id_]
        if not verdict.passed and set(verdict.failed) & set(INPUT_CRITERIA):
            results[id_] = Gated(id_, first[id_], False, first_attempt, verdict.failed, verdict.note, verdict.failed, verdict.note)
    retry = [id_ for id_ in ids if id_ not in results]
    second = dict(
        zip(
            retry,
            await asyncio.gather(
                *(
                    relabel(llm, transcripts[id_], first_attempt + 1, relabel_role, relabel_reasoning)
                    for id_ in retry
                )
            ),
            strict=True,
        )
    )
    rechecked = await check_references(llm, [(id_, transcripts[id_], second[id_]) for id_ in retry], judge)
    for id_ in retry:
        again = rechecked[id_]
        results[id_] = Gated(
            id_, second[id_], again.passed, first_attempt + 1, again.failed, again.note,
            verdicts[id_].failed, verdicts[id_].note,
        )
    return {id_: results[id_] for id_ in ids}
