"""The gates every new reference passes (todo 110, step 6).

1. The teacher writes the reference, exactly as the app would.
2. The completeness judge (`prompts/judge/completeness.md`) checks it against its own transcript:
   every fact the speaker kept, the right side of each self-correction, nothing invented.
3. Code checks run first: every protected token the transcript wrote verbatim (a URL, email, path,
   identifier or number already in digits) is in the reference, and the reference has no em dash
   and no correction cue kept from the transcript (todo 112). What code cannot settle, a spelled-out
   number next to a word or a hedge or reason the reference says less often than the transcript, goes
   to the judge as the item's FLAGS.

A reference that fails is relabelled once, as a new sample of the same teacher request (or, with
`relabel_note`, with the judge's note added), and checked again; one that fails twice is dropped.
The label audit (`audit.py`) uses the same check.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..model.metrics import EM_DASH, casual_contractions, correction_cues, dropped_kept_words, protected_tokens, spelled_counts
from .judge import Verdict, judge_batch, size_batches
from .llm import LlmClient, Role
from .paths import PROMPTS
from .schema import EMPTY
from .teacher import relabel

COMPLETENESS = PROMPTS / "judge" / "completeness.md"
FACT_CRITERIA = ("facts_complete", "corrections_right", "nothing_invented")
COMPLETENESS_CRITERIA = (*FACT_CRITERIA, "clean")
# New generated rows: the fact criteria, plus the realism and format checks stage 3's judge made.
COMPOSE = PROMPTS / "judge" / "compose.md"
COMPOSE_CRITERIA = (*FACT_CRITERIA, "input_realistic", "not_answered", "layout_ok", "clean")
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


def corrected_later(token: str, transcript: str) -> bool:
    """Whether a correction cue follows the token's last mention, however far on: a late correction
    ("... actually, forget the cork, go with bamboo") can take back a value said long before."""
    transcript = _normalize(transcript)
    mentions = [m.end() for m in re.finditer(re.escape(token), transcript)]
    return bool(mentions) and _CORRECTION_CUE.search(transcript, mentions[-1]) is not None


# A restart right after a value ("55... 58 degrees", "precinct 30— 31") may replace it.
_RESTART = re.compile(r"\s*(?:\.{2,}|…|—|--)")
# A value followed by a spoken number ("27 six", "42 million 300 thousand") may be written as part
# of a larger one ("$27,600", "42,300,000").
_SPOKEN_NUMBER = re.compile(
    r"\s+(?:hundred|thousand|million|billion|trillion|point|oh|one|two|three|four|five|six|seven|eight|nine|"
    r"ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|"
    r"fifty|sixty|seventy|eighty|ninety|\d)\b",
    re.IGNORECASE,
)
# ... or preceded by one: "point 1.8" is 0.18, "oh 600" is 0600, "17/2 61" is 17/261.
_SPOKEN_BEFORE = re.compile(r"(?:\b(?:point|oh)\s*|\d\s+)$", re.IGNORECASE)
# A value the speaker says they misspoke ("I said $18, it was $28", "I wrote 2.3 ... it's 3.2").
_MISSPOKE = re.compile(r"\bi (?:said|wrote|put|typed)\s+\$?$", re.IGNORECASE)


def why_dropped(token: str, transcript: str) -> str:
    """Why the reference may rightly lack a protected token, for the judge to settle, or "" when code
    can fail the reference for it."""
    if corrected_later(token, transcript):
        return "which is right only if a later correction took it back"
    text = _normalize(transcript)
    spans = [m.span() for m in re.finditer(rf"(?<![\w.]){re.escape(token)}(?![\w])", text)]
    if any(_RESTART.match(text, end) for _, end in spans):
        return "which is right only if the speaker restarted and replaced it"
    if any(_MISSPOKE.search(text[:start]) for start, _ in spans):
        return "which is right only if the speaker said they misspoke it"
    if any(_SPOKEN_NUMBER.match(text, end) or _SPOKEN_BEFORE.search(text[:start]) for start, end in spans):
        return "which is right only if it is written as part of a larger number"
    return ""


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
    # Rows whose transcript is longer than this many words get a request each, the rest share requests.
    solo_over_words: int | None = None


COMPLETENESS_JUDGE = Judge()


def code_check(transcript: str, reference: str) -> tuple[list[str], str]:
    """The failed code checks of a reference, and a note naming what failed."""
    failed, notes = [], []
    # A value a correction may have taken back, or a larger number may hold, goes to the judge as a
    # flag (`code_flags`).
    missing = [token for token in missing_protected(transcript, reference) if not why_dropped(token, transcript)]
    if missing:
        failed.append("protected_tokens")
        notes.append("missing protected tokens: " + ", ".join(repr(token) for token in missing[:5]))
    if EM_DASH in reference:
        failed.append("em_dash")
        notes.append("an em dash, where a comma, period, colon or parentheses belong")
    cues = correction_cues(reference, transcript)
    if cues:
        failed.append("correction_cue")
        notes.append("kept the correction cue " + ", ".join(repr(cue) for cue in cues[:3]))
    contractions = casual_contractions(reference)
    if contractions:
        failed.append("contraction")
        notes.append("kept " + ", ".join(repr(word) for word in contractions[:3]) + " instead of writing it out ('want to', 'going to')")
    return failed, "; ".join(notes)


_FILLER_WORD = re.compile(r"um+|uh+m?|erm?|hmm+|mm+|mhm|ah+", re.IGNORECASE)


def filler_only(transcript: str) -> bool:
    """Whether a transcript holds no dictated words, only fillers and punctuation, so its edit is EMPTY."""
    return all(_FILLER_WORD.fullmatch(word) for word in re.findall(r"[\w']+", transcript))


def empty_check(transcript: str, reference: str) -> tuple[list[str], str]:
    """Code settles EMPTY: it is right exactly when the transcript holds only fillers."""
    if filler_only(transcript) == (reference.strip() == EMPTY):
        return [], ""
    if reference.strip() == EMPTY:
        return ["empty"], "returned EMPTY for a transcript with dictated words"
    return ["empty"], f"the transcript holds only fillers, so the edit is exactly {EMPTY}"


def code_flags(transcript: str, reference: str) -> str:
    """What code found and the judge settles: spelled-out numbers next to a word, hedges or reasons
    the reference says less often than the transcript, and dropped values a correction may have
    taken back or a larger number may hold."""
    flags = [f"spelled-out number {phrase!r}" for phrase in spelled_counts(reference)[:5]]
    flags += [f"the transcript says {word!r} more often than the reference" for word in dropped_kept_words(transcript, reference)]
    flags += [
        f"the reference drops {token!r}, {why}"
        for token in missing_protected(transcript, reference)[:5] if (why := why_dropped(token, transcript))
    ]
    return "; ".join(flags)


async def check_references(
    llm: LlmClient, rows: Sequence[tuple[str, str, str]], judge: Judge = COMPLETENESS_JUDGE
) -> dict[str, Verdict]:
    """(id, transcript, reference) rows, checked in code and then by the judge."""
    verdicts: dict[str, Verdict] = {}
    judged = []
    for id_, transcript, reference in rows:
        if reference.strip() == EMPTY or filler_only(transcript):
            failed, note = empty_check(transcript, reference)
            verdicts[id_] = Verdict(id_, not failed, failed, note, "code")
            continue
        failed, note = code_check(transcript, reference)
        if failed:
            verdicts[id_] = Verdict(id_, False, failed, note, "code")
        else:
            style = {"style": judge.styles[id_]} if judge.styles else {}
            flags = code_flags(transcript, reference)
            judged.append((id_, {**style, "transcript": transcript, "reference": reference, **({"flags": flags} if flags else {})}))
    instructions = judge.instructions.read_text(encoding="utf-8") + (f"\n{judge.addendum}\n" if judge.addendum else "")
    judged.sort(key=lambda item: len(item[1]["transcript"]))
    solo = [item for item in judged if judge.solo_over_words is not None and len(item[1]["transcript"].split()) > judge.solo_over_words]
    shared = [item for item in judged if item not in solo]
    batches = size_batches(shared, judge.max_items, JUDGE_MAX_CHARS) + [[item] for item in solo]
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
    relabel_note: bool = False,
) -> dict[str, Gated]:
    """Labels (id, transcript) rows with the teacher, checks them, and relabels each failure once.

    `references` supplies a first reference per id instead of the teacher's attempt `first_attempt`
    (the audit passes the existing ones). A relabel is teacher attempt `first_attempt + 1`.
    `teacher_role="judge"` has the judge's model write the references; `relabel_role="judge"` only
    the relabel, so a failure gets its second reference from the stronger model, at `judge.reasoning`.
    `teacher_reasoning` overrides `LLM_REASONING` for the references the teacher role writes.
    `relabel_note` adds the first check's note to the relabel request, so the second reference fixes
    what the first got wrong.
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
                    relabel(
                        llm, transcripts[id_], first_attempt + 1, relabel_role, relabel_reasoning,
                        note=verdicts[id_].note if relabel_note else None,
                    )
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
