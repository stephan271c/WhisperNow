"""The gates every new reference passes.

1. The teacher writes the reference, exactly as the app would.
2. The completeness judge (`prompts/judge/completeness.md`) checks it against its own transcript:
   every fact the speaker kept, the right side of each self-correction, nothing invented.
3. Code checks run first: every protected token the transcript wrote verbatim (a URL, email, path,
   identifier or number already in digits) is in the reference, and the reference has no em dash
   and no correction cue kept from the transcript. What code cannot settle, a spelled-out
   number next to a word, a hedge or reason the reference says less often than the transcript, or
   a kept hedge the speaker may have settled, goes to the judge as the item's FLAGS.

A reference that fails is relabelled once, as a new sample of the same teacher request (or, with
`relabel_note`, with the judge's note added), and checked again; one that fails twice is dropped.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..text_rules import CASE_SENSITIVE, POSSIBLE_CORRECTION, filler_only, protected_tokens
from ..model.metrics import EM_DASH, HEDGES, casual_contractions, correction_cues, dropped_kept_words, spelled_counts
from .concurrency import task_group
from .judge import Verdict, judge_batch, size_batches
from .llm import LlmClient
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

_CUE_WINDOW_WORDS = 25
# "a.m."/"p.m." match the identifier pattern but are written "AM"/"PM" just as well.
_TIME_OF_DAY = re.compile(r"^[ap]\.m$", re.IGNORECASE)
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
# "23rd", "2am", "28s" and "20%" hold the numbers 23, 2, 28 and 20.
_NUMBER_SUFFIX = re.compile(r"^(\d+(?:\.\d+)?)(?:st|nd|rd|th|am|pm|s|%)$", re.IGNORECASE)


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
    return not CASE_SENSITIVE.search(token) and token.casefold() in reference.casefold()


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
            if POSSIBLE_CORRECTION.search(window) or "?" in " ".join(words[index : index + 3]):
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
    return bool(mentions) and POSSIBLE_CORRECTION.search(transcript, mentions[-1]) is not None


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


_HEDGE = re.compile(rf"\b(?:{'|'.join(map(re.escape, HEDGES))})\b", re.IGNORECASE)
# A yes or a decision soon after a hedge may settle it: "maybe some jasmine rice, actually yeah get it".
_SETTLE = re.compile(
    r"\b(?:yeah|yes|yep|yup|definitely|for sure|let'?s (?:do|get|go|just)|go with|"
    r"get (?:it|them|that|those|one|both|two)|make it|we do)\b",
    re.IGNORECASE,
)
_SETTLE_WINDOW_WORDS = 10
_FUNCTION_WORDS = frozenset(
    "some that this then with have just also from there they what like it's i'm we're you're about would "
    "could should them those these their really other thing things okay yeah".split()
) | frozenset(word for phrase in HEDGES for word in phrase.split())


def settled_hedges(transcript: str, reference: str) -> list[str]:
    """Kept hedges near a decision, quoted with the item they refer to for the judge to settle."""
    sentences = re.split(r"(?<=[.!?])\s+|\n", reference)
    matches = list(_HEDGE.finditer(transcript))
    found = []
    for index, match in enumerate(matches):
        if (
            index + 1 < len(matches)
            and match.group().casefold() == matches[index + 1].group().casefold()
            and not re.search(r"\w", transcript[match.end() : matches[index + 1].start()])
        ):
            continue
        following = list(re.finditer(r"\S+", transcript[match.end() :]))[:_SETTLE_WINDOW_WORDS]
        after = transcript[match.end() : match.end() + following[-1].end()] if following else ""
        settle = _SETTLE.search(after)
        if not settle:
            continue
        item = after[: settle.start()]
        start = match.start()
        hedged = [w for w in re.findall(r"[\w']+", item.casefold()) if len(w) >= 4 and w not in _FUNCTION_WORDS]
        if not hedged:
            preceding = list(re.finditer(r"\S+", transcript[:match.start()]))[-_SETTLE_WINDOW_WORDS:]
            start = preceding[0].start() if preceding else match.start()
            item = transcript[start : match.start()]
            hedged = [w for w in re.findall(r"[\w']+", item.casefold()) if len(w) >= 4 and w not in _FUNCTION_WORDS]
        hedge = re.compile(r"\b" + r"\s+".join(map(re.escape, match.group(0).split())) + r"\b", re.IGNORECASE)
        if any(hedge.search(s) and any(re.search(rf"\b{re.escape(w)}", s, re.IGNORECASE) for w in hedged) for s in sentences):
            found.append(transcript[start : match.end() + settle.end()].strip())
    return found


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
    # None keeps `LLM_JUDGE_REASONING`.
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


def empty_check(transcript: str, reference: str) -> tuple[list[str], str]:
    """Code settles EMPTY: it is right exactly when the transcript holds only fillers."""
    if filler_only(transcript) == (reference.strip() == EMPTY):
        return [], ""
    if reference.strip() == EMPTY:
        return ["empty"], "returned EMPTY for a transcript with dictated words"
    return ["empty"], f"the transcript holds only fillers, so the edit is exactly {EMPTY}"


def code_flags(transcript: str, reference: str) -> str:
    """What code found and the judge settles: spelled-out numbers next to a word, hedges or reasons
    the reference says less often than the transcript, dropped values a correction may have taken
    back or a larger number may hold, and kept hedges the speaker may have settled."""
    flags = [f"spelled-out number {phrase!r}" for phrase in spelled_counts(reference)[:5]]
    flags += [f"the transcript says {word!r} more often than the reference" for word in dropped_kept_words(transcript, reference)]
    flags += [
        f"the reference drops {token!r}, {why}"
        for token in missing_protected(transcript, reference)[:5] if (why := why_dropped(token, transcript))
    ]
    flags += [f"the reference keeps a hedge the speaker may settle: {phrase!r}" for phrase in settled_hedges(transcript, reference)[:3]]
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
    batches = size_batches(judged, judge.max_items, JUDGE_MAX_CHARS, judge.solo_over_words)
    async with task_group() as group:
        tasks = [group.create_task(judge_batch(llm, instructions, judge.criteria, batch, judge.reasoning))
                 for batch in batches]
    for task in tasks:
        verdicts.update((verdict.id, verdict) for verdict in task.result())
    return verdicts


def _relabel_note(note: str, earlier: str) -> str:
    if not earlier:
        return note
    return f"{note} Also avoid what an edit before it was rejected for: {earlier}"


@dataclass
class Gated:
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
    teacher_reasoning: str | None = None,
    relabel_note: bool = False,
    earlier_notes: dict[str, str] | None = None,
) -> dict[str, Gated]:
    """Labels (id, transcript) rows with the teacher, checks them, and relabels each failure once.

    `references` supplies a first reference per id instead of the teacher's attempt `first_attempt`
    instead of requesting that sample. A relabel is teacher attempt `first_attempt + 1`.
    `teacher_reasoning` overrides `LLM_REASONING` for the references the teacher writes.
    `relabel_note` adds the first check's note to the relabel request, so the second reference fixes
    what the first got wrong. `earlier_notes` adds a note per id from a check before that, so a fix
    does not bring back an earlier fault.
    """
    transcripts = dict(rows)
    ids = list(transcripts)
    first = dict(references or {})
    missing = [id_ for id_ in ids if id_ not in first]
    async with task_group() as group:
        tasks = [group.create_task(relabel(llm, transcripts[id_], first_attempt, teacher_reasoning))
                 for id_ in missing]
    for id_, task in zip(missing, tasks, strict=True):
        first[id_] = task.result()
    verdicts = await check_references(llm, [(id_, transcripts[id_], first[id_]) for id_ in ids], judge)
    results = {
        id_: Gated(id_, first[id_], True, first_attempt, [], "", [], "") for id_ in ids if verdicts[id_].passed
    }
    for id_ in ids:
        verdict = verdicts[id_]
        if not verdict.passed and set(verdict.failed) & set(INPUT_CRITERIA):
            results[id_] = Gated(id_, first[id_], False, first_attempt, verdict.failed, verdict.note, verdict.failed, verdict.note)
    retry = [id_ for id_ in ids if id_ not in results]
    async with task_group() as group:
        tasks = [group.create_task(relabel(
            llm, transcripts[id_], first_attempt + 1, teacher_reasoning,
            note=_relabel_note(verdicts[id_].note, (earlier_notes or {}).get(id_, "")) if relabel_note else None,
        )) for id_ in retry]
    second = {id_: task.result() for id_, task in zip(retry, tasks, strict=True)}
    rechecked = await check_references(llm, [(id_, transcripts[id_], second[id_]) for id_ in retry], judge)
    for id_ in retry:
        again = rechecked[id_]
        results[id_] = Gated(
            id_, second[id_], again.passed, first_attempt + 1, again.failed, again.note,
            verdicts[id_].failed, verdicts[id_].note,
        )
    return {id_: results[id_] for id_ in ids}
