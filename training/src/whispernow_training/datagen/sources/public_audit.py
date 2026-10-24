"""Add handy, disfl_qa and optionally sotto rows, audited against the target rules."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import replace

from .. import paths
from . import public
from ..concurrency import task_group
from ..judge import judge_all, pair_item
from ..llm import LlmClient
from ..schema import Example, write_examples
from ..teacher import relabel

PROMPT = paths.PROMPTS / "audit" / "handy_disfl_qa.md"
CRITERIA = ("content_preserved", "no_hallucination", "not_answered", "fillers_removed", "layout_ok")
JUDGMENTS = paths.PUBLIC / "judgments.jsonl"
RELABEL_JUDGMENTS = paths.PUBLIC / "relabel_judgments.jsonl"
REPORT = paths.PUBLIC / "report.json"
SEED = 108

# Rows kept per category. handy's questions are copies of disfl_qa rows, so questions come from disfl_qa.
CAPS = {
    public.HANDY: {
        "self_correction": 3500,
        "short_dictation": 3000,
        "no_edit": 3000,
        "dictation": 1500,
    },
    public.DISFL_QA: {"dictated_question": 3000},
    public.SOTTO: {"self_correction": 1500, "preserve_wording": 1000, "adversarial": 800, "false_start": 800},
}
OUT_FILES = {
    public.HANDY: paths.PUBLIC / "handy.jsonl",
    public.DISFL_QA: paths.PUBLIC / "disfl_qa.jsonl",
    public.SOTTO: paths.PUBLIC / "sotto.jsonl",
}

# The target writes these as digits; these sources keep them as words.
_SPELLED_NUMBER = re.compile(
    r"\b(ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|"
    r"fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b"
    r"|\b(one|two|three|four|five|six|seven|eight|nine)\s+(am|pm|a\.m\.|p\.m\.|o'clock|percent|dollars?|pounds?|euros?)\b"
    r"|\b\d+(\.\d+)?\s+percent\b",
    re.IGNORECASE,
)


def conflict(example: Example) -> str | None:
    """Why a row conflicts with the target rules in a way we don't normalize, or None."""
    if _SPELLED_NUMBER.search(example.output):
        return "number written as words"
    if example.source == public.HANDY and public.spliced(example.input):
        return "written text spliced into dictation"
    if example.source == public.DISFL_QA and public.reworded(example.input, example.output):
        return "reworded question"
    return None


def _switchboard() -> public.SwitchboardIndex:
    cache_path = paths.CACHE / "switchboard_texts.json"
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(public.switchboard_texts()))
    return public.SwitchboardIndex(json.loads(cache_path.read_text()))


def load_candidates(with_sotto: bool) -> tuple[dict[str, list[Example]], dict[str, Counter]]:
    dropped: dict[str, Counter] = defaultdict(Counter)
    handy, dropped[public.HANDY] = public.load_handy(_switchboard())
    loaded = {public.HANDY: handy, public.DISFL_QA: public.load_disfl_qa()}
    if with_sotto:
        loaded[public.SOTTO], dropped[public.SOTTO] = public.load_sotto()
    sampled = {}
    for source, rows in loaded.items():
        kept = []
        for row in rows:
            reason = conflict(row)
            if reason:
                dropped[source][reason] += 1
            else:
                kept.append(row)
        sampled[source] = public.sample_by_category(kept, CAPS[source], SEED)
    return sampled, dropped


async def add_sources(llm: LlmClient, with_sotto: bool) -> dict:
    """Audits the sampled rows with the judge. A list kept as prose is relabeled; other failures are dropped."""
    sampled, dropped = load_candidates(with_sotto)
    candidates = [row for rows in sampled.values() for row in rows]
    verdicts = await judge_all(
        llm,
        [pair_item(e) for e in candidates],
        JUDGMENTS,
        instructions_file=PROMPT,
        criteria=CRITERIA,
        batch_size=30,
    )
    layout_only = [e for e in candidates if verdicts[e.id].failed == ["layout_ok"]]
    async with task_group() as group:
        tasks = [group.create_task(relabel(llm, e.input)) for e in layout_only]
    outputs = [task.result() for task in tasks]
    relabeled = [
        replace(e, output=output, teacher_model=llm.settings.model, tags=[*e.tags, "relabeled"])
        for e, output in zip(layout_only, outputs, strict=True)
    ]
    rejudged = await judge_all(
        llm,
        [pair_item(e) for e in relabeled],
        RELABEL_JUDGMENTS,
        instructions_file=PROMPT,
        criteria=CRITERIA,
        batch_size=30,
    )
    fixed = {e.id: e for e in relabeled if rejudged[e.id].passed}

    report: dict = {"judge_model": llm.settings.judge_model, "sources": {}}
    for source, rows in sampled.items():
        kept = [fixed.get(e.id, e) for e in rows if verdicts[e.id].passed or e.id in fixed]
        write_examples(OUT_FILES[source], kept)
        failed = Counter(c for e in rows for c in verdicts[e.id].failed)
        report["sources"][source] = {
            "dropped_before_audit": dict(dropped[source]),
            "audited": len(rows),
            "flagged": sum(not verdicts[e.id].passed for e in rows),
            "relabeled": sum(e.id in fixed for e in rows),
            "kept": len(kept),
            "failed_criteria": dict(failed),
            "kept_per_category": dict(Counter(e.category for e in kept)),
        }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
