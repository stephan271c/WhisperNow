"""Flag aawaaz rows below its own standard, fix them with the teacher or drop them."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import replace

from .. import paths
from ..concurrency import task_group
from ..judge import BATCH_SIZE, Verdict, judge_all, load_verdicts, pair_item
from ..llm import LlmClient
from ..schema import Example, layout_of, read_examples, write_examples
from ..teacher import relabel

IMPORTED = paths.AAWAAZ / "aawaaz.imported.jsonl"
JUDGMENTS = paths.AAWAAZ / "judgments.jsonl"
RELABEL_JUDGMENTS = paths.AAWAAZ / "relabel_judgments.jsonl"
RELABELED = paths.AAWAAZ / "relabeled.jsonl"
CLEAN = paths.AAWAAZ / "aawaaz.clean.jsonl"
REPORT = paths.AAWAAZ / "report.json"
PROMPT = paths.PROMPTS / "audit" / "aawaaz.md"
CRITERIA = ("content_preserved", "no_hallucination", "lead_in_kept", "not_answered")
AGREEMENT_CRITERIA = ("layout_agrees", "lead_in_agrees", "facts_agree")

# A second pass for register: the hand review found casual stories shortened into notes and
# legal dictation made more formal, which the four `CRITERIA` do not catch.
REGISTER_CATEGORIES = (
    "casual_conversation",
    "creative_writing",
    "legal_contract",
    "email_professional",
    "medical_clinical",
    "meeting_notes",
)
REGISTER_CRITERIA = ("register_kept",)
REGISTER_PROMPT = paths.PROMPTS / "audit" / "aawaaz_register.md"
REGISTER_JUDGMENTS = paths.AAWAAZ / "register_judgments.jsonl"
RELABEL_REGISTER_JUDGMENTS = paths.AAWAAZ / "relabel_register_judgments.jsonl"


async def judge_aawaaz(
    llm: LlmClient, examples: list[Example], out=None, batch_size: int = BATCH_SIZE,
) -> dict[str, Verdict]:
    return await judge_all(
        llm, [pair_item(e) for e in examples], out or JUDGMENTS,
        instructions_file=PROMPT, criteria=CRITERIA, batch_size=batch_size,
    )


def needs_register_check(example: Example) -> bool:
    return example.category in REGISTER_CATEGORIES


async def judge_register(llm: LlmClient, examples: list[Example], out=None) -> dict[str, Verdict]:
    return await judge_all(
        llm,
        [pair_item(e) for e in examples if needs_register_check(e)],
        out or REGISTER_JUDGMENTS,
        instructions_file=REGISTER_PROMPT,
        criteria=REGISTER_CRITERIA,
    )


def agreement_sample(examples: list[Example], n: int) -> list[Example]:
    """Held-out rows spread evenly over categories, in a fixed order."""
    by_category: dict[str, list[Example]] = defaultdict(list)
    for example in sorted(examples, key=lambda e: e.id):
        if example.split == "test":
            by_category[example.category].append(example)
    sample: list[Example] = []
    while len(sample) < n and any(by_category.values()):
        for category in sorted(by_category):
            if by_category[category] and len(sample) < n:
                sample.append(by_category[category].pop(0))
    return sample


async def teacher_agreement(llm: LlmClient, n: int, label: str) -> dict:
    """Runs the teacher on held-out inputs and has the judge compare it with the aawaaz references."""
    sample = agreement_sample(read_examples(IMPORTED), n)
    async with task_group() as group:
        tasks = [group.create_task(relabel(llm, example.input)) for example in sample]
    outputs = [task.result() for task in tasks]
    out_dir = paths.AAWAAZ / "teacher_check"
    items = [
        (example.id, {"transcript": example.input, "reference": example.output, "candidate": output})
        for example, output in zip(sample, outputs, strict=True)
    ]
    verdict_file = out_dir / f"{label}.verdicts.jsonl"
    verdict_file.unlink(missing_ok=True)
    verdicts = await judge_all(
        llm,
        items,
        verdict_file,
        instructions_file=paths.PROMPTS / "judge" / "agreement.md",
        criteria=AGREEMENT_CRITERIA,
        batch_size=10,
    )
    rows = []
    for example, output in zip(sample, outputs, strict=True):
        verdict = verdicts[example.id]
        rows.append(
            {
                "id": example.id,
                "category": example.category,
                "input": example.input,
                "reference": example.output,
                "teacher": output,
                "reference_layout": example.layout,
                "teacher_layout": layout_of(output),
                "failed": verdict.failed,
                "note": verdict.note,
            }
        )
    summary = {
        "label": label,
        "teacher_model": llm.settings.model,
        "reasoning": llm.settings.reasoning,
        "judge_model": llm.settings.judge_model,
        "judge_reasoning": llm.settings.judge_reasoning,
        "n": len(rows),
        "all_agree": sum(not row["failed"] for row in rows) / len(rows),
        **{
            criterion: sum(criterion not in row["failed"] for row in rows) / len(rows)
            for criterion in AGREEMENT_CRITERIA
        },
        "layout_class_matches": sum(r["reference_layout"] == r["teacher_layout"] for r in rows) / len(rows),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"summary": summary, "rows": rows}
    (out_dir / f"{label}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


async def fix_flagged(llm: LlmClient) -> dict:
    """Relabels every flagged row with the teacher, re-judges it, and keeps it only if it now passes."""
    examples = read_examples(IMPORTED)
    verdicts = load_verdicts(JUDGMENTS)
    register_verdicts = load_verdicts(REGISTER_JUDGMENTS)
    missing = [e.id for e in examples if e.id not in verdicts]
    if missing:
        raise SystemExit(f"{len(missing)} rows are not judged yet; run `datagen judge-aawaaz` first")
    unchecked = [e.id for e in examples if needs_register_check(e) and e.id not in register_verdicts]
    if unchecked:
        raise SystemExit(f"{len(unchecked)} rows lack a register verdict; run `datagen judge-register` first")
    failed = {
        e.id: [*verdicts[e.id].failed, *(register_verdicts[e.id].failed if e.id in register_verdicts else [])] for e in examples
    }
    flagged = [e for e in examples if failed[e.id]]
    async with task_group() as group:
        tasks = [group.create_task(relabel(llm, e.input)) for e in flagged]
    outputs = [task.result() for task in tasks]
    relabeled = [
        replace(e, output=output, teacher_model=llm.settings.model, tags=[*e.tags, "relabeled"])
        for e, output in zip(flagged, outputs, strict=True)
    ]
    write_examples(RELABELED, relabeled)
    rejudged = await judge_aawaaz(llm, relabeled, RELABEL_JUDGMENTS)
    rejudged_register = await judge_register(llm, relabeled, RELABEL_REGISTER_JUDGMENTS)
    fixed = {
        e.id: e
        for e in relabeled
        if rejudged[e.id].passed and (e.id not in rejudged_register or rejudged_register[e.id].passed)
    }

    kept = [fixed.get(e.id, e) for e in examples if not failed[e.id] or e.id in fixed]
    write_examples(CLEAN, kept)

    per_category: dict[str, Counter] = defaultdict(Counter)
    for e in examples:
        counts = per_category[e.category]
        counts["rows"] += 1
        if failed[e.id]:
            counts["flagged"] += 1
            counts["fixed" if e.id in fixed else "dropped"] += 1
    failed_criteria = Counter(c for criteria in failed.values() for c in criteria)
    report = {
        "judge_model": llm.settings.judge_model,
        "teacher_model": llm.settings.model,
        "rows": len(examples),
        "flagged": len(flagged),
        "fixed": len(fixed),
        "dropped": len(flagged) - len(fixed),
        "kept": len(kept),
        "failed_criteria": dict(failed_criteria),
        "per_category": {category: dict(counts) for category, counts in sorted(per_category.items())},
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
