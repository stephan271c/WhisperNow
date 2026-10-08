"""Re-judges rows gated before the header rules reached the judges, and checks the long benchmark's
references once more, one row per request.

The judges now say how a cleanup treats a header: a lead-in, title or group label before a list is
optional, but a header the speaker said must be kept and one that misnames what it heads is wrong.
Rows gated earlier were judged without those rules, so the rows they can affect (a reference with
a list or a label, or a transcript that opens with a header-like phrase) go through the gates again
with their existing reference: a failure is relabelled once and dropped if it fails again.

The generators append to their files while this runs, so verdicts go to `<dir>/rejudged.jsonl`
and the loaders apply them (`apply`): a failed row is left out, a relabelled one gets its new
reference. `<dir>/rejudge.cutoff.json` holds each file's row count when the rules changed; rows
written later were judged under the new rules and are skipped. A directory without one (noise, AMI)
was complete before the change.

The strict pass (`strict.jsonl`) came from the long benchmark's hand read: 13 of 54 references the
gates had passed were wrong, 9 of the 14 over 600 words (a digit string misread, a late correction
left the old value standing, an attributed hedge flattened). The judges learned those cases, and the
pass re-gates every row of both benchmarks with the completeness judge, one row per request, and a
row whose teacher references fail twice gets references from the judge's model (`rescue`). The
noise benchmark's verdicts cover its assembled rows, since a probed row has its own id and reference.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from pathlib import Path

from . import gates
from .schema import Example, read_examples, read_jsonl

_LIST_OR_LABEL = re.compile(r"^(?:- |\d+\. )|:\s*$", re.MULTILINE)
_SPOKEN_HEADER = re.compile(
    r"^[^.?!]{0,40}?\b(?:grocery|shopping|packing|to-?do|errands?|tasks?|checklist|agenda|meeting notes|notes|list)\b",
    re.IGNORECASE,
)
CHUNK = 60
# Override files, applied in this order; each name is also the tag a row it covers gets.
PASSES = ("rejudged", "strict", "codex")
# The strict pass relabels as teacher attempt 3, a sample no earlier gate drew.
STRICT_FIRST_ATTEMPT = 2


def overrides_file(directory: Path, name: str = "rejudged") -> Path:
    return directory / f"{name}.jsonl"


def affected(example: Example) -> bool:
    """Whether the header rules can change this row's verdict."""
    return bool(_LIST_OR_LABEL.search(example.output) or _SPOKEN_HEADER.search(example.input[:160]))


def apply(rows: Sequence[Example], directory: Path, passes: Sequence[str] = PASSES) -> list[Example]:
    kept = list(rows)
    for name in passes:
        path = overrides_file(directory, name)
        if not path.exists():
            continue
        verdicts = {row["id"]: row for row in read_jsonl(path)}
        rows, kept = kept, []
        for e in rows:
            verdict = verdicts.get(e.id)
            if verdict is None:
                kept.append(e)
            elif verdict["passed"]:
                relabeled = verdict["output"] != e.output
                tags = [*e.tags, name, *(["relabeled"] if relabeled and "relabeled" not in e.tags else [])]
                if "teacher_model" in verdict:
                    tags.append("judge_model_reference")
                teacher_model = verdict.get("teacher_model") or (relabeled and verdict.get("relabel_model")) or e.teacher_model
                kept.append(replace(e, output=verdict["output"], tags=tags, teacher_model=teacher_model))
    return kept


def before_cutoff(directory: Path, files: Sequence[Path]) -> list[Example]:
    """The rows of `files` written before the rules changed, per `rejudge.cutoff.json` (all rows without one)."""
    cutoff_file = directory / "rejudge.cutoff.json"
    cutoff = json.loads(cutoff_file.read_text(encoding="utf-8")) if cutoff_file.exists() else None
    rows = []
    for path in files:
        if path.exists():
            examples = read_examples(path)
            rows += examples if cutoff is None else examples[: cutoff.get(path.name, 0)]
    return rows


def _style(example: Example) -> str:
    return next((t.removeprefix("style_") for t in example.tags if t.startswith("style_")), "recognizer")


def _styled(judge: gates.Judge, rows: Sequence[Example]) -> gates.Judge:
    return replace(judge, styles={e.id: _style(e) for e in rows}) if judge.instructions == gates.COMPOSE else judge


async def rejudge(
    llm,
    directory: Path,
    rows: Sequence[Example],
    judge: gates.Judge,
    *,
    name: str = "rejudged",
    select: Callable[[Example], bool] = affected,
    first_attempt: int | None = None,
    rescue: bool = False,
    relabel_role: str | None = None,
    teacher_reasoning: str | None = None,
) -> Counter:
    """Gates the `select`ed `rows` again with their references; appends verdicts to the pass's file.

    A relabel is a fresh teacher sample: after attempt 1 for rows relabelled before, unless
    `first_attempt` sets one for every row. With `rescue`, a row that fails twice is labelled by the
    judge's model instead, with the same hosted prompt, through the same gates, before it is dropped:
    the hardest rows (a late correction the teacher only appends) stay in the benchmark.
    A relabel by the teacher role is recorded with the teacher's model (`relabel_model`).
    """
    relabel_model = llm.settings.judge_model if relabel_role == "judge" else llm.settings.model
    path = overrides_file(directory, name)
    records = {row["id"]: row for row in read_jsonl(path)} if path.exists() else {}
    todo = [e for e in rows if select(e) and e.id not in records]
    counts: Counter = Counter(skipped=sum(not select(e) for e in rows), done_before=len(records))
    if rescue:
        # Rows an earlier run dropped without trying the judge's model.
        failed = [e for e in rows if e.id in records and not records[e.id]["passed"] and "teacher_model" not in records[e.id]]
        if failed:
            await _rescue(llm, path, failed, judge, counts)
    for start in range(0, len(todo), CHUNK):
        chunk = todo[start : start + CHUNK]
        results: dict[str, gates.Gated] = {}
        for attempt in (0, 1) if first_attempt is None else (first_attempt,):
            group = [e for e in chunk if first_attempt is not None or ("relabeled" in e.tags) == bool(attempt)]
            if group:
                results |= await gates.gate(
                    llm, [(e.id, e.input) for e in group], {e.id: e.output for e in group}, attempt, _styled(judge, group),
                    relabel_role=relabel_role, teacher_reasoning=teacher_reasoning,
                )
        with path.open("a", encoding="utf-8") as sink:
            for e in chunk:
                result = results[e.id]
                sink.write(json.dumps(asdict(result) | {"relabel_model": relabel_model}, ensure_ascii=False) + "\n")
                outcome = "kept" if result.passed and result.output == e.output else "relabeled" if result.passed else "failed"
                counts[outcome if rescue or outcome != "failed" else "dropped"] += 1
                if not result.passed or result.output != e.output:
                    counts.update(f"first_failed:{c}" for c in result.first_failed)
        failed = [e for e in chunk if not results[e.id].passed]
        if rescue and failed:
            await _rescue(llm, path, failed, judge, counts)
    return counts


async def _rescue(llm, path: Path, rows: Sequence[Example], judge: gates.Judge, counts: Counter) -> None:
    """Labels and gates `rows` with the judge's model writing the references; appends its verdicts."""
    results = await gates.gate(llm, [(e.id, e.input) for e in rows], judge=_styled(judge, rows), teacher_role="judge")
    with path.open("a", encoding="utf-8") as sink:
        for e in rows:
            result = results[e.id]
            sink.write(json.dumps(asdict(result) | {"teacher_model": llm.settings.judge_model}, ensure_ascii=False) + "\n")
            counts["rescued" if result.passed else "dropped"] += 1


async def run(llm, target: str) -> dict:
    from . import bench, compose, noise

    if target == "train":
        directory = compose.TRAIN.out_dir
        files = sorted(p for p in directory.glob("*.jsonl") if p.name.count(".") == 2)
        return dict(await rejudge(llm, directory, before_cutoff(directory, files), gates.Judge(gates.COMPOSE, gates.COMPOSE_CRITERIA)))
    if target == "bench-long":
        directory = compose.BENCH.out_dir
        files = sorted(p for p in directory.glob("*.jsonl") if p.name.count(".") == 2)
        composed = await rejudge(llm, directory, before_cutoff(directory, files), gates.Judge(gates.COMPOSE, gates.COMPOSE_CRITERIA))
        ami = await rejudge(llm, directory, read_examples(bench.AMI_FILE), bench.AMI_JUDGE)
        return {"composed": dict(composed), "ami": dict(ami)}
    if target == "bench-long-strict":
        directory = compose.BENCH.out_dir
        # The rows as the strict pass found them: its own verdicts are not applied.
        before = ("rejudged",)
        rows = [*compose.load(compose.BENCH, before), *apply(read_examples(bench.AMI_FILE), directory, before)]
        judge = replace(gates.COMPLETENESS_JUDGE, max_items=1)
        counts = await rejudge(
            llm, directory, rows, judge, name="strict", select=lambda _: True, first_attempt=STRICT_FIRST_ATTEMPT,
            rescue=True,
        )
        return dict(counts)
    if target == "train-strict":
        # The rows generated before the cascade (`compose.RELABEL_ROLE`), checked as compose now checks
        # a new row: the compose judge, one row per request at its bucket's reasoning, and a failure
        # relabelled by the judge's model.
        directory = compose.TRAIN.out_dir
        rows = [e for e in compose.load(compose.TRAIN, ("rejudged",)) if e.teacher_model != llm.settings.model]
        groups: dict[str | None, list[Example]] = {}
        for e in rows:
            groups.setdefault(compose.JUDGE_REASONING.get(compose.file_bucket(len(e.input.split()))), []).append(e)
        counts = Counter()
        for reasoning, group in groups.items():
            judge = gates.Judge(gates.COMPOSE, gates.COMPOSE_CRITERIA, max_items=1, reasoning=reasoning)
            counts += await rejudge(
                llm, directory, group, judge, name="strict", select=lambda _: True,
                first_attempt=STRICT_FIRST_ATTEMPT, relabel_role="judge",
            )
        return dict(counts)
    if target == "train-codex":
        # The rows whose references glm-5.3 wrote, gated as compose gates a new row once gpt-6.1-sol
        # became teacher and judge: a failure gets one gpt-6.1-sol reference at its bucket's reasoning.
        directory = compose.TRAIN.out_dir
        rows = [e for e in compose.load(compose.TRAIN, ("rejudged", "strict")) if e.teacher_model != llm.settings.model]
        by_bucket: dict[str | None, list[Example]] = {}
        for e in rows:
            by_bucket.setdefault(compose.file_bucket(len(e.input.split())), []).append(e)
        counts = Counter()
        for bucket, group in by_bucket.items():
            judge = gates.Judge(gates.COMPOSE, gates.COMPOSE_CRITERIA, max_items=1, reasoning=compose.JUDGE_REASONING.get(bucket))
            counts += await rejudge(
                llm, directory, group, judge, name="codex", select=lambda _: True, first_attempt=0,
                relabel_role="teacher", teacher_reasoning=compose.TEACHER_REASONING.get(bucket),
            )
        return dict(counts)
    if target == "noise-strict":
        judge = replace(gates.COMPLETENESS_JUDGE, max_items=1)
        rows = read_examples(noise.BENCH_FILE)
        counts = await rejudge(
            llm, noise.OUT_DIR, rows, judge, name="strict", select=lambda _: True, first_attempt=STRICT_FIRST_ATTEMPT,
            rescue=True,
        )
        return dict(counts)
    if target == "noise":
        return dict(await rejudge(llm, noise.OUT_DIR, noise.load_raw(), gates.Judge(gates.COMPOSE, gates.COMPOSE_CRITERIA)))
    raise ValueError(target)
