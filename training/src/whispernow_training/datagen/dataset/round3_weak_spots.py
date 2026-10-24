"""Round 3: train rows for the weak spots models trained on round 2 showed, and fixes to its references.

- Reference fixes: suspect round 2 references reviewed by the teacher and verified blind by the judge
  (`review.FIXES`), applied to its train rows.
- `disfl_qa`: Disfl-QA questions the build never used, from the source's train and validation splits
  (the build's test rows come from its test split), labelled and gated like every reference
  (`gates.gate`), then reviewed against their original question (`review.check`).
- `ami`: single-speaker turns from the meetings in AMI's train split, which share no meeting with the
  long benchmark's turns (AMI's test and validation splits), gated the same way.
- Generated rows of the categories in `generate.RUN_TARGETS[ROUND3_TAG]`, where the round 2 models lost
  more facts than round 1's or than glm-5.3 max on short rows, checked by `second_sample`: a row under
  `TWO_SAMPLE_MAX_WORDS` words with a correction cue gets a second teacher sample.
- Recognizer repair rows (`speech.repair`), which replace their origins' train rows.

Every kept row is tagged round3 and goes to train. Each step resumes from its files.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

from .. import gates, paths
from ..llm import LlmClient
from ..references import review
from ..schema import ROUND3_TAG, Example, append_jsonl, example_id, read_examples, read_jsonl
from ..sources import ami, public
from ..speech import heard, repair
from ..synthetic import generate, second_sample
from . import dedup, filters, shape, stats

DIR = paths.DATA / "weak_spots"
# The pilot's batch only: with its references fixed, Disfl-QA is no weak spot (every model beats glm-5.3 max there).
DISFL_ROWS = 50
DISFL_SEED = 111
AMI_MIN_WORDS = 50
AMI_LONG_WORDS = 150
AMI_MAX_WORDS = 1000
AMI_SHORT_ROWS = 400
AMI_SHORT_PER_MEETING = 3
AMI_SEED = 112
# Rows labelled together; each group's rows are written when it finishes.
GROUP = 50
# The relabel's split: longer transcripts get a judge request each.
JUDGE = gates.Judge(solo_over_words=300)
CATEGORIES = tuple(generate.RUN_TARGETS[ROUND3_TAG])
CHECKS = paths.GENERATED / "round3.checks.jsonl"
CHECK_STEP = "weak-spots check"
TWO_SAMPLE_MAX_WORDS = 100
CORRECTION_CUE = re.compile(
    r"\b(?:actually|no|sorry|i mean|wait|scratch that|or rather|make that|correction|or,? (?:uh|um|er))\b", re.IGNORECASE
)
# Rows generated beyond each category's target, for the build's filters, which dropped 0.5% of this round's rows.
SPARE = 0.02


def rows_file(name: str) -> Path:
    return DIR / f"{name}.jsonl"


def rejected_file(name: str) -> Path:
    return DIR / f"{name}.rejected.jsonl"


def _done(name: str) -> set[str]:
    done = {e.id for e in read_examples(rows_file(name))} if rows_file(name).exists() else set()
    return done | ({row["id"] for row in read_jsonl(rejected_file(name))} if rejected_file(name).exists() else set())


def _used_ids() -> set[str]:
    return {json.loads(line)["id"] for split in ("train", "val", "test")
            for line in (paths.ROUND2 / f"{split}.jsonl").open(encoding="utf-8")}


def disfl_pool() -> list[Example]:
    """`DISFL_ROWS` Disfl-QA questions from the source's train and validation splits that no built row
    uses, none asking the same question as a test row."""
    used = _used_ids()
    examples = public.load_disfl_qa()
    test_questions = {" ".join(review.words(e.output)) for e in examples if e.id in used and e.split == "test"}
    pool = sorted((e for e in examples if e.split != "test" and e.id not in used
                   and " ".join(review.words(e.output)) not in test_questions), key=lambda e: e.id)
    picked = random.Random(DISFL_SEED).sample(pool, min(DISFL_ROWS, len(pool)))
    return [replace(e, output="", split="train", tags=[*e.tags, ROUND3_TAG]) for e in picked]


def bench_meetings() -> set[str]:
    rows = read_examples(paths.BENCH / "long.jsonl")
    return {tag for e in rows for tag in e.tags if tag.startswith("ami:")}


def ami_pool() -> list[Example]:
    """Every train-split turn of `AMI_LONG_WORDS` to `AMI_MAX_WORDS` words, plus `AMI_SHORT_ROWS` turns of
    `AMI_MIN_WORDS` words up to that, at most `AMI_SHORT_PER_MEETING` from a meeting."""
    turns = [t for t in ami.turns(ami.fetch("train")) if AMI_MIN_WORDS <= t["words"] <= AMI_MAX_WORDS]
    rows = [
        Example(
            id=example_id(ami.SOURCE, t["text"]), input=t["text"], output="", source=ami.SOURCE,
            category="real_meeting_turn", license=ami.LICENSE, split="train",
            tags=["real_speech", "style_raw", f"ami:{t['meeting']}", "ami_split:train", ROUND3_TAG],
        )
        for t in turns
    ]
    if overlap := {tag for e in rows for tag in e.tags if tag.startswith("ami:")} & bench_meetings():
        raise SystemExit(f"AMI train turns share meetings with the long benchmark: {sorted(overlap)[:3]}")
    long = [e for e in rows if len(e.input.split()) >= AMI_LONG_WORDS]
    short = sorted((e for e in rows if len(e.input.split()) < AMI_LONG_WORDS), key=lambda e: e.id)
    random.Random(AMI_SEED).shuffle(short)
    per_meeting: Counter = Counter()
    picked = []
    for e in short:
        meeting = next(tag for tag in e.tags if tag.startswith("ami:"))
        if per_meeting[meeting] < AMI_SHORT_PER_MEETING and len(picked) < AMI_SHORT_ROWS:
            per_meeting[meeting] += 1
            picked.append(e)
    return sorted({e.id: e for e in [*long, *picked]}.values(), key=lambda e: e.id)


async def _label_group(llm: LlmClient, name: str, group: Sequence[Example]) -> Counter:
    results = await gates.gate(llm, [(e.id, e.input) for e in group], judge=JUDGE)
    kept, rejected = [], []
    for e in group:
        result = results[e.id]
        if result.passed:
            tags = [*e.tags, *(["relabeled"] if result.attempt else [])]
            kept.append(replace(e, output=result.output, tags=tags, teacher_model=llm.settings.model,
                                teacher_reasoning=llm.settings.reasoning, judge_model=llm.settings.judge_model,
                                judge_reasoning=llm.settings.judge_reasoning))
        else:
            rejected.append({**asdict(e), "failed": result.failed, "note": result.note})
    reviewed = 0
    if name == "disfl_qa":
        checks = [row for e in kept if (row := review.disfl_row(e.id, e.input, e.output))]
        final = {row["id"]: row for row in await review.check(llm, "disfl_qa", checks)}
        reviewed = sum(row["final_output"] != row["output"] for row in final.values())
        kept = [replace(e, output=final[e.id]["final_output"], tags=[*e.tags, "ref_fixed"])
                if e.id in final and final[e.id]["final_output"] != e.output else e for e in kept]
    append_jsonl(rows_file(name), [e.to_json() for e in kept])
    append_jsonl(rejected_file(name), [json.dumps(row, ensure_ascii=False) for row in rejected])
    return Counter(kept=len(kept), rejected=len(rejected), ref_fixed=reviewed)


async def label(llm: LlmClient, name: str, limit: int = 0) -> dict:
    """Labels the `disfl_qa` or `ami` pool's rows not done yet (the first `limit` of them, when given)."""
    pool = disfl_pool() if name == "disfl_qa" else ami_pool()
    done = _done(name)
    todo = [e for e in pool if e.id not in done]
    todo = todo[:limit] if limit else todo
    print(f"{name}: {len(pool)} in the pool, {len(done)} done, labelling {len(todo)}", flush=True)
    counts: Counter = Counter()

    async def one(group: Sequence[Example]) -> None:
        counts.update(await _label_group(llm, name, group))
        print(f"{name}: {dict(counts)}", flush=True)

    await asyncio.gather(*(one(todo[i:i + GROUP]) for i in range(0, len(todo), GROUP)))
    return dict(counts)


def cued(example: Example) -> bool:
    return len(example.input.split()) < TWO_SAMPLE_MAX_WORDS and bool(CORRECTION_CUE.search(example.input))


async def check(llm: LlmClient) -> dict:
    """Checks the round's generated rows not checked yet into `CHECKS`."""
    return await second_sample.check_rows(llm, second_sample.run_rows(ROUND3_TAG, CATEGORIES), CHECKS, cued)


async def top_up(llm: LlmClient) -> dict:
    """Checks the round's generated rows, then generates and checks more until each category has its
    `generate.RUN_TARGETS[ROUND3_TAG]` count of rows the check keeps, plus `SPARE`."""
    return await second_sample.top_up(llm, ROUND3_TAG, generate.RUN_TARGETS[ROUND3_TAG], CHECKS, cued, SPARE, CHECK_STEP)


def load() -> list[Example]:
    """The round's kept rows: labelled Disfl-QA, AMI and repair rows, and the generated rows the check keeps,
    with their fixes. The build cuts the target groups (`capped`, `repair.capped`)."""
    rows = [e for name in ("disfl_qa", "ami") if rows_file(name).exists() for e in read_examples(rows_file(name))]
    rows += repair.load()
    rows += [e for kept in second_sample.checked_run(ROUND3_TAG, CATEGORIES, CHECKS, CHECK_STEP).values() for e in kept]
    return [replace(e, split="train", tags=e.tags if ROUND3_TAG in e.tags else [*e.tags, ROUND3_TAG]) for e in rows]


def capped(rows: Sequence[Example]) -> list[Example]:
    """`rows` with each generated category cut to its target, keeping its earliest rows in file order. The build
    cuts after its filters, so a spare takes a filtered row's place."""
    file_rank = {e.id: i for category in CATEGORIES
                 for i, e in enumerate(second_sample.run_rows(ROUND3_TAG, [category]))}
    groups: dict[str, list[Example]] = defaultdict(list)
    out = []
    for e in rows:
        if e.id in file_rank:
            groups[e.category].append(e)
        else:
            out.append(e)
    for name, members in groups.items():
        out += sorted(members, key=lambda e: file_rank[e.id])[:generate.RUN_TARGETS[ROUND3_TAG][name]]
    return out


def build(base_dir: Path = paths.ROUND2, out_dir: Path = paths.ROUND3) -> dict:
    """Round 2's rows with the verified reference fixes on its train rows, plus the round 3 rows in train.
    val and test are copied; the test fixes are already in round 1's test file.

    A round 3 row is dropped when it fails a filter or duplicates an earlier row; then each generated
    category keeps its first target count (`capped`). A repair row replaces its origin's train row, and
    skips the duplicate check and the input variant. Every other round 2 train line is copied as written.
    """
    base = {split: read_examples(base_dir / f"{split}.jsonl") for split in ("train", "val", "test")}
    kept, dropped = filters.split_by_filters([shape.normalized(e) for e in load()])
    repairs = [e for e in kept if repair.TAG in e.tags]
    kept = [e for e in kept if repair.TAG not in e.tags]
    survivors, duplicates = dedup.deduplicate([*(e for rows in base.values() for e in rows), *kept])
    dropped += [(e, reason) for e, reason in duplicates if ROUND3_TAG in e.tags]
    added = [shape.spoken_variant(e) for e in shape.bare_filler_share(capped([e for e in survivors if ROUND3_TAG in e.tags]))]
    swapped = {heard.origin_id(e): e for e in repair.capped(repairs)}

    fixes = review.verified_fixes("train")
    lines, fixed = [], []
    for row_id, line in stats.read_split_lines(base_dir / "train.jsonl"):
        if row_id in swapped:
            continue
        if row_id in fixes:
            row = json.loads(line)
            example = replace(Example(**row), output=fixes[row_id], tags=[*row["tags"], "ref_fixed"])
            fixed.append(example)
            line = example.to_json()
        lines.append((row_id, line))
    lines = sorted([*lines, *((e.id, e.to_json()) for e in [*added, *swapped.values()])])
    held_out = stats.copy_held_out(base_dir, out_dir)
    (out_dir / "train.jsonl").write_text("".join(line + "\n" for _, line in lines), encoding="utf-8")
    stats.write_dropped(out_dir, dropped)
    fixed_ids = {e.id for e in fixed}
    train = [*fixed, *(e for e in base["train"] if e.id not in fixed_ids and e.id not in swapped)]
    report = stats.stats([*train, *added, *swapped.values(), *held_out["val"], *held_out["test"]], dropped)
    report["round3"] = {
        "added": len(added), "added_by_source": dict(Counter(e.source for e in added)), "references_fixed": len(fixed),
        "repairs_swapped": len(swapped), "dropped": dict(Counter(reason for _, reason in dropped).most_common()),
    }
    counts = Counter("repair" if repair.TAG in e.tags else e.category for e in [*added, *swapped.values()])
    targets = {"repair": repair.ROWS, **generate.RUN_TARGETS[ROUND3_TAG]}
    report["round3"]["targets"] = {name: f"{counts[name]}/{target}" for name, target in targets.items()}
    stats.write_report(out_dir, report, stats.stats_markdown(report) + f"\nround 3: {report['round3']}\n")
    return report


def report() -> dict:
    out: dict = {}
    for name in ("disfl_qa", "ami"):
        kept = read_examples(rows_file(name)) if rows_file(name).exists() else []
        rejected = list(read_jsonl(rejected_file(name))) if rejected_file(name).exists() else []
        out[name] = {"kept": len(kept), "rejected": len(rejected), "ref_fixed": sum("ref_fixed" in e.tags for e in kept),
                     "rejected_for": dict(Counter(f for row in rejected for f in row["failed"]).most_common())}
    records = {row["id"]: row for row in read_jsonl(CHECKS)} if CHECKS.exists() else {}
    by_category: dict[str, Counter] = defaultdict(Counter)
    for e in second_sample.run_rows(ROUND3_TAG, CATEGORIES):
        by_category[e.category][records[e.id]["action"] if e.id in records else "unchecked"] += 1
    out["generated"] = {category: dict(counts) for category, counts in by_category.items()}
    out["repair"] = repair.report()
    return out
