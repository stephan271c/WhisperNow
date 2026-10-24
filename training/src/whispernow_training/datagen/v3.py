"""Dataset v3's new train rows, in `data/v3/`. `build.build_v3` adds them to data/build-v2-no-heard.

- `disfl_qa`: Disfl-QA questions the build never used, from the source's train and validation splits
  (the build's test rows come from its test split), labelled and gated like every reference
  (`gates.gate`), then reviewed against their original question (`review.check`).
- `ami`: single-speaker turns from the meetings in AMI's train split, which share no meeting with the
  long benchmark's turns (AMI's test and validation splits), gated the same way.
- stage 3 rows of the categories in `generate.RUN_TARGETS["v3"]` (`generate.generate`, run `v3`), checked by `check`:
  a line that starts with a capitalized command goes to `review.check`, and a row under
  `TWO_SAMPLE_MAX_WORDS` words with a correction cue gets a second teacher sample. Two samples that
  keep different words mean the transcript does not settle what the speaker meant, and the row is dropped.
  `top_up` generates replacements until each category has its target count of rows the check keeps, plus
  `SPARE`; the build keeps each category's first target count that pass its filters.
- `repair`: the raw recognizer text of heard records (`heard.records`) whose recognizer misheard a word
  the restore had to put back, labelled from that text like the noise benchmark's probed rows
  (`noise.gate_probed`): the teacher never sees what was said, so it can fix only what the heard text
  makes clear, and the row is kept only when the judge, reading what was said, finds no lost fact. A
  mishearing nobody could undo loses a fact and drops the row. Records are labelled until `REPAIR_ROWS` are
  kept, plus `SPARE`, and the build swaps the first `REPAIR_ROWS` that pass its filters in for their origins.

Every kept row is tagged v3 and goes to train. Each step resumes from its files.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from pathlib import Path

from . import ami, gates, generate, heard, noise, paths, review, sources
from .llm import LlmClient
from .schema import V2_TAG, V3_TAG, Example, append_jsonl, example_id, read_examples, read_jsonl
from .teacher import relabel

DIR = paths.DATA / "v3"
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
STAGE3_CATEGORIES = tuple(generate.RUN_TARGETS[V3_TAG])
TWO_SAMPLE_MAX_WORDS = 100
CORRECTION_CUE = re.compile(
    r"\b(?:actually|no|sorry|i mean|wait|scratch that|or rather|make that|correction|or,? (?:uh|um|er))\b", re.IGNORECASE
)
# Words two samples may differ in without keeping different content.
_MINOR = frozenset(
    "a an the and or but of to in on at for with by from is are was were be been it its this that these those "
    "so then also just please".split()
)


def rows_file(name: str) -> Path:
    return DIR / f"{name}.jsonl"


def rejected_file(name: str) -> Path:
    return DIR / f"{name}.rejected.jsonl"


def _done(name: str) -> set[str]:
    done = {e.id for e in read_examples(rows_file(name))} if rows_file(name).exists() else set()
    return done | ({row["id"] for row in read_jsonl(rejected_file(name))} if rejected_file(name).exists() else set())


def _used_ids() -> set[str]:
    return {json.loads(line)["id"] for split in ("train", "val", "test")
            for line in (paths.BUILD_V2_NO_HEARD / f"{split}.jsonl").open(encoding="utf-8")}


def disfl_pool() -> list[Example]:
    """`DISFL_ROWS` Disfl-QA questions from the source's train and validation splits that no built row
    uses, none asking the same question as a test row."""
    used = _used_ids()
    examples = sources.load_disfl_qa()
    test_questions = {" ".join(review.words(e.output)) for e in examples if e.id in used and e.split == "test"}
    pool = sorted((e for e in examples if e.split != "test" and e.id not in used
                   and " ".join(review.words(e.output)) not in test_questions), key=lambda e: e.id)
    picked = random.Random(DISFL_SEED).sample(pool, min(DISFL_ROWS, len(pool)))
    return [replace(e, output="", split="train", tags=[*e.tags, V3_TAG]) for e in picked]


def bench_meetings() -> set[str]:
    rows = read_examples(paths.DATA / "bench" / "long.jsonl")
    return {tag for e in rows for tag in e.tags if tag.startswith("ami:")}


def ami_pool() -> list[Example]:
    """Every train-split turn of `AMI_LONG_WORDS` to `AMI_MAX_WORDS` words, plus `AMI_SHORT_ROWS` turns of
    `AMI_MIN_WORDS` words up to that, at most `AMI_SHORT_PER_MEETING` from a meeting."""
    turns = [t for t in ami.turns(ami.fetch("train")) if AMI_MIN_WORDS <= t["words"] <= AMI_MAX_WORDS]
    rows = [
        Example(
            id=example_id(ami.SOURCE, t["text"]), input=t["text"], output="", source=ami.SOURCE,
            category="real_meeting_turn", license=ami.LICENSE, split="train",
            tags=["real_speech", "style_raw", f"ami:{t['meeting']}", "ami_split:train", V3_TAG],
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


REPAIR = "repair"
REPAIR_TAG = "recognizer_repair"
REPAIR_SEED = 113
REPAIR_ROWS = 4000
# Rows labelled beyond each target, for the build's filters, which dropped 0.5% of the v3 rows.
SPARE = 0.02
REPAIR_FACTS = replace(noise.PROBED_JUDGE, solo_over_words=JUDGE.solo_over_words)


def repair_pool() -> list[tuple[Example, dict]]:
    """Heard records with a restored word, each with its train origin, in a seeded order."""
    origins = {e.id: e for e in read_examples(paths.BUILD_V2_NO_HEARD / "train.jsonl")}
    pairs = sorted(((origins[r["id"]], r) for r in heard.records() if r["fixes"] and r["id"] in origins),
                   key=lambda pair: pair[0].id)
    random.Random(REPAIR_SEED).shuffle(pairs)
    return pairs


def repair_row(llm: LlmClient, origin: Example, record: dict, output: str, attempt: int) -> Example:
    row = heard.heard_row(origin, record)
    tags = [tag for tag in row.tags if tag != V2_TAG]
    return replace(
        row, id=example_id(heard.SOURCE, record["heard"]), input=record["heard"], output=output,
        tags=[*tags, REPAIR_TAG, V3_TAG, *(["relabeled"] if attempt else [])],
        teacher_model=llm.settings.model, teacher_reasoning=llm.settings.reasoning,
        judge_model=llm.settings.judge_model, judge_reasoning=llm.settings.judge_reasoning,
    )


async def _repair_group(llm: LlmClient, group: Sequence[tuple[Example, dict]]) -> Counter:
    results = await gates.gate(llm, [(origin.id, record["heard"]) for origin, record in group], judge=JUDGE)
    facts = await gates.check_references(
        llm, [(origin.id, origin.input, results[origin.id].output) for origin, _ in group if results[origin.id].passed],
        REPAIR_FACTS,
    )
    kept, rejected = [], []
    for origin, record in group:
        result = results[origin.id]
        if not result.passed:
            reason = "gates: " + ",".join(result.failed)
        elif not facts[origin.id].passed:
            reason = "recognizer changed a fact: " + facts[origin.id].note
        else:
            kept.append(repair_row(llm, origin, record, result.output, result.attempt))
            continue
        rejected.append({"id": origin.id, "heard": record["heard"], "output": result.output, "reason": reason})
    append_jsonl(rows_file(REPAIR), [e.to_json() for e in kept])
    append_jsonl(rejected_file(REPAIR), [json.dumps(row, ensure_ascii=False) for row in rejected])
    counts = Counter(row["reason"].split(":")[0] for row in rejected)
    counts["kept"] = len(kept)
    return counts


def _repair_done() -> tuple[set[str], set[str]]:
    """Origin ids of the kept repair rows, and of every labelled record."""
    kept = {heard.origin_id(e) for e in read_examples(rows_file(REPAIR))} if rows_file(REPAIR).exists() else set()
    rejected = {row["id"] for row in read_jsonl(rejected_file(REPAIR))} if rejected_file(REPAIR).exists() else set()
    return kept, kept | rejected


async def repair(llm: LlmClient, limit: int = 0) -> dict:
    """Labels `repair_pool` records in order until `REPAIR_ROWS` plus `SPARE` are kept, or only the first `limit` not done
    yet, when given. Each wave labels as many records as the keep rate so far says the rest takes."""
    pool = repair_pool()
    counts: Counter = Counter()

    async def one(group: Sequence[tuple[Example, dict]]) -> None:
        counts.update(await _repair_group(llm, group))
        print(f"{REPAIR}: {dict(counts)}", flush=True)

    while True:
        kept, done = _repair_done()
        todo = [pair for pair in pool if pair[0].id not in done]
        missing = with_spare(REPAIR_ROWS) - len(kept)
        if not todo or (not limit and missing <= 0):
            break
        rate = len(kept) / len(done) if done else 0.5
        wave = todo[:limit] if limit else todo[:max(GROUP, math.ceil(missing / max(rate, 0.2)))]
        print(f"{REPAIR}: {len(pool)} in the pool, {len(done)} done, {len(kept)} kept, labelling {len(wave)}", flush=True)
        await asyncio.gather(*(one(wave[i:i + GROUP]) for i in range(0, len(wave), GROUP)))
        if limit:
            break
    kept, _ = _repair_done()
    if not limit and len(kept) < with_spare(REPAIR_ROWS):
        print(f"{REPAIR}: the pool ran out at {len(kept)} kept rows; `datagen heard run` hears more origins", flush=True)
    return dict(counts)


def with_spare(target: int) -> int:
    return math.ceil(target * (1 + SPARE))


def capped(rows: Sequence[Example]) -> list[Example]:
    """`rows` with each target group cut to its target, keeping its earliest rows: the repair rows
    (`REPAIR_ROWS`, in pool order) and each stage 3 category of `generate.RUN_TARGETS["v3"]` (in file order).
    The build cuts after its filters, so a spare takes a filtered row's place."""
    pool_rank = {origin.id: i for i, (origin, _) in enumerate(repair_pool())}
    file_rank = {e.id: i for category in STAGE3_CATEGORIES if generate.output_file(category, run=V3_TAG).exists()
                 for i, e in enumerate(read_examples(generate.output_file(category, run=V3_TAG)))}
    targets = {REPAIR: REPAIR_ROWS, **generate.RUN_TARGETS[V3_TAG]}
    groups: dict[str, list[Example]] = defaultdict(list)
    out = []
    for e in rows:
        if REPAIR_TAG in e.tags:
            groups[REPAIR].append(e)
        elif e.id in file_rank:
            groups[e.category].append(e)
        else:
            out.append(e)
    for name, members in groups.items():
        rank = (lambda e: pool_rank[heard.origin_id(e)]) if name == REPAIR else (lambda e: file_rank[e.id])
        out += sorted(members, key=rank)[:targets[name]]
    return out


def two_sample_disagreement(transcript: str, first: str, second: str) -> list[str]:
    """Words of the transcript that one sample keeps and the other leaves out."""
    spoken = set(review.words(transcript))
    a, b = set(review.words(first)), set(review.words(second))
    return sorted(w for w in a ^ b if w in spoken and w not in _MINOR and not w.isdigit())


def stage3_rows() -> list[Example]:
    return [e for category in STAGE3_CATEGORIES if generate.output_file(category, run=V3_TAG).exists()
            for e in read_examples(generate.output_file(category, run=V3_TAG))]


CHECKS = DIR / "stage3.checks.jsonl"


def cued(example: Example) -> bool:
    return len(example.input.split()) < TWO_SAMPLE_MAX_WORDS and bool(CORRECTION_CUE.search(example.input))


async def check(llm: LlmClient) -> dict:
    """Checks the stage 3 v3 rows not checked yet into `CHECKS`."""
    return await check_rows(llm, stage3_rows(), CHECKS, cued)


async def check_rows(llm: LlmClient, rows: Sequence[Example], checks: Path, second_sample: Callable[[Example], bool]) -> dict:
    """Checks the `rows` not in `checks` yet into it: one record per row, `keep`, `fix` (with the fixed
    reference) or `drop` (with the reason). A row `second_sample` picks is labelled again and dropped when the
    two samples keep different words."""
    done = {row["id"] for row in read_jsonl(checks)} if checks.exists() else set()
    rows = [e for e in rows if e.id not in done]
    cases = [row for e in rows if (row := review.case_row(e.id, e.input, e.output))]
    case_results = {row["id"]: row for row in await review.check(llm, "literal_case", cases)}
    asked = [e for e in rows if second_sample(e)]
    seconds = await asyncio.gather(*(relabel(llm, e.input, attempt=1) for e in asked))
    second = {e.id: text for e, text in zip(asked, seconds, strict=True)}
    records = []
    for e in rows:
        output = case_results[e.id]["final_output"] if e.id in case_results else e.output
        record = {"id": e.id, "action": "fix" if output != e.output else "keep", "output": output}
        if e.id in second and (differ := two_sample_disagreement(e.input, e.output, second[e.id])):
            record = {"id": e.id, "action": "drop", "reason": f"teacher samples keep different words: {', '.join(differ)}",
                      "second": second[e.id]}
        records.append(record)
    append_jsonl(checks, [json.dumps(r, ensure_ascii=False) for r in records])
    return {"checked": len(records), **Counter(r["action"] for r in records), "two_sample_checked": len(asked)}


def checked_rows(rows: Sequence[Example], checks: Path, check_step: str) -> list[Example]:
    """The `rows` the check keeps, fixed where it fixed them, in their order."""
    records = {row["id"]: row for row in read_jsonl(checks)} if checks.exists() else {}
    if unchecked := [e.id for e in rows if e.id not in records]:
        raise SystemExit(f"{len(unchecked)} rows are not checked yet; run `datagen {check_step}`")
    return [
        replace(e, output=records[e.id]["output"], tags=[*e.tags, "ref_fixed"]) if records[e.id]["action"] == "fix" else e
        for e in rows if records[e.id]["action"] != "drop"
    ]


def checked_stage3() -> dict[str, list[Example]]:
    """Each stage 3 v3 category's rows the check keeps, fixed where it fixed them, in file order."""
    out: dict[str, list[Example]] = {}
    for category in STAGE3_CATEGORIES:
        path = generate.output_file(category, run=V3_TAG)
        out[category] = checked_rows(read_examples(path) if path.exists() else [], CHECKS, "v3 check")
    return out


async def top_up(llm: LlmClient) -> dict:
    """Checks the stage 3 v3 rows, then generates and checks more until each category has its
    `generate.RUN_TARGETS["v3"]` count of rows the check keeps, plus `SPARE`."""
    while True:
        await check(llm)
        kept = checked_stage3()
        short = {category: with_spare(target) - len(kept[category])
                 for category, target in generate.RUN_TARGETS[V3_TAG].items() if len(kept[category]) < with_spare(target)}
        if not short:
            return {category: len(rows) for category, rows in kept.items()}
        print(f"stage 3 v3 rows short of their targets: {short}", flush=True)
        await asyncio.gather(*(
            generate.generate_category(llm, category, len(read_examples(generate.output_file(category, run=V3_TAG))) + missing, run=V3_TAG)
            for category, missing in short.items()
        ))


def load() -> list[Example]:
    """The kept v3 rows: labelled Disfl-QA, AMI and repair rows, and the stage 3 rows the check keeps, with
    their fixes. The build cuts the target groups (`capped`)."""
    rows = [e for name in ("disfl_qa", "ami", REPAIR) if rows_file(name).exists() for e in read_examples(rows_file(name))]
    rows += [e for kept in checked_stage3().values() for e in kept]
    return [replace(e, split="train", tags=e.tags if V3_TAG in e.tags else [*e.tags, V3_TAG]) for e in rows]


def report() -> dict:
    out: dict = {}
    for name in ("disfl_qa", "ami"):
        kept = read_examples(rows_file(name)) if rows_file(name).exists() else []
        rejected = list(read_jsonl(rejected_file(name))) if rejected_file(name).exists() else []
        out[name] = {"kept": len(kept), "rejected": len(rejected), "ref_fixed": sum("ref_fixed" in e.tags for e in kept),
                     "rejected_for": dict(Counter(f for row in rejected for f in row["failed"]).most_common())}
    generated = stage3_rows()
    records = {row["id"]: row for row in read_jsonl(CHECKS)} if CHECKS.exists() else {}
    by_category: dict[str, Counter] = defaultdict(Counter)
    for e in generated:
        by_category[e.category][records[e.id]["action"] if e.id in records else "unchecked"] += 1
    out["stage3"] = {category: dict(counts) for category, counts in by_category.items()}
    kept = read_examples(rows_file(REPAIR)) if rows_file(REPAIR).exists() else []
    rejected = list(read_jsonl(rejected_file(REPAIR))) if rejected_file(REPAIR).exists() else []
    out[REPAIR] = {"kept": len(kept), "target": REPAIR_ROWS, "relabeled": sum("relabeled" in e.tags for e in kept),
                   "rejected_for": dict(Counter(row["reason"].split(":")[0] for row in rejected).most_common())}
    return out
