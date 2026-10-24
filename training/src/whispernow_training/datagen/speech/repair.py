"""Recognizer repair rows: what a recognizer wrote, misheard words and all, with a reference that fixes
only what the heard text makes clear.

The pool is the heard records (`heard.records`) whose recognizer misheard a word the restore had to put
back. The teacher labels the raw heard text without seeing what was said, as the noise benchmark's
probed rows are labelled (`noise.gate_probed`), so it can fix only what the text itself shows; the row
is kept only when the judge, reading what was said, finds no lost fact. A mishearing nobody could undo
loses a fact and drops the row. Records are labelled in pool order until `ROWS` are kept, plus `SPARE`,
and round 3 swaps the first `ROWS` that pass its filters in for their origins.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace

from .. import gates, paths
from ..bench import noise
from ..llm import LlmClient
from ..schema import ROUND3_TAG, Example, append_jsonl, example_id, read_examples, read_jsonl
from . import heard

DIR = paths.DATA / "repair"
ROWS_FILE = DIR / "repair.jsonl"
REJECTED_FILE = DIR / "repair.rejected.jsonl"
TAG = "recognizer_repair"
SEED = 113
ROWS = 4000
# Rows labelled beyond the target, for the build's filters, which dropped 0.5% of round 3's rows.
SPARE = 0.02
# Rows labelled together; each group's rows are written when it finishes.
GROUP = 50
# The relabel's split: longer transcripts get a judge request each.
JUDGE = gates.Judge(solo_over_words=300)
FACTS = replace(noise.PROBED_JUDGE, solo_over_words=JUDGE.solo_over_words)


def pool() -> list[tuple[Example, dict]]:
    """Heard records with a restored word, each with its round 2 train origin, in a seeded order."""
    origins = {e.id: e for e in read_examples(paths.ROUND2 / "train.jsonl")}
    pairs = sorted(((origins[r["id"]], r) for r in heard.records() if r["fixes"] and r["id"] in origins),
                   key=lambda pair: pair[0].id)
    random.Random(SEED).shuffle(pairs)
    return pairs


def repair_row(llm: LlmClient, origin: Example, record: dict, output: str, attempt: int) -> Example:
    row = heard.heard_row(origin, record)
    return replace(
        row, id=example_id(heard.SOURCE, record["heard"]), input=record["heard"], output=output,
        tags=[*row.tags, TAG, ROUND3_TAG, *(["relabeled"] if attempt else [])],
        teacher_model=llm.settings.model, teacher_reasoning=llm.settings.reasoning,
        judge_model=llm.settings.judge_model, judge_reasoning=llm.settings.judge_reasoning,
    )


async def _label_group(llm: LlmClient, group: Sequence[tuple[Example, dict]]) -> Counter:
    results = await gates.gate(llm, [(origin.id, record["heard"]) for origin, record in group], judge=JUDGE)
    facts = await gates.check_references(
        llm, [(origin.id, origin.input, results[origin.id].output) for origin, _ in group if results[origin.id].passed],
        FACTS,
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
    append_jsonl(ROWS_FILE, [e.to_json() for e in kept])
    append_jsonl(REJECTED_FILE, [json.dumps(row, ensure_ascii=False) for row in rejected])
    counts = Counter(row["reason"].split(":")[0] for row in rejected)
    counts["kept"] = len(kept)
    return counts


def _done() -> tuple[set[str], set[str]]:
    """Origin ids of the kept repair rows, and of every labelled record."""
    kept = {heard.origin_id(e) for e in read_examples(ROWS_FILE)} if ROWS_FILE.exists() else set()
    rejected = {row["id"] for row in read_jsonl(REJECTED_FILE)} if REJECTED_FILE.exists() else set()
    return kept, kept | rejected


def with_spare(target: int) -> int:
    return math.ceil(target * (1 + SPARE))


async def label(llm: LlmClient, limit: int = 0) -> dict:
    """Labels `pool` records in order until `ROWS` plus `SPARE` are kept, or only the first `limit` not done
    yet, when given. Each wave labels as many records as the keep rate so far says the rest takes."""
    pairs = pool()
    counts: Counter = Counter()

    async def one(group: Sequence[tuple[Example, dict]]) -> None:
        counts.update(await _label_group(llm, group))
        print(f"repair: {dict(counts)}", flush=True)

    while True:
        kept, done = _done()
        todo = [pair for pair in pairs if pair[0].id not in done]
        missing = with_spare(ROWS) - len(kept)
        if not todo or (not limit and missing <= 0):
            break
        rate = len(kept) / len(done) if done else 0.5
        wave = todo[:limit] if limit else todo[:max(GROUP, math.ceil(missing / max(rate, 0.2)))]
        print(f"repair: {len(pairs)} in the pool, {len(done)} done, {len(kept)} kept, labelling {len(wave)}", flush=True)
        await asyncio.gather(*(one(wave[i:i + GROUP]) for i in range(0, len(wave), GROUP)))
        if limit:
            break
    kept, _ = _done()
    if not limit and len(kept) < with_spare(ROWS):
        print(f"repair: the pool ran out at {len(kept)} kept rows; `datagen heard run` hears more origins", flush=True)
    return dict(counts)


def load() -> list[Example]:
    return read_examples(ROWS_FILE) if ROWS_FILE.exists() else []


def capped(rows: Sequence[Example]) -> list[Example]:
    """The first `ROWS` of `rows` in pool order."""
    rank = {origin.id: i for i, (origin, _) in enumerate(pool())}
    return sorted(rows, key=lambda e: rank[heard.origin_id(e)])[:ROWS]


def report() -> dict:
    kept = load()
    rejected = list(read_jsonl(REJECTED_FILE)) if REJECTED_FILE.exists() else []
    return {"kept": len(kept), "target": ROWS, "relabeled": sum("relabeled" in e.tags for e in kept),
            "rejected_for": dict(Counter(row["reason"].split(":")[0] for row in rejected).most_common())}
