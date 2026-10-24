"""Dataset v5's new train rows, in `data/v5/`: stage 3 rows of the categories in `generate.RUN_TARGETS["v5"]`
(`generate.generate`, run `v5`), short dictations that use a correction word literally ("hold on to the
receipts"). Every row gets a second teacher sample (`v3.check_rows`), and two samples that keep different words
drop it, since the transcript then does not settle whether the word was meant. `top_up` generates replacements
until each category has its target count of rows the check keeps, plus `SPARE`. `build.build_v5` adds them
to train and keeps each category's first target count that pass its filters.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import replace

from . import generate, paths, v3
from .llm import LlmClient
from .schema import V5_TAG, Example, read_examples, read_jsonl

DIR = paths.DATA / "v5"
CHECKS = DIR / "stage3.checks.jsonl"
CATEGORIES = tuple(generate.RUN_TARGETS[V5_TAG])
# Spare kept rows over each target: the build drops near-duplicates and rows that share a phrase with a held-out
# input (3 and 14 of 630).
SPARE = 0.05


def stage3_rows() -> list[Example]:
    return [e for category in CATEGORIES if generate.output_file(category, run=V5_TAG).exists()
            for e in read_examples(generate.output_file(category, run=V5_TAG))]


async def check(llm: LlmClient) -> dict:
    DIR.mkdir(parents=True, exist_ok=True)
    return await v3.check_rows(llm, stage3_rows(), CHECKS, lambda e: True)


def checked() -> dict[str, list[Example]]:
    out = {}
    for category in CATEGORIES:
        path = generate.output_file(category, run=V5_TAG)
        out[category] = v3.checked_rows(read_examples(path) if path.exists() else [], CHECKS, "v5 check")
    return out


async def top_up(llm: LlmClient) -> dict:
    """Checks the v5 rows, then generates and checks more until each category has its target of kept rows."""
    while True:
        await check(llm)
        kept = checked()
        short = {category: with_spare(target) - len(kept[category])
                 for category, target in generate.RUN_TARGETS[V5_TAG].items() if len(kept[category]) < with_spare(target)}
        if not short:
            return {category: len(rows) for category, rows in kept.items()}
        print(f"v5 rows short of their targets: {short}", flush=True)
        for category, missing in short.items():
            done = len(read_examples(generate.output_file(category, run=V5_TAG)))
            await generate.generate_category(llm, category, done + missing, run=V5_TAG)


def with_spare(target: int) -> int:
    return math.ceil(target * (1 + SPARE))


def load() -> list[Example]:
    return [replace(e, split="train") for rows in checked().values() for e in rows]


def capped(rows: list[Example]) -> list[Example]:
    """Each category's first target count of `rows`, in file order."""
    rank = {e.id: i for i, e in enumerate(stage3_rows())}
    groups: dict[str, list[Example]] = defaultdict(list)
    for e in rows:
        groups[e.category].append(e)
    return [e for category, members in groups.items()
            for e in sorted(members, key=lambda e: rank[e.id])[: generate.RUN_TARGETS[V5_TAG][category]]]


def report() -> dict:
    records = {row["id"]: row for row in read_jsonl(CHECKS)} if CHECKS.exists() else {}
    by_category: dict[str, Counter] = defaultdict(Counter)
    reasons: Counter = Counter()
    for e in stage3_rows():
        record = records.get(e.id)
        by_category[e.category][record["action"] if record else "unchecked"] += 1
        if record and record["action"] == "drop":
            reasons[record["reason"].split(":")[0]] += 1
    lengths = Counter("<=25" if len(e.input.split()) <= 25 else "26-60" if len(e.input.split()) <= 60 else ">60"
                      for e in stage3_rows())
    tags = Counter(t for e in stage3_rows() for t in e.tags if not t.startswith("style_") and t != V5_TAG)
    return {"by_category": {c: dict(n) for c, n in by_category.items()}, "dropped_for": dict(reasons),
            "words": dict(lengths), "tags": dict(tags.most_common())}
