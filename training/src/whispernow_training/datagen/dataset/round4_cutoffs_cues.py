"""Round 4: cut-off word starts in round 3's train inputs, and short dictations that use a correction word
literally. It writes the dataset (`paths.BUILD`).

- Cut-off word starts ("interf interface", `speech.cutoff`): `cutoff.ROWS` round 3 train rows get their
  input with one to three cut-off pieces, keeping their id and reference.
- Literal correction words ("hold on to the receipts"): generated rows of the categories in
  `generate.RUN_TARGETS[ROUND4_TAG]`. Every row gets a second teacher sample (`second_sample`), and two
  samples that keep different words drop it, since the transcript then does not settle whether the word
  was meant. `top_up` generates replacements until each category has its target count of kept rows,
  plus `SPARE`; the build keeps each category's first target count that pass its filters.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path

from .. import paths
from ..llm import LlmClient
from ..schema import ROUND4_TAG, Example, read_examples, read_jsonl, write_examples
from ..speech import cutoff
from ..synthetic import generate, second_sample
from . import dedup, filters, shape, stats

CATEGORIES = tuple(generate.RUN_TARGETS[ROUND4_TAG])
CHECKS = paths.GENERATED / "round4.checks.jsonl"
CHECK_STEP = "literal-cue check"
# Spare kept rows over each target: the build drops near-duplicates and rows that share a phrase with a held-out
# input (3 and 14 of 630).
SPARE = 0.05
# A generated row sharing this many consecutive words with a val, test or benchmark input is dropped, so a
# phrase the eval asks about is not taught by name.
HELD_OUT_RUN = 6


def _every_row(example: Example) -> bool:
    return True


async def check(llm: LlmClient) -> dict:
    return await second_sample.check_rows(llm, second_sample.run_rows(ROUND4_TAG, CATEGORIES), CHECKS, _every_row)


async def top_up(llm: LlmClient) -> dict:
    """Checks the round's generated rows, then generates and checks more until each category has its target of kept rows."""
    return await second_sample.top_up(llm, ROUND4_TAG, generate.RUN_TARGETS[ROUND4_TAG], CHECKS, _every_row, SPARE, CHECK_STEP)


def load() -> list[Example]:
    kept = second_sample.checked_run(ROUND4_TAG, CATEGORIES, CHECKS, CHECK_STEP)
    return [replace(e, split="train") for rows in kept.values() for e in rows]


def capped(rows: list[Example]) -> list[Example]:
    """Each category's first target count of `rows`, in file order."""
    rank = {e.id: i for i, e in enumerate(second_sample.run_rows(ROUND4_TAG, CATEGORIES))}
    groups: dict[str, list[Example]] = defaultdict(list)
    for e in rows:
        groups[e.category].append(e)
    return [e for category, members in groups.items()
            for e in sorted(members, key=lambda e: rank[e.id])[: generate.RUN_TARGETS[ROUND4_TAG][category]]]


def _word_runs(texts: Iterable[str]) -> set[str]:
    runs = set()
    for text in texts:
        words = re.findall(r"[a-z0-9']+", text.casefold())
        runs |= {" ".join(words[i : i + HELD_OUT_RUN]) for i in range(len(words) - HELD_OUT_RUN + 1)}
    return runs


def _benchmark_rows() -> list[Example]:
    return [e for name in ("long", "noise", "cutoff") if (path := paths.BENCH / f"{name}.jsonl").exists()
            for e in read_examples(path)]


def _cutoff_records() -> dict[str, dict]:
    cut = {record["id"]: record for record in cutoff.load()}
    if len(cut) < cutoff.ROWS:
        raise SystemExit(f"{len(cut)} of {cutoff.ROWS} cut-off rows kept; run `datagen cutoff top-up` first")
    return cut


def _swap_cutoff(rows: Sequence[Example], cut: dict[str, dict]) -> tuple[list[Example], Counter]:
    """`rows` with the cut-off records' inputs swapped in, and the swapped rows' length buckets."""
    out, lengths = [], Counter()
    for e in rows:
        record = cut.get(e.id)
        if record is not None:
            lengths[cutoff.bucket(e)] += 1
            e = replace(e, input=record["input"], tags=[*e.tags, cutoff.TAG, f"tts:{record['tts']}"])
        out.append(e)
    return out, lengths


def _cutoff_report(rows: Sequence[Example], cut: dict[str, dict], lengths: Counter) -> dict:
    swapped = [e for e in rows if cutoff.TAG in e.tags]
    return {
        "cutoff_rows": f"{len(swapped)}/{cutoff.ROWS}", "by_tts": dict(Counter(t for e in swapped for t in e.tags if t.startswith("tts:"))),
        "cuts": sum(len(cut[e.id]["cuts"]) for e in swapped),
        "by_length": dict(lengths),
    }


def _words_bucket(example: Example) -> str:
    words = len(example.input.split())
    return "<=25" if words <= 25 else "26-60" if words <= 60 else ">60"


def build(base_dir: Path = paths.ROUND3, out_dir: Path = paths.BUILD) -> dict:
    """Round 3's rows with the cut-off inputs swapped in, plus the literal-cue rows in train. A literal-cue row
    is dropped when it fails a filter, duplicates an earlier row or shares `HELD_OUT_RUN` words with a held-out
    input; then each category keeps its first target count (`capped`). val and test are copied."""
    cut = _cutoff_records()
    base = {split: read_examples(base_dir / f"{split}.jsonl") for split in ("train", "val", "test")}
    held_out_runs = _word_runs(e.input for e in [*base["val"], *base["test"], *_benchmark_rows()])
    kept, dropped = filters.split_by_filters([shape.normalized(e) for e in load()])
    survivors, duplicates = dedup.deduplicate([*(e for rows in base.values() for e in rows), *kept])
    dropped += [(e, reason) for e, reason in duplicates if ROUND4_TAG in e.tags]
    shaped = [shape.spoken_variant(e) for e in shape.bare_filler_share(e for e in survivors if ROUND4_TAG in e.tags)]
    overlapping = {e.id for e in shaped if _word_runs([e.input]) & held_out_runs}
    dropped += [(e, f"shares {HELD_OUT_RUN} words with a held-out input") for e in shaped if e.id in overlapping]
    added = capped([e for e in shaped if e.id not in overlapping])
    swapped, lengths = _swap_cutoff(base["train"], cut)
    rows = sorted([*swapped, *added], key=lambda e: e.id)
    held_out = stats.copy_held_out(base_dir, out_dir)
    write_examples(out_dir / "train.jsonl", rows)
    stats.write_dropped(out_dir, dropped)
    report = stats.stats([*rows, *held_out["val"], *held_out["test"]], dropped)
    counts = Counter(e.category for e in added)
    report["round4"] = {
        "cutoff": _cutoff_report(rows, cut, lengths),
        "added": {name: f"{counts[name]}/{target}" for name, target in generate.RUN_TARGETS[ROUND4_TAG].items()},
        "added_words": dict(Counter(_words_bucket(e) for e in added)),
        "dropped": dict(Counter(reason for _, reason in dropped).most_common()),
    }
    stats.write_report(out_dir, report, stats.stats_markdown(report) + f"\nround 4: {report['round4']}\n")
    return report


def report() -> dict:
    records = {row["id"]: row for row in read_jsonl(CHECKS)} if CHECKS.exists() else {}
    rows = second_sample.run_rows(ROUND4_TAG, CATEGORIES)
    by_category: dict[str, Counter] = defaultdict(Counter)
    reasons: Counter = Counter()
    for e in rows:
        record = records.get(e.id)
        by_category[e.category][record["action"] if record else "unchecked"] += 1
        if record and record["action"] == "drop":
            reasons[record["reason"].split(":")[0]] += 1
    tags = Counter(t for e in rows for t in e.tags if not t.startswith("style_") and t != ROUND4_TAG)
    return {"by_category": {c: dict(n) for c, n in by_category.items()}, "dropped_for": dict(reasons),
            "words": dict(Counter(_words_bucket(e) for e in rows)), "tags": dict(tags.most_common())}
