"""Round 2: more long composed rows, weighted toward the categories and lengths where models trained on
round 1 lost the most facts (`compose.TARGETED`), plus an injected spoken version of each long one.

Their references were written by the relabel's teacher and passed the same gates, so they skip
`relabel.apply`. Round 1's rows are copied as written.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

from .. import paths
from ..schema import ROUND2_TAG, Example, read_examples
from ..synthetic import compose, inject
from . import dedup, filters, round1_base, shape, stats


def gather() -> list[Example]:
    """The targeted compose run's rows, plus an injected spoken version of each long reference, from a
    pool of their own as round 1's long ones are."""
    targeted = compose.load(compose.TARGETED)
    injected = inject.injected_rows(
        [e for e in targeted if len(e.output.split()) >= round1_base.LONG_INJECTION_WORDS], len(targeted),
        round1_base.LONG_INJECTION_SEED, layouts=("plain", "paragraphs"),
    )
    rows = targeted + [replace(e, tags=[*e.tags, "long_origin", ROUND2_TAG]) for e in injected]
    return [shape.normalized(e) for e in rows]


def build(base_dir: Path = paths.ROUND1, out_dir: Path = paths.ROUND2) -> dict:
    """Round 1's rows and the round 2 rows in train.

    A round 2 row is dropped when it fails a filter or duplicates an earlier row. Round 1's rows were
    deduplicated before their inputs were varied, so the duplicate check below may flag a few of them
    again; it is used only to drop round 2 rows.
    """
    base = {split: read_examples(base_dir / f"{split}.jsonl") for split in ("train", "val", "test")}
    kept, dropped = filters.split_by_filters(gather())
    survivors, duplicates = dedup.deduplicate([*(e for rows in base.values() for e in rows), *kept])
    dropped += [(e, reason) for e, reason in duplicates if ROUND2_TAG in e.tags]
    added = [shape.spoken_variant(replace(e, split="train")) for e in survivors if ROUND2_TAG in e.tags]

    held_out = stats.copy_held_out(base_dir, out_dir)
    # Round 1's train lines are copied as written, so fields added to the schema since never change them.
    lines = sorted([*stats.read_split_lines(base_dir / "train.jsonl"), *((e.id, e.to_json()) for e in added)])
    (out_dir / "train.jsonl").write_text("".join(line + "\n" for _, line in lines), encoding="utf-8")
    stats.write_dropped(out_dir, dropped)
    report = stats.stats([*base["train"], *added, *held_out["val"], *held_out["test"]], dropped)
    report["round2"] = {"added": len(added), "dropped": dict(Counter(reason for _, reason in dropped).most_common())}
    stats.write_report(out_dir, report, stats.stats_markdown(report) + f"\nround 2: {report['round2']}\n")
    return report
