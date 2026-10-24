"""A round's counts, its `stats.json` and `stats.md`, and the files every round writes besides its splits."""

from __future__ import annotations

import json
import shutil
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from ..schema import LENGTH_BUCKETS, Example, length_bucket, read_examples
from ..sources import aawaaz

# Word counts the stats report the share of rows over.
LENGTH_MARKS = (150, 300, 600)


def _percentiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def at(q: float) -> int:
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    return {"p5": at(0.05), "p50": at(0.5), "p95": at(0.95), "max": ordered[-1], "mean": round(statistics.mean(values), 1)}


def stats(examples: Sequence[Example], dropped: Sequence[tuple[Example, str]]) -> dict:
    def count(key) -> dict:
        return dict(Counter(key(e) for e in examples).most_common())

    total = len(examples)
    laid_out = sum(e.layout != "plain" for e in examples)
    return {
        "rows": total,
        "per_split": count(lambda e: e.split),
        "per_source": count(lambda e: e.source),
        "per_category": count(lambda e: e.category),
        "per_layout": count(lambda e: e.layout),
        "per_license": count(lambda e: e.license),
        "per_tag": dict(Counter(t for e in examples for t in e.tags if not t.startswith("from:")).most_common()),
        "per_split_source": {
            split: dict(Counter(e.source for e in examples if e.split == split).most_common())
            for split in ("train", "val", "test")
        },
        "aawaaz_share": round(sum(e.source == aawaaz.REPO for e in examples) / max(1, total), 3),
        "layout_share": round(laid_out / max(1, total), 3),
        "list_share": round(sum(e.layout in ("list", "mixed") for e in examples) / max(1, total), 3),
        "input_words": _percentiles([len(e.input.split()) for e in examples]),
        "output_words": _percentiles([len(e.output.split()) for e in examples]),
        "dropped": dict(Counter(reason for _, reason in dropped).most_common()),
        "length": {split: length_histogram([e for e in examples if e.split == split]) for split in ("train", "val", "test")},
    }


def length_histogram(examples: Sequence[Example]) -> dict:
    """Rows per input length bucket, and the share over each of `LENGTH_MARKS` words."""
    counts = [len(e.input.split()) for e in examples]
    buckets = Counter(length_bucket(n) for n in counts)
    return {
        "buckets": {name: buckets[name] for _, name in (*LENGTH_BUCKETS, (None, "over 1000")) if buckets[name]},
        "over": {str(mark): round(sum(n > mark for n in counts) / max(1, len(counts)), 3) for mark in LENGTH_MARKS},
    }


def stats_markdown(report: dict) -> str:
    lines = [f"# Build stats\n\n{report['rows']} rows. aawaaz share {report['aawaaz_share']:.0%}, "
             f"layout share {report['layout_share']:.0%} (lists {report['list_share']:.0%}).\n"]
    for title, key in (
        ("Split", "per_split"),
        ("Source", "per_source"),
        ("Category", "per_category"),
        ("Layout", "per_layout"),
        ("License", "per_license"),
        ("Tag", "per_tag"),
        ("Dropped", "dropped"),
    ):
        lines.append(f"## {title}\n\n| {title} | Rows |\n|---|---|")
        lines += [f"| {name} | {value} |" for name, value in report[key].items()]
        lines.append("")
    lines.append("## Length (words)\n\n| | p5 | p50 | p95 | max | mean |\n|---|---|---|---|---|---|")
    for key in ("input_words", "output_words"):
        p = report[key]
        lines.append(f"| {key.split('_')[0]} | {p['p5']} | {p['p50']} | {p['p95']} | {p['max']} | {p['mean']} |")
    names = [name for _, name in (*LENGTH_BUCKETS, (None, "over 1000"))]
    marks = [f"> {mark}" for mark in LENGTH_MARKS]
    lines.append("\n## Input length by split\n")
    lines.append("| split | " + " | ".join(names + marks) + " |\n|" + "---|" * (len(names) + len(marks) + 1))
    for split, hist in report.get("length", {}).items():
        cells = [str(hist["buckets"].get(name, 0)) for name in names]
        cells += [f"{hist['over'][str(mark)]:.1%}" for mark in LENGTH_MARKS]
        lines.append(f"| {split} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_dropped(out_dir: Path, dropped: Sequence[tuple[Example, str]]) -> None:
    with (out_dir / "dropped.jsonl").open("w", encoding="utf-8") as sink:
        for example, reason in dropped:
            sink.write(json.dumps({**asdict(example), "dropped": reason}, ensure_ascii=False) + "\n")


def write_report(out_dir: Path, report: dict, markdown: str) -> None:
    (out_dir / "stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "stats.md").write_text(markdown, encoding="utf-8")


def copy_held_out(base_dir: Path, out_dir: Path) -> dict[str, list[Example]]:
    """Copies val and test byte for byte from the round before; only round 1 makes them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        shutil.copyfile(base_dir / f"{split}.jsonl", out_dir / f"{split}.jsonl")
    return {split: read_examples(out_dir / f"{split}.jsonl") for split in ("val", "test")}


def read_split_lines(path: Path) -> list[tuple[str, str]]:
    """(id, line) for each line of a split file, the line as written."""
    return [(json.loads(line)["id"], line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
