"""Round 1, the base dataset: every source's rows, filtered, deduplicated and split by seed group.

The sources are the cleaned aawaaz rows, handy and disfl_qa, the generated short categories, the
composed long rows and the programmatic rows made here (filler injection and `EMPTY`), each with its
reference from the relabel. This is the only round that makes val and test; later rounds add train
rows and copy val and test as written.

Round 1's output (`paths.ROUND1`) is kept as built, with the test references round 3's review fixed
(`review.FIXES`) written into it, and every later round starts from it. Today's code rebuilds it with
small differences, so `datagen build` rebuilds round 1 only when asked.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from .. import paths
from ..references import relabel
from ..schema import Example, hash_bucket, read_examples, write_examples
from ..sources import aawaaz, aawaaz_audit, public, public_audit
from ..synthetic import compose, generate, inject
from . import dedup, filters, shape, stats

SEED = 108
INJECTED = 2000
# Every long one- or multi-paragraph clean text not already injected gets one injected spoken
# version too: aawaaz references and the composed rows' references.
LONG_INJECTION_WORDS = 150
LONG_INJECTION_SEED = 110
EMPTY_ROWS = 150
# More EMPTY rows pinned to test, from a second seed; a repeat of a training input stays in test.
EMPTY_TEST_ROWS = 90
VAL_PERCENT = 5
TEST_PERCENT = 5
# Sources with their own held-out test rows; other groups get a hashed test share.
PRESET_TEST_SOURCES = {aawaaz.REPO, public.HANDY, public.DISFL_QA, public.SOTTO}


def gather_with_drops(with_sotto: bool = False) -> tuple[list[Example], list[tuple[Example, str]], dict[str, dict] | None]:
    """Every source's rows, plus the programmatic rows made here from the cleaned aawaaz rows and the
    composed rows. The benchmarks in `data/bench/` are never read.

    Once `datagen relabel build` has run, every row but the injected ones takes its reference from
    the relabel (`relabel.apply`), and rows it dropped or never saw are dropped here; the injected
    rows are then made from the relabelled parents.
    """
    records = relabel.snapshot("build")
    dropped: list[tuple[Example, str]] = []

    def labelled(examples: list[Example]) -> list[Example]:
        if records is None:
            return examples
        kept, lost = relabel.apply(examples, records)
        dropped.extend(lost)
        return kept

    clean = labelled(read_examples(aawaaz_audit.CLEAN))
    rows = list(clean)
    for source, path in public_audit.OUT_FILES.items():
        if path.exists() and (with_sotto or source != public.SOTTO):
            rows += labelled(read_examples(path))
    for category in generate.TARGETS:
        for held_out in (False, True):
            if generate.output_file(category, held_out).exists():
                rows += labelled(read_examples(generate.output_file(category, held_out)))
    composed = labelled(compose.load(compose.TRAIN))
    rows += composed
    injected = inject.injected_rows(clean, INJECTED, SEED)
    used = {e.output for e in injected}
    long_origins = [
        e for e in [*clean, *composed] if len(e.output.split()) >= LONG_INJECTION_WORDS and e.output not in used
    ]
    long_injected = inject.injected_rows(
        long_origins, len(long_origins), LONG_INJECTION_SEED, layouts=("plain", "paragraphs")
    )
    rows += injected + [replace(e, tags=[*e.tags, "long_origin"]) for e in long_injected]
    rows += labelled(inject.empty_rows(EMPTY_ROWS, SEED))
    rows += labelled([replace(e, split="test") for e in inject.empty_rows(EMPTY_TEST_ROWS, SEED + 1)])
    return [shape.normalized(e) for e in rows], dropped, records


def assign_splits(examples: Iterable[Example]) -> list[Example]:
    """Splits by seed group: a group with a preset test row is test, and others are hashed into val and test."""
    groups: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        groups[example.seed_group].append(example)
    result = []
    for seed_group, members in groups.items():
        bucket = hash_bucket(f"split\0{seed_group}")
        if any(e.split == "test" for e in members):
            split = "test"
        elif bucket < VAL_PERCENT:
            split = "val"
        elif bucket < VAL_PERCENT + TEST_PERCENT and not any(
            e.source in PRESET_TEST_SOURCES for e in members
        ):
            split = "test"
        else:
            split = "train"
        result += [replace(e, split=split) for e in members]
    return result


def build(with_sotto: bool = False, out_dir: Path = paths.ROUND1) -> dict:
    rows, dropped, records = gather_with_drops(with_sotto)
    kept, filtered = filters.split_by_filters(rows)
    dropped += filtered
    kept, duplicates = dedup.deduplicate(kept)
    dropped += duplicates
    final = [shape.spoken_variant(e) for e in shape.bare_filler_share(assign_splits(shape.restyled(kept)))]

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        write_examples(out_dir / f"{split}.jsonl", sorted((e for e in final if e.split == split), key=lambda e: e.id))
    stats.write_dropped(out_dir, dropped)
    report = stats.stats(final, dropped)
    markdown = stats.stats_markdown(report)
    if records is not None:
        # Old against new by source; the old injected rows were copied aside with the relabel's inputs.
        compared = relabel.compare([*relabel.inputs("build"), *relabel.old_injected()], final, records)
        report["relabel"] = compared
        markdown += "\n" + relabel.compare_markdown(compared)
    stats.write_report(out_dir, report, markdown)
    return report
