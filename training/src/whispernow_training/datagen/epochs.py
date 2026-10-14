"""Fresh speech noise per epoch: one noisy variant of every train row per epoch.

Epoch 1 is the build's own train split. Epoch n > 1 draws the programmatic noise again with the
salt `epoch<n>`, from the same code and the same clean text, so the target is unchanged:

- **Injected rows** (`inject.inject`): fillers, clusters, "the uh the", repeats, restarts,
  fragments, known-answer corrections and pause punctuation, injected into the row's own target
  with a new seed. `EMPTY` rows stay as built.
- **Composed rows and the short sources' train rows** (`build.spoken_variant`): the recognizer
  style and the densified share are drawn again from the row's input before the variant step,
  which `build.gather_with_drops` rebuilds. The build's own variant of each row is checked to come out of it.

Every other row (aawaaz, the restyled rows, generated rows the teacher labelled) is the same in
every epoch: its noise came from an LLM, which also wrote the target. Val, test and the
benchmarks are never re-noised. Each file is a pure function of the build and the epoch number.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from . import build, inject, paths
from .schema import EMPTY, Example, read_examples, write_examples

EPOCHS = paths.BUILD / "epochs"
# Tags a re-injection replaces.
_INJECTION_TAGS = ("noise:", "noise_level:", "fillers_", "correction:")


@dataclass
class SpokenBase:
    before: Example
    built_tags: tuple[str, ...]


def epoch_file(epoch: int) -> Path:
    return paths.BUILD / "train.jsonl" if epoch == 1 else EPOCHS / f"train.epoch-{epoch}.jsonl"


def _salt(epoch: int) -> str:
    return f"epoch{epoch}"


def reinjected(row: Example, epoch: int) -> Example:
    """An injected row with new noise in its own target; `EMPTY` rows and epoch 1 are as built."""
    if epoch == 1 or row.output == EMPTY or "injected" not in row.tags:
        return row
    style = "raw" if "style_raw" in row.tags else "recognizer"
    injection = inject.inject(row.output, f"{row.id}-{_salt(epoch)}", style)
    if injection.text == " ".join(row.output.split()):
        return row
    kept = [tag for tag in row.tags if not tag.startswith(_INJECTION_TAGS)]
    return replace(row, input=injection.text, tags=[*kept, *injection.tags])


def respoken(row: Example, base: SpokenBase, epoch: int) -> Example:
    if epoch == 1:
        return row
    fresh = build.spoken_variant(base.before, _salt(epoch))
    added_tags = [tag for tag in row.tags if tag not in base.built_tags]
    return replace(row, input=fresh.input, tags=[*fresh.tags, *added_tags])


def _respoken_kind(row: Example) -> bool:
    """Whether `build.spoken_variant` applies to this train row."""
    return "composed" in row.tags or row.source in build.SHORT_DENSIFY_SOURCES


def _prepare_variants(train: Sequence[Example]) -> dict[str, SpokenBase]:
    wanted = {row.id for row in train if _respoken_kind(row)}
    gathered: dict[str, Example] = {}
    for example in build.gather_with_drops()[0]:
        if example.id in wanted and example.id not in gathered:
            gathered[example.id] = replace(example, split="train")
    missing = wanted - set(gathered)
    if missing:
        raise ValueError(f"{len(missing)} train rows are not in the rebuilt rows, e.g. {sorted(missing)[:3]}")
    before = {e.id: e for e in build.bare_filler_share(list(gathered.values()))}
    bases = {}
    for row in train:
        if _respoken_kind(row):
            original = before[row.id]
            built = build.spoken_variant(original, "")
            if built.input != row.input:
                raise ValueError(f"{row.id}: the build's variant does not come out of the rebuilt row")
            bases[row.id] = SpokenBase(original, tuple(built.tags))
    return bases


def epoch_rows(train: Sequence[Example], bases: dict[str, SpokenBase], epoch: int) -> list[Example]:
    out = []
    for row in train:
        if row.source == inject.SOURCE:
            out.append(reinjected(row, epoch))
        elif _respoken_kind(row):
            out.append(respoken(row, bases[row.id], epoch))
        else:
            out.append(row)
    return out


def _noise_counts(rows: Sequence[Example], train: Sequence[Example]) -> dict:
    changed = sum(a.input != b.input for a, b in zip(rows, train, strict=True))
    composed = [row for row in rows if "composed" in row.tags]
    short = [row for row in rows if row.source in build.SHORT_DENSIFY_SOURCES]
    return {
        "rows": len(rows),
        "input_differs_from_build": changed,
        "densified_composed_pct": round(100 * sum("densified" in r.tags for r in composed) / max(1, len(composed)), 1),
        "densified_short_pct": round(100 * sum("densified" in r.tags for r in short) / max(1, len(short)), 1),
        "injected_levels": dict(Counter(t for r in rows for t in r.tags if t.startswith("noise_level:"))),
        "asr_styles_composed": dict(Counter(t for r in composed for t in r.tags if t.startswith("asr_"))),
        "fillers_per_100_words": round(
            100 * sum(inject.filler_count(r.input) for r in rows) / sum(len(r.input.split()) for r in rows), 2
        ),
    }


def materialize(epochs: int) -> list[Path]:
    """Writes `data/build/epochs/train.epoch-<n>.jsonl` for n = 2…epochs where missing or older than
    the build's train split (a rebuild), and their noise counts to `stats.json`; returns every
    epoch's file, epoch 1 being the build's train split."""
    files = [epoch_file(epoch) for epoch in range(1, epochs + 1)]
    build_mtime = (paths.BUILD / "train.jsonl").stat().st_mtime

    def stale(path: Path) -> bool:
        return not path.exists() or path.stat().st_mtime < build_mtime

    if not any(stale(path) for path in files):
        return files
    train = read_examples(paths.BUILD / "train.jsonl")
    bases = _prepare_variants(train)
    stats_file = EPOCHS / "stats.json"
    stats = json.loads(stats_file.read_text(encoding="utf-8")) if stats_file.exists() else {}
    stats["1"] = _noise_counts(train, train)
    for epoch in range(2, epochs + 1):
        rows = epoch_rows(train, bases, epoch)
        if stale(epoch_file(epoch)):
            write_examples(epoch_file(epoch), rows)
        stats[str(epoch)] = _noise_counts(rows, train)
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    stats_file.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return files
