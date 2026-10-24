"""The dataset, built in rounds. Each round starts from the one before and writes its own directory; the
last writes `paths.BUILD`, the train, val and test splits the cleanup models train and are scored on."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from .. import paths
from ..schema import read_examples, write_examples
from . import round1_base, round2_targeted, round3_weak_spots, round4_cutoffs_cues

ROUNDS: dict[str, tuple[Callable[[], dict], Path]] = {
    "base": (round1_base.build, paths.ROUND1),
    "targeted": (round2_targeted.build, paths.ROUND2),
    "weak-spots": (round3_weak_spots.build, paths.ROUND3),
    "cutoffs-cues": (round4_cutoffs_cues.build, paths.BUILD),
}
# Round 1 is kept as built (see `round1_base`), so the default rebuild starts after it.
LATER_ROUNDS = ("targeted", "weak-spots", "cutoffs-cues")


def review_sample(layout: str, n: int, seed: int, out_dir: Path = paths.BUILD) -> str:
    """`n` random built rows of one layout as text for a hand review, with each row's source and tags."""
    rows = [e for split in ("train", "val", "test") for e in read_examples(out_dir / f"{split}.jsonl")]
    rows = sorted((e for e in rows if e.layout == layout), key=lambda e: e.id)
    picked = np.random.default_rng(seed).choice(len(rows), size=min(n, len(rows)), replace=False)
    blocks = []
    for index, row_index in enumerate(picked):
        e = rows[row_index]
        header = f"### {index} {e.id} [{e.source}/{e.category}] {','.join(e.tags)}"
        blocks.append(f"{header}\nIN: {e.input}\nOUT: {e.output}\n")
    return "\n".join(blocks)


def publish(repo_id: str, out_dir: Path = paths.BUILD) -> None:
    """Pushes the splits to a private Hugging Face dataset repo. Every row is written out again with every schema
    field, since val and test lines predate some of them and `datasets` takes a split's columns from its rows."""
    from datasets import DatasetDict, load_dataset

    staged = out_dir / "hf"
    for split in ("train", "val", "test"):
        write_examples(staged / f"{split}.jsonl", read_examples(out_dir / f"{split}.jsonl"))
    files = {split: str(staged / f"{split}.jsonl") for split in ("train", "val", "test")}
    dataset: DatasetDict = load_dataset("json", data_files=files)
    dataset.push_to_hub(repo_id, private=True)
