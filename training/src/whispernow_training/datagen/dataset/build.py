"""The dataset, built in rounds. Each round starts from the one before and writes its own directory; the
last writes `paths.BUILD`, the train, val and test splits the cleanup models train and are scored on."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .. import paths
from ..schema import Example, read_examples
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


# The dataset card's categories, named for what the cleanup has to do. The build keeps the finer categories:
# generation prompts, generated-file names and evaluation reports are keyed by them.
PUBLISHED_CATEGORY = {
    category: published
    for published, categories in {
        "self_correction": "self_correction self_corrections_heavy revised_choice email_late_correction dictated_question",
        "list": "shopping_lists list_corrections list_edits technical_list technical_steps recipe_quantities"
        " mixed_message command_question_list ambiguous_intent",
        "meeting_notes": "meeting_notes meeting_actions",
        "prose": "academic_research medical_clinical financial_business legal_contract creative_writing"
        " casual_conversation email_professional recipe_cooking dictation prose_mentions short_dictation"
        " multi_topic long_ramble dense_professional real_meeting_turn",
        "code_and_symbols": "technical_code spoken_symbols",
        "dictated_command": "dictated_command",
        "literal_cue": "literal_cue",
        "fillers": "filler_injection",
        "no_edit": "no_edit",
        "empty": "empty",
    }.items()
    for category in categories.split()
}
PUBLISHED_FIELDS = ("id", "input", "output", "category", "source", "license", "layout")


def published_row(example: Example) -> dict:
    row = {name: getattr(example, name) for name in PUBLISHED_FIELDS}
    row["category"] = PUBLISHED_CATEGORY[example.category]
    return row


def publish(repo_id: str, out_dir: Path = paths.BUILD) -> None:
    """Pushes the splits to a private Hugging Face dataset repo, with the card's fields and categories."""
    from datasets import DatasetDict, load_dataset

    staged = out_dir / "hf"
    staged.mkdir(exist_ok=True)
    files = {split: str(staged / f"{split}.jsonl") for split in ("train", "val", "test")}
    for split, path in files.items():
        rows = [json.dumps(published_row(e), ensure_ascii=False) for e in read_examples(out_dir / f"{split}.jsonl")]
        Path(path).write_text("".join(row + "\n" for row in rows), encoding="utf-8")
    dataset: DatasetDict = load_dataset("json", data_files=files)
    dataset.push_to_hub(repo_id, private=True)
