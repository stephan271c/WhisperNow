"""Apply stored reference corrections and failed-row overrides in pass order."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from .schema import Example, read_jsonl

# Each pass name is also the tag a row it covers gets.
PASSES = ("rejudged", "strict", "codex")


def overrides_file(directory: Path, name: str = "rejudged") -> Path:
    return directory / f"{name}.jsonl"


def apply(rows: Sequence[Example], directory: Path, passes: Sequence[str] = PASSES) -> list[Example]:
    kept = list(rows)
    for name in passes:
        path = overrides_file(directory, name)
        if not path.exists():
            continue
        verdicts = {row["id"]: row for row in read_jsonl(path)}
        rows, kept = kept, []
        for e in rows:
            verdict = verdicts.get(e.id)
            if verdict is None:
                kept.append(e)
            elif verdict["passed"]:
                relabeled = verdict["output"] != e.output
                tags = [*e.tags, name, *(["relabeled"] if relabeled and "relabeled" not in e.tags else [])]
                if "teacher_model" in verdict:
                    tags.append("judge_model_reference")
                teacher_model = verdict.get("teacher_model") or (relabeled and verdict.get("relabel_model")) or e.teacher_model
                kept.append(replace(e, output=verdict["output"], tags=tags, teacher_model=teacher_model))
    return kept
