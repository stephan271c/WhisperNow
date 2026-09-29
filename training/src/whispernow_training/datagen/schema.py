"""The one example schema every stage reads and writes, and JSONL helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

Layout = Literal["plain", "list", "paragraphs", "mixed"]
Split = Literal["train", "val", "test", ""]

EMPTY = "EMPTY"

LIST_ITEM = re.compile(r"^\s*(?:- |\d+\. )\S")
_LABEL_MAX_WORDS = 8


@dataclass
class Example:
    id: str
    input: str
    output: str
    source: str
    category: str
    license: str
    tags: list[str] = field(default_factory=list)
    layout: Layout = "plain"
    seed_group: str = ""
    split: Split = ""
    teacher_model: str | None = None

    def __post_init__(self) -> None:
        self.layout = layout_of(self.output)
        if not self.seed_group:
            self.seed_group = self.id

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, row: dict) -> Example:
        return cls(**row)


def example_id(source: str, text: str) -> str:
    """Stable id from the source and the input text."""
    digest = hashlib.sha256(f"{source}\0{text}".encode()).hexdigest()[:16]
    return f"{source.split('/')[-1]}-{digest}"


def _is_label(line: str) -> bool:
    """A lead-in, section label or title line that sits above list items."""
    stripped = line.strip()
    if stripped.endswith(":"):
        return True
    return len(stripped.split()) <= _LABEL_MAX_WORDS and not stripped.endswith((".", "!", "?"))


def layout_of(output: str) -> Layout:
    """Classifies an output's layout from its line structure."""
    lines = [line for line in output.strip().split("\n") if line.strip()]
    items = [line for line in lines if LIST_ITEM.match(line)]
    if items:
        others = [line for line in lines if not LIST_ITEM.match(line)]
        return "list" if all(_is_label(line) for line in others) else "mixed"
    return "paragraphs" if len(lines) > 1 else "plain"


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            if line.strip():
                yield json.loads(line)


def read_examples(path: Path) -> list[Example]:
    return [Example.from_dict(row) for row in read_jsonl(path)]


def write_examples(path: Path, examples: Iterable[Example]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as out:
        for example in examples:
            out.write(example.to_json() + "\n")
            count += 1
    return count


def append_examples(path: Path, examples: Iterable[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        for example in examples:
            out.write(example.to_json() + "\n")
