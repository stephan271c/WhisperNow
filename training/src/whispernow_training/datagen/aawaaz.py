"""Stage 1: import `shantanugoel/aawaaz-transcript-cleanup-dataset` and normalize its outputs."""

from __future__ import annotations

import hashlib
import re
from collections import Counter

from .schema import Example, example_id

REPO = "shantanugoel/aawaaz-transcript-cleanup-dataset"
LICENSE = "MIT"
# The dataset ships one JSONL file per category, so each row's category is known exactly.
CATEGORIES = (
    "academic_research",
    "casual_conversation",
    "creative_writing",
    "email_professional",
    "financial_business",
    "legal_contract",
    "medical_clinical",
    "meeting_notes",
    "recipe_cooking",
    "self_corrections_heavy",
    "shopping_lists",
    "technical_code",
)
# aawaaz has no eval split, so this share of each category is held out as test.
TEST_PERCENT = 3

# Bold markers sit at word edges; "x**2" and the "6-31G**" basis set are content and stay.
_BOLD = re.compile(r"(?<![\w*])\*\*(?=\S)(.+?)(?<=\S)\*\*(?![\w*])", re.DOTALL)
_ITALIC = re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])")
_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]+", re.MULTILINE)
_PAREN_NUMBER = re.compile(r"^([ \t]*)(\d+)\)[ \t]+", re.MULTILINE)
_STAR_BULLET = re.compile(r"^([ \t]*)[*•+][ \t]+", re.MULTILINE)
_TABLE_ROW = re.compile(r"^[ \t]*\|.*\|[ \t]*$", re.MULTILINE)
_LINK = re.compile(r"\]\([^)]*\)")


def normalize_output(text: str) -> str | None:
    """Removes markdown the app can't paste, or returns None for a row that can't be saved."""
    if _TABLE_ROW.search(text) or _LINK.search(text) or "```" in text:
        return None
    stripped = text.strip()
    if len(stripped) > 1 and stripped[0] == stripped[-1] == '"' and stripped.count('"') == 2:
        return None
    text = _BOLD.sub(r"\1", text)
    text = _ITALIC.sub(r"\1", text)
    text = text.replace("`", "")
    text = _HEADING.sub("", text)
    text = _PAREN_NUMBER.sub(r"\1\2. ", text)
    text = _STAR_BULLET.sub(r"\1- ", text)
    text = "\n".join(line.rstrip() for line in text.strip().split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text)


def is_test(example_id_: str) -> bool:
    return int(hashlib.sha256(example_id_.encode()).hexdigest(), 16) % 100 < TEST_PERCENT


def import_rows(rows_by_category: dict[str, list[dict]]) -> tuple[list[Example], Counter]:
    """Maps raw rows into the schema. Returns the examples and a count of what was dropped and why."""
    dropped: Counter = Counter()
    seen: set[str] = set()
    examples = []
    for category, rows in rows_by_category.items():
        for row in rows:
            source_input = row["input"].strip()
            if source_input in seen:
                dropped["duplicate input"] += 1
                continue
            seen.add(source_input)
            output = normalize_output(row["output"])
            if output is None:
                dropped["unfixable markdown or wrapping quotes"] += 1
                continue
            id_ = example_id(REPO, source_input)
            examples.append(
                Example(
                    id=id_,
                    input=source_input,
                    output=output,
                    source=REPO,
                    category=category,
                    license=LICENSE,
                    split="test" if is_test(id_) else "",
                )
            )
    return examples, dropped


def load_rows() -> dict[str, list[dict]]:
    from datasets import load_dataset

    return {
        category: load_dataset(REPO, data_files=f"{category}.jsonl", split="train").to_list()
        for category in CATEGORIES
    }
