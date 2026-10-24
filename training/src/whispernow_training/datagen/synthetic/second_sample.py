"""A second look at a later round's generated rows before they are built.

`check_rows` records one verdict per row: a line that starts with a capitalized command goes to
`review.check` ("literal_case"), and a row the round picks gets a second teacher sample. Two samples
that keep different words mean the transcript does not settle what the speaker meant, and the row is
dropped. `top_up` generates replacements until each category has its target of kept rows plus a spare
share, since the build's filters drop a few more.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from ..llm import LlmClient
from ..references import review
from ..schema import Example, append_jsonl, read_examples, read_jsonl
from ..teacher import relabel
from . import generate

# Words two samples may differ in without keeping different content.
_MINOR = frozenset(
    "a an the and or but of to in on at for with by from is are was were be been it its this that these those "
    "so then also just please".split()
)


def two_sample_disagreement(transcript: str, first: str, second: str) -> list[str]:
    """Words of the transcript that one sample keeps and the other leaves out."""
    spoken = set(review.words(transcript))
    a, b = set(review.words(first)), set(review.words(second))
    return sorted(w for w in a ^ b if w in spoken and w not in _MINOR and not w.isdigit())


def run_rows(run: str, categories: Sequence[str]) -> list[Example]:
    """The generated rows of `run`'s categories, in file order."""
    return [e for category in categories if generate.output_file(category, run=run).exists()
            for e in read_examples(generate.output_file(category, run=run))]


async def check_rows(llm: LlmClient, rows: Sequence[Example], checks: Path, second_sample: Callable[[Example], bool]) -> dict:
    """Checks the `rows` not in `checks` yet into it: one record per row, `keep`, `fix` (with the fixed
    reference) or `drop` (with the reason). A row `second_sample` picks is labelled again and dropped when the
    two samples keep different words."""
    done = {row["id"] for row in read_jsonl(checks)} if checks.exists() else set()
    rows = [e for e in rows if e.id not in done]
    cases = [row for e in rows if (row := review.case_row(e.id, e.input, e.output))]
    case_results = {row["id"]: row for row in await review.check(llm, "literal_case", cases)}
    asked = [e for e in rows if second_sample(e)]
    seconds = await asyncio.gather(*(relabel(llm, e.input, attempt=1) for e in asked))
    second = {e.id: text for e, text in zip(asked, seconds, strict=True)}
    records = []
    for e in rows:
        output = case_results[e.id]["final_output"] if e.id in case_results else e.output
        record = {"id": e.id, "action": "fix" if output != e.output else "keep", "output": output}
        if e.id in second and (differ := two_sample_disagreement(e.input, e.output, second[e.id])):
            record = {"id": e.id, "action": "drop", "reason": f"teacher samples keep different words: {', '.join(differ)}",
                      "second": second[e.id]}
        records.append(record)
    append_jsonl(checks, [json.dumps(r, ensure_ascii=False) for r in records])
    return {"checked": len(records), **Counter(r["action"] for r in records), "two_sample_checked": len(asked)}


def checked_rows(rows: Sequence[Example], checks: Path, check_step: str) -> list[Example]:
    """The `rows` the check keeps, fixed where it fixed them, in their order."""
    records = {row["id"]: row for row in read_jsonl(checks)} if checks.exists() else {}
    if unchecked := [e.id for e in rows if e.id not in records]:
        raise SystemExit(f"{len(unchecked)} rows are not checked yet; run `datagen {check_step}`")
    return [
        replace(e, output=records[e.id]["output"], tags=[*e.tags, "ref_fixed"]) if records[e.id]["action"] == "fix" else e
        for e in rows if records[e.id]["action"] != "drop"
    ]


def checked_run(run: str, categories: Sequence[str], checks: Path, check_step: str) -> dict[str, list[Example]]:
    """Each category's rows the check keeps, fixed where it fixed them, in file order."""
    return {category: checked_rows(run_rows(run, [category]), checks, check_step) for category in categories}


def with_spare(target: int, spare: float) -> int:
    return math.ceil(target * (1 + spare))


async def top_up(
    llm: LlmClient, run: str, targets: Mapping[str, int], checks: Path, second_sample: Callable[[Example], bool],
    spare: float, check_step: str,
) -> dict:
    """Checks `run`'s rows, then generates and checks more until each category has its target of kept rows, plus `spare`."""
    while True:
        await check_rows(llm, run_rows(run, list(targets)), checks, second_sample)
        kept = checked_run(run, list(targets), checks, check_step)
        short = {category: with_spare(target, spare) - len(kept[category])
                 for category, target in targets.items() if len(kept[category]) < with_spare(target, spare)}
        if not short:
            return {category: len(rows) for category, rows in kept.items()}
        print(f"{run} rows short of their targets: {short}", flush=True)
        await asyncio.gather(*(
            generate.generate_category(llm, category, len(read_examples(generate.output_file(category, run=run))) + missing, run=run)
            for category, missing in short.items()
        ))
