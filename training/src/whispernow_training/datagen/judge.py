"""Batched LLM judging: many pairs per request, since only a few requests run at once."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .llm import LlmClient, LlmError, parse_json_reply
from .paths import PROMPTS
from .schema import Example, read_jsonl

STAGE1_CRITERIA = ("content_preserved", "no_hallucination", "lead_in_kept", "not_answered")
BATCH_SIZE = 15


@dataclass
class Verdict:
    id: str
    passed: bool
    failed: list[str]
    note: str
    judge_model: str


Item = tuple[str, dict]
"""An id and the fields the judge sees for it."""


def pair_item(example: Example) -> Item:
    return example.id, {"transcript": example.input, "output": example.output}


def _messages(instructions: str, pairs: Sequence[Item]) -> list[dict[str, str]]:
    lines = [json.dumps({"i": index, **fields}, ensure_ascii=False) for index, (_, fields) in enumerate(pairs)]
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": "Items, one JSON object per line:\n" + "\n".join(lines)},
    ]


async def judge_batch(
    llm: LlmClient, instructions: str, criteria: Sequence[str], pairs: Sequence[Item]
) -> list[Verdict]:
    """Judges items in one request. A malformed reply is retried as two half batches, down to single items."""
    model = llm.settings.judge_model
    try:
        reply = parse_json_reply(await llm.complete(_messages(instructions, pairs), role="judge"))
        results = {int(result["i"]): result for result in reply["results"]}
        if set(results) != set(range(len(pairs))):
            raise ValueError("reply does not cover every pair")
        verdicts = []
        for index, (id_, _) in enumerate(pairs):
            result = results[index]
            failed = [name for name in criteria if result.get(name) is not True]
            verdicts.append(Verdict(id_, not failed, failed, str(result.get("note", "")), model))
        return verdicts
    except (ValueError, KeyError, TypeError, LlmError) as error:
        if isinstance(error, LlmError) and not error.retryable:
            raise
        if len(pairs) == 1:
            return [Verdict(pairs[0][0], False, ["judge_error"], str(error)[:200], model)]
        middle = len(pairs) // 2
        first, second = await asyncio.gather(
            judge_batch(llm, instructions, criteria, pairs[:middle]),
            judge_batch(llm, instructions, criteria, pairs[middle:]),
        )
        return first + second


def load_verdicts(path: Path) -> dict[str, Verdict]:
    if not path.exists():
        return {}
    return {row["id"]: Verdict(**row) for row in read_jsonl(path)}


async def judge_all(
    llm: LlmClient,
    items: Sequence[Item],
    out: Path,
    *,
    instructions_file: Path = PROMPTS / "judge" / "stage1.md",
    criteria: Sequence[str] = STAGE1_CRITERIA,
    batch_size: int = BATCH_SIZE,
    progress: Callable[[str], None] = lambda line: print(line, file=sys.stderr, flush=True),
) -> dict[str, Verdict]:
    """Judges every example not yet in `out`, appending verdicts as batches finish so a rerun resumes."""
    instructions = instructions_file.read_text(encoding="utf-8")
    verdicts = load_verdicts(out)
    pending = [item for item in items if item[0] not in verdicts]
    batches = [pending[start : start + batch_size] for start in range(0, len(pending), batch_size)]
    out.parent.mkdir(parents=True, exist_ok=True)
    done = 0

    async def run(batch: Sequence[Item]) -> None:
        nonlocal done
        for verdict in await judge_batch(llm, instructions, criteria, batch):
            verdicts[verdict.id] = verdict
            with out.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(asdict(verdict), ensure_ascii=False) + "\n")
        done += 1
        if done % 10 == 0 or done == len(batches):
            progress(f"judged {done}/{len(batches)} batches ({llm.calls} calls, {llm.cache_hits} cached)")

    await asyncio.gather(*(run(batch) for batch in batches))
    return verdicts
