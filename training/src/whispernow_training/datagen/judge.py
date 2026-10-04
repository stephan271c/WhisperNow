"""Batched LLM judging: many pairs per request, since only a few requests run at once."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .llm import LlmClient, LlmError, parse_json_reply
from .paths import PROMPTS
from .schema import Example, read_jsonl

STAGE1_CRITERIA = ("content_preserved", "no_hallucination", "lead_in_kept", "not_answered")
BATCH_SIZE = 15
# A single item whose reply cannot be read is asked again for a new sample this many times.
SINGLE_RETRIES = 2


@dataclass
class Verdict:
    id: str
    passed: bool
    failed: list[str]
    note: str
    judge_model: str
    # Each fault the judge listed, as {"kind", "fact"}, when it judges by fault kinds.
    faults: list[dict] = field(default_factory=list)

    def record(self) -> dict:
        """The verdict as written to a verdicts file; a judge without fault kinds writes no `faults`."""
        record = asdict(self)
        if not self.faults:
            del record["faults"]
        return record


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


def _faults(result: dict, fault_kinds: Mapping[str, str | None]) -> list[dict]:
    faults = result.get("faults")
    if not isinstance(faults, list):
        raise ValueError("reply has no faults list")
    parsed = [{"kind": str(fault["kind"]), "fact": str(fault.get("fact", ""))} for fault in faults]
    unknown = {fault["kind"] for fault in parsed} - set(fault_kinds)
    if unknown:
        raise ValueError(f"unknown fault kinds {sorted(unknown)}")
    return parsed


async def judge_batch(
    llm: LlmClient,
    instructions: str,
    criteria: Sequence[str],
    pairs: Sequence[Item],
    reasoning: str | None = None,
    fault_kinds: Mapping[str, str | None] | None = None,
    attempt: int = 0,
) -> list[Verdict]:
    """Judges items in one request. A malformed reply is retried as two half batches, down to single
    items, and a single item up to `SINGLE_RETRIES` more times with a new sample (`attempt`).

    `reasoning` overrides the judge's reasoning setting. With `fault_kinds` (kind → the criterion a
    fault of that kind fails, or None for one that is only counted), each result lists its faults,
    and a criterion some kind maps to fails exactly when the result has a fault of that kind.
    """
    derived = set(fault_kinds.values()) - {None} if fault_kinds else set()
    model = llm.settings.judge_model
    try:
        reply = parse_json_reply(
            await llm.complete(_messages(instructions, pairs), role="judge", attempt=attempt, reasoning=reasoning)
        )
        results = {int(result["i"]): result for result in reply["results"]}
        if set(results) != set(range(len(pairs))):
            raise ValueError("reply does not cover every pair")
        verdicts = []
        for index, (id_, _) in enumerate(pairs):
            result = results[index]
            faults = _faults(result, fault_kinds) if fault_kinds else []
            hit = {fault_kinds[fault["kind"]] for fault in faults} if fault_kinds else set()
            failed = [
                name for name in criteria if (name in hit if name in derived else result.get(name) is not True)
            ]
            note = "; ".join(filter(None, [str(result.get("note", "")), *(f"{f['kind']}: {f['fact']}" for f in faults)]))
            verdicts.append(Verdict(id_, not failed, failed, note, model, faults))
        return verdicts
    except (ValueError, KeyError, TypeError, LlmError) as error:
        if isinstance(error, LlmError) and not error.retryable:
            raise
        if len(pairs) == 1:
            if attempt < SINGLE_RETRIES:
                return await judge_batch(llm, instructions, criteria, pairs, reasoning, fault_kinds, attempt + 1)
            return [Verdict(pairs[0][0], False, ["judge_error"], str(error)[:200], model)]
        middle = len(pairs) // 2
        first, second = await asyncio.gather(
            judge_batch(llm, instructions, criteria, pairs[:middle], reasoning, fault_kinds),
            judge_batch(llm, instructions, criteria, pairs[middle:], reasoning, fault_kinds),
        )
        return first + second


def size_batches(items: Sequence[Item], max_items: int, max_chars: int | None = None) -> list[list[Item]]:
    """Consecutive batches of at most `max_items` items and, when set, `max_chars` characters of fields.

    A long transcript and its cleanup fill a request on their own, so long rows go in small batches.
    """
    batches: list[list[Item]] = []
    current: list[Item] = []
    size = 0
    for item in items:
        length = sum(len(str(value)) for value in item[1].values())
        if current and (len(current) >= max_items or (max_chars is not None and size + length > max_chars)):
            batches.append(current)
            current, size = [], 0
        current.append(item)
        size += length
    if current:
        batches.append(current)
    return batches


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
    max_chars: int | None = None,
    solo_over_words: int | None = None,
    fault_kinds: Mapping[str, str | None] | None = None,
    progress: Callable[[str], None] = lambda line: print(line, file=sys.stderr, flush=True),
) -> dict[str, Verdict]:
    """Judges every example not yet in `out`, appending verdicts as batches finish so a rerun resumes.

    Batches hold `batch_size` items, or fewer when `max_chars` caps their length; an item whose
    transcript is over `solo_over_words` words gets a request of its own. `fault_kinds` is as in
    `judge_batch`.
    """
    instructions = instructions_file.read_text(encoding="utf-8")
    verdicts = load_verdicts(out)
    pending = [item for item in items if item[0] not in verdicts]
    def solo(item: Item) -> bool:
        return solo_over_words is not None and len(str(item[1].get("transcript", "")).split()) > solo_over_words

    batches = size_batches([item for item in pending if not solo(item)], batch_size, max_chars)
    batches += [[item] for item in pending if solo(item)]
    out.parent.mkdir(parents=True, exist_ok=True)
    done = 0

    async def run(batch: Sequence[Item]) -> None:
        nonlocal done
        for verdict in await judge_batch(llm, instructions, criteria, batch, fault_kinds=fault_kinds):
            verdicts[verdict.id] = verdict
            with out.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(verdict.record(), ensure_ascii=False) + "\n")
        done += 1
        if done % 10 == 0 or done == len(batches):
            progress(f"judged {done}/{len(batches)} batches ({llm.calls} calls, {llm.cache_hits} cached)")

    await asyncio.gather(*(run(batch) for batch in batches))
    return verdicts
