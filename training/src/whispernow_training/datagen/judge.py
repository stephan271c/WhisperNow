"""Batched LLM judging: many pairs per request, since only a few requests run at once."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .concurrency import task_group
from .llm import LlmClient, parse_json_reply, request_key
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
    request_key: str = ""

    def record(self) -> dict:
        """The verdict as written to a verdicts file; a judge without fault kinds writes no `faults`."""
        record = asdict(self)
        if not self.faults:
            del record["faults"]
        if not self.request_key:
            del record["request_key"]
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
    reply = await llm.complete(_messages(instructions, pairs), role="judge", attempt=attempt, reasoning=reasoning)
    try:
        reply = parse_json_reply(reply)
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
    except (ValueError, KeyError, TypeError) as error:
        if len(pairs) == 1:
            if attempt < SINGLE_RETRIES:
                return await judge_batch(llm, instructions, criteria, pairs, reasoning, fault_kinds, attempt + 1)
            return [Verdict(pairs[0][0], False, ["judge_error"], str(error)[:200], model)]
        middle = len(pairs) // 2
        async with task_group() as group:
            tasks = [group.create_task(judge_batch(llm, instructions, criteria, half, reasoning, fault_kinds))
                     for half in (pairs[:middle], pairs[middle:])]
        return tasks[0].result() + tasks[1].result()


def size_batches(
    items: Sequence[Item], max_items: int, max_chars: int | None = None, solo_over_words: int | None = None,
) -> list[list[Item]]:
    """Consecutive batches of at most `max_items` items and, when set, `max_chars` characters of fields.

    Transcripts over `solo_over_words`, when set, get a request each.
    """
    solo: list[Item] = []
    batches: list[list[Item]] = []
    current: list[Item] = []
    size = 0
    for item in items:
        if solo_over_words is not None and len(str(item[1].get("transcript", "")).split()) > solo_over_words:
            solo.append(item)
            continue
        length = sum(len(str(value)) for value in item[1].values())
        if current and (len(current) >= max_items or (max_chars is not None and size + length > max_chars)):
            batches.append(current)
            current, size = [], 0
        current.append(item)
        size += length
    if current:
        batches.append(current)
    return batches + [[item] for item in solo]


def load_verdicts(path: Path) -> dict[str, Verdict]:
    if not path.exists():
        return {}
    return {row["id"]: Verdict(**row) for row in read_jsonl(path)}


def judge_contract(
    llm: LlmClient, instructions: str, criteria: Sequence[str],
    fault_kinds: Mapping[str, str | None] | None = None, reasoning: str | None = None,
) -> dict:
    settings = llm.settings.request("judge")
    if reasoning is not None:
        settings["reasoning"] = reasoning
    return {"settings": settings, "instructions": instructions, "criteria": list(criteria),
            "fault_kinds": dict(fault_kinds) if fault_kinds is not None else None}


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
    """Judges changed or unfinished requests, appending verdicts as batches finish.

    Batches hold `batch_size` items, or fewer when `max_chars` caps their length; an item whose
    transcript is over `solo_over_words` words gets a request of its own. `fault_kinds` is as in
    `judge_batch`.
    """
    instructions = instructions_file.read_text(encoding="utf-8")
    contract = judge_contract(llm, instructions, criteria, fault_kinds)
    keys = {id_: request_key({"contract": contract, "fields": fields}) for id_, fields in items}
    saved = load_verdicts(out)
    verdicts = {id_: verdict for id_, verdict in saved.items()
                if id_ in keys and verdict.request_key == keys[id_] and "judge_error" not in verdict.failed}
    pending = [item for item in items if item[0] not in verdicts]
    batches = size_batches(pending, batch_size, max_chars, solo_over_words)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = 0

    async def run(batch: Sequence[Item]) -> None:
        nonlocal done
        for verdict in await judge_batch(llm, instructions, criteria, batch, fault_kinds=fault_kinds):
            verdict.request_key = keys[verdict.id]
            verdicts[verdict.id] = verdict
            with out.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(verdict.record(), ensure_ascii=False) + "\n")
        done += 1
        if done % 10 == 0 or done == len(batches):
            progress(f"judged {done}/{len(batches)} batches ({llm.calls} calls, {llm.cache_hits} cached)")

    async with task_group() as group:
        for batch in batches:
            group.create_task(run(batch))
    return verdicts
