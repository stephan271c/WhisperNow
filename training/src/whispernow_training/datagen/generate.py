"""Stage 3: generate rows for gaps and hard cases.

Each batch is a fresh generator request on `LLM_MODEL` for about 50 transcripts
as JSON. The teacher writes each reference exactly as the app would, and a
validator on `LLM_JUDGE_MODEL` checks the pairs. Passing rows are appended to
`data/stage3/<category>.jsonl`; the rejection notes go into the next batch's
request. Batches are logged in `<category>.batches.jsonl`, and a batch's request
depends only on the category, its number and the previous batch's notes, so an
interrupted run resumes from the log with the finished requests served from the
response cache.
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace

from . import paths
from .concurrency import task_group
from .judge import judge_batch
from .llm import LlmClient, LlmError, parse_json_reply
from .schema import EMPTY, BatchRecord, Example, append_jsonl, example_id, is_missing_key_failure, read_examples, read_jsonl
from .teacher import relabel

SOURCE = "teacher"
LICENSE = "zai-output"
BATCH_SIZE = 50
# Long transcripts make slow replies: a batch of 50 long rambles took 12 minutes (64 KB).
BATCH_SIZES = {"long_ramble": 25}
JUDGE_BATCH_SIZE = 10
STYLES = ("recognizer", "raw")
CRITERIA = ("input_realistic", "content_preserved", "no_hallucination", "not_answered", "layout_ok", "clean")
FEEDBACK_NOTES = 15

# Rows to keep per category: gaps no dataset covers, then hard cases.
TARGETS = {
    "dictated_command": 500,
    "prose_mentions": 750,
    "spoken_symbols": 500,
    "mixed_message": 300,
    "list_edits": 300,
    "long_ramble": 300,
    "command_question_list": 300,
    "technical_list": 300,
    "ambiguous_intent": 300,
}
# Extra rows pinned to the test split, bringing each category to about 100 test rows (long
# rambles to about 50) so its test rates are not decided by one row.
TEST_TARGETS = {
    "dictated_command": 70,
    "prose_mentions": 70,
    "spoken_symbols": 75,
    "mixed_message": 85,
    "list_edits": 85,
    "long_ramble": 40,
    "command_question_list": 85,
    "technical_list": 85,
    "ambiguous_intent": 85,
}
# Categories where the validator, not the generator's guess, decides whether a list fits.
LAYOUT_OPEN = {"ambiguous_intent"}

NAMES = (
    "Aisha Ben Carlos Dana Elif Farid Grace Hiro Ines Jamal Kofi Lena Mateo Nadia Omar Priya Quinn Rosa Sven Tariq "
    "Uma Vikram Wen Xavier Yara Zoe Mike Sarah Tom Jess Raj Mei Diego Hannah Luca Fatima Noah Chloe Kenji Amara "
    "Dmitri Leah Sam Ravi Olga Emeka Bea Pat Colin Marisol"
).split()
SETTINGS = (
    "a small bakery", "a hospital ward", "a college dorm", "a software startup", "a construction site",
    "a family road trip", "a wedding", "a dentist's office", "a high school class", "a law firm",
    "a farm", "a gym", "a book club", "a camping trip", "moving house", "a car repair", "a birthday party",
    "a restaurant kitchen", "a remote team", "a volunteer group", "a garden", "a new puppy", "a job search",
    "a research lab", "a bike shop", "a band rehearsal", "a church event", "a soccer team", "a home renovation",
    "a tax return", "an online store", "a hotel front desk", "a flight delay", "a podcast", "a hackathon",
    "a school fundraiser", "a doctor's appointment", "a landlord", "a neighborhood cleanup", "a video game",
)


def _stem(category: str, held_out: bool) -> str:
    return f"{category}.test" if held_out else category


def output_file(category: str, held_out: bool = False):
    return paths.STAGE3 / f"{_stem(category, held_out)}.jsonl"


def batch_log(category: str, held_out: bool = False):
    return paths.STAGE3 / f"{_stem(category, held_out)}.batches.jsonl"


def rejected_file(category: str, held_out: bool = False):
    return paths.STAGE3 / f"{_stem(category, held_out)}.rejected.jsonl"


def _read_log(category: str, held_out: bool) -> list[dict]:
    path = batch_log(category, held_out)
    return list(read_jsonl(path)) if path.exists() else []


def generator_messages(category: str, style: str, count: int, batch: int, notes: Sequence[str]) -> list[dict]:
    base = (paths.PROMPTS / "generate" / "base.md").read_text(encoding="utf-8")
    style_text = (paths.PROMPTS / "generate" / f"style_{style}.md").read_text(encoding="utf-8")
    guidance = (paths.PROMPTS / "categories" / f"{category}.md").read_text(encoding="utf-8")
    rng = random.Random(f"{category}-{batch}")
    names = ", ".join(rng.sample(NAMES, 8))
    settings = "; ".join(rng.sample(SETTINGS, 5))
    user = (
        f"{guidance}\n\n{style_text}\n\nBatch {batch}. Write {count} transcripts for this category. "
        f"For variety, draw on some of these names ({names}) and settings ({settings}), and invent others."
    )
    if notes:
        user += (
            "\n\nThe checker rejected these transcripts from the previous batch. Avoid the same problems:\n"
            + "\n".join(f"- {note}" for note in notes)
        )
    return [{"role": "system", "content": base}, {"role": "user", "content": user}]


# One malformed item (the generator once wrote Python, `"a".split() + ["b"]`, as a tag list) fails the
# whole reply; this recovers each item's transcript and layout from the text instead.
_ITEM = re.compile(r'"input":\s*("(?:[^"\\]|\\.)*")\s*,\s*"layout":\s*"(\w+)"')
_OBJECT = re.compile(r'\{(?:[^{}"]|"(?:[^"\\]|\\.)*")*\}')


def parse_items(reply: str, count: int) -> list[dict]:
    try:
        parsed = parse_json_reply(reply)
        items = parsed["items"] if isinstance(parsed, dict) else parsed
    except ValueError:
        items = []
        for text in _OBJECT.findall(reply):
            try:
                items.append(json.loads(text))
            except ValueError:
                items.extend({"input": json.loads(value), "layout": layout} for value, layout in _ITEM.findall(text))
        if not items:
            raise
    kept, seen = [], set()
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("input"), str):
            continue
        text = " ".join(item["input"].split())
        if not text or text in seen:
            continue
        seen.add(text)
        tags = [str(tag) for tag in item.get("tags", []) if isinstance(tag, str)][:4]
        evidence = [e for e in item.get("evidence", []) if isinstance(e, str)][:6]
        kept.append({"input": text, "layout": str(item.get("layout", "prose")), "tags": tags, "evidence": evidence})
    return kept[:count]


def layout_mismatch(expected: str, layout: str) -> str | None:
    """The teacher's layout against the generator's intent: a list where prose was meant, or the reverse."""
    has_list = layout in ("list", "mixed")
    if expected == "prose" and has_list:
        return "the teacher made a list where the speaker was not making one"
    if expected in ("list", "mixed") and not has_list:
        return "the teacher kept a list as sentences"
    return None


async def run_batch(
    llm: LlmClient, category: str, batch: int, count: int, notes: Sequence[str], known_inputs: set[str]
) -> tuple[BatchRecord, list[Example], list[dict]]:
    style = STYLES[batch % len(STYLES)]
    record = BatchRecord(batch=batch, style=style, requested=count)
    started = time.monotonic()
    reply = await llm.complete(generator_messages(category, style, count, batch, notes), role="generator")
    try:
        items = [item for item in parse_items(reply, count) if item["input"] not in known_inputs]
    except (ValueError, KeyError, TypeError) as error:
        record.error = f"unreadable generator reply: {error}"[:300]
        record.seconds = round(time.monotonic() - started, 1)
        return record, [], []
    record.returned = len(items)

    async with task_group() as group:
        tasks = [group.create_task(relabel(llm, item["input"])) for item in items]
    outputs = [task.result() for task in tasks]
    examples = [
        Example(
            id=example_id(SOURCE, item["input"]),
            input=item["input"],
            output=output,
            source=SOURCE,
            category=category,
            license=LICENSE,
            tags=[*item["tags"], f"style_{style}"],
            teacher_model=llm.settings.model,
        )
        for item, output in zip(items, outputs, strict=True)
    ]

    rejected: list[dict] = []
    candidates = []
    for item, example in zip(items, examples, strict=True):
        reason = None if category in LAYOUT_OPEN else layout_mismatch(item["layout"], example.layout)
        if example.output.strip() == EMPTY:
            reason = "the teacher returned EMPTY for a transcript with dictated words"
        if reason:
            rejected.append({**asdict(example), "failed": ["layout_intent"], "note": reason})
        else:
            candidates.append(example)

    instructions = (paths.PROMPTS / "judge" / "generate.md").read_text(encoding="utf-8")
    chunks = [candidates[i : i + JUDGE_BATCH_SIZE] for i in range(0, len(candidates), JUDGE_BATCH_SIZE)]
    async with task_group() as group:
        tasks = [group.create_task(judge_batch(
            llm, instructions, CRITERIA,
            [(e.id, {"style": style, "transcript": e.input, "output": e.output}) for e in chunk],
        )) for chunk in chunks]
    verdict_lists = [task.result() for task in tasks]
    passed = []
    for chunk, verdicts in zip(chunks, verdict_lists, strict=True):
        for example, verdict in zip(chunk, verdicts, strict=True):
            if verdict.passed:
                passed.append(example)
            else:
                rejected.append({**asdict(example), "failed": verdict.failed, "note": verdict.note})

    record.passed = len(passed)
    record.notes = [
        f"{', '.join(row['failed'])}: {row['note']} (transcript began: {row['input'][:80]!r})"
        for row in rejected
        if row["note"]
    ][:FEEDBACK_NOTES]
    record.seconds = round(time.monotonic() - started, 1)
    return record, passed, rejected


async def generate_category(
    llm: LlmClient,
    category: str,
    target: int,
    *,
    batch_size: int | None = None,
    held_out: bool = False,
    progress: Callable[[str], None] = lambda line: print(line, file=sys.stderr, flush=True),
) -> int:
    """Generates batches until `target` rows pass, resuming from the category's files.

    `held_out` rows go to `<category>.test.jsonl`, pinned to the test split. Their batches are
    numbered after the training batches, so each request draws new names and settings.
    """
    batch_size = batch_size or BATCH_SIZES.get(category, BATCH_SIZE)
    out = output_file(category, held_out)
    kept = read_examples(out) if out.exists() else []
    log = _read_log(category, held_out)
    first_batch = len(_read_log(category, False)) if held_out else 0
    known = {e.input for e in kept}
    if held_out and output_file(category).exists():
        known |= {e.input for e in read_examples(output_file(category))}
    max_batches = 3 * -(-target // batch_size) + 2 + sum(is_missing_key_failure(record) for record in log)
    while len(kept) < target:
        if len(log) >= max_batches:
            progress(f"{category}: stopped after {len(log)} batches with {len(kept)}/{target} rows")
            break
        batch = first_batch + len(log)
        notes = log[-1]["notes"] if log else []
        count = min(batch_size, target - len(kept) + batch_size // 5)
        try:
            record, passed, rejected = await run_batch(llm, category, batch, count, notes, known)
        except LlmError as error:
            progress(f"{category}: batch {batch} failed: {error}")
            raise
        passed = [e for e in passed if e.input not in known][: target - len(kept)]
        if held_out:
            passed = [replace(e, split="test") for e in passed]
        known |= {e.input for e in passed}
        kept += passed
        append_jsonl(out, [e.to_json() for e in passed])
        append_jsonl(rejected_file(category, held_out), [json.dumps(row, ensure_ascii=False) for row in rejected])
        append_jsonl(batch_log(category, held_out), [json.dumps(asdict(record), ensure_ascii=False)])
        log.append(asdict(record))
        progress(
            f"{_stem(category, held_out)}: batch {batch} ({record.style}) {record.passed}/{record.returned} passed"
            f"{' ' + record.error if record.error else ''}, {len(kept)}/{target} rows, {record.seconds}s"
        )
    return len(kept)


async def generate(
    llm: LlmClient,
    categories: Sequence[str],
    scale: float = 1.0,
    batch_size: int | None = None,
    held_out: bool = False,
) -> dict[str, int]:
    """Runs every category's loop at once; the client's semaphore caps the requests in flight."""
    goals = TEST_TARGETS if held_out else TARGETS
    targets = {category: max(1, round(goals[category] * scale)) for category in categories}
    async with task_group() as group:
        tasks = [group.create_task(generate_category(llm, category, target, batch_size=batch_size, held_out=held_out))
                 for category, target in targets.items()]
    counts = [task.result() for task in tasks]
    return dict(zip(targets, counts, strict=True))
