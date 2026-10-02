"""Generate outputs for a split, check them, judge them, and summarize per group.

Three sources produce outputs: a trained checkpoint through transformers (`hf`), a GGUF behind
`llama-server` (`server`), and the hosted teacher with the app's current prompt (`teacher`).
Each writes `<dir>/<split>.outputs.jsonl`; scoring reads it back, so a finished generation is
never repeated.
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

from ..datagen.audit import length_bucket
from ..datagen.paths import BUILD, DATA, PROMPTS, TRAINING_ROOT
from ..datagen.schema import Example, read_examples, read_jsonl
from . import prompt
from .metrics import RowChecks, check, input_style

JUDGE_CRITERIA = ("not_answered", "facts_kept", "nothing_invented")
JUDGE_INSTRUCTIONS = PROMPTS / "judge" / "eval.md"
# Categories the acceptance criteria name for answered transcripts.
COMMAND_CATEGORIES = ("dictated_command", "command_question_list")
# The held-out benchmarks of todo 110, scored like a split.
BENCHMARKS = {"bench-long": DATA / "bench" / "long.jsonl", "bench-noise": DATA / "bench" / "noise.jsonl"}
SPLITS = ("train", "val", "test", *BENCHMARKS)
# Todo 109's test rows, by id: later builds add test rows, and scores report the two groups apart.
TEST_IDS_109 = TRAINING_ROOT / "configs" / "test-ids-109.txt"
# A judge request holds up to 15 rows, or fewer when long rows would pass this many characters.
JUDGE_BATCH_ROWS = 15
JUDGE_MAX_CHARS = 16_000


def load_rows(split: str, limit: int | None = None, seed: int = 109) -> list[Example]:
    rows = read_examples(BENCHMARKS.get(split, BUILD / f"{split}.jsonl"))
    if limit is not None and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)
    return rows


def _progress(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


# --- generation ---------------------------------------------------------------------------


def generate_hf(
    model_dir: Path, rows: Sequence[Example], batch_size: int = 48, repeat_penalty: float = prompt.REPEAT_PENALTY
) -> list[dict]:
    """Greedy batched generation with the served decoding settings, longest inputs first."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.chat_template = prompt.chat_template()
    tokenizer.padding_side = "left"
    end_of_turn = tokenizer.convert_tokens_to_ids(prompt.END_OF_TURN)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
    order = sorted(range(len(rows)), key=lambda index: -len(rows[index].input.split()))
    results: dict[int, dict] = {}
    started = time.monotonic()
    for start in range(0, len(order), batch_size):
        batch = order[start : start + batch_size]
        texts = [prompt.render(prompt.messages(rows[index].input), bos_token=tokenizer.bos_token) for index in batch]
        budgets = [prompt.max_tokens(rows[index].input) for index in batch]
        encoded = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max(budgets),
                do_sample=False,
                repetition_penalty=repeat_penalty,
                eos_token_id=end_of_turn,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = generated[:, encoded["input_ids"].shape[1] :].tolist()
        for index, budget, tokens in zip(batch, budgets, new_tokens, strict=True):
            tokens = tokens[:budget]
            ended = end_of_turn in tokens
            if ended:
                tokens = tokens[: tokens.index(end_of_turn)]
            results[index] = {
                "id": rows[index].id,
                "output": tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                "hit_limit": not ended,
                "tokens": len(tokens),
            }
        if (start // batch_size) % 10 == 0:
            _progress(f"generated {len(results)}/{len(rows)} in {time.monotonic() - started:.0f} s")
    return [results[index] for index in range(len(rows))]


async def generate_server(
    url: str,
    rows: Sequence[Example],
    concurrency: int = 4,
    repeat_penalty: float = prompt.REPEAT_PENALTY,
    repeat_last_n: int = prompt.REPEAT_LAST_N,
) -> list[dict]:
    """Sends each transcript to an OpenAI-compatible `llama-server` exactly as the app does."""
    import httpx

    semaphore = asyncio.Semaphore(concurrency)

    async def one(client: httpx.AsyncClient, row: Example) -> dict:
        body = {
            "messages": prompt.messages(row.input),
            "max_tokens": prompt.max_tokens(row.input),
            "temperature": prompt.TEMPERATURE,
            "repeat_penalty": repeat_penalty,
            "repeat_last_n": repeat_last_n,
        }
        async with semaphore:
            started = time.monotonic()
            response = await client.post(f"{url.rstrip('/')}/chat/completions", json=body)
            response.raise_for_status()
            seconds = time.monotonic() - started
        reply = response.json()
        choice = reply["choices"][0]
        return {
            "id": row.id,
            "output": (choice["message"]["content"] or "").strip(),
            "hit_limit": choice["finish_reason"] == "length",
            "tokens": reply["usage"]["completion_tokens"],
            "seconds": round(seconds, 3),
        }

    async with httpx.AsyncClient(timeout=600) as client:
        return await asyncio.gather(*(one(client, row) for row in rows))


async def generate_teacher(rows: Sequence[Example]) -> list[dict]:
    """`LLM_MODEL` with the app's current hosted prompt, through the datagen client and cache."""
    from ..datagen.llm import LlmClient, LlmSettings
    from ..datagen.teacher import relabel

    async with LlmClient(LlmSettings.from_env()) as llm:
        done = 0

        async def one(row: Example) -> dict:
            nonlocal done
            output = await relabel(llm, row.input)
            done += 1
            if done % 100 == 0:
                _progress(f"teacher {done}/{len(rows)} ({llm.calls} calls, {llm.cache_hits} cached)")
            return {"id": row.id, "output": output, "hit_limit": False, "tokens": None}

        return await asyncio.gather(*(one(row) for row in rows))


def write_outputs(path: Path, outputs: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs), encoding="utf-8")


def read_outputs(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in read_jsonl(path)}


# --- judging ------------------------------------------------------------------------------


async def judge(rows: Sequence[Example], outputs: dict[str, dict], out: Path) -> dict[str, list[str]]:
    """Judges every row not yet in `out`; returns the failed criteria per row id."""
    from ..datagen.judge import judge_all
    from ..datagen.llm import LlmClient, LlmSettings

    items = [
        (row.id, {"transcript": row.input, "reference": row.output, "output": outputs[row.id]["output"]})
        for row in rows
    ]
    async with LlmClient(LlmSettings.from_env()) as llm:
        verdicts = await judge_all(
            llm,
            items,
            out,
            instructions_file=JUDGE_INSTRUCTIONS,
            criteria=JUDGE_CRITERIA,
            batch_size=JUDGE_BATCH_ROWS,
            max_chars=JUDGE_MAX_CHARS,
        )
    return {id_: verdict.failed for id_, verdict in verdicts.items()}


# --- scoring ------------------------------------------------------------------------------

# Rate name → (field or judge criterion, whether a True value is the bad outcome).
RATES: dict[str, tuple[str, bool]] = {
    "answered": ("judge:not_answered", True),
    "facts_dropped": ("judge:facts_kept", True),
    "facts_invented": ("judge:nothing_invented", True),
    "judge_error": ("judge:judge_error", True),
    "assistant_opener": ("assistant_opener", True),
    "protected_kept": ("protected_kept", False),
    "formatting_added": ("formatting_added", True),
    "formatting_missed": ("formatting_missed", True),
    "layout_match": ("layout_match", False),
    "bad_list_marker": ("bad_list_marker", True),
    "markdown": ("markdown", True),
    "leftover_filler": ("leftover_filler", True),
    "passthrough_kept": ("passthrough_kept", False),
    "empty_correct": ("empty_correct", False),
    "false_empty": ("false_empty", True),
    "deleted_over_30pct": ("deleted_over_30pct", True),
    "loop": ("loop", True),
    "exact": ("exact", False),
}


def _value(name: str, checks: RowChecks, failed: list[str] | None) -> bool | None:
    field, _ = RATES[name]
    if field.startswith("judge:"):
        if failed is None:
            return None
        criterion = field.removeprefix("judge:")
        return criterion in failed if criterion == "judge_error" else (criterion in failed or "judge_error" in failed)
    return getattr(checks, field)


def _summarize(entries: Sequence[tuple[RowChecks, list[str] | None]]) -> dict:
    summary: dict = {"rows": len(entries)}
    if not entries:
        return summary
    for name in RATES:
        values = [value for checks, failed in entries if (value := _value(name, checks, failed)) is not None]
        if values:
            summary[name] = {"rate": round(100 * sum(values) / len(values), 2), "n": len(values)}
    summary["chrf"] = round(sum(checks.chrf for checks, _ in entries) / len(entries), 2)
    total = sum(checks.protected_tokens_total for checks, _ in entries)
    if total:
        summary["protected_token_recall"] = round(
            100 * sum(checks.protected_tokens_kept for checks, _ in entries) / total, 2
        )
    return summary


def _test_ids_109() -> set[str]:
    return set(TEST_IDS_109.read_text(encoding="utf-8").split()) if TEST_IDS_109.exists() else set()


def score(
    rows: Sequence[Example], outputs: dict[str, dict], failed: dict[str, list[str]] | None = None
) -> dict:
    """Overall rates, and the same rates per category, tag, input style, source, length bucket and
    noise pattern, and for test rows, todo 109's rows apart from rows added since."""
    old_test = _test_ids_109()
    entries = {
        row.id: (check(row, outputs[row.id]["output"], hit_limit=outputs[row.id]["hit_limit"]), (failed or {}).get(row.id))
        for row in rows
    }
    groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        entry = entries[row.id]
        groups["category"][row.category].append(entry)
        groups["style"][input_style(row)].append(entry)
        groups["source"][row.source].append(entry)
        groups["length"][length_bucket(len(row.input.split()))].append(entry)
        if row.split == "test":
            groups["test_group"]["todo109" if row.id in old_test else "new"].append(entry)
        for tag in row.tags:
            groups["tag"][tag].append(entry)
            if tag.startswith("pattern:"):
                groups["pattern"][tag.removeprefix("pattern:")].append(entry)
    return {
        "overall": _summarize(list(entries.values())),
        "commands": _summarize([entries[row.id] for row in rows if row.category in COMMAND_CATEGORIES]),
        **{kind: {name: _summarize(group) for name, group in sorted(by.items())} for kind, by in groups.items()},
    }


def worst_rows(
    rows: Sequence[Example], outputs: dict[str, dict], name: str, limit: int = 20
) -> list[dict]:
    """Rows failing one deterministic check, for reading by hand."""
    field, bad = RATES[name]
    picked = []
    for row in rows:
        value = getattr(check(row, outputs[row.id]["output"], hit_limit=outputs[row.id]["hit_limit"]), field)
        if value is not None and value == bad:
            picked.append({"id": row.id, "input": row.input, "reference": row.output, "output": outputs[row.id]["output"]})
    return picked[:limit]


def evaluate(
    produce: Callable[[Sequence[Example]], list[dict]],
    out_dir: Path,
    split: str,
    limit: int | None = None,
    with_judge: bool = False,
) -> dict:
    """Generates (or reuses) outputs for a split, scores them, and writes `<split>.scores.json`."""
    rows = load_rows(split, limit)
    stem = split if limit is None else f"{split}{limit}"
    outputs_file = out_dir / f"{stem}.outputs.jsonl"
    outputs = read_outputs(outputs_file) if outputs_file.exists() else {}
    # Only rows without an output are generated: a benchmark that dropped rows reuses the rest.
    missing = [row for row in rows if row.id not in outputs]
    if missing:
        started = time.monotonic()
        generated = produce(missing)
        _progress(f"generated {len(generated)} outputs in {time.monotonic() - started:.0f} s")
        outputs |= {row["id"]: row for row in generated}
        write_outputs(outputs_file, list(outputs.values()))
    outputs = {row.id: outputs[row.id] for row in rows}
    failed = asyncio.run(judge(rows, outputs, out_dir / f"{stem}.judge.jsonl")) if with_judge else None
    scores = score(rows, outputs, failed)
    scores["judged"] = with_judge
    (out_dir / f"{stem}.scores.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    return scores
