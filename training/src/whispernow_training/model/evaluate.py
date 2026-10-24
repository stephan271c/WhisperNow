"""Generate outputs for a split, check them, judge them, and summarize per group.

Three sources produce outputs: a trained checkpoint through transformers (`hf`), a GGUF behind
`llama-server` (`server`), and the hosted teacher with the app's current prompt (`teacher`).
Each appends completed requests to `<dir>/<split>.outputs.jsonl`; matching requests resume.
`rescore` scores saved outputs and verdicts again with the current checks, and `rejudge` judges
chosen rows again with the current judge.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from urllib.parse import urlparse

from ..datagen.concurrency import task_group
from ..datagen.judge import Verdict, load_verdicts
from ..datagen.llm import R112_TARGET, request_key, require_r112
from ..datagen.paths import BUILD, DATA, PROMPTS, TRAINING_ROOT
from ..datagen.schema import Example, length_bucket, read_examples, read_jsonl
from . import draft, prompt
from .metrics import RowChecks, check, input_style

JUDGE_CRITERIA = ("not_answered", "facts_kept", "nothing_invented")
JUDGE_INSTRUCTIONS = PROMPTS / "judge" / "eval.md"
# The judge lists each fault by kind, and the criterion each kind fails. An unresolved correction
# (the output still says which value was taken back) loses nothing, so it is only counted.
FAULT_KINDS: dict[str, str | None] = {
    "dropped": "facts_kept",
    "changed": "facts_kept",
    "wrong_side": "facts_kept",
    "unresolved": None,
    "invented": "nothing_invented",
}
# The kinds that lose or corrupt a fact the speaker meant, counted together as `facts_lost`.
LOST_KINDS = ("dropped", "changed", "wrong_side")
# Judge and score files carry this after the reference tag, so verdicts of the pass/fail judge
# before fault counts (`<stem>.r112.judge.jsonl`) are never resumed or overwritten.
JUDGE_VERSION = "faults"
# The definition of the mechanical checks, recorded in every scores file as `metrics_version`; bump
# it when a check here or in `text_rules` changes meaning. A file without one is version 1: it
# predates whole-token protected matching, and may predate other rule changes too.
METRICS_VERSION = 2
# Categories the acceptance criteria name for answered transcripts.
COMMAND_CATEGORIES = ("dictated_command", "command_question_list")
# The held-out benchmarks, scored like a split. bench-cutoff is test rows with cut-off word starts
# (`datagen cutoff bench`), scored by `datagen cutoff score` against the same test rows' outputs.
BENCHMARKS = {"bench-long": DATA / "bench" / "long.jsonl", "bench-noise": DATA / "bench" / "noise.jsonl",
              "bench-cutoff": DATA / "bench" / "cutoff.jsonl"}
# Fixed subsets of a split, scored like a split, for models too slow or costly to run on all of it:
# name → (split, id file). `test-sample` is 800 test rows drawn per category and length bucket in
# proportion to the split. Of 500 such draws (seeds 0–499), seed 408's came closest to the full split
# in facts lost per 1,000 words for glm-5.3 max, glm-5.3-flash and todo 113's two models: within 1.6%
# for each, where the median draw is 10% off for one of them. `test-long` is the 536 test rows over
# 200 words, where the local models lose the most facts per word.
SAMPLES = {"test-sample": ("test", TRAINING_ROOT / "configs" / "test-sample-800.txt"),
           "test-long": ("test", TRAINING_ROOT / "configs" / "test-long-536.txt")}
SPLITS = ("train", "val", "test", *BENCHMARKS, *SAMPLES)
# The original test rows, by id: later builds add test rows, and scores report the two groups apart.
TEST_IDS_109 = TRAINING_ROOT / "configs" / "test-ids-109.txt"
# A judge request holds up to 15 rows, or fewer when long rows would pass this many characters.
JUDGE_BATCH_ROWS = 15
JUDGE_MAX_CHARS = 16_000
# Rows longer than this get a judge request each.
JUDGE_SOLO_OVER_WORDS = 300
# References the relabel rewrote are judged and scored apart from the old ones, by the model that wrote
# them: `<stem>.r112.judge.jsonl` and `<stem>.r112.scores.json`.
R112 = "r112"


def reference_tag(rows: Sequence[Example]) -> str:
    """`R112` when every row's reference is the relabel's, `""` for the references before it."""
    if rows and all((row.teacher_model, row.teacher_reasoning) == R112_TARGET[1:] for row in rows):
        return R112
    return ""


def load_rows(split: str, limit: int | None = None, seed: int = 109) -> list[Example]:
    parent, ids_file = SAMPLES.get(split, (split, None))
    rows = read_examples(BENCHMARKS.get(parent, BUILD / f"{parent}.jsonl"))
    if ids_file is not None:
        ids = set(ids_file.read_text(encoding="utf-8").split())
        rows = [row for row in rows if row.id in ids]
    if limit is not None and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)
    return rows


def _progress(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def model_revision(path: Path) -> str:
    files = sorted(file for file in path.rglob("*") if file.is_file()) if path.is_dir() else [path]
    digests = {}
    for file in files:
        with file.open("rb") as source:
            name = str(file.relative_to(path)) if path.is_dir() else file.name
            digests[name] = hashlib.file_digest(source, "sha256").hexdigest()
    return request_key(digests)


def hf_generation(model_dir: Path, repeat_penalty: float = prompt.REPEAT_PENALTY) -> dict:
    from importlib.metadata import version

    return {"source": "hf", "revision": model_revision(model_dir), "chat_template": prompt.chat_template(),
            "repeat_penalty": repeat_penalty, "do_sample": False, "dtype": "bfloat16",
            "end_of_turn": prompt.END_OF_TURN, "transformers": version("transformers"), "torch": version("torch")}


def server_generation(
    url: str, repeat_penalty: float = prompt.REPEAT_PENALTY, repeat_last_n: int = prompt.REPEAT_LAST_N,
) -> dict:
    import httpx

    root = url.rstrip("/").removesuffix("/v1")
    response = httpx.get(f"{root}/props", timeout=30)
    response.raise_for_status()
    props = response.json()
    model = Path(props["model_path"])
    defaults = props["default_generation_settings"]
    is_local = urlparse(url).hostname in ("localhost", "127.0.0.1", "::1")
    return {"source": "server", "revision": model_revision(model) if is_local and model.is_absolute() and model.is_file() else None,
            "model_path": str(model), "chat_template": props["chat_template"], "build": props.get("build_info"),
            "defaults": defaults["params"], "n_ctx": defaults["n_ctx"], "temperature": prompt.TEMPERATURE,
            "repeat_penalty": repeat_penalty, "repeat_last_n": repeat_last_n}


def teacher_generation() -> dict:
    from ..datagen.llm import LlmSettings
    from ..datagen.teacher import teacher_messages

    return {"source": "teacher", "settings": LlmSettings.from_env().request("teacher"),
            "messages": teacher_messages("")}


def generate_hf(
    model_dir: Path, rows: Sequence[Example], batch_size: int = 48, repeat_penalty: float = prompt.REPEAT_PENALTY
) -> list[dict]:
    """Greedy batched generation with the served decoding settings, longest inputs first."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.chat_template = prompt.chat_template()
    tokenizer.padding_side = "left"
    end_token_id = tokenizer.convert_tokens_to_ids(prompt.END_OF_TURN)
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
                eos_token_id=end_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = generated[:, encoded["input_ids"].shape[1] :].tolist()
        for index, budget, tokens in zip(batch, budgets, new_tokens, strict=True):
            tokens = tokens[:budget]
            ended = end_token_id in tokens
            if ended:
                tokens = tokens[: tokens.index(end_token_id)]
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
            "draft_text": draft.preclean(row.input),
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
        async with task_group() as group:
            tasks = [group.create_task(one(client, row)) for row in rows]
        return [task.result() for task in tasks]


async def generate_teacher(rows: Sequence[Example]) -> list[dict]:
    """`LLM_MODEL` with the app's current hosted prompt, through the datagen client and cache.

    An OpenRouter `:batch` model gets every row in one batch first.
    """
    from ..datagen.llm import LlmClient, LlmSettings
    from ..datagen.teacher import relabel, teacher_messages

    async with LlmClient(LlmSettings.from_env()) as llm:
        if llm.settings.model.endswith(":batch"):
            await llm.complete_batch([teacher_messages(row.input) for row in rows])
        done = 0

        async def one(row: Example) -> dict:
            nonlocal done
            output = await relabel(llm, row.input)
            done += 1
            if done % 100 == 0:
                _progress(f"teacher {done}/{len(rows)} ({llm.calls} calls, {llm.cache_hits} cached)")
            return {"id": row.id, "output": output, "hit_limit": False, "tokens": None}

        async with task_group() as group:
            tasks = [group.create_task(one(row)) for row in rows]
        return [task.result() for task in tasks]


def write_outputs(path: Path, outputs: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as sink:
        sink.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs)


def read_outputs(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in read_jsonl(path)}


async def judge_outputs(llm, items: Sequence[tuple[str, dict]], out: Path,
                        progress: Callable[[str], None] | None = None) -> dict[str, Verdict]:
    from ..datagen.judge import judge_all

    return await judge_all(
        llm, items, out, instructions_file=JUDGE_INSTRUCTIONS, criteria=JUDGE_CRITERIA,
        batch_size=JUDGE_BATCH_ROWS, max_chars=JUDGE_MAX_CHARS, solo_over_words=JUDGE_SOLO_OVER_WORDS,
        fault_kinds=FAULT_KINDS, progress=progress if progress is not None else _progress,
    )


async def judge(
    rows: Sequence[Example], outputs: dict[str, dict], out: Path, r112: bool = False
) -> dict[str, Verdict]:
    """Judges current transcript/reference/output requests; returns each row's verdict by id."""
    from ..datagen.llm import LlmClient, LlmSettings

    settings = LlmSettings.from_env()
    if r112:
        require_r112(settings, "judge")

    items = [
        (row.id, {"transcript": row.input, "reference": row.output, "output": outputs[row.id]["output"]})
        for row in rows
    ]
    async with LlmClient(settings) as llm:
        return await judge_outputs(llm, items, out)


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
    "em_dash": ("em_dash", True),
    "spelled_count": ("spelled_count", True),
    "correction_cue": ("correction_cue", True),
}


def _value(name: str, checks: RowChecks, judged: Verdict | None) -> bool | None:
    field, _ = RATES[name]
    if field.startswith("judge:"):
        if judged is None:
            return None
        failed = judged.failed
        criterion = field.removeprefix("judge:")
        return criterion in failed if criterion == "judge_error" else (criterion in failed or "judge_error" in failed)
    return getattr(checks, field)


def _fault_counts(entries: Sequence[tuple[RowChecks, Verdict | None]]) -> dict | None:
    """Faults per judged row and per 1,000 reference words by kind, and the share of rows with one.
    A row the judge could not read counts in none."""
    judged = [(checks, verdict) for checks, verdict in entries if verdict is not None and "judge_error" not in verdict.failed]
    if not judged:
        return None
    words = sum(checks.reference_words for checks, _ in judged)
    kinds = {kind: (kind,) for kind in FAULT_KINDS} | {"facts_lost": LOST_KINDS}
    counts = {}
    for name, members in kinds.items():
        per_row = [sum(fault["kind"] in members for fault in verdict.faults) for _, verdict in judged]
        counts[name] = {
            "per_row": round(sum(per_row) / len(per_row), 3),
            "per_1000_words": round(1000 * sum(per_row) / words, 2) if words else None,
            "rows": round(100 * sum(count > 0 for count in per_row) / len(per_row), 2),
            "n": len(per_row),
        }
    return counts


def _summarize(entries: Sequence[tuple[RowChecks, Verdict | None]]) -> dict:
    summary: dict = {"rows": len(entries)}
    if not entries:
        return summary
    for name in RATES:
        values = [value for checks, judged in entries if (value := _value(name, checks, judged)) is not None]
        if values:
            summary[name] = {"rate": round(100 * sum(values) / len(values), 2), "n": len(values)}
    summary["chrf"] = round(sum(checks.chrf for checks, _ in entries) / len(entries), 2)
    reference_words = sum(checks.reference_words for checks, _ in entries)
    if reference_words:
        summary["length_ratio"] = round(sum(checks.output_words for checks, _ in entries) / reference_words, 3)
    # Hedges and reasons the transcript said that the output kept; a drop can be the abandoned side
    # of a correction, so the rate is read beside the reference's own.
    said, kept = Counter(), Counter()
    for checks, _ in entries:
        for word, (times, kept_times) in checks.kept_words.items():
            said[word] += times
            kept[word] += kept_times
    if said:
        summary["kept_words"] = {word: {"rate": round(100 * kept[word] / said[word], 2), "n": said[word]} for word in sorted(said)}
    total = sum(checks.protected_tokens_total for checks, _ in entries)
    if total:
        summary["protected_token_recall"] = round(
            100 * sum(checks.protected_tokens_kept for checks, _ in entries) / total, 2
        )
    if (faults := _fault_counts(entries)) is not None:
        summary["faults"] = faults
    return summary


def _test_ids_109() -> set[str]:
    return set(TEST_IDS_109.read_text(encoding="utf-8").split())


def score(rows: Sequence[Example], outputs: dict[str, dict], judged: dict[str, Verdict] | None = None) -> dict:
    """Overall rates, and the same rates per category, tag, input style, source, length bucket and
    noise pattern, and for test rows, the `TEST_IDS_109` rows apart from rows added since."""
    original_test_ids = _test_ids_109()
    entries = {
        row.id: (check(row, outputs[row.id]["output"], hit_limit=outputs[row.id]["hit_limit"]), (judged or {}).get(row.id))
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
            groups["test_group"]["todo109" if row.id in original_test_ids else "new"].append(entry)
        for tag in row.tags:
            groups["tag"][tag].append(entry)
            if tag.startswith("pattern:"):
                groups["pattern"][tag.removeprefix("pattern:")].append(entry)
    return {
        "overall": _summarize(list(entries.values())),
        "commands": _summarize([entries[row.id] for row in rows if row.category in COMMAND_CATEGORIES]),
        **{kind: {name: _summarize(group) for name, group in sorted(by.items())} for kind, by in groups.items()},
    }


def _file_stem(stem: str, tag: str, judged: bool) -> str:
    tagged = f"{stem}.{tag}" if tag else stem
    return f"{tagged}.{JUDGE_VERSION}" if judged else tagged


def rescored_path(out_dir: Path, file_stem: str) -> Path:
    """Where `rescore` puts `<file_stem>.scores.json`'s scores under the current checks."""
    return out_dir / f"{file_stem}.metrics{METRICS_VERSION}.scores.json"


def write_scores(
    path: Path, rows: Sequence[Example], outputs: dict[str, dict], judged: dict[str, Verdict] | None, tag: str
) -> dict:
    scores = score(rows, outputs, judged)
    scores["judged"] = judged is not None
    if judged is not None:
        scores["judge"] = JUDGE_VERSION
    scores["references"] = tag or "before todo 112"
    scores["metrics_version"] = METRICS_VERSION
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    return scores


def rescore(out_dir: Path, split: str, limit: int | None = None) -> Path:
    """Scores the saved outputs and fault verdicts of `out_dir` with the current checks, generating
    and judging nothing, into `rescored_path` beside the scores they had. A run with no files of its
    own for a `SAMPLES` split is scored from the whole split's."""
    rows = load_rows(split, limit)
    stem = split if limit is None else f"{split}{limit}"
    source = SAMPLES[split][0] if split in SAMPLES and not (out_dir / f"{stem}.outputs.jsonl").exists() else stem
    outputs = read_outputs(out_dir / f"{source}.outputs.jsonl")
    if unsaved := [row.id for row in rows if row.id not in outputs]:
        raise ValueError(f"{out_dir}: {len(unsaved)} of {len(rows)} {stem} rows have no saved output")
    tag = reference_tag(rows)
    judged = load_verdicts(out_dir / f"{_file_stem(source, tag, judged=True)}.judge.jsonl") or None
    if judged is not None and (unjudged := [row.id for row in rows if row.id not in judged]):
        raise ValueError(f"{out_dir}: {len(unjudged)} of {len(rows)} {stem} rows have no verdict")
    path = rescored_path(out_dir, _file_stem(stem, tag, judged is not None))
    write_scores(path, rows, {row.id: outputs[row.id] for row in rows}, judged, tag)
    return path


def rejudge(out_dir: Path, split: str, ids: Sequence[str]) -> Path:
    """Judges the saved outputs of the rows `ids` again with the current judge, then `rescore`s.

    The verdicts are appended to the split's verdict file, where a row's last line wins: every other
    row keeps its verdict, and the replaced ones stay on record. A row whose request is unchanged
    since its last verdict is not judged again."""
    rows = {row.id: row for row in load_rows(split)}
    if unknown := [id_ for id_ in ids if id_ not in rows]:
        raise ValueError(f"{len(unknown)} ids are not {split} rows, such as {unknown[0]}")
    tag = reference_tag(list(rows.values()))
    verdicts = out_dir / f"{_file_stem(split, tag, judged=True)}.judge.jsonl"
    if not verdicts.exists():
        raise ValueError(f"{out_dir}: {split} was never judged; run `cleanup eval --judge` on it first")
    outputs = read_outputs(out_dir / f"{split}.outputs.jsonl")
    picked = [rows[id_] for id_ in dict.fromkeys(ids)]
    if unsaved := [row.id for row in picked if row.id not in outputs]:
        raise ValueError(f"{out_dir}: {len(unsaved)} chosen {split} rows have no saved output, such as {unsaved[0]}")
    asyncio.run(judge(picked, {row.id: outputs[row.id] for row in picked}, verdicts, bool(tag)))
    return rescore(out_dir, split)


def evaluate(
    produce: Callable[[Sequence[Example]], list[dict]],
    out_dir: Path,
    split: str,
    limit: int | None = None,
    with_judge: bool = False,
    *,
    generation: dict,
) -> dict:
    """Scores matching generation requests; unverified server revisions are never reused."""
    rows = load_rows(split, limit)
    stem = split if limit is None else f"{split}{limit}"
    outputs_file = out_dir / f"{stem}.outputs.jsonl"
    outputs = read_outputs(outputs_file) if outputs_file.exists() else {}
    keys = {row.id: request_key({"generation": generation, "transcript": row.input,
                                "messages": prompt.messages(row.input), "max_tokens": prompt.max_tokens(row.input),
                                **({"reference": row.output} if generation["source"] == "reference" else {}),
                                **({"draft_text": draft.preclean(row.input)} if generation["source"] == "server" else {})})
            for row in rows}
    reusable = not (generation["source"] == "server" and generation["revision"] is None)
    missing = [row for row in rows if not reusable or outputs.get(row.id, {}).get("request_key") != keys[row.id]]
    if missing:
        started = time.monotonic()
        generated = produce(missing)
        for output in generated:
            output["request_key"] = keys[output["id"]]
        _progress(f"generated {len(generated)} outputs in {time.monotonic() - started:.0f} s")
        outputs |= {row["id"]: row for row in generated}
        write_outputs(outputs_file, generated)
    outputs = {row.id: outputs[row.id] for row in rows}
    tag = reference_tag(rows)
    file_stem = _file_stem(stem, tag, with_judge)
    judged = (
        asyncio.run(judge(rows, outputs, out_dir / f"{file_stem}.judge.jsonl", bool(tag)))
        if with_judge else None
    )
    return write_scores(out_dir / f"{file_stem}.scores.json", rows, outputs, judged, tag)
