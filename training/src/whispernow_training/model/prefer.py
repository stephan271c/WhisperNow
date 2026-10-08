"""The preference stage (todo 111): sample an SFT model, judge its outputs, and pair them.

Every step writes a file under `runs/<run>/prefer/` and is skipped when its file exists, so a
stopped run resumes:

1. **inputs.jsonl** — train rows only, never a benchmark row: `PER_BUCKET` rows from each long
   bucket, and `CORRECTIONS` more rows with a reasoned or re-decided correction of any length.
2. **samples.jsonl** — per input, the greedy output (`k` 0) and `SAMPLES` samples at `TEMPERATURE`,
   from `llama-server` serving the SFT model with the app's request and penalty.
3. **verdicts.jsonl** — each distinct output, checked in code (loops, fillers the reference does not
   have, Markdown, list markers, added layout, an assistant opener) and by the eval judge
   (`prompts/judge/eval.md`, `gpt-6.1-sol` at high reasoning, as `cleanup eval --judge` scores), which
   lists each fault by kind against the reference. An output fails on a dropped, changed, wrong-side
   or invented fact or an answered transcript; an unresolved correction alone does not fail it
   (todo 114). The judge's replies are kept in **verdicts.judge.jsonl**, so a stopped run resumes.
4. **restored.jsonl** — minimal pairs: a failing output whose only faults are lost facts (dropped,
   changed or wrong side) gets them fixed by the teacher (`prompts/prefer/restore.md`). It is kept
   when the code confirms the edit touches only those facts (`only_insertions` for drops,
   `only_fact_edits` otherwise) and the eval judge passes it.
5. **pairs.jsonl** — up to `PAIRS_PER_INPUT` pairs per input: the minimal pair first, then failing
   outputs against the passing output nearest in length. An input with nothing passing pairs a
   failure with its gated reference instead. `kind` names the rejected output's fault.
   **rft.jsonl** — rejection-sampling SFT, the baseline: one passing output per input.

`split` puts `VAL_SHARE` of the inputs, by id hash, in a val split for DPO's eval loss.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path

from ..datagen.audit import length_bucket
from ..datagen.paths import BUILD, PROMPTS
from ..datagen.schema import EMPTY, Example, read_examples, read_jsonl
from . import prompt
from .evaluate import (
    BENCHMARKS,
    FAULT_KINDS,
    JUDGE_BATCH_ROWS,
    JUDGE_CRITERIA,
    JUDGE_INSTRUCTIONS,
    JUDGE_MAX_CHARS,
    JUDGE_SOLO_OVER_WORDS,
    LOST_KINDS,
    R112_TARGET,
)
from .metrics import check

PER_BUCKET = {"150-300": 400, "300-600": 400, "600-1000": 400}
CORRECTIONS = 300
CORRECTION_TAGS = ("reasoned_correction", "redecision", "correction:reasoned", "correction:redecision")
SAMPLES = 4
TEMPERATURE = 0.8
PAIRS_PER_INPUT = 2
VAL_SHARE = 0.05
SEED = 111
RESTORE = PROMPTS / "prefer" / "restore.md"
# Code checks a passing output must also pass.
CODE_CHECKS = ("loop", "leftover_filler", "markdown", "bad_list_marker", "formatting_added", "assistant_opener", "false_empty")
# Rejected outputs' faults, most serious first; a pair is named by its first.
KINDS = {
    "loop": "loop",
    "answered": "answered",
    "invented": "invented",
    "wrong_side": "wrong_side",
    "changed": "changed",
    "dropped": "dropped",
    "leftover_filler": "filler",
    "assistant_opener": "format",
    "false_empty": "dropped",
    "markdown": "format",
    "bad_list_marker": "format",
    "formatting_added": "format",
}
# The judge's failures, by fault kind; an unresolved correction loses nothing and fails nothing.
JUDGE_FAILURES = (*LOST_KINDS, "invented")
_REPAIRABLE = set(LOST_KINDS)
_TOKEN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]")


def _progress(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def _write(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def split_of(id_: str) -> str:
    fraction = int(hashlib.sha256(f"prefer-{id_}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if fraction < VAL_SHARE else "train"


# --- 1. inputs ----------------------------------------------------------------------------


def select_inputs(train: Sequence[Example], benchmark: Sequence[Example], seed: int = SEED) -> list[Example]:
    """Long train rows by bucket, then rows with a reasoned or re-decided correction."""
    held_out = {row.id for row in benchmark} | {row.seed_group for row in benchmark}
    rows = [
        row for row in train
        if row.output != EMPTY and row.id not in held_out and row.seed_group not in held_out
    ]
    rng = random.Random(seed)
    picked: list[Example] = []
    for bucket, count in PER_BUCKET.items():
        in_bucket = [row for row in rows if length_bucket(len(row.input.split())) == bucket]
        picked += rng.sample(in_bucket, min(count, len(in_bucket)))
    taken = {row.id for row in picked}
    corrections = [row for row in rows if row.id not in taken and set(row.tags) & set(CORRECTION_TAGS)]
    picked += rng.sample(corrections, min(CORRECTIONS, len(corrections)))
    return picked


# --- 2. samples ---------------------------------------------------------------------------


async def sample(url: str, rows: Sequence[Example], samples: int = SAMPLES, temperature: float = TEMPERATURE,
                 concurrency: int = 4) -> list[dict]:
    """Greedy (`k` 0) and `samples` samples per row, with the served request and penalty."""
    import httpx

    semaphore = asyncio.Semaphore(concurrency)
    done = 0

    async def one(client: httpx.AsyncClient, row: Example, k: int) -> dict:
        nonlocal done
        body = {
            "messages": prompt.messages(row.input),
            "max_tokens": prompt.max_tokens(row.input),
            **prompt.decoding(),
            **({"temperature": temperature, "seed": k} if k else {}),
        }
        async with semaphore:
            response = await client.post(f"{url.rstrip('/')}/chat/completions", json=body)
            response.raise_for_status()
        choice = response.json()["choices"][0]
        done += 1
        if done % 500 == 0:
            _progress(f"sampled {done}/{len(rows) * (samples + 1)}")
        return {
            "id": row.id,
            "k": k,
            "output": (choice["message"]["content"] or "").strip(),
            "hit_limit": choice["finish_reason"] == "length",
        }

    async with httpx.AsyncClient(timeout=900) as client:
        return await asyncio.gather(*(one(client, row, k) for row in rows for k in range(samples + 1)))


# --- 3. verdicts --------------------------------------------------------------------------


def code_failures(row: Example, output: str, hit_limit: bool) -> list[str]:
    checks = check(row, output, hit_limit=hit_limit)
    return [name for name in CODE_CHECKS if getattr(checks, name)]


def require_r112(settings, role: str) -> None:
    """Every judge and restore call is `gpt-6.1-sol` at high reasoning through codex, as the scores are."""
    if settings.target(role) != R112_TARGET:
        raise SystemExit(f"the preference stage's {role} must be {'/'.join(R112_TARGET)} "
                         f"(now {'/'.join(settings.target(role))})")


async def judge_outputs(llm, rows: dict[str, Example], items: Sequence[tuple[str, str, str]], out: Path) -> dict:
    """(key, row id, output) items through the eval judge, against each row's reference; resumes from `out`."""
    from ..datagen.judge import judge_all

    payload = [(key, {"transcript": rows[id_].input, "reference": rows[id_].output, "output": output})
               for key, id_, output in items]
    return await judge_all(llm, payload, out, instructions_file=JUDGE_INSTRUCTIONS, criteria=JUDGE_CRITERIA,
                           batch_size=JUDGE_BATCH_ROWS, max_chars=JUDGE_MAX_CHARS,
                           solo_over_words=JUDGE_SOLO_OVER_WORDS, fault_kinds=FAULT_KINDS, progress=_progress)


def judge_failures(verdict) -> tuple[list[str], list[dict]]:
    """The fault kinds an output fails on, most serious first, and the faults behind them."""
    if "judge_error" in verdict.failed:
        return ["judge_error"], []
    faults = [fault for fault in verdict.faults if fault.get("kind") in JUDGE_FAILURES]
    kinds = {fault["kind"] for fault in faults}
    failed = ["answered"] if "not_answered" in verdict.failed else []
    failed += [kind for kind in ("invented", "wrong_side", "changed", "dropped") if kind in kinds]
    return failed, faults


def fault_note(faults: Sequence[dict]) -> str:
    return "\n".join(f"- {fault['kind']}: {fault.get('fact', '')}" for fault in faults)


def distinct_outputs(samples: Sequence[dict]) -> list[dict]:
    """The first sample of each distinct (id, output)."""
    seen, out = set(), []
    for sample_ in sorted(samples, key=lambda s: (s["id"], s["k"])):
        key = (sample_["id"], sample_["output"])
        if key not in seen:
            seen.add(key)
            out.append(sample_)
    return out


async def verdicts_for(rows: dict[str, Example], samples: Sequence[dict], judged: Path) -> list[dict]:
    """A verdict per distinct output: code failures, then the eval judge's on outputs the code passed."""
    from ..datagen.llm import LlmClient, LlmSettings

    distinct = distinct_outputs(samples)
    results = []
    for item in distinct:
        failed = code_failures(rows[item["id"]], item["output"], item["hit_limit"])
        results.append({"id": item["id"], "k": item["k"], "output": item["output"], "failed": failed, "faults": [],
                        "note": ""})
    to_judge = [result for result in results if "loop" not in result["failed"]]
    settings = LlmSettings.from_env()
    require_r112(settings, "judge")
    async with LlmClient(settings) as llm:
        verdicts = await judge_outputs(llm, rows, [(f"{r['id']}#{r['k']}", r["id"], r["output"]) for r in to_judge],
                                       judged)
        _progress(f"judged {len(to_judge)} outputs; llm calls {llm.calls}, cached {llm.cache_hits}")
    for result in to_judge:
        failed, faults = judge_failures(verdicts[f"{result['id']}#{result['k']}"])
        result["failed"] = [*failed, *result["failed"]]
        result["faults"] = faults
        result["note"] = fault_note(faults)
    for result in results:
        result["passed"] = not result["failed"]
    return results


# --- 4. minimal pairs ---------------------------------------------------------------------


def only_insertions(before: str, after: str) -> bool:
    """Whether `after` is `before` with words inserted: no word removed or changed. Case,
    punctuation and line breaks may change around an insertion."""
    a = [token.lower() for token in _TOKEN.findall(before)]
    b = [token.lower() for token in _TOKEN.findall(after)]
    inserted = False
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        if any(re.match(r"\w", token) for token in a[i1:i2]):
            return False
        inserted = inserted or any(re.match(r"\w", token) for token in b[j1:j2])
    return inserted


_QUOTED = re.compile(r"(?<!\w)'(.+?)'(?!\w)")


def only_fact_edits(before: str, after: str, faults: Sequence[dict]) -> bool:
    """Whether `after` is `before` with words inserted and only words the faults quote removed or
    replaced: the fix of a changed or wrong-side value touches that value and nothing else."""
    quoted = {
        token.lower()
        for fault in faults
        for phrase in _QUOTED.findall(fault.get("fact", ""))
        for token in _TOKEN.findall(phrase)
    }
    a = [token.lower() for token in _TOKEN.findall(before)]
    b = [token.lower() for token in _TOKEN.findall(after)]
    edited = False
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        removed = [token for token in a[i1:i2] if re.match(r"\w", token)]
        if any(token not in quoted for token in removed):
            return False
        edited = edited or bool(removed) or any(re.match(r"\w", token) for token in b[j1:j2])
    return edited


def restore_messages(transcript: str, output: str, note: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": RESTORE.read_text(encoding="utf-8")},
        {"role": "user", "content": f"TRANSCRIPT:\n{transcript}\n\nOUTPUT:\n{output}\n\nNOTE:\n{note}"},
    ]


def restore_candidates(verdicts: Sequence[dict]) -> list[dict]:
    """Per input, one failing output whose only faults are lost facts, the lowest `k` first."""
    picked: dict[str, dict] = {}
    for verdict in sorted(verdicts, key=lambda v: (v["id"], v["k"])):
        if verdict["failed"] and set(verdict["failed"]) <= _REPAIRABLE and verdict["id"] not in picked:
            picked[verdict["id"]] = verdict
    return list(picked.values())


def edit_ok(candidate: dict, restored: str) -> bool:
    """Drops are fixed by insertion only; a changed or wrong-side value may be replaced, and nothing else."""
    if set(candidate["failed"]) == {"dropped"}:
        return only_insertions(candidate["output"], restored)
    return only_fact_edits(candidate["output"], restored, candidate["faults"])


async def restore(rows: dict[str, Example], verdicts: Sequence[dict], judged: Path) -> list[dict]:
    from ..datagen.llm import LlmClient, LlmSettings
    from ..datagen.teacher import clean_teacher_output

    candidates = restore_candidates(verdicts)
    settings = LlmSettings.from_env()
    require_r112(settings, "teacher")
    require_r112(settings, "judge")
    async with LlmClient(settings) as llm:
        replies = await asyncio.gather(
            *(llm.complete(restore_messages(rows[c["id"]].input, c["output"], c["note"])) for c in candidates)
        )
        results = []
        for candidate, reply in zip(candidates, replies, strict=True):
            restored = clean_teacher_output(reply).strip()
            ok = edit_ok(candidate, restored)
            results.append({"id": candidate["id"], "k": candidate["k"], "rejected": candidate["output"],
                            "restored": restored, "fixed": candidate["failed"], "edit_ok": ok,
                            "failed": [] if ok else ["edited"]})
        checked = [r for r in results if r["edit_ok"]]
        verdicts_ = await judge_outputs(llm, rows, [(f"{r['id']}#restored", r["id"], r["restored"]) for r in checked],
                                        judged)
        for result in checked:
            failed, faults = judge_failures(verdicts_[f"{result['id']}#restored"])
            result["failed"] = [*failed, *code_failures(rows[result["id"]], result["restored"], False)]
            result["note"] = fault_note(faults)
    for result in results:
        result["passed"] = not result["failed"]
    _progress(f"restored {sum(r['passed'] for r in results)}/{len(results)} "
              f"({sum(r['edit_ok'] for r in results)} edited only the named facts)")
    return results


# --- 5. pairs -----------------------------------------------------------------------------


def kind_of(failed: Sequence[str]) -> str:
    for name, kind in KINDS.items():
        if name in failed:
            return kind
    return "other"


def _words(text: str) -> int:
    return len(text.split())


def build_pairs(rows: dict[str, Example], verdicts: Sequence[dict], restored: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    """Preference pairs and the rejection-sampling SFT rows, per input in `rows` order."""
    by_id: dict[str, list[dict]] = defaultdict(list)
    for verdict in verdicts:
        by_id[verdict["id"]].append(verdict)
    minimal = {r["id"]: r for r in restored if r["passed"]}
    pairs, rft = [], []
    for id_, row in rows.items():
        outputs = sorted(by_id.get(id_, []), key=lambda v: v["k"])
        passes = [v for v in outputs if v["passed"]]
        fails = [v for v in outputs if not v["passed"] and "judge_error" not in v["failed"]]
        chosen_rows: list[tuple[str, str, str, str]] = []  # (chosen, rejected, kind, chosen_from)
        if id_ in minimal:
            chosen_rows.append((minimal[id_]["restored"], minimal[id_]["rejected"], "minimal", "restored"))
        for fail in fails:
            if len(chosen_rows) >= PAIRS_PER_INPUT:
                break
            if any(rejected == fail["output"] for _, rejected, _, _ in chosen_rows):
                continue
            if passes:
                best = min(passes, key=lambda v: (abs(_words(v["output"]) - _words(fail["output"])), v["k"]))
                chosen_rows.append((best["output"], fail["output"], kind_of(fail["failed"]), "sample"))
            elif id_ not in minimal:
                chosen_rows.append((row.output, fail["output"], kind_of(fail["failed"]), "reference"))
        for chosen, rejected, kind, chosen_from in chosen_rows:
            pairs.append({"id": id_, "split": split_of(id_), "input": row.input, "chosen": chosen,
                          "rejected": rejected, "kind": kind, "chosen_from": chosen_from,
                          "bucket": length_bucket(_words(row.input))})
        keep = passes[0]["output"] if passes else minimal[id_]["restored"] if id_ in minimal else None
        if keep is not None and split_of(id_) == "train":
            rft.append(asdict(Example(id=row.id, input=row.input, output=keep, source=row.source, category=row.category,
                                      license=row.license, tags=[*row.tags, "rft"], seed_group=row.seed_group, split="train")))
    return pairs, rft


def stats(verdicts: Sequence[dict], restored: Sequence[dict], pairs: Sequence[dict], rft: Sequence[dict], rows: dict[str, Example]) -> dict:
    greedy = [v for v in verdicts if v["k"] == 0]
    by_input: dict[str, list[dict]] = defaultdict(list)
    for verdict in verdicts:
        by_input[verdict["id"]].append(verdict)
    return {
        "inputs": len(rows),
        "distinct_outputs": len(verdicts),
        "greedy_pass_pct": round(100 * sum(v["passed"] for v in greedy) / max(1, len(greedy)), 1),
        "distinct_pass_pct": round(100 * sum(v["passed"] for v in verdicts) / max(1, len(verdicts)), 1),
        "inputs_with_pass_and_fail": sum(
            any(v["passed"] for v in vs) and any(not v["passed"] for v in vs) for vs in by_input.values()
        ),
        "inputs_all_fail": sum(not any(v["passed"] for v in vs) for vs in by_input.values()),
        "failed_criteria": dict(Counter(name for v in verdicts for name in v["failed"]).most_common()),
        "greedy_failed_by_bucket": {
            bucket: f"{sum(not v['passed'] for v in greedy if length_bucket(_words(rows[v['id']].input)) == bucket)}/"
            f"{sum(length_bucket(_words(rows[v['id']].input)) == bucket for v in greedy)}"
            for bucket in sorted({length_bucket(_words(row.input)) for row in rows.values()})
        },
        "restore_attempts": len(restored),
        "restore_edit_ok": sum(r["edit_ok"] for r in restored),
        "restore_passed": sum(r["passed"] for r in restored),
        "pairs": len(pairs),
        "pairs_by_kind": dict(Counter(p["kind"] for p in pairs).most_common()),
        "pairs_by_chosen_from": dict(Counter(p["chosen_from"] for p in pairs).most_common()),
        "pairs_val": sum(p["split"] == "val" for p in pairs),
        "length_ratio_rejected_over_chosen": round(
            sum(_words(p["rejected"]) for p in pairs) / max(1, sum(_words(p["chosen"]) for p in pairs)), 3
        ),
        "rft_rows": len(rft),
    }


# --- the steps ----------------------------------------------------------------------------


def run(out: Path, server: str | None, concurrency: int = 4) -> dict:
    """Runs every step whose file is missing; sampling needs `server`."""
    out.mkdir(parents=True, exist_ok=True)
    inputs_file = out / "inputs.jsonl"
    if not inputs_file.exists():
        benchmark = [row for path in BENCHMARKS.values() for row in read_examples(path)]
        _write(inputs_file, [asdict(row) for row in select_inputs(read_examples(BUILD / "train.jsonl"), benchmark)])
    rows = {row.id: row for row in read_examples(inputs_file)}
    samples_file = out / "samples.jsonl"
    if not samples_file.exists():
        if not server:
            raise SystemExit("sampling needs --server, the SFT model behind llama-server")
        _write(samples_file, asyncio.run(sample(server, list(rows.values()), concurrency=concurrency)))
    samples = list(read_jsonl(samples_file))
    verdicts_file = out / "verdicts.jsonl"
    if not verdicts_file.exists():
        _write(verdicts_file, asyncio.run(verdicts_for(rows, samples, out / "verdicts.judge.jsonl")))
    verdicts = list(read_jsonl(verdicts_file))
    restored_file = out / "restored.jsonl"
    if not restored_file.exists():
        _write(restored_file, asyncio.run(restore(rows, verdicts, out / "restored.judge.jsonl")))
    restored = list(read_jsonl(restored_file))
    pairs, rft = build_pairs(rows, verdicts, restored)
    _write(out / "pairs.jsonl", pairs)
    _write(out / "rft.jsonl", rft)
    summary = stats(verdicts, restored, pairs, rft, rows)
    (out / "stats.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
