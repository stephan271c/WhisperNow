"""Check the new eval judge (`gpt-6.1-sol`, high, codex) before scores rely on it.

Both checks judge existing model outputs against the references they were scored against before
(the old ones, from `data/relabel/inputs/` once the relabel has copied them), so only the judge
changes:

- `reviewed`: the 33 rows of the user's hand review (`runs/judge-review.html`, verdicts in
  `runs/judge-review.verdicts.jsonl`). The user found a real fault in 30 outputs and part of one in 3.
  Under the relabel contract hedges and reasons are facts, so all 33 should fail.
- `sample`: stratified rows that the old judge (glm-5.3, low) judged, judged again by the new one.
  The report lists every disagreement for reading.

Writes `runs/judge-check/{reviewed,sample}.judge.jsonl`, `report.json` and `disagreements.md`.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from ..datagen.audit import length_bucket
from ..datagen.judge import Verdict, judge_all, load_verdicts
from ..datagen.llm import LlmClient, LlmSettings
from ..datagen.paths import DATA, TRAINING_ROOT
from ..datagen.schema import Example, read_examples, read_jsonl
from . import evaluate

RUNS = TRAINING_ROOT / "runs"
OUT = RUNS / "judge-check"
VERDICTS = RUNS / "judge-review.verdicts.jsonl"
# The evaluations the old judge scored: two SFT runs and the hosted teacher.
EVALS = {
    "lfm-e3": RUNS / "long-full-inst-1e-5-e3" / "eval-gpu-q8",
    "lfm-e3-noise": RUNS / "long-full-inst-1e-5-e3-noise" / "eval-gpu-q8",
    "teacher": RUNS / "teacher" / "eval",
}
SPLITS = ("bench-long", "bench-noise", "test")
SAMPLE_ROWS = 300
SEED = 112


def old_references(split: str) -> dict[str, Example]:
    """The references the old judge scored against, by id."""
    snapshot = {"bench-long": "bench-long", "bench-noise": "bench-noise", "test": "build"}[split]
    copied = DATA / "relabel" / "inputs" / f"{snapshot}.jsonl"
    rows = read_examples(copied) if copied.exists() else evaluate.load_rows(split)
    if split == "test" and copied.exists():
        rows += read_examples(copied.with_name("build.injected.jsonl"))
    return {e.id: e for e in rows if split != "test" or e.split == "test"}


def _item(row: Example, output: str) -> tuple[str, dict]:
    return row.id, {"transcript": row.input, "reference": row.output, "output": output}


def _settings() -> LlmSettings:
    settings = LlmSettings.from_env()
    if settings.target("judge") != evaluate.R112_TARGET:
        raise SystemExit(f"the judge check needs the judge on {'/'.join(evaluate.R112_TARGET)}")
    return settings


async def _judge(items: list[tuple[str, dict]], out: Path) -> dict[str, Verdict]:
    async with LlmClient(_settings()) as llm:
        return await judge_all(
            llm, items, out, instructions_file=evaluate.JUDGE_INSTRUCTIONS, criteria=evaluate.JUDGE_CRITERIA,
            batch_size=evaluate.JUDGE_BATCH_ROWS, max_chars=evaluate.JUDGE_MAX_CHARS,
            solo_over_words=evaluate.JUDGE_SOLO_OVER_WORDS, fault_kinds=evaluate.FAULT_KINDS,
        )


def _key(eval_dir: str, row_id: str) -> str:
    # One id can be reviewed for two models, so keys name the evaluation too.
    return f"{eval_dir}|{row_id}"


def reviewed_items() -> tuple[list[dict], list[tuple[str, dict]]]:
    reviews = list(read_jsonl(VERDICTS))
    items = []
    for review in reviews:
        row = old_references(review["split"])[review["id"]]
        output = evaluate.read_outputs(RUNS / review["eval_dir"] / f"{review['split']}.outputs.jsonl")[review["id"]]["output"]
        items.append((_key(review["eval_dir"], review["id"]), _item(row, output)[1]))
    return reviews, items


def sample_items() -> list[tuple[str, dict, Verdict]]:
    """`SAMPLE_ROWS` rows the old judge judged, spread over models, splits, old verdicts and lengths."""
    strata: dict[tuple, list] = defaultdict(list)
    for name, eval_dir in EVALS.items():
        for split in SPLITS:
            judged = load_verdicts(eval_dir / f"{split}.judge.jsonl")
            outputs = evaluate.read_outputs(eval_dir / f"{split}.outputs.jsonl")
            references = old_references(split)
            for row_id, verdict in sorted(judged.items()):
                if row_id not in references or row_id not in outputs or "judge_error" in verdict.failed:
                    continue
                row = references[row_id]
                key = (name, split, verdict.passed, length_bucket(len(row.input.split())))
                strata[key].append((_key(str(eval_dir.relative_to(RUNS)), row_id), _item(row, outputs[row_id]["output"])[1], verdict))
    rng = random.Random(SEED)
    for members in strata.values():
        rng.shuffle(members)
    picked = []
    while len(picked) < SAMPLE_ROWS and any(strata.values()):
        for key in sorted(strata, key=str):
            if strata[key] and len(picked) < SAMPLE_ROWS:
                picked.append(strata[key].pop())
    return picked


def run(step: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    if step in ("reviewed", "all"):
        asyncio.run(_judge(reviewed_items()[1], OUT / "reviewed.judge.jsonl"))
    if step in ("sample", "all"):
        asyncio.run(_judge([(key, fields) for key, fields, _ in sample_items()], OUT / "sample.judge.jsonl"))
    return report()


def report() -> dict:
    result: dict = {}
    lines = ["# Eval judge check (todo 112)\n"]
    if (OUT / "reviewed.judge.jsonl").exists():
        reviews, items = reviewed_items()
        new = load_verdicts(OUT / "reviewed.judge.jsonl")
        fields = dict(items)
        rows = []
        for review, (key, _) in zip(reviews, items, strict=True):
            verdict = new.get(key)
            rows.append({**review, "new_failed": verdict.failed if verdict else None, "new_note": verdict.note if verdict else ""})
        judged = [r for r in rows if r["new_failed"] is not None]
        result["reviewed"] = {
            "rows": len(rows),
            "judged": len(judged),
            # A real or partly real fault is matched when the new judge fails the output.
            "matched": sum(bool(r["new_failed"]) for r in judged),
            "by_user_verdict": dict(Counter(f"{r['verdict']}: {'fail' if r['new_failed'] else 'pass'}" for r in judged)),
            "missed": [{k: r[k] for k in ("n", "id", "model", "split", "verdict", "comment")} for r in judged if not r["new_failed"]],
        }
        lines.append(f"## The 33 reviewed rows\n\nThe new judge fails {result['reviewed']['matched']} of {len(judged)} outputs the user found a fault in.\n")
        for r in judged:
            if not r["new_failed"]:
                key = _key(r["eval_dir"], r["id"])
                lines.append(f"### Missed #{r['n']} {r['id']} ({r['model']}, {r['split']}, user: {r['verdict']})\n")
                lines.append(f"User: {r['comment'] or '-'}\n\nOUTPUT:\n\n```\n{fields[key]['output']}\n```\n")
    if (OUT / "sample.judge.jsonl").exists():
        picked = sample_items()
        new = load_verdicts(OUT / "sample.judge.jsonl")
        agree, by_criterion, disagreements = 0, Counter(), []
        judged = [(key, fields, old, new[key]) for key, fields, old in picked if key in new and "judge_error" not in new[key].failed]
        for key, fields, old, verdict in judged:
            if old.passed == verdict.passed:
                agree += 1
            else:
                disagreements.append((key, fields, old, verdict))
            for criterion in evaluate.JUDGE_CRITERIA:
                by_criterion[(criterion, criterion in old.failed, criterion in verdict.failed)] += 1
        result["sample"] = {
            "rows": len(picked),
            "judged": len(judged),
            "agreement": round(agree / max(1, len(judged)), 3),
            "old_fail_new_pass": sum(not old.passed for _, _, old, _ in disagreements),
            "old_pass_new_fail": sum(old.passed for _, _, old, _ in disagreements),
            "per_criterion": {
                criterion: {
                    "agree": round(sum(v for (c, a, b), v in by_criterion.items() if c == criterion and a == b) / max(1, len(judged)), 3),
                    "old_only": by_criterion[(criterion, True, False)],
                    "new_only": by_criterion[(criterion, False, True)],
                }
                for criterion in evaluate.JUDGE_CRITERIA
            },
        }
        lines.append(
            f"## Old judge (glm-5.3, low) against the new one on {len(judged)} stratified rows\n\n"
            f"Whole-row agreement {result['sample']['agreement']:.1%}: {result['sample']['old_fail_new_pass']} rows only the old "
            f"judge failed, {result['sample']['old_pass_new_fail']} only the new one.\n"
        )
        for key, fields, old, verdict in disagreements:
            lines.append(f"### {key}\n")
            lines.append(f"- old: {'pass' if old.passed else 'FAIL ' + ', '.join(old.failed)}: {old.note}")
            lines.append(f"- new: {'pass' if verdict.passed else 'FAIL ' + ', '.join(verdict.failed)}: {verdict.note}\n")
            for name in ("transcript", "reference", "output"):
                lines.append(f"{name.upper()}:\n\n```\n{fields[name]}\n```\n")
    (OUT / "report.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    (OUT / "disagreements.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
