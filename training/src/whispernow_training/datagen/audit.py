"""The label audit: are the long references complete against their own transcripts?

The stage 1 judge checked aawaaz rows against aawaaz's standard, never for completeness against the
transcript. `run` judges every built row over 150 words, and a fixed random 500 of the shorter rows
as a control, with the completeness judge (`prompts/judge/completeness.md`). `fix` sends every
failing row through the gates new rows pass (`gates.gate`): a teacher relabel, checked again, and a
second relabel if that fails. `apply` is how `build` uses the result: a relabel that passed replaces
the reference; a train or val row with none is dropped; a test row keeps its id, with the passing
relabel or a hand-written correction (`test_corrections.jsonl`).

Injected rows are complete by construction (the target is the clean text the noise was added to),
so their failures measure the judge's false positives, and they are never relabelled.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, replace

from . import gates, inject, paths
from .judge import Verdict, judge_all, load_verdicts
from .llm import LlmClient
from .schema import EMPTY, Example, read_examples, read_jsonl, write_examples

OUT_DIR = paths.DATA / "audit"
# The audited rows as they were built when the audit ran, so a later build does not change them.
ROWS = OUT_DIR / "rows.jsonl"
VERDICTS = OUT_DIR / "verdicts.jsonl"
RELABELS = OUT_DIR / "relabels.jsonl"
# Test references neither relabel fixed, corrected by hand: {"id", "output"} per line.
TEST_CORRECTIONS = OUT_DIR / "test_corrections.jsonl"
LONG_WORDS = 150
CONTROL_ROWS = 500
SEED = 110
# Upper bounds, inclusive except the first: under 50 words, 50-150, then over 150 up to 300, and so on.
BUCKETS = ((49, "<50"), (150, "50-150"), (300, "150-300"), (600, "300-600"), (1000, "600-1000"))


def words(text: str) -> int:
    return len(text.split())


def length_bucket(word_count: int) -> str:
    for limit, name in BUCKETS:
        if word_count <= limit:
            return name
    return "over 1000"


def built_rows() -> list[Example]:
    return [e for split in ("train", "val", "test") for e in read_examples(paths.BUILD / f"{split}.jsonl")]


def select(rows: Sequence[Example]) -> list[Example]:
    """Every row over `LONG_WORDS` words, then a fixed random `CONTROL_ROWS` of the shorter ones."""
    rows = sorted((e for e in rows if e.output != EMPTY), key=lambda e: e.id)
    long = [e for e in rows if words(e.input) > LONG_WORDS]
    short = [e for e in rows if words(e.input) <= LONG_WORDS]
    return long + random.Random(SEED).sample(short, min(CONTROL_ROWS, len(short)))


def audit_rows() -> list[Example]:
    """The rows `select` picked from the build the first time the audit ran."""
    if not ROWS.exists():
        write_examples(ROWS, select(built_rows()))
    return read_examples(ROWS)


async def run(llm: LlmClient) -> dict[str, Verdict]:
    rows = audit_rows()
    items = [(e.id, {"transcript": e.input, "reference": e.output}) for e in sorted(rows, key=lambda e: len(e.input))]
    return await judge_all(
        llm,
        items,
        VERDICTS,
        instructions_file=gates.COMPLETENESS,
        criteria=gates.COMPLETENESS_CRITERIA,
        batch_size=gates.JUDGE_MAX_ITEMS,
        max_chars=gates.JUDGE_MAX_CHARS,
    )


def _rates(rows: Iterable[Example], verdicts: dict[str, Verdict], key) -> dict[str, dict]:
    groups: dict[str, list[Verdict]] = defaultdict(list)
    for e in rows:
        if e.id in verdicts:
            groups[key(e)].append(verdicts[e.id])
    out = {}
    for name, group in sorted(groups.items()):
        failed = Counter(criterion for v in group for criterion in v.failed)
        out[name] = {
            "rows": len(group),
            "failed": sum(not v.passed for v in group),
            "rate": round(100 * sum(not v.passed for v in group) / len(group), 1),
            **{criterion: failed[criterion] for criterion in (*gates.COMPLETENESS_CRITERIA, "judge_error") if failed[criterion]},
        }
    return out


def report() -> dict:
    rows = audit_rows()
    verdicts = load_verdicts(VERDICTS)
    long = [e for e in rows if words(e.input) > LONG_WORDS]
    control = [e for e in rows if words(e.input) <= LONG_WORDS]
    code = Counter(
        (e.source, bool(gates.missing_protected(e.input, e.output))) for e in rows if e.id in verdicts
    )
    relabels = load_relabels()
    return {
        "judged": sum(e.id in verdicts for e in rows),
        "long": _rates(long, verdicts, lambda e: "all")["all"] if long else {},
        "control": _rates(control, verdicts, lambda e: "all")["all"] if control else {},
        "by_bucket": _rates(rows, verdicts, lambda e: length_bucket(words(e.input))),
        "by_source_long": _rates(long, verdicts, lambda e: e.source),
        "by_source_control": _rates(control, verdicts, lambda e: e.source),
        "by_category_long": _rates(long, verdicts, lambda e: e.category),
        "by_split_long": _rates(long, verdicts, lambda e: e.split),
        "protected_tokens_missing": {
            source: code[(source, True)] for source in sorted({s for s, _ in code}) if code[(source, True)]
        },
        "relabels": {
            "rows": len(relabels),
            "passed": sum(r["passed"] for r in relabels.values()),
            "passed_first": sum(r["passed"] and r["attempt"] == 1 for r in relabels.values()),
            "failed": dict(Counter(c for r in relabels.values() if not r["passed"] for c in r["failed"])),
        },
    }


def failing(rows: Sequence[Example], verdicts: dict[str, Verdict]) -> list[Example]:
    """Rows whose reference failed the audit and can be relabelled: every source but injection."""
    return [e for e in rows if e.id in verdicts and not verdicts[e.id].passed and e.source != inject.SOURCE]


async def fix(llm: LlmClient) -> dict:
    """Relabels each failing row through the gates; resumes past rows already in `RELABELS`."""
    rows = audit_rows()
    verdicts = load_verdicts(VERDICTS)
    done = load_relabels()
    todo = [e for e in failing(rows, verdicts) if e.id not in done]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(todo), 60):
        chunk = todo[start : start + 60]
        # Attempt 0 of a teacher-sourced row is its current reference, so relabels start at attempt 1.
        results = await gates.gate(llm, [(e.id, e.input) for e in chunk], first_attempt=1)
        with RELABELS.open("a", encoding="utf-8") as sink:
            for e in chunk:
                sink.write(json.dumps({**asdict(results[e.id]), "split": e.split}, ensure_ascii=False) + "\n")
        print(f"relabelled {min(start + 60, len(todo))}/{len(todo)} ({llm.calls} calls)", flush=True)
    return report()["relabels"]


def load_relabels() -> dict[str, dict]:
    return {row["id"]: row for row in read_jsonl(RELABELS)} if RELABELS.exists() else {}


def load_test_corrections() -> dict[str, str]:
    return {row["id"]: row["output"] for row in read_jsonl(TEST_CORRECTIONS)} if TEST_CORRECTIONS.exists() else {}


def apply(rows: Iterable[Example]) -> tuple[list[Example], list[tuple[Example, str]]]:
    """The audit's outcome for gathered rows: relabelled, corrected (test) or dropped (train and val).

    A row keeps its id, so the `TEST_IDS_109` test rows stay comparable. Rows the audit never judged, or
    passed, are unchanged. A test row's split is preset by its source, so it is known here.
    """
    verdicts = load_verdicts(VERDICTS)
    relabels = load_relabels()
    corrections = load_test_corrections()
    kept, dropped = [], []
    for e in rows:
        verdict = verdicts.get(e.id)
        if verdict is None or verdict.passed or e.source == inject.SOURCE:
            kept.append(e)
            continue
        relabel = relabels.get(e.id)
        if e.id in corrections:
            kept.append(replace(e, output=corrections[e.id], tags=[*e.tags, "audit_corrected"]))
        elif relabel and relabel["passed"]:
            kept.append(replace(e, output=relabel["output"], tags=[*e.tags, "audit_relabeled"]))
        elif relabel is None:
            # Not relabelled yet: keep the row as it was rather than lose it silently.
            kept.append(e)
        elif relabel["split"] == "test":
            # A test row keeps its id and needs a hand correction in `TEST_CORRECTIONS`.
            kept.append(replace(e, tags=[*e.tags, "audit_failed"]))
        else:
            dropped.append((e, "failed the completeness audit"))
    return kept, dropped
