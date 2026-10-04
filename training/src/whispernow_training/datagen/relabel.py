"""Todo 112: every reference rewritten by one model under one contract.

`gpt-6.1-sol` at high reasoning, through codex, writes the reference for every row of the build and
both benchmarks, from the app's own request (`teacher.teacher_messages`), without seeing the old
reference. The gates check it (`gates.gate`: the code checks, then the completeness judge, the same
model; rows over 300 words get a request each), a failure is written again once with the judge's
note, and a second failure is dropped.

- **Scopes.** `build`: the rows of `data/build/{train,val,test}.jsonl` with their built inputs,
  except the injected rows, which `build` makes again from the relabelled parents. `bench-long` and
  `bench-noise`: the benchmark files, judged one row per request; ids, transcripts and seed groups stay.
- **Inputs** are copied to `data/relabel/inputs/<scope>.jsonl` on the first run, so a later build or
  assemble cannot change what is being relabelled. The old references stay there for the comparison.
- **Records** go to `data/relabel/<scope>.jsonl`, one per row as its gate finishes, and a rerun goes
  on from them. The response cache makes a rerun of a finished row free.
- **The build** (`build.gather`) and the benchmark assembly apply the records by id once a scope has
  any: a passed row gets its new reference, `teacher_model` and `teacher_reasoning`, and a failed or
  missing row is dropped. Both refuse to run on a scope that is only partly relabelled.

`pilot` runs about 200 stratified rows into `data/relabel/pilot/` first and writes the hand-review
page. The full run reuses its cached responses.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path

from ..model import metrics
from . import gates, paths, teacher
from .audit import length_bucket, words
from .llm import LlmClient, LlmSettings
from .schema import EMPTY, Example, read_examples, read_jsonl, write_examples

MODEL = "gpt-6.1-sol"
REASONING = "high"
BACKEND = "codex"
ROLES = ("teacher", "judge", "generator")

DIR = paths.DATA / "relabel"
INPUTS = DIR / "inputs"
PILOT = DIR / "pilot"
REVIEW_PAGE = paths.TRAINING_ROOT / "runs" / "relabel-review.html"

SOLO_OVER_WORDS = 300
JUDGES = {
    "build": gates.Judge(solo_over_words=SOLO_OVER_WORDS),
    # The strict pass: every benchmark row gets the judge's full attention.
    "bench-long": gates.Judge(max_items=1),
    "bench-noise": gates.Judge(max_items=1),
}
SCOPES = tuple(JUDGES)
SOURCES = {
    "build": tuple(paths.BUILD / f"{split}.jsonl" for split in ("train", "val", "test")),
    "bench-long": (paths.DATA / "bench" / "long.jsonl",),
    "bench-noise": (paths.DATA / "bench" / "noise.jsonl",),
}
# Rows gated together: the judge batches them, and their records are written as the group finishes.
GROUP = 10
# The injected rows are made from their parents' references, so they are made again, not relabelled.
INJECTED_CATEGORY = "filler_injection"
TAG = "relabel112"

PILOT_ROWS = {"build": 150, "bench-long": 25, "bench-noise": 25}
PILOT_SEED = 112
REVIEW_ROWS = 60


def check_settings(settings: LlmSettings) -> None:
    """Refuses unless every role is `MODEL` at `REASONING` through codex: no other model writes or judges."""
    wrong = {
        role: target for role in ROLES if (target := settings.target(role)) != (BACKEND, MODEL, REASONING)
    }
    if wrong:
        found = ", ".join(f"{role}: {'/'.join(target)}" for role, target in wrong.items())
        raise SystemExit(
            f"relabel needs every role on {BACKEND}/{MODEL}/{REASONING}; found {found}. Set LLM_BACKEND=codex "
            f"LLM_MODEL={MODEL} LLM_REASONING={REASONING} and the same for LLM_JUDGE_* and LLM_GENERATOR_*."
        )


def in_scope(example: Example) -> bool:
    return example.category != INJECTED_CATEGORY


def inputs(scope: str) -> list[Example]:
    """The scope's rows with their old references, copied from the current files on the first call."""
    path = INPUTS / f"{scope}.jsonl"
    if path.exists():
        return read_examples(path)
    missing = [str(source) for source in SOURCES[scope] if not source.exists()]
    if missing:
        raise SystemExit(f"relabel {scope} reads {', '.join(missing)}, which do not exist")
    every = [e for source in SOURCES[scope] for e in read_examples(source)]
    rows = [e for e in every if in_scope(e)]
    write_examples(path, rows)
    if scope == "build":
        # Kept for the old-against-new stats only.
        write_examples(INPUTS / "build.injected.jsonl", [e for e in every if not in_scope(e)])
    return rows


def old_injected() -> list[Example]:
    path = INPUTS / "build.injected.jsonl"
    return read_examples(path) if path.exists() else []


def records_path(scope: str, root: Path | None = None) -> Path:
    """`root` defaults to `DIR`, looked up per call."""
    return (root or DIR) / f"{scope}.jsonl"


def load_records(scope: str, root: Path | None = None) -> dict[str, dict]:
    path = records_path(scope, root)
    return {row["id"]: row for row in read_jsonl(path)} if path.exists() else {}


def active(scope: str, root: Path | None = None) -> bool:
    """Whether the scope has been relabelled, so its rows take their references from the records."""
    return records_path(scope, root).exists()


def require_complete(scope: str, root: Path | None = None) -> None:
    records = load_records(scope, root)
    pending = [e.id for e in inputs(scope) if e.id not in records]
    if pending:
        raise SystemExit(
            f"the {scope} relabel is partial ({len(pending)} rows left); run `datagen relabel {scope}` to finish it"
        )


def relabelled_license(example: Example) -> str:
    """The row's license with the reference now `MODEL`'s output; the transcript's author stays.

    A dataset original keeps its license, as rows the teacher relabelled before did. A row of model
    output names each writer: a single author wrote both parts, a pair names the transcript's writer
    and the old reference's.
    """
    parts = set(example.license.split("+"))
    if not parts & {"zai-output", "openai-output"}:
        return example.license
    old_reference = "openai-output" if (example.teacher_model or "").startswith("gpt") else "zai-output"
    transcript = (parts - {old_reference}) or {old_reference}
    return "+".join(sorted(transcript | {"openai-output"}))


def apply(rows: Iterable[Example], scope: str, root: Path | None = None) -> tuple[list[Example], list[tuple[Example, str]]]:
    """Rows with their relabelled references; a row that failed twice, or has no record, is dropped."""
    records = load_records(scope, root)
    kept, dropped = [], []
    for e in rows:
        record = records.get(e.id)
        if record is None:
            dropped.append((e, "not relabelled"))
        elif not record["passed"]:
            dropped.append((e, "failed the relabel gate"))
        else:
            tags = [*e.tags, TAG, *(["relabel112_retry"] if record["attempt"] else [])]
            kept.append(
                replace(
                    e, output=record["output"], tags=tags, license=relabelled_license(e),
                    teacher_model=record["teacher_model"], teacher_reasoning=record["teacher_reasoning"],
                )
            )
    return kept, dropped


@dataclass
class Progress:
    rows: int
    done: int = 0
    passed: int = 0
    started: float = 0.0
    # When the last group finished, for the stall watch.
    last: float = 0.0

    def line(self, llm: LlmClient) -> str:
        elapsed = time.monotonic() - self.started
        rate = self.done / elapsed if elapsed else 0.0
        left = (self.rows - self.done) / rate if rate else 0.0
        return (
            f"{time.strftime('%H:%M')} {self.done}/{self.rows} rows, {self.passed} passed, "
            f"{rate * 3600:.0f} rows/h, about {left / 3600:.1f} h left ({llm.calls} calls, "
            f"{llm.cache_hits} cached, {llm.tokens:,} tokens)"
        )


# No group finishing for this long means the run is stuck (one group takes a few minutes, and a
# request times out after 15): dump every task's stack and stop, so a restart resumes from the records.
STALL_SECONDS = 20 * 60


class Stalled(RuntimeError):
    """No group finished for `STALL_SECONDS`."""


async def _watch(state: Progress, limit: float) -> None:
    while True:
        await asyncio.sleep(min(60.0, limit))
        if time.monotonic() - state.last >= limit:
            print(f"{time.strftime('%H:%M')} STALLED: no group finished for {limit / 60:.0f} min; pending tasks:", file=sys.stderr)
            for task in asyncio.all_tasks():
                task.print_stack(limit=12, file=sys.stderr)
            sys.stderr.flush()
            raise Stalled(f"no group finished for {limit / 60:.0f} min")


def _groups(rows: Sequence[Example]) -> list[list[Example]]:
    """Rows of similar length together, so a group's judge batch is evenly sized."""
    ordered = sorted(rows, key=lambda e: (len(e.input), e.id))
    return [list(ordered[start : start + GROUP]) for start in range(0, len(ordered), GROUP)]


def _record(e: Example, result: gates.Gated, model: str, reasoning: str, **extra: object) -> dict:
    return {
        "id": e.id, "output": result.output, "passed": result.passed, "attempt": result.attempt,
        "failed": result.failed, "note": result.note, "first_failed": result.first_failed,
        "first_note": result.first_note, "source": e.source, "category": e.category,
        "split": e.split, "words": words(e.input), "teacher_model": model, "teacher_reasoning": reasoning, **extra,
    }


async def retry(
    llm: LlmClient, scope: str, rows: Sequence[Example], root: Path | None = None, round_: int = 1
) -> dict:
    """Retry round `round_` over the scope's dropped rows: their last reference is checked again (the
    gates may have changed since), and one that still fails gets one more rewrite with the new note.
    Each new record is appended with `retry: round_`, so it replaces the dropped one and a rerun of
    the round skips it."""
    check_settings(llm.settings)
    _, model, reasoning = llm.settings.target("teacher")
    records = load_records(scope, root)
    failed = [
        e for e in rows if e.id in records and not records[e.id]["passed"] and records[e.id].get("retry", 0) < round_
    ]
    path = records_path(scope, root)
    lanes = asyncio.Semaphore(max(4, llm.settings.codex_concurrency))
    passed = 0

    async def one(group: list[Example]) -> None:
        nonlocal passed
        async with lanes:
            results = await gates.gate(
                llm, [(e.id, e.input) for e in group], references={e.id: records[e.id]["output"] for e in group},
                first_attempt=max(records[e.id]["attempt"] for e in group), judge=JUDGES[scope], relabel_note=True,
            )
        with path.open("a", encoding="utf-8") as sink:
            for e in group:
                sink.write(json.dumps(_record(e, results[e.id], model, reasoning, retry=round_), ensure_ascii=False) + "\n")
                passed += results[e.id].passed

    await asyncio.gather(*(one(group) for group in _groups(failed)))
    return {"scope": scope, "round": round_, "retried": len(failed), "passed": passed}


async def run(
    llm: LlmClient,
    scope: str,
    rows: Sequence[Example],
    root: Path | None = None,
    progress: Callable[[str], None] = lambda line: print(line, file=sys.stderr, flush=True),
    stall_seconds: float = STALL_SECONDS,
) -> dict:
    """Relabels the rows without a record in `root`, appending each group's records as it finishes.

    Raises `Stalled` when no group finishes for `stall_seconds`."""
    check_settings(llm.settings)
    _, model, reasoning = llm.settings.target("teacher")
    records = load_records(scope, root)
    todo = [e for e in rows if e.id not in records]
    path = records_path(scope, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = Progress(len(todo), started=time.monotonic(), last=time.monotonic())
    # Enough groups in flight to keep every codex slot busy through the judge step.
    lanes = asyncio.Semaphore(max(4, llm.settings.codex_concurrency))
    started_tokens, started_seconds, started_calls = llm.tokens, llm.request_seconds, llm.calls

    async def one(group: list[Example]) -> None:
        async with lanes:
            results = await gates.gate(
                llm, [(e.id, e.input) for e in group], judge=JUDGES[scope], relabel_note=True
            )
        with path.open("a", encoding="utf-8") as sink:
            for e in group:
                result = results[e.id]
                sink.write(json.dumps(_record(e, result, model, reasoning), ensure_ascii=False) + "\n")
                state.passed += result.passed
        state.done += len(group)
        state.last = time.monotonic()
        if state.done % 200 < len(group) or state.done == state.rows:
            progress(state.line(llm))

    work = asyncio.gather(*(one(group) for group in _groups(todo)))
    watch = asyncio.ensure_future(_watch(state, stall_seconds))
    try:
        await asyncio.wait({work, watch}, return_when=asyncio.FIRST_COMPLETED)
        if watch.done():
            work.cancel()
            watch.result()
        await work
    finally:
        watch.cancel()
    return {
        "scope": scope,
        "rows": len(todo),
        "passed": state.passed,
        "seconds": round(time.monotonic() - state.started, 1),
        "calls": llm.calls - started_calls,
        "tokens": llm.tokens - started_tokens,
        "request_seconds": round(llm.request_seconds - started_seconds, 1),
        "limit_waits": llm.limit_waits,
    }


# --- The checks, old against new ------------------------------------------------------------

_HESITATIONS = ("um", "umm", "uh", "uhh", "uhm", "erm", "er", "hmm", "mhm")
# Lower case or capitalized only: "ER" and "UM" are acronyms.
_HESITATION = re.compile(rf"\b(?:{'|'.join(f'[{w[0]}{w[0].upper()}]{w[1:]}' for w in _HESITATIONS)})\b")


def writer(example: Example) -> str:
    """Who wrote a reference, as todo 112's table names them."""
    if example.output == EMPTY and example.category == "empty":
        return "EMPTY (code)"
    model = example.teacher_model or "dataset authors"
    return f"{model}/{example.teacher_reasoning}" if example.teacher_reasoning else model


def kind(example: Example) -> str:
    """The row's source, with the generated rows told apart: composed (todo 110) or stage 3 (todo 108)."""
    if example.source == "teacher":
        return "composed" if "composed" in example.tags else "generated"
    return example.source.split("/")[-1]


def reference_checks(rows: Sequence[Example]) -> dict:
    """The contract's code checks over references: shares of rows, and the hedge and reason keep rates.
    `kept` leaves out the occurrences a reference may drop (repeats, closing asides, corrections);
    `kept_raw` counts every one."""
    texts = [e for e in rows if e.output != EMPTY]

    def rates(excuse: bool) -> dict:
        said: Counter = Counter()
        kept: Counter = Counter()
        for e in texts:
            for word, (count, keep) in metrics.kept_counts(e.input, e.output, excuse).items():
                said[word] += count
                kept[word] += keep
        return {word: {"rate": round(kept[word] / said[word], 3), "n": said[word]} for word in metrics.KEPT_WORDS if said[word]}

    total = max(1, len(texts))
    ratio = [len(e.output.split()) / max(1, len(e.input.split())) for e in texts]
    return {
        "rows": len(rows),
        "em_dash": round(sum(metrics.EM_DASH in e.output for e in texts) / total, 4),
        "spelled_count": round(sum(bool(metrics.spelled_counts(e.output)) for e in texts) / total, 4),
        "correction_cue": round(sum(bool(metrics.correction_cues(e.output, e.input)) for e in texts) / total, 4),
        "contraction": round(sum(bool(metrics.casual_contractions(e.output)) for e in texts) / total, 4),
        "hesitation_filler": round(sum(bool(_HESITATION.search(e.output)) for e in texts) / total, 4),
        "kept": rates(True),
        "kept_raw": rates(False),
        "out_in_words": round(sum(ratio) / max(1, len(ratio)), 3),
    }


def compare(old: Sequence[Example], new: Sequence[Example], records: dict[str, dict]) -> dict:
    """Old references against new ones, overall and by source kind: writers, checks and drops."""
    by_kind_old: dict[str, list[Example]] = defaultdict(list)
    by_kind_new: dict[str, list[Example]] = defaultdict(list)
    for e in old:
        by_kind_old[kind(e)].append(e)
    for e in new:
        by_kind_new[kind(e)].append(e)
    dropped = Counter(kind(e) for e in old if e.id in records and not records[e.id]["passed"])
    drops_by_category = Counter(e.category for e in old if e.id in records and not records[e.id]["passed"])
    report = {
        "all": {"old": reference_checks(old), "new": reference_checks(new)},
        "writers": {"old": dict(Counter(writer(e) for e in old).most_common()), "new": dict(Counter(writer(e) for e in new).most_common())},
        "by_kind": {},
        "dropped_by_category": dict(drops_by_category.most_common()),
        "retried": sum(1 for record in records.values() if record["attempt"]),
    }
    for name in sorted(by_kind_old):
        rows = by_kind_old[name]
        report["by_kind"][name] = {
            "old": reference_checks(rows),
            "new": reference_checks(by_kind_new.get(name, [])),
            "dropped": dropped[name],
            "drop_rate": round(dropped[name] / max(1, len(rows)), 4),
        }
    return report


def _pct(value: float) -> str:
    return f"{value:.1%}"


def compare_markdown(report: dict, title: str = "References, old against new (todo 112)") -> str:
    """`compare`'s report as tables for `stats.md` and the pilot report."""
    lines = [f"## {title}\n"]
    lines.append("### Writers\n\n| writer | old rows | new rows |\n|---|---|---|")
    writers = sorted(set(report["writers"]["old"]) | set(report["writers"]["new"]))
    lines += [f"| {w} | {report['writers']['old'].get(w, 0)} | {report['writers']['new'].get(w, 0)} |" for w in writers]
    lines.append(
        "\n### Code checks by source\n\nShares of non-EMPTY references. Spelled counts are flags the judge settles, "
        "so some are names and idioms.\n\n"
        "| source | rows old / new | dropped | em dash | spelled count | correction cue | um/uh | wanna etc. | out/in words |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    for name, row in [*report["by_kind"].items(), ("all", {**report["all"], "dropped": sum(r["dropped"] for r in report["by_kind"].values()), "drop_rate": None})]:
        old, new = row["old"], row["new"]
        drop = f"{row['dropped']} ({_pct(row['drop_rate'])})" if row["drop_rate"] is not None else str(row["dropped"])
        cells = [f"{_pct(old[key])} → {_pct(new[key])}" for key in ("em_dash", "spelled_count", "correction_cue", "hesitation_filler", "contraction")]
        lines.append(
            f"| {name} | {old['rows']} / {new['rows']} | {drop} | " + " | ".join(cells)
            + f" | {old['out_in_words']} → {new['out_in_words']} |"
        )
    lines.append(
        "\n### Hedges and reasons kept\n\nShare of the transcript's occurrences the reference keeps (n occurrences), leaving out "
        "the ones a reference may drop: a hedge said again straight away, a closing aside (\"I think that's everything\") "
        "and one followed within 10 words by a correction cue. Every occurrence counted: `kept_raw` in `report.json`.\n"
    )
    lines.append("| source | " + " | ".join(metrics.KEPT_WORDS) + " |\n|" + "---|" * (len(metrics.KEPT_WORDS) + 1))
    for name, row in [*report["by_kind"].items(), ("all", report["all"])]:
        cells = []
        for word in metrics.KEPT_WORDS:
            old, new = row["old"]["kept"].get(word), row["new"]["kept"].get(word)
            cells.append(f"{_pct(old['rate']) if old else '-'} → {_pct(new['rate']) if new else '-'} ({new['n'] if new else 0})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    if report["dropped_by_category"]:
        lines.append("\n### Dropped by category\n\n| category | rows |\n|---|---|")
        lines += [f"| {name} | {count} |" for name, count in report["dropped_by_category"].items()]
    return "\n".join(lines) + "\n"


def scope_report(scope: str, root: Path | None = None) -> dict:
    old = inputs(scope)
    records = load_records(scope, root)
    old = [e for e in old if e.id in records]
    new, _ = apply(old, scope, root)
    return compare(old, new, records)


# --- Pilot -----------------------------------------------------------------------------------


def stratum(scope: str, example: Example) -> tuple[str, str]:
    return (f"{scope}:{kind(example)}:{writer(example)}", length_bucket(words(example.input)))


def stratified(rows: Sequence[Example], count: int, key: Callable[[Example], tuple], seed: int) -> list[Example]:
    """`count` rows, drawn in turn from each stratum so small strata are covered too."""
    rng = random.Random(seed)
    strata: dict[tuple, list[Example]] = defaultdict(list)
    for e in sorted(rows, key=lambda e: e.id):
        strata[key(e)].append(e)
    for members in strata.values():
        rng.shuffle(members)
    picked: list[Example] = []
    order = sorted(strata)
    while len(picked) < count and any(strata.values()):
        for name in order:
            if strata[name] and len(picked) < count:
                picked.append(strata[name].pop())
    return picked


def pilot_rows(scope: str) -> list[Example]:
    return stratified(inputs(scope), PILOT_ROWS[scope], lambda e: stratum(scope, e), PILOT_SEED)


async def pilot(llm: LlmClient) -> dict:
    """Relabels the pilot rows of every scope into `PILOT` and reports cost, pass rate and the checks."""
    runs = {}
    for scope in SCOPES:
        rows = pilot_rows(scope)
        runs[scope] = await run(llm, scope, rows, PILOT)
    return pilot_report(runs)


def full_rows() -> dict[str, int]:
    return {scope: len(inputs(scope)) for scope in SCOPES}


def pilot_report(runs: dict[str, dict] | None = None) -> dict:
    """The pilot's measures, with the full run's time and tokens scaled from them.

    `runs` holds the cost of the pilot's own run; a later call keeps the costs saved before.
    """
    path = PILOT / "report.json"
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    costs = saved.get("cost", {})
    for scope, cost in (runs or {}).items():
        if cost["rows"]:
            costs[scope] = cost
    report: dict = {"cost": costs, "scopes": {}}
    for scope in SCOPES:
        if not active(scope, PILOT):
            continue
        records = load_records(scope, PILOT)
        compared = scope_report(scope, PILOT)
        report["scopes"][scope] = {
            "rows": len(records),
            "passed": sum(r["passed"] for r in records.values()),
            "pass_rate": round(sum(r["passed"] for r in records.values()) / max(1, len(records)), 3),
            "first_try": round(sum(r["passed"] and not r["attempt"] for r in records.values()) / max(1, len(records)), 3),
            "first_failed": dict(Counter(c for r in records.values() for c in r["first_failed"]).most_common()),
            "failed": dict(Counter(c for r in records.values() if not r["passed"] for c in r["failed"]).most_common()),
            "compare": compared,
        }
    totals = full_rows()
    estimate = {}
    for scope, cost in costs.items():
        rows = max(1, cost["rows"])
        estimate[scope] = {
            "rows": totals[scope],
            "seconds_per_row_wall": round(cost["seconds"] / rows, 2),
            "request_seconds_per_row": round(cost["request_seconds"] / rows, 1),
            "tokens_per_row": round(cost["tokens"] / rows),
            "tokens": round(cost["tokens"] / rows * totals[scope]),
            # Requests run `LLM_CODEX_CONCURRENCY` at a time, so a full run takes the summed request time
            # over that many slots, plus whatever usage-limit pauses the account needs.
            "request_hours": round(cost["request_seconds"] / rows * totals[scope] / 3600, 1),
        }
    report["estimate"] = estimate
    PILOT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return report


# --- Review page -----------------------------------------------------------------------------

_TOKEN = re.compile(r"\s+|[^\s]+")


def word_diff(old: str, new: str) -> str:
    """`new` as HTML, with the words it adds in <ins> and the old reference's words it drops in <del>."""
    a, b = _TOKEN.findall(old), _TOKEN.findall(new)
    out = []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op in ("delete", "replace") and "".join(a[i1:i2]).strip():
            out.append(f"<del>{html.escape(''.join(a[i1:i2]))}</del>")
        if op in ("insert", "replace", "equal"):
            text = html.escape("".join(b[j1:j2]))
            out.append(text if op == "equal" or not text.strip() else f"<ins>{text}</ins>")
    return "".join(out)


_STYLE = """
:root { --bg:#fafaf8; --fg:#1d1d1b; --muted:#6b6b66; --card:#fff; --line:#e3e2dc; --del:#fbe1df; --ins:#dff3e4; --pill:#ece9ff; --note:#fff7e0; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#171716; --fg:#ecebe6; --muted:#a3a29b; --card:#212120; --line:#34332f; --del:#5a2a27; --ins:#21452c; --pill:#35305a; --note:#3d3420; } }
:root[data-theme="dark"] { --bg:#171716; --fg:#ecebe6; --muted:#a3a29b; --card:#212120; --line:#34332f; --del:#5a2a27; --ins:#21452c; --pill:#35305a; --note:#3d3420; }
body { background:var(--bg); color:var(--fg); font:15px/1.5 system-ui,sans-serif; margin:0; padding:24px 16px 80px; }
main { max-width:1200px; margin:0 auto; }
h1 { font-size:22px; margin:0 0 4px; } p.lead { color:var(--muted); margin:0 0 16px; }
table { border-collapse:collapse; font-size:13px; margin-bottom:20px; } td,th { border:1px solid var(--line); padding:3px 8px; text-align:left; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin:0 0 18px; }
.card header { display:flex; flex-wrap:wrap; gap:8px; align-items:baseline; margin-bottom:8px; }
.num { font-weight:700; } .pill { background:var(--pill); border-radius:999px; padding:1px 10px; font-size:13px; }
.meta { color:var(--muted); font-size:13px; } .note { background:var(--note); border-radius:6px; padding:8px 10px; margin-bottom:8px; }
.cols { display:grid; grid-template-columns:1fr 1fr; gap:12px; } @media (max-width:760px) { .cols { grid-template-columns:1fr; } }
h4 { margin:6px 0; font-size:14px; } h4 small { color:var(--muted); font-weight:400; }
pre { white-space:pre-wrap; word-wrap:break-word; font:13.5px/1.5 ui-monospace,monospace; margin:0; background:transparent; }
del { background:var(--del); text-decoration:line-through; } ins { background:var(--ins); text-decoration:none; }
details summary { cursor:pointer; color:var(--muted); margin:4px 0; }
.fb { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-top:10px; border-top:1px solid var(--line); padding-top:8px; }
.fb textarea { flex:1 1 100%; background:var(--bg); color:var(--fg); border:1px solid var(--line); border-radius:6px; padding:6px; font:inherit; }
#bar { position:fixed; bottom:0; left:0; right:0; background:var(--card); border-top:1px solid var(--line); padding:10px 16px; display:flex; gap:10px; align-items:center; }
#bar button { font:inherit; padding:6px 14px; border-radius:6px; border:1px solid var(--line); background:var(--pill); color:var(--fg); cursor:pointer; }
#out { width:100%; max-width:1200px; margin:12px auto 0; display:none; }
"""

def _contract_tag() -> str:
    """Names the edit request the page reviews, so verdicts saved for an earlier contract don't fill in."""
    return hashlib.sha256(teacher.EDIT_REQUEST_FILE.read_bytes()).hexdigest()[:8]


_SCRIPT = """
const KEY = "relabel-review-112-__CONTRACT__";
const cards = [...document.querySelectorAll(".card")];
function collect() {
  return cards.map(c => { const n = c.dataset.n; const v = c.querySelector(`input[name=v${n}]:checked`); const t = c.querySelector(`textarea[name=c${n}]`).value.trim();
    return { n: +n, id: c.dataset.id, verdict: v ? v.value : "", comment: t }; }).filter(x => x.verdict || x.comment);
}
function save() { try { localStorage.setItem(KEY, JSON.stringify(collect())); } catch (e) {} status(); }
function status() { document.getElementById("status").textContent = `${collect().length} of ${cards.length} rows marked`; }
try { for (const x of JSON.parse(localStorage.getItem(KEY) || "[]")) { const c = document.getElementById("r" + x.n); if (!c) continue;
  if (x.verdict) { const r = c.querySelector(`input[name=v${x.n}][value=${x.verdict}]`); if (r) r.checked = true; }
  c.querySelector(`textarea[name=c${x.n}]`).value = x.comment || ""; } } catch (e) {}
document.addEventListener("input", save); document.addEventListener("change", save); status();
document.getElementById("copy").onclick = async () => {
  const lines = collect().map(x => `#${x.n} ${x.id}: ${x.verdict || "-"}${x.comment ? " | " + x.comment : ""}`).join("\\n");
  const out = document.getElementById("out"); out.style.display = "block"; out.value = lines;
  try { await navigator.clipboard.writeText(lines); document.getElementById("status").textContent = "Copied. Paste it into the chat."; }
  catch (e) { out.select(); document.getElementById("status").textContent = "Select the text below and copy it."; }
};
"""


def _flags(example: Example, output: str) -> list[str]:
    flags = []
    if metrics.EM_DASH in output:
        flags.append("em dash")
    flags += [f"spelled: {phrase}" for phrase in metrics.spelled_counts(output)[:3]]
    flags += [f"cue: {cue}" for cue in metrics.correction_cues(output, example.input)]
    flags += [f"fewer '{word}'" for word in metrics.dropped_kept_words(example.input, output)]
    return flags


def review_rows(count: int = REVIEW_ROWS) -> list[tuple[str, Example, dict]]:
    """`count` pilot rows for the hand read, spread over scopes, sources and lengths."""
    pool = []
    for scope in SCOPES:
        if not active(scope, PILOT):
            continue
        records = load_records(scope, PILOT)
        pool += [(scope, e, records[e.id]) for e in inputs(scope) if e.id in records]
    by_id = {e.id: (scope, e, record) for scope, e, record in pool}
    picked = stratified([e for _, e, _ in pool], count, lambda e: stratum(by_id[e.id][0], e), PILOT_SEED + 1)
    return [by_id[e.id] for e in sorted(picked, key=lambda e: (by_id[e.id][0], kind(e), words(e.input)))]


def review_page(rows: Sequence[tuple[str, Example, dict]], report: dict, out: Path = REVIEW_PAGE) -> Path:
    """The hand-review page: transcript, old reference and new reference with the change marked."""
    summary = ["<tr><th>scope</th><th>rows</th><th>passed</th><th>first try</th><th>tokens/row</th><th>full run</th></tr>"]
    for scope, info in report["scopes"].items():
        estimate = report["estimate"].get(scope, {})
        tokens = f"{estimate['tokens_per_row']:,}" if "tokens_per_row" in estimate else "-"
        summary.append(
            f"<tr><td>{scope}</td><td>{info['rows']}</td><td>{_pct(info['pass_rate'])}</td><td>{_pct(info['first_try'])}</td>"
            f"<td>{tokens}</td><td>{estimate.get('rows', '-')} rows</td></tr>"
        )
    cards = []
    for n, (scope, e, record) in enumerate(rows, 1):
        new = record["output"]
        flags = _flags(e, new)
        verdict = (
            "passed the gate" + (" on the second try" if record["attempt"] else "")
            if record["passed"] else f"<b>dropped</b>: failed {', '.join(record['failed'])}"
        )
        notes = []
        if record["attempt"]:
            notes.append(f"First reference failed <code>{html.escape(', '.join(record['first_failed']))}</code>: {html.escape(record['first_note'])}")
        if not record["passed"]:
            notes.append(f"Second reference failed <code>{html.escape(', '.join(record['failed']))}</code>: {html.escape(record['note'])}")
        cards.append(f"""<section class="card" id="r{n}" data-n="{n}" data-id="{html.escape(e.id)}">
  <header><span class="num">#{n}</span> <span class="pill">{html.escape(kind(e))}</span>
  <span class="meta">{scope} · {html.escape(e.category)} · {words(e.input)} words · old by {html.escape(writer(e))} · {verdict} · <code>{html.escape(e.id)}</code></span></header>
  {"".join(f'<div class="note">{note}</div>' for note in notes)}
  {f'<div class="meta">Code flags on the new reference: {html.escape("; ".join(flags))}</div>' if flags else ""}
  <details{" open" if words(e.input) <= 150 else ""}><summary>Transcript</summary><pre>{html.escape(e.input)}</pre></details>
  <div class="cols"><div><h4>Old reference</h4><pre>{html.escape(e.output)}</pre></div>
  <div><h4>New reference <small>(added words green, the old reference's dropped words struck)</small></h4><pre>{word_diff(e.output, new)}</pre></div></div>
  <div class="fb"><span>Is the new reference right?</span>
   <label><input type="radio" name="v{n}" value="right"> right</label>
   <label><input type="radio" name="v{n}" value="fault"> has a fault</label>
   <label><input type="radio" name="v{n}" value="unsure"> unsure</label>
   <textarea name="c{n}" rows="2" placeholder="Comment (optional): what is wrong, or what the contract should say"></textarea></div>
</section>""")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Relabel Pilot Review</title>
<style>{_STYLE}</style></head><body><main>
<h1>Relabel pilot review (todo 112)</h1>
<p class="lead"><b>How to read a row.</b> The <b>transcript</b> is unchanged. The <b>old reference</b> is the one the build or benchmark has now. The <b>new reference</b> is {MODEL}'s, at {REASONING} reasoning, written from the app's own Clean Up request (with the tightened contract) without seeing the old one, then checked in code and by the completeness judge (the same model). A reference that failed was written once more with the judge's note; a second failure is dropped, and such rows are marked. Judge the <b>new</b> reference against the transcript: are all facts there, hedges and reasons kept, the corrected side kept and its cue gone, amounts and counts in digits, no em dashes, and the layout right? When you're done, press <b>Copy feedback</b> and paste it into the chat.</p>
<table>{"".join(summary)}</table>
{"".join(cards)}
<textarea id="out" rows="10"></textarea>
</main>
<div id="bar"><button id="copy">Copy feedback</button><span id="status" class="meta"></span></div>
<script>{_SCRIPT.replace("__CONTRACT__", _contract_tag())}</script></body></html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out


# --- Applying the records ----------------------------------------------------------------------


def apply_bench(rows: list[Example], scope: str) -> list[Example]:
    """A benchmark's rows with their new references once the scope is relabelled; failed rows go."""
    if not active(scope):
        return rows
    require_complete(scope)
    kept, dropped = apply(rows, scope)
    missing = [e.id for e, reason in dropped if reason == "not relabelled"]
    if missing:
        raise SystemExit(f"{len(missing)} {scope} rows were never relabelled, such as {missing[0]}; the benchmark changed since")
    return kept
