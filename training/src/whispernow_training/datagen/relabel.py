"""Every reference rewritten by one model under one contract.

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
- **The build** (`build.build`) and the benchmark assembly apply the records by id once a scope has
  any: a passed row gets its new reference, `teacher_model` and `teacher_reasoning`, and a failed or
  missing row is dropped. Both refuse to run on a scope that is only partly relabelled.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from ..text_rules import FILLER, REFERENCE_FILLER_EXCLUSIONS
from ..model import metrics
from . import gates, paths
from .concurrency import task_group
from .judge import judge_contract
from .llm import LlmClient, request_key
from .schema import EMPTY, Example, read_examples, read_jsonl, words, write_examples
from .teacher import teacher_messages

DIR = paths.DATA / "relabel"
INPUTS = DIR / "inputs"

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


def records_path(scope: str) -> Path:
    return DIR / f"{scope}.jsonl"


def load_records(scope: str) -> dict[str, dict]:
    path = records_path(scope)
    return {row["id"]: row for row in read_jsonl(path)} if path.exists() else {}


def active(scope: str) -> bool:
    """Whether the scope has been relabelled, so its rows take their references from the records."""
    return records_path(scope).exists()


def snapshot(scope: str) -> dict[str, dict] | None:
    records = load_records(scope)
    if not records and not active(scope):
        return None
    require_complete(scope, records)
    return records


def require_complete(scope: str, records: Mapping[str, dict]) -> None:
    pending = [e.id for e in inputs(scope) if e.id not in records or (
        not records[e.id].get("ruling") and "judge_error" in records[e.id]["failed"]
    )]
    if pending:
        raise SystemExit(
            f"the {scope} relabel is partial ({len(pending)} rows left); run `datagen relabel {scope}` to finish it"
        )


def relabelled_license(example: Example) -> str:
    """The row's license with the new teacher reference; the transcript's author stays.

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


def apply(rows: Iterable[Example], records: Mapping[str, dict]) -> tuple[list[Example], list[tuple[Example, str]]]:
    """Rows with their relabelled references; a row that failed twice, or has no record, is dropped."""
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


def _contract(llm: LlmClient, scope: str) -> dict:
    judge = JUDGES[scope]
    instructions = judge.instructions.read_text(encoding="utf-8") + (f"\n{judge.addendum}\n" if judge.addendum else "")
    return {"teacher": llm.settings.request("teacher"), "messages": teacher_messages(""),
            "judge": judge_contract(llm, instructions, judge.criteria, reasoning=judge.reasoning)}


def _matches(example: Example, record: dict, contract: dict) -> bool:
    return record.get("request_key") == request_key({
        "contract": contract, "transcript": example.input, "reference": record["output"],
    }) and "judge_error" not in record["failed"]


async def _gate_groups(
    llm: LlmClient,
    scope: str,
    rows: Sequence[Example],
    references: dict[str, dict],
    earlier_notes: dict[str, str],
    round_: int | None,
    contract: dict,
    *,
    progress: Callable[[str], None] = lambda line: print(line, file=sys.stderr, flush=True),
    stall_seconds: float = STALL_SECONDS,
) -> Progress:
    _, model, reasoning = llm.settings.target("teacher")
    path = records_path(scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = Progress(len(rows), started=time.monotonic(), last=time.monotonic())
    # Enough groups in flight to keep every codex slot busy through the judge step.
    lanes = asyncio.Semaphore(max(4, llm.settings.codex_concurrency))

    async def one(group: list[Example]) -> None:
        async with lanes:
            results = await gates.gate(
                llm, [(e.id, e.input) for e in group],
                references={e.id: references[e.id]["output"] for e in group} if references else None,
                first_attempt=max((references[e.id]["attempt"] for e in group), default=0) if references else 0,
                judge=JUDGES[scope], relabel_note=True, earlier_notes=earlier_notes,
            )
        with path.open("a", encoding="utf-8") as sink:
            for e in group:
                result = results[e.id]
                previous = references.get(e.id)
                if previous and result.passed and previous["passed"] and result.output == previous["output"]:
                    # A passed reference checked again and unchanged keeps its attempt and first note.
                    record = {**previous, "retry": round_}
                else:
                    if previous and result.passed and not result.first_failed:
                        result = replace(result, attempt=previous["attempt"])
                    extra = {"retry": round_} if round_ is not None else {}
                    record = _record(e, result, model, reasoning, **extra)
                record["request_key"] = request_key({"contract": contract, "transcript": e.input,
                                                     "reference": result.output})
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                state.passed += result.passed
        state.done += len(group)
        state.last = time.monotonic()
        if state.done % 200 < len(group) or state.done == state.rows:
            progress(state.line(llm))

    async with task_group() as group:
        watch = group.create_task(_watch(state, stall_seconds))
        tasks = [group.create_task(one(batch)) for batch in _groups(rows)]
        for task in tasks:
            await task
        watch.cancel()
    return state


async def retry(
    llm: LlmClient,
    scope: str,
    rows: Sequence[Example],
    round_: int = 1,
    settled_hedges: bool = False,
) -> dict:
    """Checks dropped references again and rewrites failures with the current and earlier notes.

    `settled_hedges` also checks passed references whose hedge may have been settled. A record with
    a user `ruling` is never retried. Matching records carrying this round or a later one are skipped.
    """
    records = load_records(scope)
    contract = _contract(llm, scope)
    to_check = []
    for e in rows:
        record = records.get(e.id)
        if record is None or record.get("ruling"):
            continue
        if _matches(e, record, contract):
            if record.get("retry", 0) >= round_:
                continue
            if record["passed"] and not (settled_hedges and gates.settled_hedges(e.input, record["output"])):
                continue
        to_check.append(e)
    rechecked = sum(records[e.id]["passed"] for e in to_check)
    earlier_notes = {e.id: records[e.id]["first_note"] for e in to_check if not records[e.id]["passed"]}
    state = await _gate_groups(llm, scope, to_check, records, earlier_notes, round_, contract)
    return {"scope": scope, "round": round_, "retried": len(to_check), "rechecked": rechecked, "passed": state.passed}


async def run(
    llm: LlmClient,
    scope: str,
    rows: Sequence[Example],
    progress: Callable[[str], None] = lambda line: print(line, file=sys.stderr, flush=True),
    stall_seconds: float = STALL_SECONDS,
) -> dict:
    """Relabels changed or unfinished gate requests, appending each group's records as it finishes.

    Raises `Stalled` when no group finishes for `stall_seconds`.
    """
    records = load_records(scope)
    contract = _contract(llm, scope)
    pending = [e for e in rows if e.id not in records or (
        not records[e.id].get("ruling") and not _matches(e, records[e.id], contract)
    )]
    started_tokens, started_seconds, started_calls = llm.tokens, llm.request_seconds, llm.calls
    state = await _gate_groups(
        llm, scope, pending, {}, {}, None, contract, progress=progress, stall_seconds=stall_seconds,
    )
    return {
        "scope": scope,
        "rows": len(pending),
        "passed": state.passed,
        "seconds": round(time.monotonic() - state.started, 1),
        "calls": llm.calls - started_calls,
        "tokens": llm.tokens - started_tokens,
        "request_seconds": round(llm.request_seconds - started_seconds, 1),
        "limit_waits": llm.limit_waits,
    }


def writer(example: Example) -> str:
    if example.output == EMPTY and example.category == "empty":
        return "EMPTY (code)"
    model = example.teacher_model or "dataset authors"
    return f"{model}/{example.teacher_reasoning}" if example.teacher_reasoning else model


def kind(example: Example) -> str:
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
        "hesitation_filler": round(sum(any(not match.group(0).isupper() and match.group(0).casefold() not in REFERENCE_FILLER_EXCLUSIONS
                                            for match in FILLER.finditer(e.output)) for e in texts) / total, 4),
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
    """`compare`'s report as tables for `stats.md`."""
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


def scope_report(scope: str) -> dict:
    old = inputs(scope)
    records = load_records(scope)
    old = [e for e in old if e.id in records]
    new, _ = apply(old, records)
    return compare(old, new, records)


def apply_bench(rows: list[Example], scope: str) -> list[Example]:
    """A benchmark's rows with their new references once the scope is relabelled; failed rows go."""
    records = snapshot(scope)
    if records is None:
        return rows
    kept, dropped = apply(rows, records)
    missing = [e.id for e, reason in dropped if reason == "not relabelled"]
    if missing:
        raise SystemExit(f"{len(missing)} {scope} rows were never relabelled, such as {missing[0]}; the benchmark changed since")
    return kept
