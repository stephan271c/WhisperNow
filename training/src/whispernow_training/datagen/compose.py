"""Long composed-task rows (todo 110, step 5) and the long-dictation benchmark (step 4).

One dictation combines several behaviours: a list with corrections, choices revised with a reason,
meeting notes whose owners and dates change, a ramble with an embedded list and a question. Each
request asks for a few transcripts of one length bucket (150-300, 300-600 or 600-1,000 words). The
teacher writes each reference and every reference passes the gates (`gates.gate` with the
`compose` judge): the completeness criteria, a protected-token code check, and the realism and
format checks stage 3's judge made. A failing reference is relabelled once, then dropped; an
unrealistic transcript is dropped at once. Drops go to `<stem>.rejected.jsonl` with their reason.

Two runs share the code and nothing else:
- `TRAIN` writes `data/stage4/`, which `build` reads. Batches draw names, settings and seeds from
  the stage 3 pools with a "train" prefix.
- `BENCH` writes `data/bench/long/`, which `build` never reads. Its requests use their own base
  prompt (`prompts/bench/base.md`), their own name and setting pools, and a "bench" seed prefix, so
  no benchmark row shares a template or `seed_group` with a training row.

Batches are logged per category and bucket, and a batch's request depends only on the run, the
category, the bucket, its number and the previous batch's notes, so an interrupted run resumes with
the finished requests served from the response cache.
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from . import gates, generate, paths, rejudge
from .audit import length_bucket
from .llm import LlmClient, LlmError
from .schema import EMPTY, Example, example_id, layout_of, read_examples, read_jsonl
from .teacher import relabel

CATEGORIES = (
    "list_corrections", "revised_choice", "meeting_actions", "multi_topic", "recipe_quantities",
    "technical_steps", "email_late_correction", "dense_professional",
)
BUCKETS = {"150-300": (150, 300), "300-600": (300, 600), "600-1000": (600, 1000)}
SHARES = {"150-300": 0.40, "300-600": 0.35, "600-1000": 0.25}
# The benchmark needs at least 100 rows in every bucket, so its buckets are equal.
BENCH_SHARES = {bucket: 1 / 3 for bucket in BUCKETS}
# Transcripts per request: a request's reply takes about as long as its words.
PER_REQUEST = {"150-300": 6, "300-600": 3, "600-1000": 2}
RECOGNIZER_PERCENT = 70
# Batches a cell runs at once. A batch over 600 words takes about 10 minutes for about one kept row,
# so with one batch at a time the long cells would run on for a day after the rest, leaving most
# codex slots idle. A cell can overshoot its target by the batches still running when it is reached.
LANES = {"150-300": 2, "300-600": 6, "600-1000": 12}
FEEDBACK_NOTES = 8
# A reference that fails the gates gets a second sample from the teacher at its bucket's reasoning.
# With glm-5.3 as teacher this was the judge's model (`"judge"`), which got 32 of 40 long references
# past the strict judge where the teacher at low got 24-25. With gpt-6.1-sol as both, the teacher's
# setting is the stronger one for every bucket.
RELABEL_ROLE = "teacher"
# The teacher's reasoning per length bucket; a bucket not listed keeps `LLM_REASONING` (high). On 40
# long references, gpt-6.1-sol passed the judge 7 of 14 times over 600 words at xhigh against 3 at
# high; below 600 words, xhigh was no better and took 2.5 times as long (report, section 5).
TEACHER_REASONING = {"600-1000": "xhigh"}
# One row per judge request: batched at low reasoning, the gates missed every fault a hand read found
# in 13 of 54 long references; one row at a time, the judge caught 6 of them at low reasoning and 9 at
# max (report, section 4).
JUDGE_ITEMS = 1
# The judge's reasoning per length bucket; a bucket not listed keeps `LLM_JUDGE_REASONING` (max for
# the training run). 9 of the 13 faults were in rows over 600 words, and max costs about 7 times low
# per row, so the shorter rows are checked at low.
JUDGE_REASONING = {"150-300": "low", "300-600": "low"}

_BENCH_NAMES = (
    "Adaeze Bjorn Camila Dev Esme Felix Gideon Harriet Ivo Jun Kalani Lorenzo Maren Nnamdi Orla Pavel Rhea Soren "
    "Tamsin Ulrich Valentina Wendell Ximena Yusuf Zara Anouk Bashir Cosima Dariusz Eero Freya Gustavo Hamid Iris "
    "Joaquin Kirra Leif Mbali Nikhil Odette Phuong Rafferty Saoirse Teodor Uzma Vesna Wilhelmina Yohannes Zeynep"
).split()
_BENCH_SETTINGS = (
    "a veterinary clinic", "a ferry terminal", "a community theater", "a solar installer", "a food truck",
    "a university admissions office", "a brewery", "a school bus depot", "a museum archive", "a yoga studio",
    "a pharmacy", "an orchard", "a recording studio", "a print shop", "a ski resort", "a daycare",
    "a charity auction", "an architecture firm", "a fishing charter", "a climbing gym", "a dental lab",
    "a city council office", "a moving company", "a tailor", "a greenhouse nursery", "a marathon organizing team",
    "a local newspaper", "an animal shelter", "a wine shop", "a robotics club", "an insurance claim",
    "a family reunion", "a kitchen remodel", "a visa application", "a startup board meeting", "a chess club",
)


@dataclass(frozen=True)
class Run:
    name: str
    out_dir: Path
    base: Path
    names: tuple[str, ...]
    settings: tuple[str, ...]
    split: str
    shares: dict[str, float]


TRAIN = Run("train", paths.DATA / "stage4", paths.PROMPTS / "generate" / "base.md", tuple(generate._NAMES), generate._SETTINGS, "", SHARES)
BENCH = Run("bench", paths.DATA / "bench" / "long", paths.PROMPTS / "bench" / "base.md", tuple(_BENCH_NAMES), _BENCH_SETTINGS, "", BENCH_SHARES)
RUNS = {run.name: run for run in (TRAIN, BENCH)}


@dataclass
class BatchRecord:
    batch: int
    style: str
    requested: int
    returned: int = 0
    passed: int = 0
    relabeled: int = 0
    notes: list[str] | None = None
    seconds: float = 0.0
    error: str = ""


def stem(category: str, bucket: str) -> str:
    return f"{category}.{bucket}"


def output_file(run: Run, category: str, bucket: str) -> Path:
    return run.out_dir / f"{stem(category, bucket)}.jsonl"


def _log_file(run: Run, category: str, bucket: str) -> Path:
    return run.out_dir / f"{stem(category, bucket)}.batches.jsonl"


def _rejected_file(run: Run, category: str, bucket: str) -> Path:
    return run.out_dir / f"{stem(category, bucket)}.rejected.jsonl"


def style_for(run: Run, category: str, bucket: str, batch: int) -> str:
    rng = random.Random(f"{run.name}-style-{category}-{bucket}-{batch}")
    return "recognizer" if rng.randrange(100) < RECOGNIZER_PERCENT else "raw"


def generator_messages(run: Run, category: str, bucket: str, style: str, count: int, batch: int, notes: Sequence[str]) -> list[dict]:
    base = run.base.read_text(encoding="utf-8")
    style_text = (paths.PROMPTS / "generate" / f"style_{style}.md").read_text(encoding="utf-8")
    guidance = (paths.PROMPTS / "compose" / f"{category}.md").read_text(encoding="utf-8")
    low, high = BUCKETS[bucket]
    rng = random.Random(f"{run.name}-{category}-{bucket}-{batch}")
    names = ", ".join(rng.sample(run.names, 8))
    settings = "; ".join(rng.sample(run.settings, 4))
    user = (
        f"{guidance}\n\n{style_text}\n\nBatch {batch}. Write {count} transcripts for this category, each "
        f"{low} to {high} words long, never over {high}; aim for about {(low + high) // 2} words, and count them. These are "
        "long dictations: pack each with many specific facts (names, numbers, dates, times, amounts, items, "
        "places), keep it noisy and spoken the whole way through, and put at least one correction late in the "
        "transcript that changes something said much earlier. Each transcript is a different situation. "
        f"For variety, draw on some of these names ({names}) and settings ({settings}), and invent others."
    )
    if notes:
        user += (
            "\n\nThe checker rejected these transcripts from the previous batch. Avoid the same problems:\n"
            + "\n".join(f"- {note}" for note in notes)
        )
    return [{"role": "system", "content": base}, {"role": "user", "content": user}]


def _bucket_of(item: dict) -> str | None:
    return file_bucket(len(item["input"].split()))


def file_bucket(words: int) -> str | None:
    """The bucket a transcript of `words` words is filed under, whatever was asked; None outside 150-1,000."""
    bucket = length_bucket(words)
    return bucket if bucket in BUCKETS else None


async def run_batch(
    llm: LlmClient, run: Run, category: str, bucket: str, batch: int, count: int, notes: Sequence[str], known: set[str]
) -> tuple[BatchRecord, list[Example], list[dict]]:
    style = style_for(run, category, bucket, batch)
    record = BatchRecord(batch=batch, style=style, requested=count, notes=[])
    started = time.monotonic()
    try:
        reply = await llm.complete(generator_messages(run, category, bucket, style, count, batch, notes), role="generator")
        items = [item for item in generate._items(reply, count) if item["input"] not in known]
    except (ValueError, KeyError, TypeError) as error:
        record.error = f"unreadable generator reply: {error}"[:300]
        record.seconds = round(time.monotonic() - started, 1)
        return record, [], []
    record.returned = len(items)
    rejected: list[dict] = []
    candidates = []
    for item in items:
        words = len(item["input"].split())
        id_ = example_id(generate.SOURCE, item["input"])
        if file_bucket(words) is None:
            rejected.append({"id": id_, "input": item["input"], "failed": ["length"], "note": f"{words} words, asked for {bucket}"})
        else:
            candidates.append((id_, item))
    # The teacher's first references. One whose layout contradicts the generator's intent would be
    # dropped below whatever the judge said, so it is dropped before the judge's cost is paid.
    outputs = await asyncio.gather(
        *(relabel(llm, item["input"], reasoning=TEACHER_REASONING.get(_bucket_of(item))) for _, item in candidates)
    )
    firsts = {id_: output for (id_, _), output in zip(candidates, outputs, strict=True)}
    unjudged = []
    for id_, item in candidates:
        mismatch = generate.layout_mismatch(item["layout"], layout_of(firsts[id_]))
        if mismatch:
            rejected.append({"id": id_, "input": item["input"], "output": firsts[id_], "failed": ["layout_intent"], "note": mismatch})
        else:
            unjudged.append((id_, item))
    candidates = unjudged
    groups: dict[str | None, list[tuple[str, dict]]] = {}
    for id_, item in candidates:
        groups.setdefault(_bucket_of(item), []).append((id_, item))
    results: dict[str, gates.Gated] = {}
    for bucket_, group in groups.items():
        judge = gates.Judge(
            gates.COMPOSE, gates.COMPOSE_CRITERIA, {id_: style for id_, _ in group}, max_items=JUDGE_ITEMS,
            reasoning=JUDGE_REASONING.get(bucket_),
        )
        results |= await gates.gate(
            llm, [(id_, item["input"]) for id_, item in group], {id_: firsts[id_] for id_, _ in group}, judge=judge,
            relabel_role=RELABEL_ROLE, teacher_reasoning=TEACHER_REASONING.get(bucket_),
        )
    passed = []
    for id_, item in candidates:
        result = results[id_]
        if result.passed and result.output.strip() == EMPTY:
            result = replace(result, passed=False, failed=["empty"], note="the teacher returned EMPTY")
        if not result.passed:
            rejected.append({"id": id_, "input": item["input"], **asdict(result)})
            continue
        words = len(item["input"].split())
        example = Example(
            id=id_,
            input=item["input"],
            output=result.output,
            source=generate.SOURCE,
            category=category,
            license=generate.LICENSE,
            tags=[
                *item["tags"], f"style_{style}", "composed", f"length:{length_bucket(words)}",
                *(["relabeled"] if result.attempt else []),
            ],
            split=run.split,
            teacher_model=llm.settings.judge_model if result.attempt and RELABEL_ROLE == "judge" else llm.settings.model,
        )
        mismatch = generate.layout_mismatch(item["layout"], example.layout)
        if mismatch:
            rejected.append({"id": id_, "input": item["input"], **asdict(result), "failed": ["layout_intent"], "note": mismatch})
            continue
        passed.append(example)
        record.relabeled += bool(result.attempt)
    record.passed = len(passed)
    # Only notes about the transcript help the generator; a reference's faults are the teacher's.
    record.notes = [
        f"{', '.join(row['failed'])}: {row['note']} (transcript began: {row['input'][:80]!r})"
        for row in rejected
        if set(row["failed"]) & {"length", "input_realistic"} and row.get("note")
    ][:FEEDBACK_NOTES]
    record.seconds = round(time.monotonic() - started, 1)
    return record, passed, rejected


def _append(path: Path, rows: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as sink:
        sink.writelines(row + "\n" for row in rows)


def output_license(example: Example, glm_inputs: set[str]) -> str:
    """Whose model output a composed row is: the transcript's writer and the reference's.

    The first rows had `glm-5.3` write the transcript; `rejudge train-codex` re-gated exactly those
    (`glm_inputs`), and every later transcript is `gpt-6.1-sol`'s. Both providers' terms restrict
    training competing models on their outputs (README, Terms and licenses).
    """
    teacher = example.teacher_model or ""
    authors = {"zai" if example.id in glm_inputs else "openai", "openai" if teacher.startswith("gpt") else "zai"}
    return "+".join(f"{author}-output" for author in sorted(authors))


def load(run: Run, passes: Sequence[str] = rejudge.PASSES) -> list[Example]:
    """The run's rows, with the re-judge passes (`rejudge.py`) applied and each row's output license."""
    rows = [e for path in sorted(run.out_dir.glob("*.jsonl")) if path.name.count(".") == 2 for e in read_examples(path)]
    codex = rejudge.overrides_file(run.out_dir, "codex")
    glm_inputs = {row["id"] for row in read_jsonl(codex)} if codex.exists() else set()
    return [replace(e, license=output_license(e, glm_inputs)) for e in rejudge.apply(rows, run.out_dir, passes)]


def counts_by_cell(run: Run) -> Counter:
    return Counter((e.category, file_bucket(len(e.input.split()))) for e in load(run))


async def generate_cell(
    llm: LlmClient,
    run: Run,
    category: str,
    bucket: str,
    target: int,
    known: set[str],
    counts: Counter | None = None,
    progress: Callable[[str], None] = lambda line: print(line, file=sys.stderr, flush=True),
) -> int:
    """Generates batches for one category and bucket until the bucket holds `target` rows; resumes from its files.

    A passing transcript that misses the requested bucket but is 150-1,000 words long is filed under
    the bucket its length puts it in, and counts there. `counts` is shared by the cells running at once.
    """
    counts = counts_by_cell(run) if counts is None else counts
    log_path = _log_file(run, category, bucket)
    log = list(read_jsonl(log_path)) if log_path.exists() else []
    per_request = PER_REQUEST[bucket]
    max_batches = 3 * -(-target // per_request) + 3
    # A batch still running when a run stopped was never logged; its number is skipped on resume.
    next_batch = max((record["batch"] for record in log), default=-1) + 1
    stopped = False

    async def lane() -> None:
        nonlocal next_batch, stopped
        while not stopped and counts[(category, bucket)] < target:
            if next_batch >= max_batches:
                stopped = True
                progress(f"{run.name} {stem(category, bucket)}: stopped after {next_batch} batches with {counts[(category, bucket)]}/{target}")
                break
            batch = next_batch
            next_batch += 1
            notes = log[-1]["notes"] if log else []
            try:
                record, passed, rejected = await run_batch(llm, run, category, bucket, batch, per_request, notes, known)
            except LlmError as error:
                progress(f"{run.name} {stem(category, bucket)}: batch {batch} failed: {error}")
                raise
            passed = [e for e in passed if e.input not in known]
            known.update(e.input for e in passed)
            for e in passed:
                actual = file_bucket(len(e.input.split()))
                counts[(category, actual)] += 1
                _append(output_file(run, category, actual), [e.to_json()])
            _append(_rejected_file(run, category, bucket), [json.dumps(row, ensure_ascii=False) for row in rejected])
            _append(log_path, [json.dumps(asdict(record), ensure_ascii=False)])
            log.append(asdict(record))
            elsewhere = sum(file_bucket(len(e.input.split())) != bucket for e in passed)
            progress(
                f"{run.name} {stem(category, bucket)}: batch {batch} ({record.style}) {record.passed}/{record.returned} passed"
                f"{f' ({elsewhere} filed under another bucket)' if elsewhere else ''}"
                f"{' ' + record.error if record.error else ''}, {counts[(category, bucket)]}/{target}, {record.seconds}s"
            )

    await asyncio.gather(*(lane() for _ in range(LANES[bucket])))
    return counts[(category, bucket)]


def targets(total: int, shares: dict[str, float] = SHARES, categories: Sequence[str] = CATEGORIES) -> dict[tuple[str, str], int]:
    """Rows per category and bucket for `total` rows, split between the buckets by `shares`."""
    return {
        (category, bucket): max(1, round(total * share / len(categories)))
        for category in categories
        for bucket, share in shares.items()
    }


async def generate_all(llm: LlmClient, run: Run, total: int, categories: Sequence[str] = CATEGORIES) -> dict[str, int]:
    """Every category and bucket at once; the client's semaphore caps the requests in flight."""
    known = {e.input for e in load(run)}
    shared = counts_by_cell(run)
    goals = targets(total, run.shares, categories)
    counts = await asyncio.gather(
        *(generate_cell(llm, run, category, bucket, target, known, shared) for (category, bucket), target in goals.items())
    )
    return {stem(category, bucket): count for (category, bucket), count in zip(goals, counts, strict=True)}


def rejections(run: Run) -> dict[str, int]:
    """Why rows were dropped, by the first failed criterion."""
    counts: Counter = Counter()
    for path in run.out_dir.glob("*.rejected.jsonl"):
        for row in read_jsonl(path):
            counts[row["failed"][0] if row["failed"] else "unknown"] += 1
    return dict(counts.most_common())
