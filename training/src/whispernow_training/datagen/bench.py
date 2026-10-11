"""The two held-out benchmarks: `data/bench/long.jsonl` and `data/bench/noise.jsonl`.

The long benchmark is the `compose` bench run's rows plus long single-speaker turns from the AMI
meeting corpus (`ami.py`), the only real speech the license survey kept. The teacher writes an AMI
turn's reference and it passes the completeness gates; the realism and format criteria of the
`compose` judge are not asked, since the speech is real and a meeting turn is not dictation.

The noise benchmark is assembled by `noise.assemble`.

Neither file shares a template or seed group with the build, and `build` never reads
`data/bench/`. `overlap` confirms no benchmark input nearly repeats a training input.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, replace

from . import ami, compose, gates, noise, paths, rejudge, relabel
from .llm import LlmClient
from .schema import EMPTY, Example, length_bucket, read_examples, read_jsonl, words, write_examples

LONG_FILE = paths.DATA / "bench" / "long.jsonl"
AMI_FILE = paths.DATA / "bench" / "long" / "ami.jsonl"
AMI_REJECTED = paths.DATA / "bench" / "long" / "ami.rejected.jsonl"
# AMI turns to try per bucket; most real turns are short, so the longest buckets take what there is.
AMI_PER_BUCKET = {"150-300": 25, "300-600": 20, "600-1000": 15}
AMI_SEED = 110
AMI_JUDGE = gates.Judge(gates.COMPLETENESS, gates.COMPLETENESS_CRITERIA)
SHINGLE = 8
OVERLAP_LIMIT = 0.3


def pick_ami(turns: Sequence[Example]) -> list[Example]:
    """Up to `AMI_PER_BUCKET` turns per bucket, at most two per meeting and bucket."""
    rng = random.Random(AMI_SEED)
    picked = []
    for bucket, count in AMI_PER_BUCKET.items():
        pool = sorted((t for t in turns if length_bucket(words(t.input)) == bucket), key=lambda t: t.id)
        rng.shuffle(pool)
        per_meeting: Counter = Counter()
        chosen = []
        for turn in pool:
            meeting = next(tag for tag in turn.tags if tag.startswith("ami:"))
            if per_meeting[meeting] < 2 and len(chosen) < count:
                per_meeting[meeting] += 1
                chosen.append(turn)
        picked += chosen
    return picked


async def label_ami(llm: LlmClient) -> dict[str, int]:
    """Teacher references for the picked AMI turns, through the completeness gates; resumes."""
    done = {e.id for e in read_examples(AMI_FILE)} if AMI_FILE.exists() else set()
    done |= {row["id"] for row in read_jsonl(AMI_REJECTED)} if AMI_REJECTED.exists() else set()
    todo = [t for t in pick_ami(ami.long_turns()) if t.id not in done]
    results = await gates.gate(llm, [(t.id, t.input) for t in todo], judge=AMI_JUDGE)
    kept, rejected = [], []
    for turn in todo:
        result = results[turn.id]
        if result.passed and result.output.strip() != EMPTY:
            tags = [*turn.tags, f"length:{length_bucket(words(turn.input))}", *(["relabeled"] if result.attempt else [])]
            kept.append(replace(turn, output=result.output, tags=tags, teacher_model=llm.settings.model))
        else:
            rejected.append({"id": turn.id, "input": turn.input, **asdict(result)})
    AMI_FILE.parent.mkdir(parents=True, exist_ok=True)
    with AMI_FILE.open("a", encoding="utf-8") as sink:
        sink.writelines(e.to_json() + "\n" for e in kept)
    with AMI_REJECTED.open("a", encoding="utf-8") as sink:
        sink.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rejected)
    return {"kept": len(kept), "rejected": len(rejected)}


def assemble_long() -> dict:
    real = rejudge.apply(read_examples(AMI_FILE), AMI_FILE.parent) if AMI_FILE.exists() else []
    rows = [*compose.load(compose.BENCH), *real]
    rows = [replace(e, seed_group=f"long-bench:{e.id}") for e in sorted(rows, key=lambda e: e.id)]
    rows = relabel.apply_bench(rows, "bench-long")
    write_examples(LONG_FILE, rows)
    return long_summary(rows)


def long_summary(rows: Sequence[Example]) -> dict:
    return {
        "rows": len(rows),
        "per_bucket": dict(sorted(Counter(length_bucket(words(e.input)) for e in rows).items())),
        "per_source": dict(Counter(e.source for e in rows)),
        "per_category": dict(sorted(Counter(e.category for e in rows).items())),
        "per_bucket_real": dict(sorted(Counter(length_bucket(words(e.input)) for e in rows if "real_speech" in e.tags).items())),
        "style": dict(Counter(next((t for t in e.tags if t.startswith("style_")), "?") for e in rows)),
        "layout": dict(Counter(e.layout for e in rows)),
        "relabeled": sum("relabeled" in e.tags for e in rows),
        "longest_words": max((words(e.input) for e in rows), default=0),
    }


def _shingles(text: str) -> set[int]:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    return {
        int.from_bytes(hashlib.blake2b(" ".join(tokens[i : i + SHINGLE]).encode(), digest_size=8).digest(), "big")
        for i in range(max(0, len(tokens) - SHINGLE + 1))
    }


def overlap(bench_rows: Sequence[Example], build_rows: Sequence[Example]) -> list[dict]:
    """Benchmark rows whose input shares more than `OVERLAP_LIMIT` of its 8-word shingles with the build's inputs."""
    seen: set[int] = set()
    for e in build_rows:
        seen |= _shingles(e.input)
    flagged = []
    for e in bench_rows:
        shingles = _shingles(e.input)
        if shingles:
            share = len(shingles & seen) / len(shingles)
            if share > OVERLAP_LIMIT:
                flagged.append({"id": e.id, "share": round(share, 2)})
    return flagged


def assemble() -> dict:
    """Writes both benchmark files and checks them against every built split."""
    report = {"long": assemble_long(), "noise": noise.assemble()}
    build_rows = [e for split in ("train", "val", "test") if (paths.BUILD / f"{split}.jsonl").exists() for e in read_examples(paths.BUILD / f"{split}.jsonl")]
    for name, path in (("long", LONG_FILE), ("noise", noise.BENCH_FILE)):
        report[name]["near_duplicates_of_build"] = overlap(read_examples(path), build_rows)
    out = paths.DATA / "bench" / "report.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return report
