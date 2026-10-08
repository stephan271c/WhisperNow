"""The noise benchmark: short rows, each built around one hesitation or correction pattern.

Nine patterns, plus a control set where filler-like words carry meaning ("I'd like the red one", "I
mean it"), so a model that deletes too much is caught. The generator writes each transcript with
the exact phrases that show its pattern ("evidence"); code confirms the pattern is there
(`detect`), the teacher writes the reference, and the reference passes the gates (`gates.gate`
with the `compose` judge). A control reference must also keep its meaningful words.

About a third of the rows are then spoken by Kokoro and transcribed by Parakeet (`probe`), so the
benchmark has real recognizer output. A probed row gets a new reference from its new transcript,
through the same gates, and is kept only if that reference still has every fact of the original
transcript (the completeness judge, reading the original transcript against the new reference)
and the pattern survived the recognizer.

The requests use their own prompt (`prompts/bench/noise.md`), their own name and setting pools and a
"noise-bench" seed prefix; `build` never reads `data/bench/`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace

from . import gates, generate, paths, rejudge, relabel
from .compose import _BENCH_NAMES, _BENCH_SETTINGS
from .llm import LlmClient, LlmError, parse_json_reply
from .schema import EMPTY, Example, example_id, read_examples, read_jsonl, write_examples

PATTERNS = (
    "filler_cluster", "determiner_filler", "repeats", "restart", "short_cue", "redecision", "reasoned",
    "thinking_aloud", "pause_punct",
)
CONTROL = "control"
KINDS = (*PATTERNS, CONTROL)
TARGETS = {**{pattern: 45 for pattern in PATTERNS}, CONTROL: 60}
PER_REQUEST = 10
RECOGNIZER_PERCENT = 75
# Rows whose inputs are spoken and transcribed again: about a third.
# Over half are probed: about 40% of the probed rows keep their facts and pattern, for about a third of the file.
PROBE_PERCENT = 55
OUT_DIR = paths.DATA / "bench" / "noise"
BENCH_FILE = paths.DATA / "bench" / "noise.jsonl"
PROBED = OUT_DIR / "probed.jsonl"
PROBE_SEED = 110
FEEDBACK_NOTES = 8

_F = r"(?:um+|uh+|er|erm|hmm+)"
_FILLER = re.compile(rf"\b{_F}\b", re.IGNORECASE)
_CLUSTER = re.compile(rf"\b{_F}\b[,.]? {_F}\b", re.IGNORECASE)
_DETERMINER = re.compile(
    rf"\b(the|a|an|this|that|these|those|my|our|your|his|her|their|some)\b,? (?:{_F},? )+\1\b", re.IGNORECASE
)
_MEANINGFUL_REPEATS = {"that", "had", "very", "really", "so", "bye", "no", "yeah", "okay", "ha", "knock", "is", "far", "long", "more", "over", "again"}
_REPEAT = re.compile(r"\b([a-z']+(?: [a-z']+){0,2})(?:,? \1\b)+", re.IGNORECASE)
_BIGRAM_RESTART = re.compile(r"\b(\w+ \w+)\b(?: [\w',]+){1,8}? \1\b", re.IGNORECASE)
_SHORT_CUE = re.compile(r"\b(?:no|nope|sorry|wait|i mean|make (?:that|it)|scratch that|or rather|rather|oops)\b", re.IGNORECASE)
_REDECISION = re.compile(
    r"\b(?:let's|let us|I'll|we'll|we should|I want|I'd|let me|we can|we could|I think we|I guess)\b|"
    r"\b(?:go with|make it|make that|switch to|stick with)\b",
    re.IGNORECASE,
)
_REASON = re.compile(
    r"\b(?:because|'cause|cause|since|except|but|though|too|so|won't|doesn't|don't|can't|isn't|aren't|wasn't|not|out of|closed|busy|full)\b",
    re.IGNORECASE,
)
_INSTEAD = re.compile(r"\binstead\b", re.IGNORECASE)
_THINKING = re.compile(
    r"\b(?:let's see|let me think|let me see|what else|where was I|so yeah,? I don't know|what was it|hello\?|can you hear me|"
    r"is this on|testing|is it recording|is this recording|mic check)\b",
    re.IGNORECASE,
)
# A period after a word no sentence ends on ("in the. Back", "about 12 of the. Navy"), any period or
# question mark before a lowercase word, or a list item said with rising intonation ("garlic? Onions,").
_PAUSE = re.compile(
    r"\b(?i:the|a|an|to|of|in|on|at|for|with|about|and|or|but|my|our|your|their|his|her|its|some|until|"
    r"from|into|is|are|was|be|should|probably|need|needs|will|would|can|could|than|this|that|these|those)"
    r"[.?] [A-Za-z]|\w[.?] [a-z]|\b\w+\? [A-Z]\w*,"
)
_CONTROL_WORD = re.compile(
    r"\b(?:like|i mean|actually|that that|you know|so|well|right|basically|kind of|sort of|literally)\b", re.IGNORECASE
)


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def detect(text: str) -> list[str]:
    """The patterns code can see in a transcript. Restarts and corrections need the evidence too (`shows`)."""
    found = []
    if _CLUSTER.search(text):
        found.append("filler_cluster")
    if _DETERMINER.search(text):
        found.append("determiner_filler")
    repeats = [m.group(1).lower() for m in _REPEAT.finditer(text)]
    if any(" " in word or (word not in _MEANINGFUL_REPEATS and not _FILLER.fullmatch(word)) for word in repeats):
        found.append("repeats")
    if _THINKING.search(text):
        found.append("thinking_aloud")
    if _PAUSE.search(text):
        found.append("pause_punct")
    return found


def _restarted_word(phrase: str) -> bool:
    """A content word said, abandoned and said again: "Buy, um we need to buy"."""
    tokens = [t for t in _norm(phrase).split() if not _FILLER.fullmatch(t)]
    return any(len(t) > 2 and t in tokens[i + 2 : i + 10] for i, t in enumerate(tokens))


def shows(kind: str, text: str, evidence: Sequence[str]) -> bool:
    """Whether the transcript has the pattern: in code where code can see it, else in its evidence."""
    evidence = [e for e in evidence if e and _norm(e) and _norm(e) in _norm(text)]
    if kind in ("filler_cluster", "determiner_filler", "repeats", "thinking_aloud", "pause_punct"):
        return kind in detect(text)
    if kind == CONTROL:
        return not _FILLER.search(text) and bool(evidence) and all(_CONTROL_WORD.search(e) for e in evidence)
    if not evidence:
        return False
    if kind == "restart":
        return any(_BIGRAM_RESTART.search(_norm(e)) or _restarted_word(e) for e in evidence)
    if kind == "short_cue":
        return any(_SHORT_CUE.search(e) and not _INSTEAD.search(e) for e in evidence)
    if kind == "redecision":
        return any(_REDECISION.search(e) for e in evidence)
    if kind == "reasoned":
        # The reason often comes in the clause before the evidence's "instead".
        return any(_INSTEAD.search(e) for e in evidence) and bool(_REASON.search(text))
    raise ValueError(kind)


# Words a faithful cleanup may drop from a control phrase: discourse markers and a repeated "that".
_DROPPABLE = {"so", "actually", "you", "know", "basically", "literally", "just", "that", "oh", "okay"}


def keeps_control_words(reference: str, evidence: Sequence[str]) -> bool:
    """A control reference keeps every word of each phrase whose filler-like word carries meaning ("I'd
    like the red one", "I mean it", "it went well"), except discourse markers it may drop."""
    target = set(_norm(reference).split())
    for phrase in evidence:
        words = _norm(phrase).split()
        if any(word not in target for word in words if word not in _DROPPABLE):
            return False
    return True


def style_for(kind: str, batch: int) -> str:
    if kind in ("pause_punct", "thinking_aloud"):
        return "recognizer"
    rng = random.Random(f"noise-bench-style-{kind}-{batch}")
    return "recognizer" if rng.randrange(100) < RECOGNIZER_PERCENT else "raw"


def generator_messages(kind: str, style: str, count: int, batch: int, notes: Sequence[str]) -> list[dict]:
    base = (paths.PROMPTS / "bench" / "noise.md").read_text(encoding="utf-8")
    style_text = (paths.PROMPTS / "generate" / f"style_{style}.md").read_text(encoding="utf-8")
    guidance = (paths.PROMPTS / "bench" / "noise" / f"{kind}.md").read_text(encoding="utf-8")
    rng = random.Random(f"noise-bench-{kind}-{batch}")
    names = ", ".join(rng.sample(_BENCH_NAMES, 8))
    settings = "; ".join(rng.sample(_BENCH_SETTINGS, 5))
    user = (
        f"{guidance}\n\n{style_text}\n\nBatch {batch}. Write {count} transcripts with this pattern. "
        f"For variety, draw on some of these names ({names}) and settings ({settings}), and invent others."
    )
    if notes:
        user += (
            "\n\nThe checker rejected these transcripts from the previous batch. Avoid the same problems:\n"
            + "\n".join(f"- {note}" for note in notes)
        )
    return [{"role": "system", "content": base}, {"role": "user", "content": user}]


# One malformed item fails the whole reply; this recovers every item object that parses on its own.
_OBJECT = re.compile(r"\{[^{}]*\}")


def _salvage(reply: str) -> list[dict]:
    items = []
    for text in _OBJECT.findall(reply):
        try:
            items.append(json.loads(text))
        except ValueError:
            continue
    if not items:
        raise ValueError("no item in the reply parses")
    return items


def _items(reply: str, count: int) -> list[dict]:
    try:
        parsed = parse_json_reply(reply)
        items = parsed["items"] if isinstance(parsed, dict) else parsed
    except ValueError:
        items = _salvage(reply)
    kept, seen = [], set()
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("input"), str):
            continue
        text = " ".join(item["input"].split())
        if not text or text in seen:
            continue
        seen.add(text)
        evidence = [str(e) for e in item.get("evidence", []) if isinstance(e, str)][:6]
        tags = [str(tag) for tag in item.get("tags", []) if isinstance(tag, str)][:4]
        kept.append({"input": text, "evidence": evidence, "layout": str(item.get("layout", "prose")), "tags": tags})
    return kept[:count]


def pattern_tags(kind: str, text: str) -> list[str]:
    """`pattern:<kind>` for the row's own pattern, `also:<pattern>` for others code sees in it."""
    return [f"pattern:{kind}", *(f"also:{p}" for p in detect(text) if p != kind)]


async def run_batch(
    llm: LlmClient, kind: str, batch: int, count: int, notes: Sequence[str], known: set[str]
) -> tuple[dict, list[Example], list[dict]]:
    style = style_for(kind, batch)
    record = {"batch": batch, "style": style, "requested": count, "returned": 0, "passed": 0, "notes": [], "seconds": 0.0, "error": ""}
    started = time.monotonic()
    try:
        reply = await llm.complete(generator_messages(kind, style, count, batch, notes), role="generator")
        items = [item for item in _items(reply, count) if item["input"] not in known]
    except (ValueError, KeyError, TypeError) as error:
        record["error"] = f"unreadable generator reply: {error}"[:300]
        record["seconds"] = round(time.monotonic() - started, 1)
        return record, [], []
    record["returned"] = len(items)
    rejected: list[dict] = []
    candidates = []
    for item in items:
        id_ = example_id(generate.SOURCE, item["input"])
        words = len(item["input"].split())
        if words >= 150:
            rejected.append({"id": id_, "input": item["input"], "failed": ["length"], "note": f"{words} words; keep it under 150"})
        elif not shows(kind, item["input"], item["evidence"]):
            rejected.append({"id": id_, "input": item["input"], "failed": ["pattern"], "note": f"the {kind} pattern is not in the transcript as asked", "evidence": item["evidence"]})
        else:
            candidates.append((id_, item))
    judge = gates.Judge(gates.COMPOSE, gates.COMPOSE_CRITERIA, {id_: style for id_, _ in candidates})
    results = await gates.gate(llm, [(id_, item["input"]) for id_, item in candidates], judge=judge) if candidates else {}
    passed = []
    for id_, item in candidates:
        result = results[id_]
        if result.passed and result.output.strip() == EMPTY:
            result = replace(result, passed=False, failed=["empty"], note="the teacher returned EMPTY")
        if result.passed and kind == CONTROL and not keeps_control_words(result.output, item["evidence"]):
            result = replace(result, passed=False, failed=["control_words"], note="the reference dropped a meaningful word")
        if not result.passed:
            rejected.append({"id": id_, "input": item["input"], **asdict(result)})
            continue
        passed.append(
            Example(
                id=id_,
                input=item["input"],
                output=result.output,
                source=generate.SOURCE,
                category=f"noise_{kind}",
                license=generate.LICENSE,
                tags=[
                    *item["tags"], f"style_{style}", "noise_bench", *pattern_tags(kind, item["input"]),
                    *(f"evidence:{e}" for e in item["evidence"]), *(["relabeled"] if result.attempt else []),
                ],
                teacher_model=llm.settings.model,
            )
        )
    record["passed"] = len(passed)
    record["notes"] = [
        f"{', '.join(row['failed'])}: {row['note']} (transcript began: {row['input'][:80]!r})"
        for row in rejected
        if set(row["failed"]) & {"length", "pattern", "input_realistic"} and row.get("note")
    ][:FEEDBACK_NOTES]
    record["seconds"] = round(time.monotonic() - started, 1)
    return record, passed, rejected


def _append(path, rows: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as sink:
        sink.writelines(row + "\n" for row in rows)


def output_file(kind: str):
    return OUT_DIR / f"{kind}.jsonl"


async def generate_kind(
    llm: LlmClient,
    kind: str,
    target: int,
    known: set[str],
    progress: Callable[[str], None] = lambda line: print(line, file=sys.stderr, flush=True),
) -> int:
    out = output_file(kind)
    kept = read_examples(out) if out.exists() else []
    log_path = OUT_DIR / f"{kind}.batches.jsonl"
    log = list(read_jsonl(log_path)) if log_path.exists() else []
    max_batches = 3 * -(-target // PER_REQUEST) + 3
    while len(kept) < target:
        if len(log) >= max_batches:
            progress(f"noise {kind}: stopped after {len(log)} batches with {len(kept)}/{target}")
            break
        batch = len(log)
        notes = log[-1]["notes"] if log else []
        try:
            record, passed, rejected = await run_batch(llm, kind, batch, PER_REQUEST, notes, known)
        except LlmError as error:
            progress(f"noise {kind}: batch {batch} failed: {error}")
            raise
        passed = [e for e in passed if e.input not in known][: target - len(kept)]
        known |= {e.input for e in passed}
        kept += passed
        _append(out, [e.to_json() for e in passed])
        _append(OUT_DIR / f"{kind}.rejected.jsonl", [json.dumps(row, ensure_ascii=False) for row in rejected])
        _append(log_path, [json.dumps(record, ensure_ascii=False)])
        log.append(record)
        progress(
            f"noise {kind}: batch {batch} ({record['style']}) {record['passed']}/{record['returned']} passed"
            f"{' ' + record['error'] if record['error'] else ''}, {len(kept)}/{target}, {record['seconds']}s"
        )
    return len(kept)


def load_raw() -> list[Example]:
    return [e for kind in KINDS if output_file(kind).exists() for e in read_examples(output_file(kind))]


def load() -> list[Example]:
    """The generated rows, with the header re-judge (`rejudge.py`) applied; `assemble` applies the strict pass."""
    return rejudge.apply(load_raw(), OUT_DIR, passes=("rejudged",))


async def generate_all(llm: LlmClient, kinds: Sequence[str] = KINDS) -> dict[str, int]:
    known = {e.input for e in load()}
    counts = await asyncio.gather(*(generate_kind(llm, kind, TARGETS[kind], known) for kind in kinds))
    return dict(zip(kinds, counts, strict=True))


# A cleanup of what the recognizer heard cannot recover a name it misspelled.
PROBED_JUDGE = gates.Judge(
    addendum=(
        "Here the REFERENCE was written from a speech recognizer's version of the TRANSCRIPT, not the TRANSCRIPT "
        "itself. The recognizer may spell a name differently ('Cosima' as 'Casima', 'Okafor' as 'Ockhaffer'): a "
        "differently spelled name is not a changed fact. Judge every other difference as usual."
    )
)


def chosen_for_probe(example: Example) -> bool:
    digest = hashlib.sha256(f"noise-probe-{example.id}".encode()).hexdigest()
    return int(digest, 16) % 100 < PROBE_PERCENT


def probe_audio() -> int:
    """Speaks and transcribes the chosen rows (Kokoro, then Parakeet); resumes past rows already heard."""
    from . import asr_probe

    rows = [e for e in load() if chosen_for_probe(e)]
    return asr_probe.probe_texts([(e.id, e.input) for e in rows], PROBED, PROBE_SEED)


def _evidence(example: Example) -> list[str]:
    return [tag.removeprefix("evidence:") for tag in example.tags if tag.startswith("evidence:")]


def _kind(example: Example) -> str:
    return next(tag.removeprefix("pattern:") for tag in example.tags if tag.startswith("pattern:"))


async def gate_probed(llm: LlmClient) -> dict[str, int]:
    """New references for the heard transcripts; keeps a row whose facts and pattern survived."""
    by_id = {e.id: e for e in load()}
    heard = {row["id"]: row for row in read_jsonl(PROBED)} if PROBED.exists() else {}
    out = OUT_DIR / "probed.gated.jsonl"
    done = {row["id"] for row in read_jsonl(out)} if out.exists() else set()
    todo = [by_id[id_] for id_ in heard if id_ in by_id and id_ not in done]
    counts: Counter = Counter()
    rows = []
    judge = gates.Judge(gates.COMPOSE, gates.COMPOSE_CRITERIA, {e.id: "recognizer" for e in todo})
    results = await gates.gate(llm, [(e.id, heard[e.id]["heard"]) for e in todo], judge=judge) if todo else {}
    # The new reference against the original transcript: did the recognizer change a fact?
    same = await gates.check_references(
        llm, [(e.id, e.input, results[e.id].output) for e in todo if results[e.id].passed], PROBED_JUDGE
    )
    for example in todo:
        text = heard[example.id]["heard"]
        result = results[example.id]
        kind = _kind(example)
        evidence = [e for e in _evidence(example) if _norm(e) in _norm(text)]
        if not result.passed:
            reason = "gates: " + ",".join(result.failed)
        elif not same[example.id].passed:
            reason = "recognizer changed a fact: " + same[example.id].note
        elif not shows(kind, text, evidence):
            reason = "pattern lost"
        elif kind == CONTROL and not keeps_control_words(result.output, evidence):
            reason = "control words lost"
        else:
            reason = ""
        counts[reason.split(":")[0] or "kept"] += 1
        rows.append({"id": example.id, "heard": text, "output": result.output, "attempt": result.attempt, "kept": not reason, "reason": reason})
    _append(out, [json.dumps(row, ensure_ascii=False) for row in rows])
    return dict(counts)


def assemble() -> dict:
    """Writes `data/bench/noise.jsonl`: generated rows, with the kept probed rows in place of their originals."""
    gated_path = OUT_DIR / "probed.gated.jsonl"
    gated = {row["id"]: row for row in read_jsonl(gated_path)} if gated_path.exists() else {}
    rows = []
    for example in load():
        probe = gated.get(example.id)
        if probe and probe["kept"]:
            kind = _kind(example)
            tags = [t for t in example.tags if not t.startswith(("also:", "style_"))]
            tags = [t for t in tags if t != f"pattern:{kind}"]
            example = replace(
                example,
                id=example_id(generate.SOURCE + ":probe", probe["heard"]),
                input=probe["heard"],
                output=probe["output"],
                tags=[*tags, *pattern_tags(kind, probe["heard"]), "style_recognizer", "asr_probe", f"probed_from:{example.id}"],
            )
        rows.append(replace(example, seed_group=f"noise-bench:{example.id}"))
    rows = rejudge.apply(rows, OUT_DIR, passes=("strict",))
    rows = relabel.apply_bench(rows, "bench-noise")
    write_examples(BENCH_FILE, rows)
    return summary(rows)


def summary(rows: Sequence[Example]) -> dict:
    by_kind = Counter(_kind(e) for e in rows)
    also = Counter(t.removeprefix("also:") for e in rows for t in e.tags if t.startswith("also:"))
    return {
        "rows": len(rows),
        "per_pattern": dict(sorted(by_kind.items())),
        "also_seen": dict(also.most_common()),
        "probed": sum("asr_probe" in e.tags for e in rows),
        "style": dict(Counter(next((t for t in e.tags if t.startswith("style_")), "?") for e in rows)),
        "relabeled": sum("relabeled" in e.tags for e in rows),
    }
