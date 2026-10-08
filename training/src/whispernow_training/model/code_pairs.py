"""DPO pairs made in code (todo 114): the reference against the same reference with one fact broken.

No sampling and no judge. Each rejected reply differs from its reference in one contiguous span:
- `dropped_sentence`: a sentence of a line that has more than one, with the space before or after it;
- `dropped_item`: a bulleted item of a list of three or more, or the last item of a numbered list;
- `dropped_reason`: a "because", "so that" or "in case" clause after the main clause;
- `dropped_hedge`: "maybe", "probably", "I think" and the like, or "about" before an amount;
- `changed`: a number, time, weekday, month or colour said as another value the text never has;
- `wrong_side`: a known-answer correction's abandoned value in place of the corrected one, for rows
  whose input has one of the noise injector's corrections (`datagen/inject.py`).

Pairs are written in `cleanup prefer`'s `pairs.jsonl` format, so DPO trains on them unchanged.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

from ..datagen.audit import length_bucket
from ..datagen.inject import _wrong_value
from ..datagen.paths import BUILD
from ..datagen.schema import EMPTY, Example, read_examples
from . import facts
from .prefer import split_of

SEED = 114
# Share of pairs per kind, from the fault mix todo 113's model makes on bench-long (dropped 3.7,
# changed 1.8, wrong side 0.5 per row); `build` re-weights when a row has no place for a kind.
MIX = {
    "dropped_sentence": 1.4,
    "dropped_item": 1.0,
    "dropped_reason": 0.8,
    "dropped_hedge": 0.5,
    "changed": 1.8,
    "wrong_side": 0.5,
}
MIN_UNIT_WORDS = 4
_HEDGE_DROP = re.compile(
    r"\b(?:maybe|probably|possibly|perhaps|I think|I guess|I believe|I suppose)\b,? "
    r"|\b(?:about|around|roughly|approximately) (?=\$?\d)",
    re.IGNORECASE,
)
_REASON_DROP = re.compile(r",? (?:because|so that|in case) [^.,;:!?\n()]*?(?=[.,;:!?\n)]|$)", re.IGNORECASE)
_VALUE_TOKEN = re.compile(r"(?<![\w$.:,])(\$?\d{1,4}%?|\d{1,2}:\d{2}|[A-Z][a-z]+|[a-z]+)(?![\w.:]\w)")
# A known-answer correction as `inject._correction` writes it: the wrong value, a cue, the right one.
_CUE = (
    r"(?:no,? wait,? sorry|sorry,? no,? I mean|no,? sorry|no,? wait|I mean|no|sorry|wait)"
    r"|or (?:\w+ )?actually no,? let's (?:make it|go with)"
    r"|except,? uh,? well,? (?:that's not right|that's wrong|I got that mixed up|no,? that's not it|I had that wrong),?"
    r" so actually let's just (?:make it|go with)"
)
_CORRECTION = re.compile(rf"(?<![\w$])(\$?[\w:%]+),? (?:{_CUE}),? (\$?[\w:%]+)", re.IGNORECASE)


def _strip(token: str) -> str:
    return token.rstrip(".,;:!?)")


def _word_in(value: str, text: str) -> list[re.Match]:
    return list(re.finditer(rf"(?<![\w$.]){re.escape(value)}(?![\w]|\.\d)", text))


# --- breakers: each returns the rejected text, or None when the row has no place for its kind ----


def drop_sentence(text: str, rng: random.Random) -> str | None:
    found = facts.units(text)
    options = [
        u for u in found
        if u.kind == "sentence" and not (u.first_in_line and u.last_in_line) and len(text[u.start:u.end].split()) >= MIN_UNIT_WORDS
    ]
    if not options:
        return None
    unit = rng.choice(options)
    if unit.last_in_line:
        start = unit.start
        while start > unit.line_start and text[start - 1] == " ":
            start -= 1
        return text[:start] + text[unit.end:]
    end = unit.end
    while end < unit.line_end and text[end] == " ":
        end += 1
    return text[:unit.start] + text[end:]


def drop_item(text: str, rng: random.Random) -> str | None:
    found = facts.units(text)
    lines = [u for u in found if u.kind in ("bullet", "numbered")]
    runs: list[list[facts.Unit]] = []
    for unit in lines:
        if runs and runs[-1][-1].kind == unit.kind and runs[-1][-1].line_end + 1 == unit.line_start:
            runs[-1].append(unit)
        else:
            runs.append([unit])
    options = []
    for run in runs:
        if len(run) < 3:
            continue
        # Dropping a numbered item mid-list would leave a gap in the numbers, a tell that is not the fact.
        candidates = run if run[0].kind == "bullet" else run[-1:]
        options += [u for u in candidates if len(text[u.start:u.end].split()) >= MIN_UNIT_WORDS - 1]
    if not options:
        return None
    unit = rng.choice(options)
    if unit.line_end < len(text) and text[unit.line_end] == "\n":
        return text[:unit.line_start] + text[unit.line_end + 1:]
    return text[:unit.line_start - 1] + text[unit.line_end:]


def drop_reason(text: str, rng: random.Random) -> str | None:
    starts = {u.start for u in facts.units(text)}
    options = [
        m for m in _REASON_DROP.finditer(text)
        if m.start() not in starts and len(m.group(0).split()) >= 3 and text[m.start() - 1:m.start()] not in ("(", "\n")
    ]
    if not options:
        return None
    match = rng.choice(options)
    return text[:match.start()] + text[match.end():]


def drop_hedge(text: str, rng: random.Random) -> str | None:
    options = [m for m in _HEDGE_DROP.finditer(text) if not re.match(r"that\b", text[m.end():])]
    if not options:
        return None
    match = rng.choice(options)
    rest = text[match.end():]
    at_start = match.start() == 0 or text[:match.start()].rstrip(" ").endswith((".", "!", "?", "\n", "- ")) or \
        re.search(r"(?:^|\n)(?:-|\d+\.) $", text[:match.start()]) is not None
    if at_start and rest[:1].islower():
        rest = rest[0].upper() + rest[1:]
    return text[:match.start()] + rest


def change_value(text: str, rng: random.Random) -> str | None:
    options = []
    for match in _VALUE_TOKEN.finditer(text):
        core = match.group(1)
        wrong = _wrong_value(core, text, random.Random(f"{rng.random()}"))
        if wrong:
            options.append((match, wrong[0]))
    if not options:
        return None
    match, wrong = rng.choice(options)
    return text[:match.start(1)] + wrong + text[match.end(1):]


def corrections(row: Example) -> list[tuple[str, str]]:
    """(wrong, right) for each known-answer correction in a noised row's input that its reference resolves."""
    if not any(tag.startswith("correction:") for tag in row.tags):
        return []
    found = []
    for match in _CORRECTION.finditer(row.input):
        wrong, right = _strip(match.group(1)), _strip(match.group(2))
        if wrong.lower() == right.lower() or _word_in(wrong, row.output) or len(_word_in(right, row.output)) != 1:
            continue
        found.append((wrong, right))
    return found


def wrong_side(row: Example, rng: random.Random) -> str | None:
    options = corrections(row)
    if not options:
        return None
    wrong, right = rng.choice(options)
    match = _word_in(right, row.output)[0]
    # The input capitalizes a value that opens a sentence; the reference keeps the right one's case.
    if right[:1].isupper() and wrong[:1].islower():
        wrong = wrong[0].upper() + wrong[1:]
    elif right[:1].islower() and wrong[:1].isupper():
        wrong = wrong[0].lower() + wrong[1:]
    return row.output[:match.start()] + wrong + row.output[match.end():]


BREAKERS: dict[str, Callable[[Example, random.Random], str | None]] = {
    "dropped_sentence": lambda row, rng: drop_sentence(row.output, rng),
    "dropped_item": lambda row, rng: drop_item(row.output, rng),
    "dropped_reason": lambda row, rng: drop_reason(row.output, rng),
    "dropped_hedge": lambda row, rng: drop_hedge(row.output, rng),
    "changed": lambda row, rng: change_value(row.output, rng),
    "wrong_side": wrong_side,
}


def _middles(chosen: str, rejected: str) -> tuple[str, str]:
    """What is left of each text between their common prefix and suffix."""
    prefix = 0
    while prefix < min(len(chosen), len(rejected)) and chosen[prefix] == rejected[prefix]:
        prefix += 1
    suffix = 0
    while suffix < min(len(chosen), len(rejected)) - prefix and chosen[-1 - suffix] == rejected[-1 - suffix]:
        suffix += 1
    return chosen[prefix:len(chosen) - suffix], rejected[prefix:len(rejected) - suffix]


def one_span(chosen: str, rejected: str) -> bool:
    """`rejected` is `chosen` with one span deleted (the next letter may be recapitalized) or one word changed."""
    removed, added = _middles(chosen, rejected)
    if not removed and not added:
        return False
    if not added:
        return True
    if len(added) == 1 and removed and removed[-1].lower() == added.lower() and len(removed) > 1:
        return True
    return not re.search(r"\s", removed) and not re.search(r"\s", added)


def pair(row: Example, kind: str, rng: random.Random) -> dict | None:
    rejected = BREAKERS[kind](row, rng)
    if rejected is None or not one_span(row.output, rejected):
        return None
    return {
        "id": row.id, "split": split_of(row.id), "input": row.input, "chosen": row.output, "rejected": rejected,
        "kind": kind, "chosen_from": "reference", "bucket": length_bucket(len(row.input.split())),
    }


def build(train: Sequence[Example], benchmark: Sequence[Example], count: int, mix: dict[str, float] = MIX,
          long_share: float = 0.7, seed: int = SEED) -> list[dict]:
    """`count` pairs, one per row, `long_share` of them from rows of 150 words or more.

    Kinds are drawn by `mix`; a row with no place for the drawn kind tries the others in mix order.
    Correction rows go to `wrong_side` first, as they are the only ones that can make it.
    """
    held_out = {row.id for row in benchmark} | {row.seed_group for row in benchmark}
    rows = [r for r in train if r.output != EMPTY and r.id not in held_out and r.seed_group not in held_out]
    rng = random.Random(seed)
    rng.shuffle(rows)
    long_rows = [r for r in rows if len(r.input.split()) >= 150]
    short_rows = [r for r in rows if len(r.input.split()) < 150]
    target = {"long": round(count * long_share), "short": count - round(count * long_share)}
    kinds, weights = list(mix), list(mix.values())
    want = Counter({kind: round(count * w / sum(weights)) for kind, w in mix.items()})
    pairs: list[dict] = []
    made = Counter()
    for group, pool in (("long", long_rows), ("short", short_rows)):
        taken = 0
        # Correction rows first, so wrong-side pairs are not starved by rows drawn for other kinds.
        ordered = sorted(pool, key=lambda r: not any(t.startswith("correction:") for t in r.tags))
        for row in ordered:
            if taken >= target[group]:
                break
            short = [k for k in kinds if made[k] < want[k]] or kinds
            first = rng.choices(short, [mix[k] for k in short])[0]
            order = [first] + [k for k in short if k != first]
            if "wrong_side" in short and corrections(row):
                order = ["wrong_side"] + [k for k in order if k != "wrong_side"]
            # A row with no place for the kinds still short makes one of the others.
            order += [k for k in kinds if k not in order]
            for kind in order:
                made_pair = pair(row, kind, rng)
                if made_pair:
                    pairs.append(made_pair)
                    made[kind] += 1
                    taken += 1
                    break
    return pairs


def stats(pairs: Sequence[dict]) -> dict:
    words = lambda text: len(text.split())  # noqa: E731
    return {
        "pairs": len(pairs),
        "by_kind": dict(Counter(p["kind"] for p in pairs).most_common()),
        "by_bucket": dict(Counter(p["bucket"] for p in pairs).most_common()),
        "val": sum(p["split"] == "val" for p in pairs),
        "rejected_over_chosen_words": round(sum(words(p["rejected"]) for p in pairs) / max(1, sum(words(p["chosen"]) for p in pairs)), 3),
    }


def run(out: Path, count: int, long_share: float = 0.7, mix: dict[str, float] = MIX) -> dict:
    """Writes `pairs.jsonl` and `stats.json` under `out`, from the build's train split."""
    from .evaluate import BENCHMARKS

    benchmark = [row for path in BENCHMARKS.values() for row in read_examples(path)]
    pairs = build(read_examples(BUILD / "train.jsonl"), benchmark, count, mix, long_share)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pairs.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")
    summary = {"count": count, "long_share": long_share, "mix": mix, **stats(pairs)}
    (out / "stats.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
