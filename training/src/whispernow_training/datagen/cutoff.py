"""Cut-off word starts: a word's beginning said just before the word ("interf interface", "set s settings").

Speakers cut a word off and say it again, and the cleanup should drop the piece, so a row with one
keeps its own reference.

- `run` gives `ROWS` train rows of data/build-v3 one to three cuts each, a fixed count per length
  bucket in proportion to the pool (`targets`), so rows too short to cut don't thin out the short
  bucket. In most rows a short window around each cut word is spoken with the word's start said first, by Kokoro or Phonon, and
  transcribed by Parakeet, which writes the piece as a bare prefix ("inv invoice"), a sound-alike word
  ("throw, through") or letters ("EV every"); the heard piece goes into the input only when the words
  around it were heard as said, lowercased and unpunctuated in a raw-style row. The other rows get a
  written piece: a bare prefix, a prefix and a letter ("set s settings"), or a word the cut word starts with.
- A piece is kept only where the text shows it is a stumble: it starts with the cut word's first two
  letters; it is one word, or one word and spelled letters; that word is no common word ("not" before
  "notify", "do" before "doctor"), number or acronym the references use ("SEC" before "security");
  and the two words are no pair the references say on purpose ("per person"). Parakeet's "new number"
  for "nu, number" fails, as nothing marks "new" as a fragment.
- `check` then asks Mercury Decide about every cut, one call per failure mode
  (prompts/judge/cutoff.json): could the piece be a word the speaker meant, would keeping it state a
  different fact, could it be a name or acronym. A row with any cut it flags is replaced (`top_up`), and
  only rows it passed are built.
- `bench` writes data/bench/cutoff.jsonl, test rows with cuts, and `score` compares each model's
  output on them with its output on the same test rows.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import cache
from pathlib import Path

import httpx

from ..text_rules import FILLER_TOKEN
from . import heard, paths
from .asr_probe import ONES, TENS
from .inject import _core, _raw_token
from .schema import EMPTY, Example, example_id, length_bucket, read_examples, write_examples

TAG = "cutoff"
OUT_DIR = paths.DATA / "cutoff"
BENCH = paths.DATA / "bench" / "cutoff.jsonl"
BENCH_CUTS = paths.DATA / "bench" / "cutoff.cuts.jsonl"
CHECK = OUT_DIR / "check.jsonl"
CHECK_PROMPT = paths.PROMPTS / "judge" / "cutoff.json"
# A cut fails when Mercury Decide's P(yes) reaches this on any question. On 44 hand-labelled cuts it failed
# 15 of the 19 unsafe ones (the rules already stop three of the rest), and 3 of 25 safe-looking ones whose
# piece could be meant after all ("straw," before "strawberries", "hose" before "hospital", "CAF" before
# "cafeteria").
CHECK_LIMIT = 0.4
# Words of the transcript shown on each side of the piece.
CHECK_WINDOW = 25
ROWS = 3000
# The <50-word bucket's share of its proportional count; the longer buckets split the rest in proportion. With
# the full share, the 1.2B lost 1.12x the facts of the 1.2B without cut rows on test dictations of up to 25
# words (Sol).
SHORT_SHARE = 0.5
SHORT = "<50"
BENCH_ROWS = 150
SEED = 121
# Words spoken on each side of a cut word.
CONTEXT = 4
MIN_WORD = 5
# A cut word is one the references use at least this often: no names or rare jargon.
COMMON_WORD = 20
# A written bare prefix is no word the references use this often; a written sound-alike is one they use this often.
RARE_PREFIX = 3
REAL_PREFIX = 50
# Who renders a row's pieces.
TTS_SHARES = {"kokoro": 37.5, "phonon": 37.5, "written": 25}
# A one-word piece the references use this often (about once in 10,000 words) reads as meant: "not", "do", "set".
COMMON_PIECE = 1000
FILLER_BETWEEN = 0.2
# Words tried per cut a row needs; 66% of Parakeet's pieces passed in a pilot of 80.
ATTEMPTS = 4
_VOWELS = set("aeiouy")
# A number said before a word is a fact, whatever the word ("at nine night" for "ni, night").
_NUMBER_WORDS = {*ONES, *TENS[2:], "hundred", "thousand", "million", "billion"}
_WRITTEN_SHAPES = {"bare": 40, "letter": 30, "word": 30}


def _norm(token: str) -> str:
    return re.sub(r"[^\w']", "", token).lower()


@dataclass(frozen=True)
class Vocabulary:
    words: Counter
    # Adjacent reference words where the second starts with the first's first two letters ("per person").
    pairs: Counter
    # Words the references write in capitals ("SEC", "EV").
    acronyms: Counter = field(default_factory=Counter)


def vocabulary(rows: Sequence[Example]) -> Vocabulary:
    words, pairs, acronyms = Counter(), Counter(), Counter()
    for e in rows:
        acronyms.update(re.findall(r"\b[A-Z]{2,}\b", e.output))
        tokens = re.findall(r"[\w']+", e.output.lower())
        words.update(tokens)
        pairs.update(f"{a} {b}" for a, b in zip(tokens, tokens[1:]) if a.isalpha() and b.startswith(a[:2]))
    return Vocabulary(words, pairs, acronyms)


def cuts_needed(words: int) -> int:
    return 1 if words <= 60 else 2 if words <= 200 else 3


def slots(tokens: Sequence[str], said: set[str], vocab: Vocabulary) -> list[int]:
    """Words that can be cut: common words the reference keeps, outside quotes, and no names."""
    out, quotes = [], 0
    for i, token in enumerate(tokens):
        opened = quotes % 2
        quotes += sum(token.count(q) for q in '"“”')
        if opened or quotes % 2:
            continue
        core = _core(token)
        if token != core and not token.endswith((".", ",", "!", "?", ";", ":")):
            continue
        if not core.isalpha() or len(core) < MIN_WORD or vocab.words[core.lower()] < COMMON_WORD or core.lower() not in said:
            continue
        sentence_start = i == 0 or tokens[i - 1].endswith((".", "!", "?"))
        if core[0].isupper() and not (sentence_start and core[1:].islower()):
            continue
        out.append(i)
    return out


def cut_points(word: str) -> list[int]:
    """Lengths of a start of `word` a speaker says before cutting off: a quarter to two thirds of it, with a vowel."""
    return [k for k in range(2, len(word) - 2) if 0.25 <= k / len(word) <= 0.65 and _VOWELS & set(word[:k])]


def splice(tokens: Sequence[str], at: int, heard_text: str) -> list[str] | None:
    """The heard words from the cut through the cut word, or None when the words around it were misheard."""
    lo, hi = max(0, at - CONTEXT), min(len(tokens), at + CONTEXT + 1)
    got = heard_text.split()
    ops = SequenceMatcher(None, [_norm(t) for t in tokens[lo:hi]], [_norm(t) for t in got], autojunk=False).get_opcodes()
    k = at - lo
    left = right = None
    for tag, i1, i2, j1, _ in ops:
        if tag == "equal" and i1 <= k - 1 < i2:
            left = j1 + k - 1 - i1
        if tag == "equal" and i1 <= k < i2:
            right = j1 + k - i1
    if right is None or (k and left is None):
        return None
    if k + 1 < hi - lo and not any(tag == "equal" and i1 <= k + 1 < i2 for tag, i1, i2, _, _ in ops):
        return None
    start = left + 1 if k else 0
    return got[start : right + 1] if start < right else None


def stumble_reason(piece: Sequence[str], word: str, vocab: Vocabulary) -> str | None:
    """Why `piece` before `word` would not read as a cut-off start, or None when it does."""
    words = [w for w in (_norm(t) for t in piece if not FILLER_TOKEN.match(t)) if w]
    joined = "".join(words)
    if joined[:2] != word[:2]:
        return "no shared start"
    if any(vocab.acronyms[token.strip(".,!?;:-")] >= RARE_PREFIX for token in piece):
        return "an acronym the references use"
    if any(len(w) > 1 or w in ("a", "i") for w in words[1:]):
        return "more than one word"
    if words[0] in _NUMBER_WORDS or words[0].isdigit():
        return "a number"
    if len(words) == 1 and vocab.words[words[0]] >= COMMON_PIECE:
        return "a common word"
    if vocab.pairs[f"{joined} {word}"]:
        return "a pair said on purpose"
    return None


def written_piece(word: str, prefix: str, rng: random.Random, vocab: Vocabulary) -> list[str]:
    shape = rng.choices(list(_WRITTEN_SHAPES), list(_WRITTEN_SHAPES.values()))[0]
    if shape == "word":
        heads = [word[:k] for k in range(3, len(word) - 2) if REAL_PREFIX <= vocab.words[word[:k]] < COMMON_PIECE]
        if heads:
            return [rng.choice(heads)]
    if shape == "letter":
        letter = word[len(prefix)]
        return [prefix, letter if letter not in _VOWELS else "s"]
    if vocab.words[prefix] >= RARE_PREFIX:
        return [prefix, word[len(prefix)] if word[len(prefix)] not in _VOWELS else "s"]
    return [prefix]


def _cased(token: str, heard_word: str) -> str:
    """`token` starting in the case Parakeet gave the word, which a new sentence before it changes."""
    first = token[0].upper() if heard_word[:1].isupper() else token[0].lower()
    return first + token[1:]


def cut_row(example: Example, speakers: dict | None, recognizer, vocab: Vocabulary, seed: int = SEED) -> dict:
    """One row's cuts: a record with the new input when every cut it needs passed, else why not."""
    rng = random.Random(f"{seed}-{example.id}")
    spans = [m.span() for m in re.finditer(r"\S+", example.input)]
    tokens = [example.input[a:b] for a, b in spans]
    candidates = slots(tokens, set(re.findall(r"[\w']+", example.output.lower())), vocab)
    rng.shuffle(candidates)
    need = cuts_needed(len(tokens))
    raw = "style_raw" in example.tags or not example.input[:1].isupper()
    tts = rng.choices(list(TTS_SHARES), list(TTS_SHARES.values()))[0]
    cuts: dict[int, dict] = {}
    rejected: Counter = Counter()
    voice = None
    tried = 0
    for at in candidates:
        if len(cuts) == need or tried == need * ATTEMPTS:
            break
        word = _core(tokens[at]).lower()
        points = cut_points(word)
        if not points or any(abs(at - other) <= CONTEXT for other in cuts):
            continue
        tried += 1
        prefix = word[: rng.choice(points)]
        filler = [rng.choice(("um", "uh"))] if rng.random() < FILLER_BETWEEN else []
        if tts == "written":
            piece, token = written_piece(word, prefix, rng, vocab) + filler, tokens[at]
            if not raw and token[:1].isupper():
                piece[0], token = piece[0][:1].upper() + piece[0][1:], token[:1].lower() + token[1:]
        else:
            lo, hi = max(0, at - CONTEXT), min(len(tokens), at + CONTEXT + 1)
            spoken_text = " ".join([*tokens[lo:at], prefix + ",", *(f"{f}," for f in filler), *tokens[at:hi]])
            spoken = speakers[tts].speak(spoken_text, random.Random(f"{seed}-{example.id}-voice"))
            voice = spoken.voice
            got = splice(tokens, at, recognizer.transcribe(spoken.samples)[0])
            if got is None:
                rejected["context misheard"] += 1
                continue
            if raw:
                piece, token = [t for t in map(_raw_token, got[:-1]) if t], tokens[at]
            else:
                piece, token = got[:-1], _cased(tokens[at], got[-1])
        reason = stumble_reason(piece, word, vocab)
        if reason:
            rejected[reason] += 1
            continue
        cuts[at] = {"word": word, "prefix": prefix, "piece": piece, "token": token}
    record = {"id": example.id, "tts": tts, "voice": voice, "rejected": dict(rejected)}
    if len(cuts) < need:
        return {**record, "kept": False, "reason": f"{len(cuts)} of {need} cuts"}
    # Spliced into the input as written, so its line breaks and spacing stay.
    parts, last = [], 0
    for i, c in sorted(cuts.items()):
        parts += [example.input[last : spans[i][0]], " ".join([*c["piece"], c["token"]])]
        last = spans[i][1]
    text = "".join([*parts, example.input[last:]])
    if not raw and text[:1].islower() and example.input[:1].isupper():
        text = text[0].upper() + text[1:]
    cut_list = [{key: c[key] for key in ("word", "prefix", "piece")} for _, c in sorted(cuts.items())]
    return {**record, "kept": True, "input": text, "cuts": cut_list}


@cache
def _check_spec() -> dict:
    return json.loads(CHECK_PROMPT.read_text(encoding="utf-8"))


def _input_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def check_state(text: str, cut: dict) -> str | None:
    """What Mercury Decide reads about one cut: the passage around the piece, with and without it."""
    tokens, piece = text.split(), [t.lower() for t in cut["piece"]]
    for i in range(len(tokens) - len(piece)):
        if [t.lower() for t in tokens[i : i + len(piece)]] == piece and _core(tokens[i + len(piece)]).lower() == cut["word"]:
            lo, hi = max(0, i - CHECK_WINDOW), min(len(tokens), i + len(piece) + 1 + CHECK_WINDOW)
            transcript = " ".join(tokens[lo:hi])
            clean = " ".join(tokens[lo:i] + tokens[i + len(piece) : hi])
            return (f"{_check_spec()['preface']}\n\nTRANSCRIPT:\n{transcript}\n\nCLEAN:\n{clean}\n\n"
                    f"PIECE: {' '.join(tokens[i : i + len(piece)])}\nNEXT WORD: {_core(tokens[i + len(piece)])}")
    return None


def verdicts() -> dict[tuple[str, str, int, str], float]:
    """Mercury Decide's answers, by row id, input, cut and question."""
    if not CHECK.exists():
        return {}
    rows = (json.loads(line) for line in CHECK.open(encoding="utf-8") if line.strip())
    return {(v["id"], v["input"], v["cut"], v["question"]): v["p"] for v in rows}


def screen(record: dict, seen: dict) -> str | None:
    """Why Mercury Decide failed a kept record, "unchecked" while a question is unanswered, or None when it passed.
    A cut `check_state` cannot find fails the record, since it can never be asked about."""
    key, unchecked = _input_key(record["input"]), False
    for i, cut in enumerate(record["cuts"]):
        for question in _check_spec()["questions"]:
            p = seen.get((record["id"], key, i, question))
            if p is None and check_state(record["input"], cut) is None:
                return f"piece not found: '{' '.join(cut['piece'])}' before '{cut['word']}'"
            if p is None:
                unchecked = True
            elif p >= CHECK_LIMIT:
                return f"{question}: '{' '.join(cut['piece'])}' before '{cut['word']}'"
    return "unchecked" if unchecked else None


async def check(rows: Sequence[dict] | None = None, concurrency: int = 8) -> Counter:
    """Asks Mercury Decide every question about every cut of the kept records it has not answered for, one call each."""
    seen = verdicts()
    questions = _check_spec()["questions"]
    todo = [
        (r, i, question) for r in (records() if rows is None else rows) if r["kept"] for i in range(len(r["cuts"]))
        for question in questions if (r["id"], _input_key(r["input"]), i, question) not in seen
    ]
    counts: Counter = Counter()
    if not todo:
        return counts
    key = heard.openrouter_key()
    semaphore = asyncio.Semaphore(concurrency)
    CHECK.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=120) as client:

        async def one(record: dict, i: int, question: str) -> None:
            state = check_state(record["input"], record["cuts"][i])
            if state is None:
                counts["piece not found"] += 1
                return
            async with semaphore:
                p = await heard.decide(client, key, state, questions[question])
            if p is None:
                counts["no answer"] += 1
                return
            line = {"id": record["id"], "input": _input_key(record["input"]), "cut": i, "question": question, "p": p,
                    "model": heard.MEANING_MODEL}
            with CHECK.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(line) + "\n")
            counts[f"{question} {'flagged' if p >= CHECK_LIMIT else 'passed'}"] += 1

        await asyncio.gather(*(one(*item) for item in todo))
    return counts


def pool(base_dir: Path = paths.BUILD_V3) -> list[Example]:
    """Train rows that can take a cut, in the order `run` tries them; repair rows are recognizer text already."""
    rows = [e for e in read_examples(base_dir / "train.jsonl") if e.output != EMPTY and e.source != heard.SOURCE]
    rows.sort(key=lambda e: e.id)
    random.Random(SEED).shuffle(rows)
    return rows


def bucket(example: Example) -> str:
    return length_bucket(len(example.input.split()))


def targets(rows: Sequence[Example], n: int, short_share: float = 1.0) -> dict[str, int]:
    """`n` rows split over the length buckets in proportion to `rows`, by largest remainder. The `SHORT` bucket
    gets `short_share` of its count and the other buckets split the rest the same way."""
    sizes = Counter(bucket(e) for e in rows)
    out = _split(sizes, n)
    if short_share != 1.0 and SHORT in out:
        short = round(out[SHORT] * short_share)
        out = {SHORT: short, **_split(Counter({name: size for name, size in sizes.items() if name != SHORT}), n - short)}
    return out


def _split(sizes: Counter, n: int) -> dict[str, int]:
    exact = {name: n * size / sum(sizes.values()) for name, size in sizes.items()}
    out = {name: math.floor(share) for name, share in exact.items()}
    for name in sorted(exact, key=lambda name: (exact[name] - out[name], name), reverse=True)[: n - sum(out.values())]:
        out[name] += 1
    return out


def _share(short_share: float | None) -> float:
    return SHORT_SHARE if short_share is None else short_share


def _files() -> list[Path]:
    return sorted(OUT_DIR.glob("cutoff.*.jsonl"))


def records() -> list[dict]:
    """One record per row, the first written."""
    out: dict[str, dict] = {}
    for line in (line for path in _files() for line in path.open(encoding="utf-8") if line.strip()):
        record = json.loads(line)
        out.setdefault(record["id"], record)
    return list(out.values())


def _speakers(tts_provider: str) -> dict:
    from .asr_probe import speaker

    return {"kokoro": speaker("kokoro", tts_provider), "phonon": speaker("phonon")}


def run(shard: int = 0, shards: int = 1, tts_provider: str = "cuda", short_share: float | None = None) -> Counter:
    """Cuts this shard's share of the pool in pool order, skipping full buckets, until each length bucket has its
    share of its target; resumes past rows any shard's file has. A row whose cuts fail, or that `check` failed, is
    recorded and the next one tried."""
    from .asr_probe import Recognizer

    rows = pool()
    by_id = {e.id: e for e in rows}
    mine = {e.id for i, e in enumerate(rows) if i % shards == shard}
    want = {name: math.ceil(target / shards) for name, target in targets(rows, ROWS, _share(short_share)).items()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"cutoff.{shard}.jsonl"
    tried, seen = records(), verdicts()
    done = {r["id"] for r in tried}
    kept = Counter(
        bucket(by_id[r["id"]]) for r in tried if r["id"] in mine and r["kept"] and screen(r, seen) in (None, "unchecked")
    )

    def full() -> bool:
        return all(kept[name] >= count for name, count in want.items())

    counts: Counter = Counter()
    if full():
        return counts
    vocab = vocabulary(rows)
    speakers, recognizer = _speakers(tts_provider), Recognizer("parakeet", "cpu")
    with out.open("a", encoding="utf-8") as sink:
        for e in (e for i, e in enumerate(rows) if i % shards == shard and e.id not in done):
            name = bucket(e)
            if kept[name] >= want[name]:
                continue
            record = cut_row(e, speakers, recognizer, vocab)
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            kept[name] += record["kept"]
            counts[f"{name} {'kept' if record['kept'] else 'short'}"] += 1
            total = sum(min(kept[b], count) for b, count in want.items())
            print(f"shard {shard}: {total}/{sum(want.values())} kept; {name} {kept[name]}/{want[name]}, {e.id} "
                  f"{record['tts']}: {'kept' if record['kept'] else record['reason']}", flush=True)
            if full():
                break
    if not full():
        print(f"shard {shard}: the pool ran out with {dict(kept)} of {want}", flush=True)
    return counts


def load(short_share: float | None = None) -> list[dict]:
    """The kept records `check` passed: in each length bucket, the first of its target in pool order."""
    rows = pool()
    rank, by_id = {e.id: i for i, e in enumerate(rows)}, {e.id: e for e in rows}
    left, seen = targets(rows, ROWS, _share(short_share)), verdicts()
    out = []
    passed = (r for r in records() if r["kept"] and screen(r, seen) is None)
    for record in sorted(passed, key=lambda r: rank[r["id"]]):
        name = bucket(by_id[record["id"]])
        if left[name]:
            left[name] -= 1
            out.append(record)
    return out


def top_up(shards: int = 2, tts_provider: str = "cuda") -> dict:
    """Checks every kept row and cuts more where the check failed some, until each bucket has its target of passed rows."""
    while True:
        for _ in range(3):
            counts = asyncio.run(check())
            print(f"check: {dict(counts)}", flush=True)
            if not counts["no answer"]:
                break
        added: Counter = Counter()
        for shard in range(shards):
            added += run(shard, shards, tts_provider)
        if not added:
            return report()


def _piece_kind(cut: dict) -> str:
    words = [_norm(t) for t in cut["piece"] if not FILLER_TOKEN.match(t)]
    if words == [cut["prefix"]]:
        return "bare prefix"
    if words[0] == cut["prefix"] and all(len(w) == 1 for w in words[1:]):
        return "prefix and letter"
    return "heard or written otherwise"


def report() -> dict:
    rows = records()
    kept, seen = load(), verdicts()
    screened = Counter((screen(r, seen) or "passed").split(":")[0] for r in rows if r["kept"])
    by_id = {e.id: e for e in pool()}
    used = Counter(bucket(by_id[r["id"]]) for r in kept)
    pieces = Counter(_piece_kind(c) for r in kept for c in r["cuts"])
    return {
        "rows tried": len(rows), "kept": len(kept), "target": ROWS,
        "by length": {name: f"{used[name]}/{target}" for name, target in targets(list(by_id.values()), ROWS, SHORT_SHARE).items()},
        "tried by length": dict(Counter(bucket(by_id[r["id"]]) for r in rows)),
        "kept by tts": dict(Counter(r["tts"] for r in kept)), "cuts": sum(len(r["cuts"]) for r in kept), "pieces": dict(pieces),
        "short rows": dict(Counter(r["reason"] for r in rows if not r["kept"])),
        "rejected pieces": dict(sum((Counter(r["rejected"]) for r in rows), Counter())),
        "mercury check of kept rows": dict(screened),
    }


def bench(n: int = BENCH_ROWS, tts_provider: str = "cuda") -> int:
    """`n` test rows of 15-120 words with cuts `check` passed, each with its own id and a `cutoff_from:` tag naming
    its test row."""
    from .asr_probe import Recognizer

    vocab = vocabulary(read_examples(paths.BUILD_V3 / "train.jsonl"))
    rows = sorted(
        (e for e in read_examples(paths.BUILD_V3 / "test.jsonl") if e.output != EMPTY and 15 <= len(e.input.split()) <= 120),
        key=lambda e: e.id,
    )
    random.Random(SEED + 1).shuffle(rows)
    speakers, recognizer = _speakers(tts_provider), Recognizer("parakeet", "cpu")
    out, cuts = [], []
    for e in rows:
        record = cut_row(e, speakers, recognizer, vocab, SEED + 1)
        if not record["kept"]:
            continue
        record = {**record, "id": example_id(TAG, record["input"])}
        asyncio.run(check([record]))
        if screen(record, verdicts()) is not None:
            continue
        row = Example(
            id=record["id"], input=record["input"], output=e.output, source=e.source,
            category=e.category, license=e.license, tags=[*e.tags, TAG, f"cutoff_from:{e.id}", f"tts:{record['tts']}"],
            layout=e.layout, seed_group=e.seed_group, split="test",
        )
        out.append(row)
        cuts.append({"id": row.id, "origin": e.id, "tts": record["tts"], "cuts": record["cuts"]})
        print(f"{len(out)}/{n} {e.id} {record['tts']}", flush=True)
        if len(out) == n:
            break
    write_examples(BENCH, out)
    BENCH_CUTS.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cuts), encoding="utf-8")
    return len(out)


def _count(piece: str, text: str) -> int:
    return len(re.findall(rf"(?<![\w-]){re.escape(piece)}(?![\w])", text, re.IGNORECASE))


def _pairs(text: str) -> list[str]:
    tokens = re.findall(r"[\w']+", text.lower())
    return [f"{a} {b}" for a, b in zip(tokens, tokens[1:]) if a.isalpha() and len(a) >= 2 and b.startswith(a) and len(b) >= len(a) + 2]


def score(eval_dirs: Sequence[Path]) -> str:
    """Per eval dir: cut pieces left in the bench-cutoff outputs, cut words lost, outputs equal to the same test
    row's output, and, as an over-deletion control, how many look-alike pairs references keep ("we were", "per
    person") the test and benchmark outputs keep too."""
    from ..model.evaluate import load_rows, read_outputs

    cut_rows = [json.loads(line) for line in BENCH_CUTS.open(encoding="utf-8")]
    references = {r.id: r for split in ("test", "bench-long", "bench-noise") for r in load_rows(split)}
    lines = ["| eval | rows | cuts | pieces left | words lost | same as on the test row | look-alike pairs kept |",
             "|---|---|---|---|---|---|---|"]
    for eval_dir in eval_dirs:
        outputs = read_outputs(eval_dir / "bench-cutoff.outputs.jsonl")
        plain = read_outputs(eval_dir / "test.outputs.jsonl")
        cuts = left = lost = same = 0
        for row in cut_rows:
            noisy, before = outputs[row["id"]]["output"], plain[row["origin"]]["output"]
            same += noisy == before
            for c in row["cuts"]:
                cuts += 1
                words = [_norm(t) for t in c["piece"] if not FILLER_TOKEN.match(t)]
                left += any(_count(w, noisy) > _count(w, before) for w in words if w)
                lost += _count(c["word"], noisy) < _count(c["word"], before)
        pairs = kept_pairs = 0
        for split in ("test", "bench-long", "bench-noise"):
            for id_, output in read_outputs(eval_dir / f"{split}.outputs.jsonl").items():
                reference = references.get(id_)
                if reference is None:
                    continue
                for pair in set(_pairs(reference.output)) & set(_pairs(reference.input)):
                    pairs += 1
                    kept_pairs += pair in " ".join(re.findall(r"[\w']+", output["output"].lower()))
        lines.append(
            f"| {eval_dir} | {len(cut_rows)} | {cuts} | {100 * left / cuts:.1f}% | {100 * lost / cuts:.1f}% | "
            f"{100 * same / len(cut_rows):.1f}% | {100 * kept_pairs / max(pairs, 1):.1f}% of {pairs} |"
        )
    return "\n".join(lines)
