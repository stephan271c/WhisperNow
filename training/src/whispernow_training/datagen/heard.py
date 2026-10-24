"""Training rows whose input is what a recognizer heard, with the origin row's reference.

Kokoro speaks a v1 train row's transcript and the app's segmenter and Parakeet or Whisper
transcribe it (`asr_probe`). The reference stays right only where the cleanup could recover the
spoken words from the heard ones, so `restore` aligns the two and puts the spoken words back
wherever a difference touches a content word, a number, a name, a negation or a correction cue, or
the recognizer made up a word. Formatting, sentence breaks, number formats and function-word slips
stay as heard. `check` asks Mercury Decide whether each kept row still says what was spoken, and only
rows it passes are built; `audit` measures what both miss on a sample with the reference judge.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import cache

import httpx
from dotenv import load_dotenv

from ..text_rules import CORRECTION, filler_only
from . import gates, paths
from .asr_probe import ONES, TENS, normalize
from .schema import EMPTY, V2_TAG, Example, example_id, hash_bucket, read_examples

SOURCE = "heard"
TAG = "heard"
OUT_DIR = paths.DATA / "heard"
AUDIT = OUT_DIR / "audit.jsonl"
MEANING = OUT_DIR / "meaning.jsonl"
MEANING_PROMPT = paths.PROMPTS / "judge" / "meaning.json"
# Inception's free yes/no model on OpenRouter's System One endpoint; it answers with P(yes).
MEANING_MODEL = "inception/mercury-decide:free"
MEANING_URL = "https://openrouter.ai/api/v1/systemone"
# On the 277 pilot rows it flags 2%, about half real (a filler heard as "a" plus a sentence break);
# on rows restored by looser rules it caught 9 of the 13 the reference judge failed for a changed word.
MEANING_LIMIT = 0.5
SEED = 120
RECOGNIZERS = ("parakeet", "whisper")
# Parakeet TDT 0.6B v2 is the app's default recognizer; Whisper is the other family it offers.
PARAKEET_PERCENT = 67
# A word in fewer v1 train inputs than this is rare: a name, brand, drug or term the cleanup could
# not spell from a mishearing.
RARE_ROWS = 20
NEGATIONS = frozenset({"not", "no", "never", "nothing", "none", "nobody", "nor", "without"})
UNIT_WORDS = frozenset({"dollar", "dollars", "bucks", "cent", "cents", "percent", "euro", "euros", "pound", "pounds", "grand"})
# Words a recognizer may swap or drop without changing what the reference says. Prepositions,
# pronouns and conjunctions are left out: "ice in the cooler" heard as "ice and the cooler" loses the
# instruction, and the audit's judge failed such rows.
SWAPPABLE = frozenset({"a", "an", "the", "so", "well", "like", "yeah", "yes", "ok", "oh", "just", "also", "it's", "its"})
# What a recognizer writes for a filler ("uh" as "a"); an inserted word outside this is made up. Its
# "U" for "uh" reads as a letter ("the U landing", "U G F R"), so it is put back.
FILLER_LIKE = frozenset({"a", "ah", "eh", "er", "erm", "mm", "hmm", "mhm", "huh"})
# Past either limit the recognizer or Kokoro lost whole phrases and the row is mostly the
# spoken text again. Set from the 300-row probe, where they drop 3-6% of rows.
MAX_RESTORED_RUN = 6
MAX_RESTORED_SHARE = 0.15
# Voices tried per row before it is dropped.
ATTEMPTS = 3
# Tags that describe the origin's input, which the heard text replaces.
_INPUT_TAGS = (
    "style_", "asr_", "densified", "restyled", "fillers_", "composed", "injected", "from:", "noise:", "noise_level:",
)

_TOKEN = re.compile(r"[^\s$]+|\$\S*")
_ORDINAL = re.compile(r"(\d+)(?:st|nd|rd|th)([,.?!;:]*)", re.IGNORECASE)
_LETTER_DIGIT = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")
_TRAILING = re.compile(r"[,.?!;:]+$")
_CURRENCY = {"£": "pounds", "€": "euros"}
# A sign the comparison would drop ("±15%" compares as "15 percent"), so a token holding one only
# matches the same sign on the other side.
_SIGN = re.compile(r"[±°×÷~#^*+=<>|§¶•√∞≈≠≤≥]")
_ODD_TEXT = re.compile(r"[\\\x00-\x08\x0b-\x1f\x7f]")
_UNITS = {word: value for value, word in enumerate(ONES[:10])}
_TEENS = {word: value for value, word in enumerate(ONES) if value >= 10}
_TENS = {word: 10 * value for value, word in enumerate(TENS) if value >= 2}
_VALUES = {**_UNITS, **_TEENS, **_TENS}
_SCALES = {"thousand": 10**3, "million": 10**6, "billion": 10**9}
_NUMBER_WORDS = {*_VALUES, *_SCALES, "hundred"}
_ORDINALS = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}


@dataclass(frozen=True)
class Unit:
    """A compared word, or a number run as "#value", and the tokens it came from."""

    word: str
    first: int
    last: int


@dataclass(frozen=True)
class Restored:
    text: str
    # (heard, restored) text of each stretch put back.
    fixes: list[tuple[str, str]] = field(default_factory=list)
    restored_words: int = 0
    longest: int = 0
    reason: str = ""


def _ordinal(word: str) -> str:
    if word in _ORDINALS:
        return _ORDINALS[word]
    return word[:-1] + "ieth" if word.endswith("y") else word + "th"


# "fifteenth" reads as 15, so "April fifteenth" matches "April 15".
_ORDINAL_VALUES = {_ordinal(word): value for word, value in _VALUES.items() if value}


def _token_words(token: str) -> list[str]:
    token = re.sub(r"([£€])(\d[\d,.]*)", lambda m: f"{m.group(2)} {_CURRENCY[m.group(1)]}", token)
    ordinal = _ORDINAL.fullmatch(token)
    if ordinal:
        words = normalize(ordinal.group(1))
        return [*words[:-1], _ordinal(words[-1])]
    return normalize(_LETTER_DIGIT.sub(" ", token))


def _cardinal(words: Sequence[str]) -> int | None:
    total = current = 0
    last = ""
    for index, word in enumerate(words):
        if word == "and" and last in ("hundred", "scale"):
            continue
        if word == "a" and index == 0:
            word = "one"
        word = ONES[_ORDINAL_VALUES[word]] if _ORDINAL_VALUES.get(word, 20) < 20 else word
        word = TENS[_ORDINAL_VALUES[word] // 10] if word in _ORDINAL_VALUES else word
        kind = "unit" if word in _UNITS else "teen" if word in _TEENS else "tens" if word in _TENS else word
        if word in _VALUES:
            if last in ("unit", "teen") or (last == "tens" and kind != "unit"):
                return None
            current += _VALUES[word]
        elif word == "hundred":
            if last not in ("unit", "teen", "tens"):
                return None
            current *= 100
        elif word in _SCALES:
            if current == 0 or last == "scale":
                return None
            total, current, kind = total + current * _SCALES[word], 0, "scale"
        else:
            return None
        last = kind
    return total + current if last else None


def _digits(words: Sequence[str]) -> str | None:
    """A run read in groups: digit by digit ("nine six two oh one" as 96201) and years ("twenty twenty six" as
    2026) join up; other groups stay apart ("ten thirty" as "10 30", "eighteen twenty" as "18 20", not 1820)."""
    groups = _groups(words)
    if groups is None:
        return None
    if all(len(g) == 1 for g in groups) or (len(groups) > 1 and groups[0] in ("19", "20") and len("".join(groups[1:])) == 2):
        return "".join(groups)
    return " ".join(groups)


def _groups(words: Sequence[str]) -> list[str] | None:
    out, index = [], 0
    while index < len(words):
        word = words[index]
        following = words[index + 1] if index + 1 < len(words) else ""
        if word in _TENS and following in _UNITS and following != "zero":
            out.append(str(_TENS[word] + _UNITS[following]))
            index += 2
            continue
        if word == "oh":
            out.append("0")
        elif word in _ORDINAL_VALUES:
            out.append(str(_ORDINAL_VALUES[word]))
        elif word in _VALUES:
            out.append(str(_VALUES[word]))
        else:
            return None
        index += 1
    return out


def _number(words: Sequence[str]) -> str:
    """A number run by value, so "$1,200" matches "twelve hundred dollars"; the words when it reads as none."""
    if "point" in words:
        at = words.index("point")
        whole = _cardinal(words[:at]) if at else 0
        fraction = words[at + 1 :]
        if whole is not None and fraction and all(w in _UNITS or w == "oh" for w in fraction):
            return f"#{whole}.{_digits(fraction)}"
        return " ".join(words)
    value = _cardinal(words)
    digits = str(value) if value is not None else _digits(words)
    return f"#{digits}" if digits is not None else " ".join(words)


def _numeric(words: Sequence[str], index: int) -> bool:
    word = words[index]
    if word in _NUMBER_WORDS or word in _ORDINAL_VALUES:
        return True
    following = words[index + 1] if index + 1 < len(words) else ""
    if word == "point":
        return index > 0 and words[index - 1] in _NUMBER_WORDS and following in _UNITS
    if word == "oh":
        return following in _VALUES or (index > 0 and words[index - 1] in _VALUES)
    if word == "a":
        return following in ("hundred", *_SCALES)
    if word == "and":
        return index > 0 and words[index - 1] in ("hundred", *_SCALES) and following in _VALUES
    return False


def units(text: str) -> tuple[list[tuple[int, int]], list[Unit]]:
    """The text's token spans and its compared units, number runs collapsed to their value."""
    tokens = [(m.start(), m.end()) for m in _TOKEN.finditer(text)]
    pairs = [(word, index) for index, (start, end) in enumerate(tokens) for word in _token_words(text[start:end])]
    for k in range(len(pairs) - 1):
        (word, token), (following, next_token) = pairs[k], pairs[k + 1]
        if word == "dollars" and following in _SCALES and token != next_token:
            pairs[k], pairs[k + 1] = pairs[k + 1], pairs[k]
    for k in range(len(pairs) - 1):
        (word, token), (following, next_token) = pairs[k], pairs[k + 1]
        if word in FILLER_LIKE and token == next_token and _numeric([following], 0):
            pairs[k] = (f"{word}+", token)
    words = [word for word, _ in pairs]
    out, k = [], 0
    while k < len(pairs):
        if not _numeric(words, k):
            out.append(Unit(words[k], pairs[k][1], pairs[k][1]))
            k += 1
            continue
        j = k + 1
        while (
            j < len(pairs) and words[j - 1] not in _ORDINAL_VALUES and _numeric(words, j)
            and not _separate(text, tokens, pairs[j - 1][1], pairs[j][1])
        ):
            j += 1
        out.append(Unit(_number(words[k:j]), pairs[k][1], pairs[j - 1][1]))
        k = j
    signed = set()
    for k, unit in enumerate(out):
        sign = _SIGN.search(text[slice(*tokens[unit.first])])
        if sign and unit.first not in signed:
            out[k] = Unit(sign.group(0) + unit.word, unit.first, unit.last)
            signed.add(unit.first)
    return tokens, out


def _separate(text: str, tokens: list[tuple[int, int]], left: int, right: int) -> bool:
    """Whether two number tokens are separate numbers: "15, 2024" or "15 2024", but not "2 1 4" or "$2 million"."""
    if left == right:
        return False
    a, b = (text[slice(*tokens[i])] for i in (left, right))
    if _TRAILING.search(a):
        return True
    digits = [re.sub(r"\D", "", t) for t in (a, b)]
    return all(digits) and not all(len(d) == 1 for d in digits)


def names(reference: str) -> dict[str, str]:
    """Reference words written with a capital that a sentence start does not explain, by lower case.

    A word the reference also writes in lower case is not a name ("The" opening a quote).
    """
    lower = set(re.findall(r"\b[a-z][\w']*", reference))
    found = {}
    for sentence in re.split(r"(?<=[.!?:])\s+|(?<=[.!?:][\"”])\s+|\n+|(?<!\S)[\"“]", reference):
        for index, word in enumerate(re.findall(r"[A-Za-z][\w']*", sentence)):
            if word in ("I", "I'm", "I'll", "I've", "I'd"):
                continue
            if (index > 0 and word != word.lower() and word.lower() not in lower) or word[1:] != word[1:].lower():
                found[word.lower()] = word
    return found


@cache
def train_row_counts() -> Counter:
    """How many v1 train inputs each compared word appears in."""
    counts: Counter = Counter()
    for example in read_examples(paths.BUILD / "train.jsonl"):
        counts.update({unit.word for unit in units(example.input)[1]})
    return counts


def _fact(word: str, name_map: Mapping[str, str], counts: Mapping[str, int]) -> bool:
    """Whether the reference could not follow from a mishearing of this word."""
    return (
        word.startswith("#") or word in UNIT_WORDS or word in NEGATIONS or word.endswith("n't") or word in name_map
        or counts.get(word, 0) < RARE_ROWS or word not in SWAPPABLE | FILLER_LIKE
    )


def _protected(spoken: Sequence[str], heard: Sequence[str], name_map: Mapping[str, str], counts: Mapping[str, int]) -> bool:
    if not spoken:
        return not all(word in FILLER_LIKE for word in heard)
    if any(_fact(word, name_map, counts) for word in (*spoken, *heard)):
        return True
    return any(CORRECTION.search(" ".join(side)) for side in (spoken, heard))


def _widen(lo: int, hi: int, side: list[Unit]) -> tuple[int, int]:
    """Widens a unit range to every unit that shares a token with it."""
    if hi <= lo:
        return lo, hi
    while lo > 0 and side[lo - 1].last >= side[lo].first:
        lo -= 1
    while hi < len(side) and side[hi].first <= side[hi - 1].last:
        hi += 1
    return lo, hi


def _inside(lo: int, hi: int, side: list[Unit]) -> tuple[int, int]:
    """An empty range between two units of one token ("https" and "updates" in "https://updates") as those units."""
    if hi == lo and 0 < lo < len(side) and side[lo - 1].last >= side[lo].first:
        return lo - 1, lo + 1
    return lo, hi


def _aligned(lo: int, hi: int, ops: list, from_spoken: bool) -> tuple[int, int] | None:
    """The other side's unit range aligned with [lo, hi)."""
    if hi <= lo:
        return None
    found = None
    for tag, i1, i2, j1, j2 in ops:
        a1, a2, b1, b2 = (i1, i2, j1, j2) if from_spoken else (j1, j2, i1, i2)
        if not (a1 < hi and a2 > lo if a2 > a1 else lo < a1 < hi):
            continue
        if tag == "equal":
            c1, c2 = b1 + max(lo, a1) - a1, b1 + min(hi, a2) - a1
        else:
            c1, c2 = b1, b2
        found = (c1, c2) if found is None else (min(found[0], c1), max(found[1], c2))
    return found


def _expand(region: tuple[int, int, int, int], ops: list, spoken: list[Unit], heard: list[Unit]):
    """Grows a differing stretch until both sides cover whole tokens and align with each other."""
    i1, i2, j1, j2 = region
    while True:
        n1, n2 = _widen(*_inside(i1, i2, spoken), spoken)
        m1, m2 = _widen(*_inside(j1, j2, heard), heard)
        other = _aligned(m1, m2, ops, from_spoken=False)
        if other:
            n1, n2 = min(n1, other[0]), max(n2, other[1])
        other = _aligned(n1, n2, ops, from_spoken=True)
        if other:
            m1, m2 = min(m1, other[0]), max(m2, other[1])
        if (n1, n2, m1, m2) == (i1, i2, j1, j2):
            return region
        region = i1, i2, j1, j2 = n1, n2, m1, m2


def _token_span(side: list[Unit], lo: int, hi: int, tokens: int) -> tuple[int, int]:
    """The tokens of units [lo, hi) and of the fillers between them and the unit before."""
    end = max(u.last for u in side[lo:hi]) + 1 if hi > lo else (side[lo].first if lo < len(side) else tokens)
    start = side[lo - 1].last + 1 if lo > 0 else 0
    if hi > lo:
        start = min(start, min(u.first for u in side[lo:hi]))
    return start, max(start, end)


def _core(token: str) -> str:
    return re.sub(r"^\W+|\W+$", "", token)


def _letter_a(heard: str, spans: list[tuple[int, int]], side: Sequence[Unit]) -> bool:
    """Whether a heard "A" stands mid-sentence, where it reads as the letter ("Garden A type 3")."""
    for unit in side:
        start, end = spans[unit.first]
        if unit.word == "a" and _core(heard[start:end]) == "A" and heard[:start].rstrip()[-1:] not in ("", ".", "?", "!"):
            return True
    return False


def _spoken_text(spoken: str, spans: list[tuple[int, int]], first: int, last: int, name_map: Mapping[str, str]) -> str:
    text = " ".join(spoken[spans[first][0] : spans[last][1]].split())
    text = re.sub(r"^[\"'(\[]+|[,.?!;:\"')\]]+$", "", text)
    return re.sub(r"[A-Za-z][\w']*", lambda m: name_map.get(m.group(0).lower(), m.group(0)), text)


def restore(spoken: str, heard: str, reference: str, counts: Mapping[str, int] | None = None) -> Restored:
    """The heard text with the spoken words put back wherever the reference could not follow from it."""
    counts = train_row_counts() if counts is None else counts
    name_map = names(reference)
    spoken_spans, spoken_units = units(spoken)
    heard_spans, heard_units = units(heard)
    a, b = [u.word for u in spoken_units], [u.word for u in heard_units]
    ops = SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    regions = []
    for tag, i1, i2, j1, j2 in ops:
        if tag != "equal" and (
            _protected(a[i1:i2], b[j1:j2], name_map, counts) or _letter_a(heard, heard_spans, heard_units[j1:j2])
        ):
            regions.append(_expand((i1, i2, j1, j2), ops, spoken_units, heard_units))
    # Token spans, heard [h1, h2) and spoken [s1, s2), each taking the fillers and marks before it, so
    # "uh the" comes back for "other"; an empty heard span inserts before h1.
    spans = []
    for i1, i2, j1, j2 in regions:
        h = _token_span(heard_units, j1, j2, len(heard_spans))
        s = _token_span(spoken_units, i1, i2, len(spoken_spans))
        spans.append([*h, *(s if s[1] > s[0] else (None, None)), i2 - i1, j2 - j1])
    spans.sort(key=lambda r: (r[0], r[1]))
    merged: list[list] = []
    for span in spans:
        last = merged[-1] if merged else None
        if last and (span[0] < last[1] or (span[2] is not None and last[3] is not None and span[2] < last[3])):
            last[1] = max(last[1], span[1])
            if span[2] is not None:
                last[2] = span[2] if last[2] is None else min(last[2], span[2])
                last[3] = span[3] if last[3] is None else max(last[3], span[3])
            last[4] += span[4]
            last[5] += span[5]
        else:
            merged.append(span)
    # A name heard with other capitals takes the reference's ("Grainlab" as "GrainLab").
    restored_tokens = {token for h1, h2, *_ in merged for token in range(h1, h2)}
    for tag, i1, i2, j1, j2 in ops:
        for unit in heard_units[j1:j2] if tag == "equal" else ():
            core = _core(heard[slice(*heard_spans[unit.first])])
            name = name_map.get(unit.word)
            if unit.first == unit.last and unit.first not in restored_tokens and name and core != name and core.lower() == unit.word:
                merged.append([unit.first, unit.first + 1, None, None, 0, 0, name])
    merged.sort(key=lambda r: r[0])
    text, fixes = heard, []
    for h1, h2, s1, s2, _, _, *recased in reversed(merged):
        if recased:
            start, end = heard_spans[h1]
            old = text[start:end]
            token = old.replace(_core(old), recased[0], 1)
            text = text[:start] + token + text[end:]
            fixes.append((old, token))
            continue
        new = _spoken_text(spoken, spoken_spans, s1, s2 - 1, name_map) if s1 is not None else ""
        if h2 > h1:
            start, end = heard_spans[h1][0], heard_spans[h2 - 1][1]
            old = text[start:end]
            tail = _TRAILING.search(old)
            if new and text[:start].rstrip()[-1:] in ("", ".", "?", "!") and old[:1].isupper():
                new = new[:1].upper() + new[1:]
            if tail and (new or text[:start].rstrip()[-1:] not in ",.?!;:"):
                new += tail.group(0)
            if new[:1].isalnum() and text[start - 1 : start].isalnum():
                new = " " + new
            if new[-1:].isalnum() and text[end : end + 1].isalnum():
                new += " "
        else:
            start = end = heard_spans[h1][0] if h1 < len(heard_spans) else len(text)
            old = ""
            new = f"{new} " if h1 < len(heard_spans) else f" {new}"
        text = text[:start] + new + text[end:]
        fixes.append((old, new.strip()))
    if merged:
        text = re.sub(r" +([,.?!;:])", r"\1", re.sub(r"[ \t]{2,}", " ", text)).strip()
    restored_words = sum(r[4] for r in merged)
    longest = max((max(r[4], r[5]) for r in merged), default=0)
    failed, note = gates.code_check(text, reference)
    reason = ""
    if longest > MAX_RESTORED_RUN:
        reason = f"restored run of {longest} words"
    elif restored_words > max(MAX_RESTORED_RUN, MAX_RESTORED_SHARE * len(a)):
        reason = f"restored {restored_words} of {len(a)} words"
    elif failed:
        reason = f"code check: {note}"
    else:
        lost = _lost(spoken_units, units(text)[1], name_map, counts)
        if lost:
            reason = "lost after restore: " + ", ".join(lost[:5])
    return Restored(text, fixes[::-1], restored_words, longest, reason)


def _lost(spoken: list[Unit], restored: list[Unit], name_map: Mapping[str, str], counts: Mapping[str, int]) -> list[str]:
    """Protected spoken words missing from the restored text."""
    kept = Counter(u.word for u in restored)
    wanted = Counter(u.word for u in spoken if _fact(u.word, name_map, counts))
    return sorted(word for word, count in wanted.items() if kept[word] < count)


def recognizer_of(origin_id: str) -> str:
    return "parakeet" if hash_bucket(f"heard-recognizer-{origin_id}") < PARAKEET_PERCENT else "whisper"


def sample_origins(n: int, seed: int = SEED) -> list[Example]:
    """The first `n` v1 train rows in a seeded order, so a larger `n` extends a smaller one's sample."""
    rows = sorted(
        (
            e for e in read_examples(paths.BUILD / "train.jsonl")
            if e.output != EMPTY and not filler_only(e.input) and not _ODD_TEXT.search(e.input)
        ),
        key=lambda e: e.id,
    )
    random.Random(seed).shuffle(rows)
    return rows[:n]


def _record_files() -> list:
    return sorted(OUT_DIR.glob("heard.*.jsonl"))


def records() -> list[dict]:
    return [json.loads(line) for path in _record_files() for line in path.open(encoding="utf-8") if line.strip()]


def hear(origin: Example, speaker, recognizer, recognizer_name: str, seed: int = SEED) -> dict:
    """Speaks and transcribes one origin, trying new voices until the restore keeps it."""
    for attempt in range(1, ATTEMPTS + 1):
        spoken = speaker.speak(origin.input, random.Random(f"{seed}-{origin.id}-{attempt}"))
        heard, _ = recognizer.transcribe(spoken.samples)
        result = _checked(origin, restore(origin.input, heard, origin.output))
        if not result.reason or result.reason == "same as spoken":
            break
    return {
        "id": origin.id, "recognizer": recognizer_name, "tts": speaker.name, "asr": recognizer.name, "attempt": attempt,
        "voice": spoken.voice,
        "speed": spoken.speed,
        "seconds": round(len(spoken.samples) / 16_000, 1), "heard": heard, "input": result.text,
        "fixes": result.fixes, "kept": not result.reason, "reason": result.reason,
    }


def rerestore() -> Counter:
    """Restores every record's heard text again with the current rules, without speaking anything."""
    origins = {e.id: e for e in read_examples(paths.BUILD / "train.jsonl")}
    counts: Counter = Counter()
    for path in _record_files():
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        for record in rows:
            origin = origins[record["id"]]
            result = _checked(origin, restore(origin.input, record["heard"], origin.output))
            record |= {"input": result.text, "fixes": result.fixes, "kept": not result.reason, "reason": result.reason}
            counts["kept" if record["kept"] else result.reason.split(":")[0].split(" of ")[0]] += 1
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return counts


def _checked(origin: Example, result: Restored) -> Restored:
    if not result.reason and result.text.split() == origin.input.split():
        return Restored(result.text, result.fixes, result.restored_words, result.longest, "same as spoken")
    return result


def run(
    n: int, recognizer: str, seed: int = SEED, shard: int = 0, shards: int = 1, tts_provider: str = "cuda",
    asr_provider: str = "cpu", tts: str = "kokoro",
) -> Counter:
    """Hears this recognizer's and shard's share of the first `n` sampled origins; resumes past rows done."""
    from .asr_probe import Recognizer, speaker as make_speaker

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = {record["id"] for record in records()}
    mine = [e for e in sample_origins(n, seed) if recognizer_of(e.id) == recognizer]
    todo = [e for index, e in enumerate(mine) if index % shards == shard and e.id not in done]
    counts: Counter = Counter()
    if not todo:
        return counts
    speaker = make_speaker(tts, tts_provider)
    model = Recognizer(recognizer, asr_provider)
    with (OUT_DIR / f"heard.{recognizer}.{shard}.jsonl").open("a", encoding="utf-8") as sink:
        for index, origin in enumerate(todo, 1):
            record = hear(origin, speaker, model, recognizer, seed)
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            outcome = "kept" if record["kept"] else record["reason"]
            counts[outcome.split(":")[0].split(" of ")[0]] += 1
            print(f"{recognizer} {shard}: {index}/{len(todo)} {origin.id} try {record['attempt']}: {outcome}", flush=True)
    return counts


def heard_row(origin: Example, record: dict) -> Example:
    kept_tags = [tag for tag in origin.tags if not tag.startswith(_INPUT_TAGS)]
    return Example(
        id=example_id(SOURCE, record["input"]),
        input=record["input"],
        output=origin.output,
        source=SOURCE,
        category=origin.category,
        license=origin.license,
        tags=[
            TAG, "style_recognizer", f"recognizer:{record['recognizer']}", f"from:{origin.source}",
            f"heard_from:{origin.id}", V2_TAG, *kept_tags,
        ],
        seed_group=origin.seed_group,
        split="train",
        teacher_model=origin.teacher_model,
        teacher_reasoning=origin.teacher_reasoning,
        generator_provider=origin.generator_provider,
        generator_model=origin.generator_model,
        generator_reasoning=origin.generator_reasoning,
        judge_model=origin.judge_model,
        judge_reasoning=origin.judge_reasoning,
    )


def verdicts() -> dict[str, dict]:
    """Each row's latest meaning verdict, by origin id."""
    if not MEANING.exists():
        return {}
    return {v["id"]: v for v in map(json.loads, MEANING.open(encoding="utf-8"))}


def meaning_kept(record: dict, verdict: dict | None) -> bool | None:
    """Whether the meaning check passed this record's current input; None if it has not seen it."""
    if verdict is None or verdict["input"] != record["input"]:
        return None
    return verdict["p"] < MEANING_LIMIT


async def decide(client: httpx.AsyncClient, key: str, state: str, question: str) -> float | None:
    """Mercury Decide's probability that the answer to `question` about `state` is yes; None if it never answered."""
    body = {"model": MEANING_MODEL, "state": state, "questions": {"q": {"type": "noul", "instructions": question}}}
    for attempt in range(8):
        try:
            response = await client.post(MEANING_URL, json=body, headers={"Authorization": f"Bearer {key}"})
        except httpx.TransportError:
            pass
        else:
            if response.status_code == 200:
                return response.json()["answers"]["q"]["noul"]
            if response.status_code not in (408, 429, 500, 502, 503, 504):
                print(f"Mercury Decide: HTTP {response.status_code}: {response.text[:200]}", flush=True)
                return None
        await asyncio.sleep(min(2**attempt + random.random(), 60))
    return None


def openrouter_key() -> str:
    load_dotenv(paths.REPO_ROOT / ".env")
    if not (key := os.environ.get("OPENROUTER_API_KEY")):
        raise SystemExit("set OPENROUTER_API_KEY in the repo-root .env")
    return key


async def _ask_meaning(client: httpx.AsyncClient, key: str, spoken: str, heard: str) -> float | None:
    spec = json.loads(MEANING_PROMPT.read_text(encoding="utf-8"))
    return await decide(client, key, f"{spec['preface']}\n\nSPOKEN:\n{spoken}\n\nRECOGNIZED:\n{heard}", spec["question"])


async def check(concurrency: int = 8) -> Counter:
    """Asks the meaning check about every kept record whose current input it has not seen."""
    key = openrouter_key()
    seen = verdicts()
    todo = [r for r in records() if r["kept"] and meaning_kept(r, seen.get(r["id"])) is None]
    origins = {e.id: e for e in read_examples(paths.BUILD / "train.jsonl")} if todo else {}
    counts: Counter = Counter()
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=120) as client:

        async def one(record: dict) -> None:
            async with semaphore:
                p = await _ask_meaning(client, key, origins[record["id"]].input, record["input"])
            if p is None:
                counts["no answer"] += 1
                return
            with MEANING.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps({"id": record["id"], "input": record["input"], "p": p, "model": MEANING_MODEL}) + "\n")
            counts["flagged" if p >= MEANING_LIMIT else "passed"] += 1

        await asyncio.gather(*(one(r) for r in todo))
    return counts


def load(v1_dir=paths.BUILD) -> list[Example]:
    """The kept heard rows the meaning check passed, built against their v1 train origins."""
    seen = verdicts()
    kept = [record for record in records() if record["kept"] and meaning_kept(record, seen.get(record["id"]))]
    if not kept:
        return []
    origins = {e.id: e for e in read_examples(v1_dir / "train.jsonl")}
    return [heard_row(origins[record["id"]], record) for record in kept]


def origin_id(row: Example) -> str:
    return next(tag.removeprefix("heard_from:") for tag in row.tags if tag.startswith("heard_from:"))


def report(rows: Sequence[dict] | None = None) -> dict:
    rows = records() if rows is None else rows
    origins = {e.id: e for e in read_examples(paths.BUILD / "train.jsonl")}
    out: dict = {"rows": len(rows), "kept": sum(r["kept"] for r in rows)}
    out["outcomes"] = dict(Counter("kept" if r["kept"] else r["reason"].split(":")[0].split(" of ")[0] for r in rows))
    out["by_recognizer"] = {
        name: f"{sum(r['kept'] for r in group)}/{len(group)}"
        for name in RECOGNIZERS if (group := [r for r in rows if r["recognizer"] == name])
    }
    buckets: dict[str, list[dict]] = {}
    for r in rows:
        words = len(origins[r["id"]].input.split())
        buckets.setdefault("<=100" if words <= 100 else "100-200" if words <= 200 else ">200", []).append(r)
    out["by_length"] = {
        name: {"kept": f"{sum(r['kept'] for r in group)}/{len(group)}", "fixes_per_row": round(sum(len(r["fixes"]) for r in group) / len(group), 1)}
        for name, group in sorted(buckets.items())
    }
    out["attempts"] = dict(Counter(r["attempt"] for r in rows if r["kept"]))
    seen = verdicts()
    meaning = Counter(meaning_kept(r, seen.get(r["id"])) for r in rows if r["kept"])
    out["meaning_check"] = {"passed": meaning[True], "flagged": meaning[False], "not_checked": meaning[None]}
    out["audio_hours"] = round(sum(r["seconds"] for r in rows) / 3600, 1)
    return out


async def audit(llm, n: int, seed: int = SEED) -> dict:
    """The judge on `n` kept rows, heard input and origin input against the same reference.

    A row the judge passes with the origin's input but fails with the heard one is one the restore
    should have caught; the origin side measures the judge's own failure rate on these references.
    """
    from . import gates

    rows = sorted(load(), key=origin_id)
    picked = random.Random(seed).sample(rows, min(n, len(rows)))
    origins = {e.id: e for e in read_examples(paths.BUILD / "train.jsonl")}
    heard_verdicts = await gates.check_references(llm, [(e.id, e.input, e.output) for e in picked])
    origin_verdicts = await gates.check_references(
        llm, [(origin_id(e), origins[origin_id(e)].input, e.output) for e in picked]
    )
    lines = []
    for e in picked:
        heard_v, origin_v = heard_verdicts[e.id], origin_verdicts[origin_id(e)]
        lines.append(json.dumps({
            "id": e.id, "origin": origin_id(e), "heard_passed": heard_v.passed, "heard_note": heard_v.note,
            "origin_passed": origin_v.passed, "origin_note": origin_v.note,
        }, ensure_ascii=False))
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    results = [json.loads(line) for line in lines]
    return {
        "rows": len(results),
        "heard_failed": sum(not r["heard_passed"] for r in results),
        "origin_failed": sum(not r["origin_passed"] for r in results),
        "failed_only_heard": sum(r["origin_passed"] and not r["heard_passed"] for r in results),
    }
