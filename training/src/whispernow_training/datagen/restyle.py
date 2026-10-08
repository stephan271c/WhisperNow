"""Step 7: give a share of the aawaaz inputs the punctuated style of the app's recognizers.

aawaaz inputs are lowercase and unpunctuated, but Parakeet, Whisper and Qwen3-ASR all write
cased, punctuated text, each in its own way (see `asr_probe` and the README). The teacher rewrites a
hashed half of the aawaaz inputs in that style without changing a spoken word; `same_words`
checks each rewrite in code, and a rewrite that fails keeps the original input. The references
are untouched, since the words they were written from are the same.

Rewrites go to `data/restyle/aawaaz.jsonl` (id, input), and `build` swaps them in through
`recognizer_quirks` in the row's `asr_style` (Parakeet, Whisper, Qwen3 or plain), tagging the
row `style_recognizer`, `restyled` and `asr_<style>`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
from collections.abc import Sequence
from typing import NamedTuple

from . import paths, stage1
from .asr_probe import join_segment_texts
from .llm import LlmClient, LlmError, parse_json_reply
from .schema import Example, read_examples

OUT = paths.DATA / "restyle/aawaaz.jsonl"
REJECTED = paths.DATA / "restyle/rejected.jsonl"
PROMPT = paths.PROMPTS / "restyle/recognizer.md"
SHARE_PERCENT = 50
BATCH_CHARS = 5000
MAX_BATCH = 8

_UNITS = (
    "zero oh one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen"
).split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
_SCALES = {"hundred", "thousand", "million", "billion", "trillion"}
_FRACTIONS = {"half", "halves", "quarter", "quarters"}
_ORDINALS = set(
    (
        "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth "
        "fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth thirtieth fortieth "
        "fiftieth sixtieth seventieth eightieth ninetieth hundredth thousandth"
    ).split()
)
_DECADES = {tens[:-1] + "ies" for tens in _TENS}  # "seventies", written "70s"
_NUMBER_WORDS = set(_UNITS) | set(_TENS) | _SCALES | _FRACTIONS | _ORDINALS | _DECADES
# Words a recognizer folds into a written number: "twenty percent" as "20%", "point five" as "0.5".
_NUMBER_JOINERS = {"point", "and", "a"}
_UNIT_JOINERS = {"percent", "dollars", "dollar", "cents", "o'clock"}
NUMBER = "#"


def chosen(example: Example) -> bool:
    """A fixed hashed share of the rows, independent of order and of the other rows."""
    digest = hashlib.sha256(f"restyle-{example.id}".encode()).digest()
    return digest[0] * 100 // 256 < SHARE_PERCENT


class Token(NamedTuple):
    word: str
    start: int
    end: int
    inside: bool = False  # part of a token like "Q3" that mixes letters and digits


def _tokens(text: str) -> list[Token]:
    """Lowercase words with their character spans; apostrophes and dots inside a word are ignored
    ("its" and "it's", "nextjs" and "next.js"), and "&" is "and" ("M&A").

    Letters and digits in one token are split ("Q3" as "q", "3"; "50k" as "50", "k"), after
    dropping an ordinal or plural ending on the digits ("16th", "70s").
    """
    out = []
    for match in re.finditer(r"[A-Za-z0-9]+(?:['’.:,/][A-Za-z0-9]+)*['’]?|[$%½¼¾&]", text):
        token = match.group().casefold().replace("’", "'")
        token = "and" if token == "&" else re.sub(r"(?<=[a-z])[.'](?=[a-z])|'", "", token).rstrip(".")
        if not token:
            continue
        if re.search(r"\d", token) and re.search(r"[a-z]", token):
            token = re.sub(r"(?<=\d)(st|nd|rd|th|s)$", "", token)
        parts = re.findall(r"\d[\d.:,/]*|[a-z]+", token) if re.search(r"\d", token) else [token]
        out += [Token(part, match.start(), match.end(), len(parts) > 1) for part in parts]
    return out


def _is_number(token: str) -> bool:
    return any(c.isdigit() for c in token) or token in _NUMBER_WORDS or token in {"$", "%", "½", "¼", "¾"}


def _number_run_end(tokens: list[str], index: int) -> int:
    """One past the end of the written number that starts at `index`."""
    index += 1
    while index < len(tokens):
        token = tokens[index]
        after = tokens[index + 1 : index + 3]
        continues = bool(after) and (
            _is_number(after[0]) or (len(after) == 2 and after[0] == "a" and after[1] in _FRACTIONS)
        )
        if _is_number(token):
            index += 1
        elif token in _UNIT_JOINERS or (token in _NUMBER_JOINERS and continues):
            index += 1
        else:
            break
    return index


def _words(text: str) -> list[Token]:
    """The spoken words, each number run as one `#` token spanning the whole run."""
    tokens = _tokens(text)
    plain = [t.word for t in tokens]
    words: list[Token] = []
    index = 0
    while index < len(tokens):
        start = index
        if plain[index] == "a" and index + 1 < len(tokens) and plain[index + 1] in _SCALES | _FRACTIONS:
            index += 1  # "a hundred" is written "100", "a half" "½"
        starts_decimal = plain[index] == "point" and index + 1 < len(tokens) and _is_number(plain[index + 1])
        if _is_number(plain[index]) or starts_decimal:
            index = _number_run_end(plain, index)
            run = tokens[start:index]
            inside = any(t.inside for t in run)
            if words and words[-1].word == NUMBER:  # "June 16th, 2025": one run
                previous = words.pop()
                words.append(Token(NUMBER, previous.start, run[-1].end, previous.inside or inside))
            else:
                words.append(Token(NUMBER, run[0].start, run[-1].end, inside))
            continue
        words.append(tokens[start])
        index = start + 1
    return words


def normalize_words(text: str) -> list[str]:
    return [t.word for t in _words(text)]


def same_words(spoken: str, restyled: str) -> bool:
    """The same words in the same order, ignoring where the spaces fall ("a v" and "AV", "not equal" and "notEqual")."""
    return "".join(normalize_words(spoken)) == "".join(normalize_words(restyled))


def first_difference(spoken: str, restyled: str) -> str:
    a, b = normalize_words(spoken), normalize_words(restyled)
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"word {index}: {' '.join(a[max(0, index - 3) : index + 3])!r} vs {' '.join(b[max(0, index - 3) : index + 3])!r}"
    return f"lengths {len(a)} vs {len(b)}"


def restyle_messages(texts: Sequence[str]) -> list[dict[str, str]]:
    items = json.dumps({"items": [{"n": i, "text": t} for i, t in enumerate(texts)]}, ensure_ascii=False)
    return [
        {"role": "system", "content": PROMPT.read_text(encoding="utf-8")},
        {"role": "user", "content": items},
    ]


def _batches(rows: Sequence[Example]) -> list[list[Example]]:
    batches: list[list[Example]] = []
    current: list[Example] = []
    size = 0
    for row in rows:
        if current and (size + len(row.input) > BATCH_CHARS or len(current) >= MAX_BATCH):
            batches.append(current)
            current, size = [], 0
        current.append(row)
        size += len(row.input)
    if current:
        batches.append(current)
    return batches


def _check(example: Example, text: str) -> str:
    """Why `text` cannot replace the input, or "" when it can."""
    if not text:
        return "missing"
    if not same_words(example.input, text):
        return first_difference(example.input, text)
    return ""


async def _restyle_batch(llm: LlmClient, batch: Sequence[Example]) -> list[tuple[Example, str, str]]:
    """(row, rewrite, why it failed or "")."""
    try:
        reply = await llm.complete(restyle_messages([e.input for e in batch]), role="teacher")
        parsed = parse_json_reply(reply)
        items = parsed["items"] if isinstance(parsed, dict) else parsed
        texts = {int(item["n"]): str(item["text"]) for item in items if isinstance(item, dict)}
    except (LlmError, ValueError, KeyError, TypeError) as error:
        return [(e, "", f"reply: {error}") for e in batch]
    results = []
    for index, example in enumerate(batch):
        text = " ".join(texts.get(index, "").split())
        results.append((example, text, _check(example, text)))
    return results


def _write(good, bad, example: Example, text: str, why: str) -> None:
    if why:
        bad.write(json.dumps({"id": example.id, "why": why, "text": text}, ensure_ascii=False) + "\n")
    else:
        good.write(json.dumps({"id": example.id, "input": text}, ensure_ascii=False) + "\n")


async def restyle_all(llm: LlmClient, limit: int = 0) -> dict:
    """Restyles the chosen aawaaz rows not yet done; resumes from `OUT` and `REJECTED`."""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    for path in (OUT, REJECTED):
        if path.exists():
            done |= {json.loads(line)["id"] for line in path.open(encoding="utf-8")}
    rows = [e for e in read_examples(stage1.CLEAN) if chosen(e) and e.id not in done]
    rows.sort(key=lambda e: e.id)
    if limit:
        rows = rows[:limit]
    batches = _batches(rows)
    passed = failed = 0
    with OUT.open("a", encoding="utf-8") as good, REJECTED.open("a", encoding="utf-8") as bad:
        for start in range(0, len(batches), 32):
            chunk = batches[start : start + 32]
            for results in await asyncio.gather(*(_restyle_batch(llm, b) for b in chunk)):
                for example, text, why in results:
                    _write(good, bad, example, text, why)
                    passed += not why
                    failed += bool(why)
            good.flush()
            bad.flush()
            print(f"batches {min(start + 32, len(batches))}/{len(batches)}: {passed} restyled, {failed} kept as spoken", flush=True)
    return {"restyled": passed, "failed": failed, "remaining_batches": 0}


def recheck() -> dict:
    """Checks the rejected rewrites again, after a change to `same_words`, and keeps the ones that now pass."""
    if not REJECTED.exists():
        return {"restyled": 0, "failed": 0}
    rows = {e.id: e for e in read_examples(stage1.CLEAN)}
    rejected = [json.loads(line) for line in REJECTED.open(encoding="utf-8")]
    passed = failed = 0
    with OUT.open("a", encoding="utf-8") as good, REJECTED.with_suffix(".tmp").open("w", encoding="utf-8") as bad:
        for row in rejected:
            example, text = rows[row["id"]], row.get("text", "")
            why = _check(example, text) if text else row["why"]
            _write(good, bad, example, text, why)
            passed += not why
            failed += bool(why)
    REJECTED.with_suffix(".tmp").replace(REJECTED)
    return {"restyled": passed, "failed": failed}


def load() -> dict[str, str]:
    if not OUT.exists():
        return {}
    return {row["id"]: row["input"] for row in map(json.loads, OUT.open(encoding="utf-8"))}


class AsrStyle(NamedTuple):
    """How one recognizer formats a transcript, measured by `asr_probe` on the same 300 recordings
    (see `data/probe/report.<model>.json`). Chances apply per comma, filler, number or segment."""

    share: int  # percent of the recognizer-style rows
    comma_keep: float  # the teacher writes about 14 commas per 100 words
    comma_to_period: float  # a comma before a new clause becomes a sentence end
    filler_comma_before: float  # a comma before a mid-sentence "um" or "uh" is kept
    filler_comma_after: float  # a mid-sentence "um" or "uh" is followed by a comma, whether or not it had one
    spoken_numbers: bool  # numbers written as the words said ("thirty-one percent")
    number_hyphen: float  # "thirty-one" rather than "thirty one"
    glue_dollar: bool  # "that's$4,200"
    command_sentence: float  # a spoken "period" is its own sentence ("hear. Period. Our")
    segment_period: float  # a segment ends with a period
    segment_lower_start: float  # a segment starts in lower case
    segment_raw: float  # a whole segment comes back lowercase and unpunctuated
    segments: bool = True


# Parakeet TDT 0.6B v2: 5.0 periods and 6.1 commas per 100 words, bare fillers, 41% of numbers as
# digits (the teacher's mix), every "$" glued. Whisper distil-large-v3.5: 4.7 and 5.7, 63% digits,
# a comma after 32% of fillers, and 17% of segments all lowercase without punctuation. Qwen3-ASR
# 0.6B: 8.4 and 7.4 (sentences of 10 words at the median), 96% of numbers as words, a comma after
# 62% of fillers, "Period." as a sentence 37% of the time, and every segment capitalized and
# closed. `plain` is the teacher's rewrite untouched, for recognizers that write clean text.
ASR_STYLES = {
    "parakeet": AsrStyle(30, 0.40, 0.0, 0.10, 0.15, False, 0.0, True, 0.0, 0.81, 0.24, 0.0),
    "whisper": AsrStyle(30, 0.45, 0.0, 0.15, 0.30, False, 0.0, False, 0.0, 0.70, 0.38, 0.17),
    "qwen3": AsrStyle(20, 0.55, 0.65, 0.10, 0.60, True, 0.58, False, 0.37, 0.96, 0.0, 0.0),
    "plain": AsrStyle(20, 1.0, 0.0, 1.0, 0.0, False, 0.0, False, 0.0, 0.0, 0.0, 0.0, segments=False),
}
# The app decodes a long dictation in segments of about 75 words.
SEGMENT_WORDS = (40, 110)
_CLAUSE_STARTS = {"and", "so", "but", "then", "because", "i", "we", "you", "it", "they", "he", "she", "that's", "which"}


def asr_style(key: str) -> str:
    """The recognizer style for the row with id `key`, in the `ASR_STYLES` shares."""
    bucket = hashlib.sha256(f"asr-{key}".encode()).digest()[0] * 100 // 256
    for name, style in ASR_STYLES.items():
        if bucket < style.share:
            return name
        bucket -= style.share
    return name


def _capitalize(word: str) -> str:
    return word[:1].upper() + word[1:]


def spoken_numbers(text: str, spoken: str, rng: random.Random, hyphen: float = 0.0) -> str:
    """`text` with each written number replaced by the words `spoken` has for it ("31%" as
    "thirty-one percent"). Unchanged when the numbers do not line up one to one."""
    written = [t for t in _words(text) if t.word == NUMBER]
    said = [t for t in _words(spoken) if t.word == NUMBER]
    if len(written) != len(said) or any(t.inside for t in written):
        return text
    for number, words in reversed(list(zip(written, said))):
        phrase = spoken[words.start : words.end].casefold()
        phrase = re.sub(
            r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety) (one|two|three|four|five|six|seven|eight|nine)\b",
            lambda m: f"{m.group(1)}-{m.group(2)}" if rng.random() < hyphen else m.group(),
            phrase,
        )
        if not text[: number.start].rstrip() or text[: number.start].rstrip()[-1] in ".?!":
            phrase = _capitalize(phrase)
        text = text[: number.start] + phrase + text[number.end :]
    return text


def _fillers(text: str, rng: random.Random, before: float, after: float) -> str:
    """Commas around mid-sentence "um" and "uh"; a sentence-initial "Um," is left."""

    def comma(match: re.Match) -> str:
        keep_before = "," if match.group(1) and rng.random() < before else ""
        mark = match.group(3) if match.group(3) not in ("", ",") else ("," if rng.random() < after else "")
        return f"{keep_before} {match.group(2)}{mark}"

    return re.sub(r"(,?) (um|uh)\b([,.?!]?)", comma, text)


def filler_commas(text: str, key: str) -> str:
    """Parakeet's filler commas for a text written with commas around every filler, seeded by `key`."""
    style = ASR_STYLES["parakeet"]
    return _fillers(text, random.Random(f"fillers-{key}"), style.filler_comma_before, style.filler_comma_after)


def recognizer_quirks(text: str, key: str, style_name: str, spoken: str | None = None) -> str:
    """`text` as the app would hand it over from a recognizer in `style_name`.

    Seeded by `key`, so a row always gets the same changes. Only punctuation, casing, spacing and
    the written form of numbers change, so `same_words` still holds. `spoken` is the transcript
    the rewrite came from, for styles that write numbers as words.
    """
    style = ASR_STYLES[style_name]
    rng = random.Random(f"quirks-{key}")
    if style.spoken_numbers and spoken is not None:
        text = spoken_numbers(text, spoken, rng, style.number_hyphen)
    if style.glue_dollar:
        text = re.sub(r"(?<=\w) \$(?=\d)", "$", text)

    def comma(match: re.Match) -> str:
        if match.group(1).casefold() in _CLAUSE_STARTS and rng.random() < style.comma_to_period:
            return ". " + _capitalize(match.group(1))
        return (", " if rng.random() < style.comma_keep else " ") + match.group(1)

    text = re.sub(r", (\S+)", comma, text)
    if style.filler_comma_before < 1 or style.filler_comma_after:
        text = _fillers(text, rng, style.filler_comma_before, style.filler_comma_after)
    if style.command_sentence:
        text = re.sub(
            r"(?<=\w) period\b[.,]?( ?)(\S?)",
            lambda m: f". Period.{m.group(1)}{m.group(2).upper()}" if rng.random() < style.command_sentence else m.group(),
            text,
        )
    if not style.segments:
        return text
    words = text.split()
    segments: list[list[str]] = []
    while len(words) > SEGMENT_WORDS[1]:
        take = rng.randint(*SEGMENT_WORDS)
        segments.append(words[:take])
        words = words[take:]
    segments.append(words)
    texts = []
    for index, segment in enumerate(segments):
        last = index == len(segments) - 1
        if not last and rng.random() < style.segment_period and not segment[-1].endswith((".", "?", "!")):
            segment[-1] = segment[-1].rstrip(",;:") + "."
        # A segment that starts mid-sentence is usually capitalized; a word the rewrite already
        # capitalized (a name, "I") stays so.
        if index and segment[0][:1].islower() and rng.random() >= style.segment_lower_start:
            segment[0] = _capitalize(segment[0])
        texts.append(" ".join(segment))
        if rng.random() < style.segment_raw:
            texts[-1] = re.sub(r"[.,?!;:](?=\s|$)", "", texts[-1]).lower()
    return join_segment_texts(texts)
