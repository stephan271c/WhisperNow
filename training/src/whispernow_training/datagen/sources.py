"""Stage 2: adapters for existing datasets that fill aawaaz's gaps.

- handy: short dictations, no-edit rows, dictated questions and self-corrections.
  Its real-speech rows re-record Switchboard-1 utterances (LDC-licensed text) and
  are dropped; see `SwitchboardIndex`.
- sotto: only its self-correction, preserve-wording, adversarial and false-start
  rows. The card has no category column, so `sotto_category` infers it. Part of
  sotto was written by Grok, whose API terms forbid training on its output, so
  the default build leaves sotto out.
- disfl_qa: disfluent questions and their fluent rewrites; answers are dropped.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Iterable

from .schema import Example, example_id

HANDY = "MagicNoThief/handy-dictation-editing"
SOTTO = "juanquivilla/sotto-transcript-cleanup"
DISFL_QA = "google-research-datasets/disfl_qa"
LICENSES = {HANDY: "CC-BY-4.0", SOTTO: "MIT", DISFL_QA: "CC-BY-4.0"}
SWITCHBOARD_DERIVED = ("amaai-lab/DisfluencySpeech", "nyralabs/disfluency_speech_english")

FILLERS = {"um", "uh", "uhm", "er", "ah", "erm", "hmm", "mm"}
CORRECTION = re.compile(
    r"\b(no wait|wait no|i mean|actually|scratch that|make that|sorry|or rather|no no|rather|correction|let me rephrase)\b"
)
_WORD = re.compile(r"[a-z0-9']+")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def skeleton(text: str) -> str:
    return " ".join(words(text))


_UK_TO_US = {
    "colour": "color", "colours": "colors", "coloured": "colored", "favour": "favor", "favourite": "favorite",
    "favourites": "favorites", "honour": "honor", "labour": "labor", "neighbour": "neighbor",
    "neighbours": "neighbors", "behaviour": "behavior", "humour": "humor", "rumour": "rumor", "flavour": "flavor",
    "flavours": "flavors", "harbour": "harbor", "centre": "center", "centres": "centers", "theatre": "theater",
    "metre": "meter", "metres": "meters", "litre": "liter", "litres": "liters", "fibre": "fiber",
    "organise": "organize", "organised": "organized", "organising": "organizing", "organisation": "organization",
    "organisations": "organizations", "realise": "realize", "realised": "realized", "realising": "realizing",
    "recognise": "recognize", "recognised": "recognized", "apologise": "apologize", "apologised": "apologized",
    "prioritise": "prioritize", "prioritised": "prioritized", "summarise": "summarize", "summarised": "summarized",
    "finalise": "finalize", "finalised": "finalized", "optimise": "optimize", "optimised": "optimized",
    "optimisation": "optimization", "customise": "customize", "customised": "customized", "authorise": "authorize",
    "authorised": "authorized", "authorisation": "authorization", "standardise": "standardize",
    "minimise": "minimize", "maximise": "maximize", "emphasise": "emphasize", "criticise": "criticize",
    "analyse": "analyze", "analysed": "analyzed", "analysing": "analyzing", "catalogue": "catalog",
    "dialogue": "dialog", "programme": "program", "programmes": "programs", "cheque": "check", "cheques": "checks",
    "grey": "gray", "defence": "defense", "offence": "offense", "licence": "license", "practise": "practice",
    "travelled": "traveled", "travelling": "traveling", "traveller": "traveler", "cancelled": "canceled",
    "cancelling": "canceling", "modelling": "modeling", "labelled": "labeled", "labelling": "labeling",
    "fulfil": "fulfill", "enrol": "enroll", "jewellery": "jewelry", "aluminium": "aluminum", "tyre": "tire",
    "tyres": "tires", "kerb": "curb", "mum": "mom", "mums": "moms", "aeroplane": "airplane", "whilst": "while",
    "cosy": "cozy", "plough": "plow", "sceptical": "skeptical", "manoeuvre": "maneuver", "paediatric": "pediatric",
    "anaesthetic": "anesthetic", "oestrogen": "estrogen", "haemoglobin": "hemoglobin", "diarrhoea": "diarrhea",
    "judgement": "judgment", "ageing": "aging", "storey": "story", "pyjamas": "pajamas", "mould": "mold",
    "moustache": "mustache", "sulphur": "sulfur", "towards": "toward",
}
_UK_WORD = re.compile(r"\b(" + "|".join(sorted(_UK_TO_US, key=len, reverse=True)) + r")\b", re.IGNORECASE)


def americanize(text: str) -> str:
    """Rewrites common British spellings, keeping each word's capitalization."""

    def swap(match: re.Match[str]) -> str:
        word = match.group(0)
        us = _UK_TO_US[word.lower()]
        if word.isupper():
            return us.upper()
        return us.capitalize() if word[0].isupper() else us

    return _UK_WORD.sub(swap, text)


_ORDINALS = {
    "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17,
    "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "twenty-first": 21, "twenty-second": 22,
    "twenty-third": 23, "twenty-fourth": 24, "twenty-fifth": 25, "twenty-sixth": 26, "twenty-seventh": 27,
    "twenty-eighth": 28, "twenty-ninth": 29, "thirtieth": 30, "thirty-first": 31,
}
_ORDINAL = re.compile(r"\b(the|on) (" + "|".join(sorted(_ORDINALS, key=len, reverse=True)) + r")\b", re.IGNORECASE)
_WH_QUESTION = re.compile(r"(what|when|where|who|whom|whose|why|which|how)\b", re.IGNORECASE)


def ordinal_suffix(n: int) -> str:
    return "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def ordinals_to_digits(text: str) -> str:
    """Writes day ordinals as digits, the way the target writes dates: "on the fifteenth" becomes "on the 15th"."""

    def swap(match: re.Match[str]) -> str:
        n = _ORDINALS[match.group(2).lower()]
        return f"{match.group(1)} {n}{ordinal_suffix(n)}"

    return _ORDINAL.sub(swap, text)


_MAY = re.compile(r"\b([a-z]+)(,? )May ([a-z]+)")
_BEFORE_MONTH = set("in of for on by since until till early late mid last next this every through from during".split())
_AFTER_MONTH = set("and or once in if to unless so before because when while until the of for at on is was will through".split())


def modal_may(text: str) -> str:
    """handy capitalizes the verb "may" as the month mid-sentence: "factors May contribute"."""

    def swap(match: re.Match[str]) -> str:
        before, gap, after = match.groups()
        if before in _BEFORE_MONTH or after in _AFTER_MONTH:
            return match.group(0)
        return f"{before}{gap}may {after}"

    return _MAY.sub(swap, text)


def question_mark(text: str) -> str:
    if _WH_QUESTION.match(text) and text.endswith(".") and text.count(".") == 1:
        return text[:-1] + "?"
    return text


_NGRAM = 6


def ngrams(text: str, n: int = _NGRAM) -> set[str]:
    tokens = [w for w in words(text) if w not in FILLERS]
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def switchboard_texts() -> list[str]:
    """Every DisfluencySpeech transcript, reading only the transcript columns and never the audio."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    texts = []
    for repo in SWITCHBOARD_DERIVED:
        for path in fs.glob(f"datasets/{repo}/data/*.parquet"):
            parquet = pq.ParquetFile(fs.open(path))
            columns = [c for c in parquet.schema_arrow.names if "transcript" in c]
            for row in parquet.read(columns=columns).to_pylist():
                texts += [text for text in row.values() if text]
    return texts


class SwitchboardIndex:
    """Word n-grams and whole lines of every DisfluencySpeech transcript."""

    def __init__(self, texts: Iterable[str]) -> None:
        self.grams: set[str] = set()
        self.lines: set[str] = set()
        for text in texts:
            self.grams |= ngrams(text)
            self.lines.add(" ".join(w for w in words(text) if w not in FILLERS))

    def contains(self, example_input: str, example_output: str) -> bool:
        for text in (example_input, example_output):
            if " ".join(w for w in words(text) if w not in FILLERS) in self.lines:
                return True
        row_grams = ngrams(example_input) | ngrams(example_output)
        return bool(row_grams) and len(row_grams & self.grams) >= max(1, len(row_grams) // 4)


def handy_tags(source_input: str, output: str) -> list[str]:
    tags = []
    if skeleton(source_input) == skeleton(output):
        tags.append("no_edit")
    if output.rstrip().endswith("?"):
        tags.append("question")
    if CORRECTION.search(source_input.lower()):
        tags.append("self_correction")
    if len(words(source_input)) <= 12:
        tags.append("short")
    elif len(words(source_input)) >= 40:
        tags.append("long")
    return tags


def handy_category(tags: list[str]) -> str:
    for tag, category in (
        ("question", "dictated_question"),
        ("no_edit", "no_edit"),
        ("self_correction", "self_correction"),
    ):
        if tag in tags:
            return category
    return "short_dictation" if "short" in tags else "dictation"


_LOOKALIKES = ("like", "so", "well", "right", "actually", "basically", "literally", "you know", "i mean", "okay")


def sotto_category(source_input: str, output: str) -> str | None:
    """Infers the sotto categories we keep; None for the rest."""
    lowered = source_input.lower()
    source_words, output_words = words(source_input), words(output)
    if skeleton(source_input) == skeleton(output):
        # A clean input the model must not edit. It is adversarial when it holds a filler look-alike.
        kept = [w for w in _LOOKALIKES if re.search(rf"\b{w}\b", lowered)]
        return "adversarial" if kept else "preserve_wording"
    if any(w in FILLERS for w in source_words):
        return None
    if CORRECTION.search(lowered):
        return "self_correction"
    if len(output_words) < len(source_words) and source_words[-len(output_words) :] == output_words:
        return "false_start"
    return None


def _handy_example(row: dict, split: str) -> Example:
    source_input, raw_output = row["input"].strip(), row["output"].strip()
    output = americanize(raw_output)
    tags = handy_tags(source_input, output)
    if americanize(source_input) != source_input or output != raw_output:
        tags.append("americanized")
    tidied = question_mark(ordinals_to_digits(output))
    if tidied != output:
        tags.append("normalized")
        output = tidied
    return Example(
        id=example_id(HANDY, source_input),
        input=americanize(source_input),
        output=output,
        source=HANDY,
        category=handy_category(tags),
        license=LICENSES[HANDY],
        tags=tags,
        split=split,
    )


def load_handy(switchboard: SwitchboardIndex) -> tuple[list[Example], Counter]:
    from datasets import load_dataset

    data = load_dataset(HANDY, "alpaca")
    dropped: Counter = Counter()
    examples = []
    for split_name, split in (("train", ""), ("test", "test")):
        for row in data[split_name]:
            if switchboard.contains(row["input"], row["output"]):
                dropped["Switchboard-derived text"] += 1
                continue
            examples.append(_handy_example(row, split))
    return examples, dropped


def load_sotto() -> tuple[list[Example], Counter]:
    from datasets import load_dataset

    data = load_dataset(SOTTO)
    dropped: Counter = Counter()
    examples = []
    for split_name in ("train", "validation"):
        for row in data[split_name]:
            source_input, output = row["input"].strip(), row["output"].strip()
            category = sotto_category(source_input, output)
            if category is None:
                dropped["other category"] += 1
                continue
            examples.append(
                Example(
                    id=example_id(SOTTO, source_input),
                    input=source_input,
                    output=output,
                    source=SOTTO,
                    category=category,
                    license=LICENSES[SOTTO],
                    tags=[category],
                    split="test" if split_name == "validation" else "",
                )
            )
    return examples, dropped


def tidy_question(text: str) -> str:
    """SQuAD's questions sometimes lack the question mark or the capital; an empty result means drop."""
    text = " ".join(text.split())
    if not text or "\\" in text:
        return ""
    text = text[0].upper() + text[1:]
    if text.endswith("."):
        text = text[:-1]
    return text if text.endswith("?") else text + "?"


def spliced(source_input: str) -> bool:
    """handy pastes written Wikipedia sentences into lowercase dictation; a recognizer never writes that."""
    if not source_input[:1].islower():
        return False
    return bool(re.search(r"\. [A-Z]", source_input) or _WRITTEN_SENTENCE_START.search(source_input))


_WRITTEN_SENTENCE_START = re.compile(
    r"\b(Thus|The|In|However|This|These|It|He|She|They|A|An|As|After|Before|During|Some|Many|Most)\b"
)


def reworded(source_input: str, output: str) -> bool:
    return not set(words(output)) <= set(words(source_input))


def load_disfl_qa() -> list[Example]:
    from datasets import load_dataset

    data = load_dataset(DISFL_QA)
    examples = []
    for split_name, split in (("train", ""), ("validation", ""), ("test", "test")):
        for row in data[split_name]:
            source_input = " ".join(row["disfluent question"].split())
            output = tidy_question(row["original question"])
            if not source_input or not output:
                continue
            examples.append(
                Example(
                    id=example_id(DISFL_QA, source_input),
                    input=source_input,
                    output=output,
                    source=DISFL_QA,
                    category="dictated_question",
                    license=LICENSES[DISFL_QA],
                    tags=["question", "self_correction"],
                    split=split,
                )
            )
    return examples


def sample_by_category(
    examples: Iterable[Example], caps: dict[str, int], seed: int, test_share: float = 0.1
) -> list[Example]:
    """Keeps up to `caps[category]` rows per category, plus a tenth as many of the source's own test rows."""
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[Example]] = {}
    for example in examples:
        if example.category in caps:
            groups.setdefault((example.category, example.split), []).append(example)
    kept = []
    for (category, split), rows in sorted(groups.items()):
        rows.sort(key=lambda e: e.id)
        rng.shuffle(rows)
        cap = caps[category] if split != "test" else max(1, round(caps[category] * test_share))
        kept += rows[:cap]
    return kept
