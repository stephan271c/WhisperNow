"""Steps 8 and 10: gather every stage's rows, filter, split by seed group, and report.

Writes `data/build/{train,val,test}.jsonl`, `dropped.jsonl` with a reason per
row, `stats.json` and `stats.md`.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ..text_rules import (
    CASE_SENSITIVE, CODE_TOKEN, CORRECTION, FILLER, FILLER_TOKEN, TRAILING_PUNCTUATION,
    assistant_opener, filler_only, has_markdown, malformed_list, protected_tokens,
)
from . import aawaaz, compose, generate, inject, paths, relabel, restyle, sources, stage1, stage2
from .schema import EMPTY, LENGTH_BUCKETS, Example, hash_bucket, length_bucket, read_examples, write_examples

SEED = 108
INJECTED = 2000
# Every long one- or multi-paragraph clean text not already injected gets one injected spoken
# version too: aawaaz references and the composed rows' references.
LONG_INJECTION_WORDS = 150
LONG_INJECTION_SEED = 110
# Shares of rows whose spoken input gets more hesitation (`inject.densify`), by kind of row. The
# generator writes about 0.8 fillers per 100 words in long rows and the short sources 1.6 to 1.7;
# real dictation measured 3.6. The rest keep their rate, as fluent dictation does.
DENSIFY_PERCENT = {"composed": 50, "short": 30}
# Short sources whose train and val inputs are densified. Their test rows (`TEST_IDS_109`) stay
# as built; aawaaz (4.2 per 100 words) and the injected rows already hesitate enough.
SHORT_DENSIFY_SOURCES = {generate.SOURCE, sources.HANDY, sources.DISFL_QA}
EMPTY_ROWS = 150
# More EMPTY rows pinned to test, from a second seed; a repeat of a training input stays in test.
EMPTY_TEST_ROWS = 90
VAL_PERCENT = 5
TEST_PERCENT = 5
NEAR_DUPLICATE_JACCARD = 0.85
MAX_SAME_OUTPUT = 3
# Share of the teacher-generated recognizer-style train and val rows whose filler commas are
# stripped, towards Parakeet's bare fillers; the rest keep the comma-wrapped form other recognizers
# and pasted text produce.
BARE_FILLER_PERCENT = 67
# Word counts the stats report the share of rows over.
LENGTH_MARKS = (150, 300, 600)

# Earlier sources win when two rows are duplicates.
SOURCE_ORDER = (aawaaz.REPO, sources.HANDY, sources.DISFL_QA, sources.SOTTO, generate.SOURCE, inject.SOURCE)
# Sources with their own held-out test rows; other groups get a hashed test share.
PRESET_TEST_SOURCES = {aawaaz.REPO, sources.HANDY, sources.DISFL_QA, sources.SOTTO}


def _source_rank(example: Example) -> tuple[int, int]:
    rank = SOURCE_ORDER.index(example.source) if example.source in SOURCE_ORDER else len(SOURCE_ORDER)
    return (0 if example.split == "test" else 1, rank)


def gather_with_drops(with_sotto: bool = False) -> tuple[list[Example], list[tuple[Example, str]], dict[str, dict] | None]:
    """Every stage's rows, plus the programmatic rows made here from the cleaned aawaaz rows and the
    composed rows. The benchmarks in `data/bench/` are never read.

    Once `datagen relabel build` has run, every row but the injected ones takes its reference from
    the relabel (`relabel.apply`), and rows it dropped or never saw are dropped here; the injected
    rows are then made from the relabelled parents.
    """
    records = relabel.snapshot("build")
    dropped: list[tuple[Example, str]] = []

    def labelled(examples: list[Example]) -> list[Example]:
        if records is None:
            return examples
        kept, lost = relabel.apply(examples, records)
        dropped.extend(lost)
        return kept

    clean = labelled(read_examples(stage1.CLEAN))
    rows = list(clean)
    for source, path in stage2.OUT_FILES.items():
        if path.exists() and (with_sotto or source != sources.SOTTO):
            rows += labelled(read_examples(path))
    for category in generate.TARGETS:
        for held_out in (False, True):
            if generate.output_file(category, held_out).exists():
                rows += labelled(read_examples(generate.output_file(category, held_out)))
    composed = labelled(compose.load(compose.TRAIN))
    rows += composed
    injected = inject.injected_rows(clean, INJECTED, SEED)
    used = {e.output for e in injected}
    long_origins = [
        e for e in [*clean, *composed] if len(e.output.split()) >= LONG_INJECTION_WORDS and e.output not in used
    ]
    long_injected = inject.injected_rows(
        long_origins, len(long_origins), LONG_INJECTION_SEED, layouts=("plain", "paragraphs")
    )
    rows += injected + [replace(e, tags=[*e.tags, "long_origin"]) for e in long_injected]
    rows += labelled(inject.empty_rows(EMPTY_ROWS, SEED))
    rows += labelled([replace(e, split="test") for e in inject.empty_rows(EMPTY_TEST_ROWS, SEED + 1)])
    return [_empty_sentinel(_bare_fillers(_modal_may(e))) for e in rows], dropped, records


def spoken_variant(example: Example, salt: str = "") -> Example:
    """`example` with its input as a given recognizer and a given speaker might produce it.

    - A composed recognizer-style row gets a recognizer's punctuation (`restyle.ASR_STYLES`), so
      long rows cover Whisper and Qwen3 as well as the Parakeet-like style the generator writes;
      `plain` keeps the generator's text.
    - A share of rows gets denser hesitation (`DENSIFY_PERCENT`), with Parakeet's pause
      punctuation only where the row is Parakeet's.

    Only fillers, repeats, punctuation and casing change, all of which the reference drops or
    rewrites anyway, so the reference stays right. A pure function of the row and `salt`: the
    build uses no salt, and training can pass one per epoch to draw fresh variants of train rows.
    Applied after deduplication (the near-duplicate check ignores fillers, so a copy that only adds
    them would be dropped as a duplicate), so a row is changed in place rather than copied.
    """
    composed = "composed" in example.tags
    if not composed and not (example.source in SHORT_DENSIFY_SOURCES and example.split in ("train", "val")):
        return example
    key = f"{example.id}{salt}"
    text, tags = example.input, list(example.tags)
    if "style_raw" in tags or not text[:1].isupper():
        style, asr = "raw", None
    else:
        style = "recognizer"
        asr = restyle.asr_style(key) if composed else None
    if hash_bucket(f"densify-{key}") < DENSIFY_PERCENT["composed" if composed else "short"]:
        dense = inject.densify(text, key, style, pauses=asr in (None, "parakeet")).text
        if dense != text:
            text, tags = dense, [*tags, "densified"]
    if asr:
        if asr != "plain":
            text = restyle.recognizer_quirks(text, key, asr)
        tags.append(f"asr_{asr}")
    if text == example.input and tags == example.tags:
        return example
    return replace(example, input=text, tags=tags)


def restyled(rows: list[Example]) -> list[Example]:
    """aawaaz rows with their recognizer-style rewrite from `restyle`, where one passed.

    `build` swaps the rewrites in after filtering and deduplication, which judge a row by the
    spoken input its reference was written and checked against: a rewrite's "401k" or "PhD"
    against the reference's "401(k)" or "Ph.D." is the cleanup, not a changed code token.
    """
    rewrites = restyle.load()
    out = []
    for e in rows:
        if e.id in rewrites:
            style = restyle.asr_style(e.id)
            text = restyle.recognizer_quirks(rewrites[e.id], e.id, style, spoken=e.input)
            e = replace(e, input=text, tags=[*e.tags, "style_recognizer", "restyled", f"asr_{style}"])
        out.append(e)
    return out


def _bare_fillers(example: Example) -> Example:
    """The `parakeet` share of the stage 3 recognizer-style inputs, which put commas around nearly
    every filler, with the fillers mostly bare as Parakeet writes them. The `TEST_IDS_109` test rows
    were built this way and stay so; `strip_filler_commas` goes further for train and val."""
    if "style_recognizer" not in example.tags or "restyled" in example.tags or example.source == inject.SOURCE:
        return example
    if restyle.asr_style(example.id) != "parakeet":
        return example
    text = restyle.filler_commas(example.input, example.id)
    return replace(example, input=text, tags=[*example.tags, "asr_parakeet"])


# A comma before a filler stays after a discourse word, as in Parakeet's "So, um let's".
_DISCOURSE = {"so", "and", "but", "okay", "ok", "well", "yeah", "oh", "alright", "right", "now", "then", "also"}


def strip_filler_commas(text: str) -> str:
    """`text` with no comma after a filler, and none before one except after a discourse word."""
    tokens = text.split(" ")
    for i, token in enumerate(tokens):
        match = FILLER_TOKEN.match(token)
        if not match:
            continue
        if match.group(2) == ",":
            tokens[i] = match.group(1)
        previous = tokens[i - 1] if i else ""
        if previous.endswith(",") and previous[:-1].lower() not in _DISCOURSE and not FILLER_TOKEN.match(previous):
            tokens[i - 1] = previous[:-1]
    return " ".join(tokens)


def bare_filler_share(examples: Iterable[Example]) -> list[Example]:
    """Strips filler commas from `BARE_FILLER_PERCENT` of the teacher-generated recognizer-style
    train and val inputs, chosen by id, and tags them `fillers_stripped`. Test rows stay as built."""
    out = []
    for e in examples:
        if (
            e.source == generate.SOURCE and e.split in ("train", "val") and "style_recognizer" in e.tags
            and "composed" not in e.tags
            and hash_bucket(f"bare-{e.id}") < BARE_FILLER_PERCENT
        ):
            text = strip_filler_commas(e.input)
            if text != e.input:
                e = replace(e, input=text, tags=[*e.tags, "fillers_stripped"])
        out.append(e)
    return out


def _modal_may(example: Example) -> Example:
    # A relabelled reference is the teacher's own, with no dataset quirk to fix.
    if example.source != sources.HANDY or example.teacher_reasoning:
        return example
    output = sources.modal_may(example.output)
    if output == example.output:
        return example
    return replace(example, output=output, tags=[*example.tags, "normalized"])


def _empty_sentinel(example: Example) -> Example:
    """aawaaz writes an empty output for filler-only input, where the app expects EMPTY."""
    if not example.output.strip() and filler_only(example.input):
        return replace(example, output=EMPTY)
    return example


def answered(example: Example) -> bool:
    return assistant_opener(example.output, example.input)


def _alnum(token: str) -> str:
    return re.sub(r"[^0-9a-z]", "", token.casefold())


def changed_protected_token(example: Example) -> str | None:
    """A code-like token (path, URL, email, identifier, flag) that the output rewrote.

    A token the output drops is not flagged, since a self-correction may remove it;
    one that reappears in another form ("x25" as "X.25") is.
    """
    output_tokens = {token.strip(TRAILING_PUNCTUATION) for token in example.output.split()}
    output_forms = {_alnum(token) for token in output_tokens}
    # Recasing a dictated word-like token ("co2" as "CO2", "typeScript" as "TypeScript") is a fix;
    # paths, files, emails, URLs and flags stay case-sensitive.
    output_folded = example.output.casefold()
    for token in protected_tokens(example.input):
        if not CODE_TOKEN.search(token) or token in example.output:
            continue
        if not CASE_SENSITIVE.search(token) and token.casefold() in output_folded:
            continue
        if _alnum(token) and _alnum(token) in output_forms:
            return token
    return None


_SUBJECT_LINE = re.compile(r"(?m)^Subject:")


def invented_subject(example: Example) -> bool:
    """An email subject line the speaker never dictated; aawaaz writes one for some emails."""
    return bool(_SUBJECT_LINE.search(example.output)) and "subject" not in example.input.casefold()


_LEGALESE = re.compile(r"\b(?:shall|hereby|herein|hereinafter|thereof|pursuant|whereas)\b", re.IGNORECASE)


def added_legalese(example: Example) -> bool:
    """Contract wording the speaker never used, such as "shall pay" for "is going to pay"."""
    spoken = {word.casefold() for word in _LEGALESE.findall(example.input)}
    return any(word.casefold() not in spoken for word in _LEGALESE.findall(example.output))


_SIGN_OFF = re.compile(
    r"(?im)^(thanks|thank you|many thanks|thanks so much|thanks again|best|best regards|kind regards|"
    r"warm regards|regards|cheers|sincerely|warmly|talk soon|all the best|bisous)[,!.]?[ \t]*$"
)


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in text if not unicodedata.combining(c))


def invented_signature(example: Example) -> bool:
    """A name or title under the sign-off that the speaker never dictated; aawaaz fills one in from the body.

    Each word under the last sign-off line must be spoken after the same sign-off in the input.
    """
    output = _fold(example.output)
    sign_offs = list(_SIGN_OFF.finditer(output))
    if not sign_offs:
        return False
    sign_off = sign_offs[-1]
    spoken = _fold(example.input)
    start = spoken.rfind(sign_off.group(1))
    tail = spoken[start + len(sign_off.group(1)) :] if start >= 0 else spoken
    tail = re.sub(r"[^a-z]", "", re.sub(r"\bdoctor\b", "dr", tail))
    return any(word not in tail for word in re.findall(r"[a-z]+", output[sign_off.end() :]))


def implausible_length(example: Example) -> bool:
    if example.output == EMPTY:
        return len(sources.words(example.input)) > 6
    source_words = len([w for w in sources.words(example.input) if not FILLER.fullmatch(w)])
    output_words = len(sources.words(example.output))
    # A list sheds the talk around its items, and a self-correction sheds the discarded versions,
    # so only prose without corrections must keep 30% of the words.
    shrinks = example.layout in ("list", "mixed") or CORRECTION.search(example.input)
    if source_words >= 20 and output_words < 0.3 * source_words and not shrinks:
        return True
    return output_words > 1.3 * source_words + 8


def filter_reason(example: Example) -> str | None:
    if not example.output.strip():
        return "empty output"
    if has_markdown(example.output):
        return "markdown"
    reason = malformed_list(example.output)
    if reason:
        return f"malformed list: {reason}"
    if answered(example):
        return "answered"
    token = changed_protected_token(example)
    if token:
        return "changed protected token"
    if invented_subject(example):
        return "invented subject line"
    if added_legalese(example):
        return "added legal formality"
    if invented_signature(example):
        return "invented signature"
    if implausible_length(example):
        return "implausible length ratio"
    return None


_PRIME = (1 << 61) - 1
_PERMUTATIONS = 128
_BANDS = 16


def _shingles(text: str) -> set[str]:
    tokens = [w for w in sources.words(text) if not FILLER.fullmatch(w)]
    if len(tokens) < 3:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + 3]) for i in range(len(tokens) - 2)}


def _minhash(shingles: set[str], a: np.ndarray, b: np.ndarray) -> np.ndarray:
    hashes = np.array(
        [int.from_bytes(hashlib.blake2b(s.encode(), digest_size=7).digest()) for s in shingles], dtype=np.uint64
    )
    # Shifting the 56-bit hashes to 28 bits keeps a * x + b below 2**61, so uint64 arithmetic stays exact.
    return ((a[:, None] * (hashes[None, :] >> np.uint64(28)) + b[:, None]) % np.uint64(_PRIME)).min(axis=1)


def near_duplicates(examples: Sequence[Example]) -> dict[int, int]:
    """Maps the index of each near-duplicate input to the index of the row it duplicates."""
    rng = np.random.default_rng(SEED)
    a = rng.integers(1, 1 << 32, _PERMUTATIONS, dtype=np.uint64)
    b = rng.integers(0, 1 << 32, _PERMUTATIONS, dtype=np.uint64)
    shingles = [_shingles(e.input) for e in examples]
    band_size = _PERMUTATIONS // _BANDS
    buckets: dict[tuple[int, bytes], int] = {}
    duplicate_of: dict[int, int] = {}
    for index, shingle_set in enumerate(shingles):
        signature = _minhash(shingle_set, a, b)
        candidates = set()
        keys = [(band, signature[band * band_size : (band + 1) * band_size].tobytes()) for band in range(_BANDS)]
        for key in keys:
            if key in buckets:
                candidates.add(buckets[key])
        match = next(
            (
                other
                for other in sorted(candidates)
                if len(shingle_set & shingles[other]) / len(shingle_set | shingles[other]) >= NEAR_DUPLICATE_JACCARD
            ),
            None,
        )
        if match is not None:
            duplicate_of[index] = match
            continue
        for key in keys:
            buckets.setdefault(key, index)
    return duplicate_of


def deduplicate(examples: Iterable[Example]) -> tuple[list[Example], list[tuple[Example, str]]]:
    """Drops exact and near-duplicate inputs, and outputs repeated more than `MAX_SAME_OUTPUT` times."""
    ordered = sorted(examples, key=_source_rank)
    kept, dropped, seen_inputs = [], [], set()
    for example in ordered:
        key = " ".join(example.input.split()).casefold()
        if key in seen_inputs:
            dropped.append((example, "exact duplicate input"))
        else:
            seen_inputs.add(key)
            kept.append(example)
    # Filler-only EMPTY inputs share almost no words once fillers are removed, so they would all
    # match each other; only exact duplicates of them are dropped.
    candidates = [e for e in kept if e.output != EMPTY]
    duplicate_ids = {id(candidates[index]) for index in near_duplicates(candidates)}
    survivors = []
    for example in kept:
        if id(example) in duplicate_ids:
            dropped.append((example, "near-duplicate input"))
        else:
            survivors.append(example)
    same_output: Counter = Counter()
    final = []
    for example in survivors:
        key = sources.skeleton(example.output)
        same_output[key] += 1
        if same_output[key] > MAX_SAME_OUTPUT and example.output != EMPTY:
            dropped.append((example, "output repeated too often"))
        else:
            final.append(example)
    return final, dropped


def assign_splits(examples: Iterable[Example]) -> list[Example]:
    """Splits by seed group: a group with a preset test row is test, and others are hashed into val and test."""
    groups: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        groups[example.seed_group].append(example)
    result = []
    for seed_group, members in groups.items():
        bucket = hash_bucket(f"split\0{seed_group}")
        if any(e.split == "test" for e in members):
            split = "test"
        elif bucket < VAL_PERCENT:
            split = "val"
        elif bucket < VAL_PERCENT + TEST_PERCENT and not any(
            e.source in PRESET_TEST_SOURCES for e in members
        ):
            split = "test"
        else:
            split = "train"
        result += [replace(e, split=split) for e in members]
    return result


def _percentiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def at(q: float) -> int:
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    return {"p5": at(0.05), "p50": at(0.5), "p95": at(0.95), "max": ordered[-1], "mean": round(statistics.mean(values), 1)}


def stats(examples: Sequence[Example], dropped: Sequence[tuple[Example, str]]) -> dict:
    def count(key) -> dict:
        return dict(Counter(key(e) for e in examples).most_common())

    total = len(examples)
    laid_out = sum(e.layout != "plain" for e in examples)
    return {
        "rows": total,
        "per_split": count(lambda e: e.split),
        "per_source": count(lambda e: e.source),
        "per_category": count(lambda e: e.category),
        "per_layout": count(lambda e: e.layout),
        "per_license": count(lambda e: e.license),
        "per_tag": dict(Counter(t for e in examples for t in e.tags if not t.startswith("from:")).most_common()),
        "per_split_source": {
            split: dict(Counter(e.source for e in examples if e.split == split).most_common())
            for split in ("train", "val", "test")
        },
        "aawaaz_share": round(sum(e.source == aawaaz.REPO for e in examples) / max(1, total), 3),
        "layout_share": round(laid_out / max(1, total), 3),
        "list_share": round(sum(e.layout in ("list", "mixed") for e in examples) / max(1, total), 3),
        "input_words": _percentiles([len(e.input.split()) for e in examples]),
        "output_words": _percentiles([len(e.output.split()) for e in examples]),
        "dropped": dict(Counter(reason for _, reason in dropped).most_common()),
        "length": {split: length_histogram([e for e in examples if e.split == split]) for split in ("train", "val", "test")},
    }


def length_histogram(examples: Sequence[Example]) -> dict:
    """Rows per input length bucket, and the share over each of `LENGTH_MARKS` words."""
    counts = [len(e.input.split()) for e in examples]
    buckets = Counter(length_bucket(n) for n in counts)
    return {
        "buckets": {name: buckets[name] for _, name in (*LENGTH_BUCKETS, (None, "over 1000")) if buckets[name]},
        "over": {str(mark): round(sum(n > mark for n in counts) / max(1, len(counts)), 3) for mark in LENGTH_MARKS},
    }


def stats_markdown(report: dict) -> str:
    lines = [f"# Build stats\n\n{report['rows']} rows. aawaaz share {report['aawaaz_share']:.0%}, "
             f"layout share {report['layout_share']:.0%} (lists {report['list_share']:.0%}).\n"]
    for title, key in (
        ("Split", "per_split"),
        ("Source", "per_source"),
        ("Category", "per_category"),
        ("Layout", "per_layout"),
        ("License", "per_license"),
        ("Tag", "per_tag"),
        ("Dropped", "dropped"),
    ):
        lines.append(f"## {title}\n\n| {title} | Rows |\n|---|---|")
        lines += [f"| {name} | {value} |" for name, value in report[key].items()]
        lines.append("")
    lines.append("## Length (words)\n\n| | p5 | p50 | p95 | max | mean |\n|---|---|---|---|---|---|")
    for key in ("input_words", "output_words"):
        p = report[key]
        lines.append(f"| {key.split('_')[0]} | {p['p5']} | {p['p50']} | {p['p95']} | {p['max']} | {p['mean']} |")
    names = [name for _, name in (*LENGTH_BUCKETS, (None, "over 1000"))]
    marks = [f"> {mark}" for mark in LENGTH_MARKS]
    lines.append("\n## Input length by split\n")
    lines.append("| split | " + " | ".join(names + marks) + " |\n|" + "---|" * (len(names) + len(marks) + 1))
    for split, hist in report.get("length", {}).items():
        cells = [str(hist["buckets"].get(name, 0)) for name in names]
        cells += [f"{hist['over'][str(mark)]:.1%}" for mark in LENGTH_MARKS]
        lines.append(f"| {split} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build(with_sotto: bool = False, out_dir: Path = paths.BUILD) -> dict:
    rows, dropped, records = gather_with_drops(with_sotto)
    kept = []
    for example in rows:
        reason = filter_reason(example)
        if reason:
            dropped.append((example, reason))
        else:
            kept.append(example)
    kept, duplicates = deduplicate(kept)
    dropped += duplicates
    final = [spoken_variant(e) for e in bare_filler_share(assign_splits(restyled(kept)))]

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        write_examples(out_dir / f"{split}.jsonl", sorted((e for e in final if e.split == split), key=lambda e: e.id))
    with (out_dir / "dropped.jsonl").open("w", encoding="utf-8") as sink:
        for example, reason in dropped:
            sink.write(json.dumps({**asdict(example), "dropped": reason}, ensure_ascii=False) + "\n")
    report = stats(final, dropped)
    markdown = stats_markdown(report)
    if records is not None:
        # Old against new by source; the old injected rows were copied aside with the relabel's inputs.
        compared = relabel.compare([*relabel.inputs("build"), *relabel.old_injected()], final, records)
        report["relabel"] = compared
        markdown += "\n" + relabel.compare_markdown(compared)
    (out_dir / "stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "stats.md").write_text(markdown, encoding="utf-8")
    return report


def review_sample(layout: str, n: int, seed: int, out_dir: Path = paths.BUILD) -> str:
    """`n` random built rows of one layout as text for a hand review, with each row's source and tags."""
    rows = [e for split in ("train", "val", "test") for e in read_examples(out_dir / f"{split}.jsonl")]
    rows = sorted((e for e in rows if e.layout == layout), key=lambda e: e.id)
    picked = np.random.default_rng(seed).choice(len(rows), size=min(n, len(rows)), replace=False)
    blocks = []
    for index, row_index in enumerate(picked):
        e = rows[row_index]
        header = f"### {index} {e.id} [{e.source}/{e.category}] {','.join(e.tags)}"
        blocks.append(f"{header}\nIN: {e.input}\nOUT: {e.output}\n")
    return "\n".join(blocks)


def publish(repo_id: str, out_dir: Path = paths.BUILD) -> None:
    """Pushes the built splits to a private Hugging Face dataset repo."""
    from datasets import DatasetDict, load_dataset

    files = {split: str(out_dir / f"{split}.jsonl") for split in ("train", "val", "test")}
    dataset: DatasetDict = load_dataset("json", data_files=files)
    dataset.save_to_disk(str(out_dir / "hf"))
    dataset.push_to_hub(repo_id, private=True)
