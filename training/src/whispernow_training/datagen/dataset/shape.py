"""Row fixes and input variants every round applies to the rows it adds."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from ...text_rules import FILLER_TOKEN, filler_only
from ..schema import EMPTY, Example, hash_bucket
from ..sources import public
from ..synthetic import generate, inject, restyle

# Shares of rows whose spoken input gets more hesitation (`inject.densify`), by kind of row. The
# generator writes about 0.8 fillers per 100 words in long rows and the short sources 1.6 to 1.7;
# real dictation measured 3.6. The rest keep their rate, as fluent dictation does.
DENSIFY_PERCENT = {"composed": 50, "short": 30}
# Short sources whose train and val inputs are densified. Their test rows (`TEST_IDS_109`) stay
# as built; aawaaz (4.2 per 100 words) and the injected rows already hesitate enough.
SHORT_DENSIFY_SOURCES = {generate.SOURCE, public.HANDY, public.DISFL_QA}
# Share of the teacher-generated recognizer-style train and val rows whose filler commas are
# stripped, towards Parakeet's bare fillers; the rest keep the comma-wrapped form other recognizers
# and pasted text produce.
BARE_FILLER_PERCENT = 67


def normalized(example: Example) -> Example:
    """`example` with the fixes every gathered row gets: Parakeet's bare fillers on its share of the
    generated recognizer-style inputs, handy's month-cased "May" lowered, and EMPTY for an empty
    output to a filler-only input."""
    return _empty_sentinel(_bare_fillers(_modal_may(example)))


def spoken_variant(example: Example) -> Example:
    """`example` with its input as a given recognizer and a given speaker might produce it.

    - A composed recognizer-style row gets a recognizer's punctuation (`restyle.ASR_STYLES`), so
      long rows cover Whisper and Qwen3 as well as the Parakeet-like style the generator writes;
      `plain` keeps the generator's text.
    - A share of rows gets denser hesitation (`DENSIFY_PERCENT`), with Parakeet's pause
      punctuation only where the row is Parakeet's.

    Only fillers, repeats, punctuation and casing change, all of which the reference drops or
    rewrites anyway, so the reference stays right. A pure function of the row. Applied
    after deduplication (the near-duplicate check ignores fillers, so a copy that only adds them
    would be dropped as a duplicate), so a row is changed in place rather than copied.
    """
    composed = "composed" in example.tags
    if not composed and not (example.source in SHORT_DENSIFY_SOURCES and example.split in ("train", "val")):
        return example
    key = example.id
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

    Swapped in after filtering and deduplication, which judge a row by the spoken input its
    reference was written and checked against: a rewrite's "401k" or "PhD" against the reference's
    "401(k)" or "Ph.D." is the cleanup, not a changed code token.
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
    """The `parakeet` share of the generated recognizer-style inputs, which put commas around nearly
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
    if example.source != public.HANDY or example.teacher_reasoning:
        return example
    output = public.modal_may(example.output)
    if output == example.output:
        return example
    return replace(example, output=output, tags=[*example.tags, "normalized"])


def _empty_sentinel(example: Example) -> Example:
    """aawaaz writes an empty output for filler-only input, where the app expects EMPTY."""
    if not example.output.strip() and filler_only(example.input):
        return replace(example, output=EMPTY)
    return example
