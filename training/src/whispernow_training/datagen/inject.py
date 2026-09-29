"""Programmatic rows: filler injection into clean text, and filler-only inputs that return EMPTY.

Injection takes a clean, punctuated one-paragraph output, adds fillers, repeated
words and a spoken opener, and uses the clean text as the target. It only adds
whole tokens between words, so code, numbers, names and URLs pass through
unchanged. Each injected row keeps its origin's `seed_group`, so it never lands
in a different split from the row it came from.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

from .schema import EMPTY, Example, example_id

SOURCE = "injected"
EMPTY_LICENSE = "MIT"
STYLES = ("recognizer", "raw")

_FILLERS = (("um", 30), ("uh", 30), ("you know", 10), ("I mean", 8), ("like", 8), ("er", 6), ("hmm", 4), ("uh, um", 4))
_OPENERS = ("So", "Okay, so", "Um, so", "Uh", "Yeah, so", "Alright")
_REPEATABLE = {"the", "a", "an", "i", "we", "to", "and", "it", "is", "that", "my", "our", "you", "they", "for", "in"}
_EDGE_PUNCTUATION = ".,!?;:"


def _is_plain_word(token: str) -> bool:
    return token.strip(_EDGE_PUNCTUATION).replace("'", "").isalpha()


def _pick_filler(rng: random.Random) -> str:
    words, weights = zip(*_FILLERS, strict=True)
    return rng.choices(words, weights)[0]


def inject(clean: str, rng: random.Random, style: str) -> str:
    """A spoken version of `clean`. `recognizer` keeps casing and punctuation; `raw` drops both."""
    tokens = clean.split()
    if not tokens:
        return clean
    edits = max(1, round(len(tokens) / rng.randint(6, 12)))
    # A slot i means "before token i"; only between plain words, so protected tokens stay intact.
    slots = [i for i in range(1, len(tokens)) if _is_plain_word(tokens[i]) and _is_plain_word(tokens[i - 1])]
    chosen = sorted(rng.sample(slots, min(edits, len(slots))), reverse=True)
    for slot in chosen:
        if rng.random() < 0.25 and tokens[slot].lower().strip(_EDGE_PUNCTUATION) in _REPEATABLE:
            tokens.insert(slot, tokens[slot].strip(_EDGE_PUNCTUATION))
            continue
        filler = _pick_filler(rng)
        if style == "recognizer":
            previous = tokens[slot - 1]
            if previous.endswith((".", "!", "?")):
                filler = filler[0].upper() + filler[1:] + ","
            elif previous.endswith((",", ";", ":")):
                filler += ","
            else:
                tokens[slot - 1] = previous + ","
                filler += ","
        tokens.insert(slot, filler)
    if rng.random() < 0.4:
        opener = rng.choice(_OPENERS)
        first = tokens[0]
        if _is_plain_word(first) and not first.isupper() and first != "I":
            tokens[0] = first[0].lower() + first[1:]
        tokens.insert(0, opener + ("," if style == "recognizer" and not opener.endswith("so") else ""))
    # Multi-word fillers and openers become separate tokens.
    tokens = " ".join(tokens).split()
    if style == "raw":
        tokens = [_raw_token(token) for token in tokens]
    return " ".join(token for token in tokens if token)


def _raw_token(token: str) -> str:
    """Lowercases capitalized words and drops trailing punctuation, as an older recognizer would write them.

    Leading punctuation stays: it is part of the token, as in "./data" or ".env".
    """
    core = token.rstrip(_EDGE_PUNCTUATION)
    if "." in core:
        # An abbreviation such as "U.S." keeps its periods.
        return token.rstrip(",;:!?")
    word = core.replace("'", "")
    if word.isalpha() and word[0].isupper() and word[1:].islower() and not core.startswith("I'") and core != "I":
        core = core.lower()
    return core


def injected_rows(origins: Iterable[Example], count: int, seed: int) -> list[Example]:
    """Injects fillers into `count` one-paragraph origins, alternating the two input styles."""
    rng = random.Random(seed)
    pool = sorted((e for e in origins if e.layout == "plain" and e.split != "test"), key=lambda e: e.id)
    rng.shuffle(pool)
    rows = []
    for index, origin in enumerate(pool[:count]):
        style = STYLES[index % len(STYLES)]
        row_rng = random.Random(f"{seed}-{origin.id}")
        spoken = inject(origin.output, row_rng, style)
        if spoken == origin.output:
            continue
        rows.append(
            Example(
                id=example_id(SOURCE, spoken),
                input=spoken,
                output=origin.output,
                source=SOURCE,
                category="filler_injection",
                license=origin.license,
                tags=["injected", f"style_{style}", f"from:{origin.source}"],
                seed_group=origin.seed_group,
                teacher_model=origin.teacher_model,
            )
        )
    return rows


_EMPTY_FILLERS = ("um", "uh", "er", "erm", "hmm", "umm", "uhh", "mm")
_EMPTY_JOINS = (" ", ", ", "... ", ". ")
_EMPTY_ENDS = ("", ".", "...", "?")


def empty_rows(count: int, seed: int) -> list[Example]:
    """Inputs with no dictated words, only fillers and punctuation, whose target is EMPTY."""
    rng = random.Random(seed)
    seen: set[str] = set()
    rows = []
    for _ in range(count * 20):
        if len(rows) == count:
            break
        fillers = [rng.choice(_EMPTY_FILLERS) for _ in range(rng.randint(1, 4))]
        text = rng.choice(_EMPTY_JOINS).join(fillers) + rng.choice(_EMPTY_ENDS)
        if rng.random() < 0.5:
            text = text[0].upper() + text[1:]
        if text in seen:
            continue
        seen.add(text)
        rows.append(
            Example(
                id=example_id(SOURCE, text),
                input=text,
                output=EMPTY,
                source=SOURCE,
                category="empty",
                license=EMPTY_LICENSE,
                tags=["empty"],
            )
        )
    return rows
