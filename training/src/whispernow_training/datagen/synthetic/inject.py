"""Programmatic rows: speech noise injected into clean text, and filler-only inputs that return EMPTY.

Injection takes a clean, punctuated one-paragraph output, makes it sound spoken, and uses the clean
text as the target. Every kind of noise is safe without a judge, because the target is the clean
text by construction:

- `filler`, `cluster`: um, uh, er and hmm, one or several in a row. Nothing that can carry meaning
  ("like", "I mean", "you know", "actually", "basically") is injected as a filler; generated rows,
  whose target the teacher writes, cover those.
- `determiner`: a filler between a repeated determiner ("the uh the green one").
- `repeat`: a doubled word or two ("we we", "I don't I don't know") or a tripled conjunction.
- `restart`: a phrase started, dropped and started again with no cue ("ask them for I need to ask
  them for"), using only the clause's own words.
- `fragment`: a word of the coming clause said first and left hanging ("it right now it's").
- `correction`: a known-answer self-correction. A number, time, weekday, month or colour in the text
  is said wrong first and then corrected, in one of three forms: a short cue ("Tuesday, no,
  Wednesday"), a re-decision ("Tuesday or actually no, let's make it Wednesday") or a reasoned one
  ("Tuesday, except uh well that's not right, so actually let's just make it Wednesday instead").
  The wrong value never appears in the clean text. The reason is about the mistake, not the
  options, so the target has nothing to keep from it; reasons that are content are left to
  generated rows too.
- `pause`: recognizer punctuation at a pause, a period or question mark mid-sentence, or a question
  mark after a list item said with rising intonation.

Fillers are written as Parakeet writes them: bare, capitalized with no comma at a sentence start,
with at most the comma already before them. A small share of rows wraps them in commas, as other
recognizers and pasted text do. Protected tokens (numbers, code, names) are never split, copied or
recased, only moved apart from their neighbours by inserted words.

`inject` is a pure function of the clean text, a seed and the style, so `epochs` can draw fresh
noise per epoch. Each injected row keeps its origin's `seed_group`, so it never lands in a different
split from the row it came from.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ...text_rules import FILLER_TOKEN
from ..schema import EMPTY, OWN_LICENSE, Example, example_id

SOURCE = "injected"
STYLES = ("recognizer", "raw")
KINDS = ("filler", "cluster", "determiner", "repeat", "restart", "fragment", "correction", "pause")
# Words that can carry meaning, which random injection never adds as a filler.
AMBIGUOUS = ("like", "i mean", "you know", "actually", "basically", "so", "well", "okay", "right")
# Words per edit, and how often each level is drawn.
LEVELS = {"light": 30, "medium": 15, "heavy": 7}
_LEVEL_WEIGHTS = {"light": 4, "medium": 4, "heavy": 2}
_KIND_WEIGHTS = {
    "filler": 30, "cluster": 10, "determiner": 8, "repeat": 15, "restart": 7, "fragment": 4, "correction": 10, "pause": 12,
}
# Share of recognizer-style rows whose fillers are wrapped in commas.
COMMA_FILLERS = 0.15
CORRECTION_FORMS = ("short", "redecision", "reasoned")

_FILLERS = (("um", 40), ("uh", 40), ("er", 8), ("hmm", 6))
_EDGE_PUNCTUATION = ".,!?;:"
_SENTENCE_END = (".", "!", "?")
_DETERMINERS = {"the", "a", "an", "this", "that", "my", "our", "your", "his", "her", "their", "some"}
_CONJUNCTIONS = {"and", "but", "because"}
_PRONOUNS = {"i", "we", "you", "they", "he", "she", "it", "i'm", "we're", "you're", "they're", "it's", "i'll", "we'll", "i've", "we've"}
_REPEATABLE = _PRONOUNS | _CONJUNCTIONS | {
    "the", "a", "to", "is", "that", "my", "our", "for", "in", "can", "will", "should", "need", "was", "do", "don't",
    "have", "if", "when",
}
# Sentence-initial words a recognizer writes in lower case after a capitalized filler or restart;
# anything else could be a name.
_LOWERABLE = {
    "the", "a", "an", "we", "it", "it's", "is", "can", "could", "should", "and", "but", "this", "that", "there",
    "they", "you", "my", "our", "what", "when", "how", "let's", "please", "if", "also", "then", "we're", "we'll",
    "we've", "to", "for", "in", "on", "at", "do", "don't", "just", "maybe", "make", "send", "call", "get", "add",
    "tell", "ask", "remind", "need", "have", "one", "next", "after", "before", "once", "all", "some", "these", "those",
    "bring", "book", "buy", "pick", "check", "let", "put", "take", "remember", "try", "keep", "use", "write", "set",
    "move", "change", "find", "look", "so", "not", "no", "yes", "or", "with", "from", "by", "as", "about", "if",
}
# Where a filler lands, by the word said before it. Speakers hesitate where they plan what comes
# next: after a conjunction, subordinator or discourse word, and after a preposition, verb or
# determiner before the phrase it introduces ("ask for uh the invoice", "the uh the green one"). Inside a
# phrase, after a content word, a filler is rare.
_PLANNING = {
    "and", "but", "or", "so", "because", "since", "that", "when", "if", "then", "except", "while", "though",
    "which", "where", "until", "unless", "also", "plus", "like", "yeah", "okay", "ok", "well", "now", "maybe",
    "actually", "probably", "just", "think", "guess", "mean",
}
_LEAD_IN = {
    "with", "for", "to", "of", "about", "from", "on", "at", "in", "into", "by", "after", "before", "get", "got",
    "go", "buy", "order", "need", "needs", "want", "wants", "call", "ask", "tell", "try", "use", "add", "bring",
    "pick", "make", "send", "is", "was", "are", "be", "it's", "there's", "do", "have", "grab", "check", "put",
}
_SLOT_WEIGHTS = {"clause": 4.0, "planning": 4.0, "lead_in": 2.0, "determiner": 1.5, "pronoun": 1.0, "inside": 0.3}
# Spoken symbols and layout commands, which a filler never splits.
_SPOKEN_SYMBOLS = {
    "slash", "dot", "dash", "underscore", "colon", "space", "hash", "backslash", "tilde", "asterisk", "equals",
    "new", "line", "paragraph", "bullet", "point", "period", "comma", "quote", "unquote", "bracket", "brace",
    "parenthesis", "semicolon", "hyphen", "ampersand", "pipe", "caret",
}
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
# "May" is left out: it is also a modal.
_MONTHS = ("January", "February", "March", "April", "June", "July", "August", "September", "October", "November", "December")
_COLOURS = ("red", "blue", "green", "yellow", "black", "white", "purple", "pink", "grey", "brown", "silver")
_NUMBER = re.compile(r"^(\$?)(\d{1,4})(%?)$")
_CLOCK = re.compile(r"^(\d{1,2}):(\d{2})$")
_SHORT_CUES = (("no",), ("sorry",), ("wait",), ("I", "mean"), ("no", "sorry"), ("no", "wait"))
_STACKED_CUES = (("no", "wait", "sorry"), ("sorry", "no", "I", "mean"))
_REASONS = (
    ("that's", "not", "right"), ("that's", "wrong"), ("I", "got", "that", "mixed", "up"), ("no", "that's", "not", "it"),
    ("I", "had", "that", "wrong"),
)


def _core(token: str) -> str:
    return token.strip(_EDGE_PUNCTUATION)


def _is_plain_word(token: str) -> bool:
    return _core(token).replace("'", "").isalpha()


def _bare_word(token: str) -> bool:
    return _is_plain_word(token) and token == token.rstrip(_EDGE_PUNCTUATION) and not token.startswith(tuple(_EDGE_PUNCTUATION))


def _pick_filler(rng: random.Random, avoid: str = "") -> str:
    words, weights = zip(*((w, n) for w, n in _FILLERS if w != avoid), strict=True)
    return rng.choices(words, weights)[0]


def _capitalize(word: str) -> str:
    return word[:1].upper() + word[1:]


@dataclass(frozen=True)
class Injection:
    """A spoken version of a clean text, and what was done to it."""

    text: str
    level: str
    fillers: str  # "bare" or "commas"
    kinds: tuple[str, ...] = ()
    # (wrong, right) for each inserted correction, with its form.
    corrections: tuple[tuple[str, str, str], ...] = ()

    @property
    def tags(self) -> list[str]:
        tags = [f"noise_level:{self.level}", f"fillers_{self.fillers}"]
        tags += [f"noise:{kind}" for kind in sorted(set(self.kinds))]
        tags += [f"correction:{form}" for form in sorted({form for _, _, form in self.corrections})]
        return tags


@dataclass
class _Draft:
    tokens: list[str]
    style: str
    commas: bool
    rng: random.Random
    before: dict[int, list[str]] = field(default_factory=dict)
    after: dict[int, list[str]] = field(default_factory=dict)
    claimed: set[int] = field(default_factory=set)
    # False when the text is already spoken and in its style: only the inserted words change.
    restyle: bool = True

    def __post_init__(self) -> None:
        self.original = list(self.tokens)

    def sentence_start(self, i: int) -> bool:
        return i == 0 or self.original[i - 1].endswith(_SENTENCE_END)

    def free(self, *indices: int) -> bool:
        return all(0 <= i < len(self.tokens) and i not in self.claimed for i in indices)

    def claim(self, *indices: int) -> None:
        self.claimed.update(indices)

    def insert(self, i: int, words: Sequence[str]) -> None:
        self.before.setdefault(i, []).extend(words)

    def lower_start(self, i: int) -> None:
        """Lower-cases the word at a sentence start that something was said before, if it is not a name."""
        word = self.tokens[i]
        if self.sentence_start(i) and _core(word).lower() in _LOWERABLE and word[:1].isupper():
            self.tokens[i] = word[:1].lower() + word[1:]

    def fillers(self, i: int, fillers: Sequence[str]) -> list[str]:
        """Fillers said before token i, written as this row writes them."""
        words = list(fillers)
        if self.sentence_start(i) and self.style != "raw":
            words[0] = _capitalize(words[0])
        if self.commas and self.style == "recognizer":
            if i > 0 and not self.tokens[i - 1].endswith(tuple(_EDGE_PUNCTUATION)):
                self.tokens[i - 1] += ","
            words = [word + "," for word in words]
        return words

    def render(self) -> str:
        out: list[str] = []
        for i, token in enumerate(self.tokens):
            out += self.before.get(i, [])
            out.append(token)
            out += self.after.get(i, [])
        tokens = " ".join(out).split()
        if self.style == "raw" and self.restyle:
            tokens = [_raw_token(token) for token in tokens]
        return " ".join(token for token in tokens if token)


def _slot_weight(draft: _Draft, i: int) -> float:
    """How likely a filler is before token i, from the word said before it; 0 where it never lands."""
    word, before = _core(draft.tokens[i]), _core(draft.original[i - 1]) if i else ""
    if word.lower() in _SPOKEN_SYMBOLS or before.lower() in _SPOKEN_SYMBOLS:
        return 0.0
    # Not inside a name such as "Bell Street Motors".
    if i and word[:1].isupper() and before[:1].isupper() and not draft.sentence_start(i):
        return 0.0
    if i and draft.original[i - 1].endswith((",", ";", ":")):
        return _SLOT_WEIGHTS["clause"]
    before = before.lower()
    for kind, words in (
        ("planning", _PLANNING), ("lead_in", _LEAD_IN), ("determiner", _DETERMINERS), ("pronoun", _PRONOUNS)
    ):
        if before in words:
            return _SLOT_WEIGHTS[kind]
    return _SLOT_WEIGHTS["inside"]


def _filler(draft: _Draft, cluster: bool) -> bool:
    # Only between plain words: not inside "3 PM" or next to a code token.
    slots = [
        i for i in range(len(draft.tokens))
        if draft.free(i, i - 1 if i else i) and _is_plain_word(draft.tokens[i]) and (i == 0 or _is_plain_word(draft.tokens[i - 1]))
    ]
    weights = [_slot_weight(draft, i) for i in slots]
    slots = [i for i, weight in zip(slots, weights, strict=True) if weight]
    if not slots:
        return False
    # Real Parakeet output puts about a third of its fillers at a sentence start ("Uh is it").
    starts = [i for i in slots if draft.sentence_start(i)]
    if starts and draft.rng.random() < 0.35:
        i = draft.rng.choice(starts)
    else:
        i = draft.rng.choices(slots, [w for w in weights if w])[0]
    count = (3 if draft.rng.random() < 0.15 else 2) if cluster else 1
    fillers = [_pick_filler(draft.rng)]
    while len(fillers) < count:
        fillers.append(_pick_filler(draft.rng, avoid=fillers[-1]))
    draft.insert(i, draft.fillers(i, fillers))
    draft.lower_start(i)
    draft.claim(i, max(0, i - 1))
    return True


def _determiner(draft: _Draft) -> bool:
    slots = [
        i for i in range(1, len(draft.tokens) - 1)
        if draft.free(i - 1, i) and draft.tokens[i] in _DETERMINERS and _is_plain_word(draft.tokens[i + 1]) and not draft.sentence_start(i)
    ]
    if not slots:
        return False
    i = draft.rng.choice(slots)
    determiner = [draft.tokens[i] + ("," if draft.commas and draft.style == "recognizer" else "")]
    filler = [_pick_filler(draft.rng)]
    if draft.rng.random() < 0.2:
        filler.append(_pick_filler(draft.rng, avoid=filler[0]))
    draft.insert(i, determiner + draft.fillers(i, filler))
    draft.claim(i - 1, i)
    return True


def _repeat(draft: _Draft) -> bool:
    slots = [i for i in range(len(draft.tokens)) if draft.free(i) and _bare_word(draft.tokens[i]) and draft.tokens[i].lower() in _REPEATABLE]
    if not slots:
        return False
    i = draft.rng.choice(slots)
    word = draft.tokens[i]
    if word.lower() in _CONJUNCTIONS and draft.rng.random() < 0.5:
        copy = [word, word.lower()]
    elif word.lower() in _PRONOUNS and i + 1 < len(draft.tokens) and _bare_word(draft.tokens[i + 1]) and draft.rng.random() < 0.4:
        copy = [word, draft.tokens[i + 1]]
    else:
        copy = [word]
    draft.insert(i, copy)
    draft.lower_start(i)
    draft.claim(i)
    return True


def _clause_starts(draft: _Draft) -> list[int]:
    return [
        i for i in range(len(draft.tokens))
        if draft.sentence_start(i) or (draft.original[i - 1].endswith(",") and draft.tokens[i].lower() in _LOWERABLE | _PRONOUNS)
    ]


def _restart(draft: _Draft) -> bool:
    """The clause's first words, or a later piece of it, said before the clause starts properly."""
    options = []
    for start in _clause_starts(draft):
        run = 0
        while start + run < len(draft.tokens) and _bare_word(draft.tokens[start + run]) and draft.tokens[start + run].lower() not in AMBIGUOUS:
            run += 1
        if run >= 3 and draft.free(*range(max(0, start - 1), start + run)):
            options.append((start, run))
    if not options:
        return False
    start, run = draft.rng.choice(options)
    if draft.rng.random() < 0.5:
        copy = draft.tokens[start : start + draft.rng.randint(2, min(4, run))]
    else:
        offset = draft.rng.randint(1, run - 2)
        copy = draft.tokens[start + offset : start + offset + draft.rng.randint(2, min(3, run - offset))]
        if draft.sentence_start(start):
            copy = [_capitalize(copy[0]), *copy[1:]]
    draft.insert(start, copy)
    draft.lower_start(start)
    draft.claim(*range(max(0, start - 1), start + run))
    return True


def _fragment(draft: _Draft) -> bool:
    """A pronoun from the coming clause, said first and dropped."""
    options = []
    for start in _clause_starts(draft):
        for index in range(start + 1, min(start + 6, len(draft.tokens))):
            base = _core(draft.tokens[index]).split("'")[0]
            if base.lower() in _PRONOUNS and draft.free(max(0, start - 1), start):
                options.append((start, base))
                break
    if not options:
        return False
    start, base = draft.rng.choice(options)
    draft.insert(start, [_capitalize(base) if draft.sentence_start(start) or base == "I" else base.lower()])
    draft.lower_start(start)
    draft.claim(max(0, start - 1), start)
    return True


def _wrong_value(core: str, clean: str, rng: random.Random) -> tuple[str, str] | None:
    """A wrong value of the same kind as `core` that the clean text never mentions, and the verb used to pick it."""

    def absent(value: str) -> bool:
        if value[:1].isdigit() or value.startswith("$"):
            digits = re.escape(value.lstrip("$").rstrip("%"))
            return re.search(rf"(?<![\d.]){digits}(?![\d])", clean) is None
        return re.search(rf"\b{re.escape(value)}\b", clean, re.IGNORECASE) is None

    candidates: list[str] = []
    verb = ("make", "it")
    if match := _NUMBER.match(core):
        prefix, digits, suffix = match.groups()
        number = int(digits)
        if digits.startswith("0") and number:
            return None
        steps = (1, -1, 2, -2, 5, 10) if number < 100 else (1, -1, 10, -10, 50, 100)
        candidates = [f"{prefix}{number + step}{suffix}" for step in steps if number + step > 0]
    elif match := _CLOCK.match(core):
        hour, minutes = int(match.group(1)), match.group(2)
        candidates = [f"{h}:{minutes}" for h in (hour + 1, hour - 1, hour + 2) if 1 <= h <= 12]
    elif core in _WEEKDAYS:
        candidates = [day for day in _WEEKDAYS if day != core]
    elif core in _MONTHS:
        candidates = [month for month in _MONTHS if month != core]
    elif core in _COLOURS:
        candidates = [colour for colour in _COLOURS if colour != core]
        verb = ("go", "with")
    rng.shuffle(candidates)
    for candidate in candidates:
        if absent(candidate):
            return candidate, " ".join(verb)
    return None


def _correction(draft: _Draft, clean: str, corrections: list[tuple[str, str, str]]) -> bool:
    options = []
    for i, token in enumerate(draft.tokens):
        core = _core(token)
        if not draft.free(i - 1 if i else i, i) or token.startswith(tuple(_EDGE_PUNCTUATION)):
            continue
        wrong = _wrong_value(core, clean, random.Random(f"{draft.rng.random()}"))
        if wrong and all(wrong[0] != w for w, _, _ in corrections):
            options.append((i, core, *wrong))
    if not options:
        return False
    i, right, wrong, verb = draft.rng.choice(options)
    form = draft.rng.choices(CORRECTION_FORMS, (5, 3, 2))[0]
    punctuate = draft.style == "recognizer" and (draft.commas or draft.rng.random() < 0.6)
    comma = "," if punctuate else ""
    if form == "short":
        cue = list(draft.rng.choice(_STACKED_CUES if draft.rng.random() < 0.05 else _SHORT_CUES))
        words = [wrong + comma, *cue[:-1], cue[-1] + comma]
    elif form == "redecision":
        middle = [_pick_filler(draft.rng)] if draft.rng.random() < 0.3 else []
        words = [wrong, "or", *middle, "actually", "no" + comma, "let's", *verb.split()]
    else:
        reason = list(draft.rng.choice(_REASONS))
        words = [wrong + comma, "except", "uh", "well", *reason[:-1], reason[-1] + comma, "so", "actually", "let's", "just",
                 *verb.split()]
        token = draft.tokens[i]
        mark = token[len(token.rstrip(_EDGE_PUNCTUATION)) :]
        if mark and draft.free(i + 1 if i + 1 < len(draft.tokens) else i):
            draft.tokens[i] = token[: len(token) - len(mark)]
            draft.after[i] = ["instead" + mark]
    if draft.sentence_start(i):
        words[0] = _capitalize(words[0])
    draft.insert(i, words)
    draft.claim(max(0, i - 1), i, min(i + 1, len(draft.tokens) - 1))
    corrections.append((wrong, right, form))
    return True


def _pause(draft: _Draft) -> bool:
    """A period or question mark where the speaker paused, or a "?" after a list item."""
    listed = [
        i for i in range(1, len(draft.tokens) - 1)
        if draft.free(i, i + 1) and _is_plain_word(draft.tokens[i]) and draft.tokens[i].endswith(",") and not draft.tokens[i].endswith(",,")
        and _is_plain_word(draft.tokens[i + 1]) and draft.original[i - 1].endswith(",")
    ]
    mid = [
        i for i in range(2, len(draft.tokens) - 1)
        if draft.free(i, i + 1) and _bare_word(draft.tokens[i]) and _bare_word(draft.tokens[i + 1])
        and not draft.sentence_start(i) and not draft.sentence_start(i - 1)
    ]
    if listed and draft.rng.random() < 0.3:
        i = draft.rng.choice(listed)
        draft.tokens[i] = draft.tokens[i][:-1] + "?"
        capitalize = draft.rng.random() < 0.6
    elif mid:
        i = draft.rng.choice(mid)
        draft.tokens[i] += "?" if draft.rng.random() < 0.2 else "."
        capitalize = draft.rng.random() < 0.5
    else:
        return False
    following = draft.tokens[i + 1]
    if capitalize and following[:1].islower():
        draft.tokens[i + 1] = _capitalize(following)
    draft.claim(i, i + 1)
    return True


def _edit(kind: str, draft: _Draft, clean: str, corrections: list[tuple[str, str, str]]) -> bool:
    return {
        "filler": lambda: _filler(draft, cluster=False),
        "cluster": lambda: _filler(draft, cluster=True),
        "determiner": lambda: _determiner(draft),
        "repeat": lambda: _repeat(draft),
        "restart": lambda: _restart(draft),
        "fragment": lambda: _fragment(draft),
        "correction": lambda: _correction(draft, clean, corrections),
        "pause": lambda: _pause(draft),
    }[kind]()


def inject(
    clean: str, seed: str | int, style: str, *, kinds: Sequence[str] = KINDS, level: str | None = None
) -> Injection:
    """A spoken version of `clean`, the same for the same text, seed, style, kinds and level.

    `recognizer` keeps casing and punctuation; `raw` drops both, and has no pause punctuation.
    """
    rng = random.Random(f"inject-{seed}")
    level = level or rng.choices(list(_LEVEL_WEIGHTS), list(_LEVEL_WEIGHTS.values()))[0]
    commas = style == "recognizer" and rng.random() < COMMA_FILLERS
    tokens = clean.split()
    draft = _Draft(tokens, style, commas, rng)
    allowed = [kind for kind in kinds if not (kind == "pause" and style == "raw")]
    if not tokens or not allowed:
        return Injection(clean, level, "commas" if commas else "bare")
    edits = max(1, round(len(tokens) / LEVELS[level] * rng.uniform(0.7, 1.3)))
    max_corrections = max(1, len(tokens) // 60)
    applied_kinds: list[str] = []
    corrections: list[tuple[str, str, str]] = []
    # A kind with no place to go in this text is tried again as another kind.
    for _ in range(edits * 4):
        if len(applied_kinds) == edits:
            break
        choices = [kind for kind in allowed if kind != "correction" or len(corrections) < max_corrections]
        if not choices:
            break
        kind = rng.choices(choices, [_KIND_WEIGHTS[kind] for kind in choices])[0]
        applied = _edit(kind, draft, clean, corrections)
        if applied:
            applied_kinds.append(kind)
    text = draft.render()
    return Injection(text, level, "commas" if commas else "bare", tuple(applied_kinds), tuple(corrections))


_DENSIFY_KINDS = {"filler": 45, "cluster": 15, "determiner": 15, "repeat": 12, "pause": 13}
# Fillers per 100 words a densified input is brought up to, drawn per row. Real Parakeet dictation
# measured 4.5; the generator writes about 0.8.
DENSIFY_RATE = (3.0, 6.0)


def filler_count(text: str) -> int:
    return sum(1 for token in text.split() if FILLER_TOKEN.match(token))


def densify(spoken: str, seed: str | int, style: str, rate: float | None = None, pauses: bool = True) -> Injection:
    """`spoken` with more hesitation: fillers, clusters, "the uh the", repeated words and, in the
    recognizer style, pause punctuation (Parakeet's habit; `pauses=False` for other recognizers),
    until it has `rate` fillers per 100 words.

    For a generated transcript whose target is already written: every edit is one the target drops
    anyway, so the target stays right. The text's own words, casing and punctuation are kept.
    """
    rng = random.Random(f"densify-{seed}")
    rate = rate if rate is not None else rng.uniform(*DENSIFY_RATE)
    tokens = spoken.split()
    draft = _Draft(tokens, style, False, rng, restyle=False)
    fillers_needed = round(len(tokens) * rate / 100) - filler_count(spoken)
    kinds = [kind for kind in _DENSIFY_KINDS if not (kind == "pause" and (style == "raw" or not pauses))]
    applied_kinds: list[str] = []
    fillers_added = 0
    for _ in range(max(0, fillers_needed) * 6):
        if fillers_added >= fillers_needed:
            break
        kind = rng.choices(kinds, [_DENSIFY_KINDS[kind] for kind in kinds])[0]
        applied = _edit(kind, draft, spoken, [])
        if applied:
            applied_kinds.append(kind)
            fillers_added = sum(1 for words in draft.before.values() for word in words if FILLER_TOKEN.match(word))
    text = draft.render() if applied_kinds else spoken
    return Injection(text, "densified", "bare", tuple(applied_kinds))


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


def injected_rows(
    origins: Iterable[Example], count: int, seed: int, layouts: Sequence[str] = ("plain",)
) -> list[Example]:
    """Injects noise into `count` origins of the given layouts, alternating the two input styles.

    The input is one block of text, as a recognizer writes it; a "paragraphs" origin keeps its
    paragraphs in the target, as when a speaker dictates without layout commands.
    """
    rng = random.Random(seed)
    pool = sorted(
        (e for e in origins if e.layout in layouts and e.split != "test" and e.output != EMPTY), key=lambda e: e.id
    )
    rng.shuffle(pool)
    rows = []
    for index, origin in enumerate(pool[:count]):
        style = STYLES[index % len(STYLES)]
        injection = inject(origin.output, f"{seed}-{origin.id}", style)
        if injection.text == " ".join(origin.output.split()):
            continue
        rows.append(
            Example(
                id=example_id(SOURCE, injection.text),
                input=injection.text,
                output=origin.output,
                source=SOURCE,
                category="filler_injection",
                license=origin.license,
                tags=["injected", f"style_{style}", f"from:{origin.source}", *injection.tags],
                seed_group=origin.seed_group,
                teacher_model=origin.teacher_model,
                teacher_reasoning=origin.teacher_reasoning,
                generator_provider=origin.generator_provider,
                generator_model=origin.generator_model,
                generator_reasoning=origin.generator_reasoning,
                judge_model=origin.judge_model,
                judge_reasoning=origin.judge_reasoning,
            )
        )
    return rows


_EMPTY_GENERATION_WORDS = ("um", "uh", "er", "erm", "hmm", "umm", "uhh", "mm")
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
        fillers = [rng.choice(_EMPTY_GENERATION_WORDS) for _ in range(rng.randint(1, 4))]
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
                license=OWN_LICENSE,
                tags=["empty"],
            )
        )
    return rows
