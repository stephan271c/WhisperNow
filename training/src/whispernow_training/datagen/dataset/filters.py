"""The build's filters: a row is dropped for the first reason `filter_reason` finds."""

from __future__ import annotations

import re
import unicodedata

from ...text_rules import (
    CASE_SENSITIVE, CODE_TOKEN, CORRECTION, FILLER, TRAILING_PUNCTUATION, assistant_opener, has_markdown, malformed_list,
    protected_tokens,
)
from ..schema import EMPTY, Example
from ..sources import public


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
        return len(public.words(example.input)) > 6
    source_words = len([w for w in public.words(example.input) if not FILLER.fullmatch(w)])
    output_words = len(public.words(example.output))
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


def split_by_filters(rows: list[Example]) -> tuple[list[Example], list[tuple[Example, str]]]:
    """The rows every filter passes, and each other row with its reason."""
    kept, dropped = [], []
    for example in rows:
        reason = filter_reason(example)
        if reason:
            dropped.append((example, reason))
        else:
            kept.append(example)
    return kept, dropped
