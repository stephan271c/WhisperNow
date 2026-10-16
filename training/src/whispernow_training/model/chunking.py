"""Cleaning a long transcript in pieces: each piece is its own request, and the replies are joined.
An experiment on whether length itself makes the local models drop facts."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Sequence

from ..datagen.schema import Example
from ..text_rules import CORRECTION

Produce = Callable[[Sequence[Example]], list[dict]]

_WORD = re.compile(r"\S+")
_CLOSERS = "\"'”’)\\]"
_SENTENCE_END = re.compile(rf"[.?!…][{_CLOSERS}]*$")
_CLAUSE_END = re.compile(rf"[,;:—–][{_CLOSERS}]*$|^[-—–]$")
# Unpunctuated speech (AMI turns) has no ends to cut at; a new thought often opens with one of these.
_OPENERS = {"so", "but", "okay", "anyway"}
# A cut this close to a correction cue can part the cue from what it corrects: the cue often comes a
# sentence before the new value ("No, the door opens at 8 … So make it 8:15 instead") or a word into
# a sentence ("So actually …"). Chosen by reading cuts in test-long.
_CORRECTION_BEFORE = 20
_CORRECTION_AFTER = 3


def _rank(text: str, words: list[re.Match[str]], cut: int) -> int:
    """How bad a cut before `words[cut]` is. A cut near a correction is the last resort."""
    before = text[words[max(0, cut - _CORRECTION_BEFORE)].start() : words[cut - 1].end()]
    after = text[words[cut].start() : words[min(cut + _CORRECTION_AFTER, len(words)) - 1].end()]
    if CORRECTION.search(before) or CORRECTION.search(after):
        return 3
    last = words[cut - 1].group()
    if _SENTENCE_END.search(last):
        return 0
    if _CLAUSE_END.search(last) or words[cut].group().casefold().strip(".,?!") in _OPENERS:
        return 1
    return 2


def split_transcript(text: str, max_words: int) -> list[str]:
    """`text` in pieces of `max_words // 2` to `max_words` words, or `text` itself when it has no more
    than `max_words`. Of all such splits it takes the one with the best cuts, then the fewest pieces,
    then the most even."""
    words = list(_WORD.finditer(text))
    if len(words) <= max_words:
        return [text.strip()]
    shortest = max(1, max_words // 2)
    ranks = [0] + [_rank(text, words, cut) for cut in range(1, len(words))] + [0]
    # best[end]: (summed cut ranks, pieces, summed squared piece lengths) of the best split of
    # words[:end], and where its last piece starts.
    best: list[tuple[tuple[int, int, int], int] | None] = [((0, 0, 0), 0)] + [None] * len(words)
    for end in range(shortest, len(words) + 1):
        for start in range(max(0, end - max_words), end - shortest + 1):
            if best[start] is None:
                continue
            summed, count, squares = best[start][0]
            option = ((summed + ranks[end], count + 1, squares + (end - start) ** 2), start)
            if best[end] is None or option < best[end]:
                best[end] = option
    pieces, end = [], len(words)
    while end:
        start = best[end][1]
        pieces.append(text[words[start].start() : words[end - 1].end()])
        end = start
    return pieces[::-1]


def join_replies(replies: Sequence[str]) -> str:
    """One paragraph break between pieces; a piece is about a paragraph of speech."""
    return "\n\n".join(reply.strip() for reply in replies if reply.strip())


def chunked(produce: Produce, max_words: int) -> Produce:
    """`produce`, sending each piece of a row over `max_words` words as a request of its own. Such a
    row's output joins the replies, and `pieces` keeps each piece's input and reply."""

    def produce_pieces(rows: Sequence[Example]) -> list[dict]:
        split = {row.id: split_transcript(row.input, max_words) for row in rows}
        requests = [
            dataclasses.replace(row, id=f"{row.id}#{index}", input=piece) if len(split[row.id]) > 1 else row
            for row in rows for index, piece in enumerate(split[row.id])
        ]
        replies = {reply["id"]: reply for reply in produce(requests)}
        outputs = []
        for row in rows:
            if len(split[row.id]) == 1:
                outputs.append(replies[row.id])
                continue
            parts = [replies[f"{row.id}#{index}"] for index in range(len(split[row.id]))]
            tokens = [part["tokens"] for part in parts]
            outputs.append({
                "id": row.id,
                "output": join_replies([part["output"] for part in parts]),
                "hit_limit": any(part["hit_limit"] for part in parts),
                "tokens": None if None in tokens else sum(tokens),
                "pieces": [{"input": piece} | {key: value for key, value in part.items() if key != "id"}
                           for piece, part in zip(split[row.id], parts, strict=True)],
            })
        return outputs

    return produce_pieces
