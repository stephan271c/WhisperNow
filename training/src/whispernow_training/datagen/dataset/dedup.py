"""Exact and near-duplicate inputs, and outputs repeated too often."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np

from ...text_rules import FILLER
from ..schema import EMPTY, ROUND2_TAG, ROUND3_TAG, ROUND4_TAG, Example
from ..sources import aawaaz, public
from ..synthetic import generate, inject

NEAR_DUPLICATE_JACCARD = 0.85
MAX_SAME_OUTPUT = 3
MINHASH_SEED = 108

# Earlier sources win when two rows are duplicates.
SOURCE_ORDER = (aawaaz.REPO, public.HANDY, public.DISFL_QA, public.SOTTO, generate.SOURCE, inject.SOURCE)
_LATER_ROUNDS = ((ROUND4_TAG, 4), (ROUND3_TAG, 3), (ROUND2_TAG, 2))


def _source_rank(example: Example) -> tuple[int, int]:
    # Later rounds' rows come last, so a duplicate check never drops an earlier row for one.
    for tag, round_ in _LATER_ROUNDS:
        if tag in example.tags:
            return (round_, 0)
    rank = SOURCE_ORDER.index(example.source) if example.source in SOURCE_ORDER else len(SOURCE_ORDER)
    return (0 if example.split == "test" else 1, rank)


_PRIME = (1 << 61) - 1
_PERMUTATIONS = 128
_BANDS = 16


def _shingles(text: str) -> set[str]:
    tokens = [w for w in public.words(text) if not FILLER.fullmatch(w)]
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
    rng = np.random.default_rng(MINHASH_SEED)
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
        key = public.skeleton(example.output)
        same_output[key] += 1
        if same_output[key] > MAX_SAME_OUTPUT and example.output != EMPTY:
            dropped.append((example, "output repeated too often"))
        else:
            final.append(example)
    return final, dropped
