"""What the teacher does with thinking aloud and mic checks (todo 110, step 3).

The hosted prompt says nothing about "let's see", "so yeah, I don't know" or "can you hear me?".
This asks the generator for about 30 transcripts that contain them, each with the exact marker
phrases, has the teacher clean them as the app would, and records per marker whether the reference
kept it. The result decides whether the judges accept the teacher's handling or such rows are kept
out of the benchmarks.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter

from . import paths
from .llm import LlmClient, parse_json_reply
from .teacher import relabel

OUT = paths.DATA / "audit" / "thinking.jsonl"
KINDS = {"thinking_aloud": 16, "mic_check": 10, "both": 6}

_PROMPT = """You write realistic dictation transcripts, as a speech recognizer (Parakeet) writes them: punctuated and cased, fillers bare ("so um we"), one block of text. Write {count} transcripts of 20 to 120 words in which the speaker {what}. Vary the topic (messages, notes, lists, plans, questions), the length, and where the phrase lands (start, middle, end). The rest of the transcript is ordinary dictated content a cleanup should keep.

Return ONLY a JSON object:
{{"items": [{{"input": "the transcript", "markers": ["each thinking-aloud or mic-check phrase exactly as it appears in the input"]}}]}}"""
_WHAT = {
    "thinking_aloud": 'thinks aloud once or twice, with phrases such as "let\'s see", "and now let\'s see", "let me think", "so yeah, I don\'t know", "what else", "hmm, what was it"',
    "mic_check": 'starts or restarts with a mic check such as "Hello? Can you hear me?", "testing, testing", "is this on?", "okay, is it recording?"',
    "both": "does a mic check at the start and thinks aloud later",
}


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z']+", text.lower()))


def kept(marker: str, reference: str) -> bool:
    return _norm(marker) in _norm(reference)


async def run(llm: LlmClient) -> list[dict]:
    rows: list[dict] = []
    for kind, count in KINDS.items():
        reply = await llm.complete([{"role": "user", "content": _PROMPT.format(count=count, what=_WHAT[kind])}])
        items = parse_json_reply(reply)["items"][:count]
        references = await asyncio.gather(*(relabel(llm, item["input"]) for item in items))
        for item, reference in zip(items, references, strict=True):
            markers = [m for m in item.get("markers", []) if m and _norm(m) in _norm(item["input"])]
            rows.append(
                {
                    "kind": kind,
                    "input": item["input"],
                    "reference": reference,
                    "markers": {marker: kept(marker, reference) for marker in markers},
                }
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return rows


def _mic(marker: str) -> bool:
    return bool(re.search(r"hear me|testing|is this on|recording|hello\??$|mic", marker, re.IGNORECASE))


def summary(rows: list[dict]) -> dict:
    counts: Counter = Counter()
    for row in rows:
        for marker, was_kept in row["markers"].items():
            group = "mic_check" if _mic(marker) else "thinking_aloud"
            counts[(group, "kept" if was_kept else "dropped")] += 1
    return {
        "transcripts": len(rows),
        **{f"{group}_{outcome}": counts[(group, outcome)] for group in ("thinking_aloud", "mic_check") for outcome in ("kept", "dropped")},
    }
