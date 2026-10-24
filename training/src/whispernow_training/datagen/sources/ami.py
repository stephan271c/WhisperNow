"""Real spontaneous speech for the benchmarks: long single-speaker turns from AMI.

The AMI Meeting Corpus (`edinburghcstr/ami`, CC BY 4.0) has hand-made, word-level transcripts of
meetings, fillers and false starts included, split into short per-speaker segments. A turn here is
one speaker's consecutive segments in a meeting, read through the other speakers' backchannels
("yeah", "mm-hmm") and ended by anything longer from someone else or a long silence. Turns of 150
words or more go to the long benchmark only, never into training.

Only the text is fetched, through the Hugging Face datasets-server; no audio is downloaded. AMI
writes words in upper case without punctuation, so a turn becomes a "raw" style transcript: lower
case, with "I" and its contractions capitalized.

Attribution: AMI Meeting Corpus, University of Edinburgh and partners, CC BY 4.0,
https://groups.inf.ed.ac.uk/ami/corpus/
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from collections.abc import Iterable

import httpx

from .. import paths
from ..schema import Example, example_id

REPO = "edinburghcstr/ami"
LICENSE = "CC-BY-4.0"
SOURCE = "ami"
CACHE = paths.CACHE / "ami"
PAGE = 100
# Segments from others this short are backchannels the speaker talks through.
BACKCHANNEL_WORDS = 3
MAX_GAP_SECONDS = 4.0
_ROWS_URL = "https://datasets-server.huggingface.co/rows"


def fetch(split: str, config: str = "ihm") -> list[dict]:
    """Every segment's text and timing in one split, cached under `.cache/ami/`; resumes a partial fetch."""
    out = CACHE / f"{config}.{split}.jsonl"
    if out.exists():
        return [json.loads(line) for line in out.open(encoding="utf-8")]
    partial = out.with_suffix(".partial")
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in partial.open(encoding="utf-8")] if partial.exists() else []
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
    with httpx.Client(timeout=60, headers=headers) as client, partial.open("a", encoding="utf-8") as sink:
        total = None
        while total is None or len(rows) < total:
            for attempt in range(8):
                response = client.get(
                    _ROWS_URL, params={"dataset": REPO, "config": config, "split": split, "offset": len(rows), "length": PAGE}
                )
                if response.status_code == 200:
                    break
                # The datasets-server rate-limits bursts; back off for up to several minutes.
                time.sleep(min(300, 15 * 2**attempt))
            response.raise_for_status()
            page = response.json()
            total = page["num_rows_total"]
            if not page["rows"]:
                break
            for item in page["rows"]:
                row = item["row"]
                row = {key: row[key] for key in ("meeting_id", "audio_id", "speaker_id", "begin_time", "end_time", "text")}
                rows.append(row)
                sink.write(json.dumps(row) + "\n")
            sink.flush()
            time.sleep(1.0)
    partial.rename(out)
    return rows


def turns(segments: Iterable[dict]) -> list[dict]:
    """Each speaker's runs of speech, read through other speakers' backchannels."""
    meetings: dict[str, list[dict]] = defaultdict(list)
    for segment in segments:
        meetings[segment["meeting_id"]].append(segment)
    out = []
    for meeting, rows in sorted(meetings.items()):
        rows.sort(key=lambda r: (r["begin_time"], r["end_time"]))
        current: dict | None = None
        for row in rows:
            words = row["text"].split()
            if current and row["speaker_id"] == current["speaker"]:
                if row["begin_time"] - current["end"] > MAX_GAP_SECONDS:
                    out.append(current)
                    current = None
                else:
                    current["texts"].append(row["text"])
                    current["end"] = max(current["end"], row["end_time"])
                    current["segments"] += 1
                    continue
            elif current and len(words) <= BACKCHANNEL_WORDS:
                continue
            elif current:
                out.append(current)
                current = None
            current = {
                "meeting": meeting, "speaker": row["speaker_id"], "begin": row["begin_time"], "end": row["end_time"],
                "texts": [row["text"]], "segments": 1,
            }
        if current:
            out.append(current)
    for turn in out:
        turn["text"] = raw_text(" ".join(turn.pop("texts")))
        turn["words"] = len(turn["text"].split())
    return out


_SPELLED = re.compile(r"\b(?:[a-z]\. ?){2,}")


def raw_text(upper: str) -> str:
    """AMI's upper-case words as a raw-style transcript; letters it spells out ("T. V.") run together ("tv")."""
    text = upper.lower()
    text = _SPELLED.sub(lambda m: m.group(0).replace(".", "").replace(" ", "") + " ", text).strip()
    text = re.sub(r" +", " ", text)
    return re.sub(r"\bi(?=\b|')", "I", text)


def long_turns(min_words: int = 150, max_words: int = 1000, splits: tuple[str, ...] = ("test", "validation")) -> list[Example]:
    """Turns of `min_words` to `max_words` words as benchmark candidates, without references yet."""
    rows = []
    for split in splits:
        for turn in turns(fetch(split)):
            if not min_words <= turn["words"] <= max_words:
                continue
            rows.append(
                Example(
                    id=example_id(SOURCE, turn["text"]),
                    input=turn["text"],
                    output="",
                    source=SOURCE,
                    category="real_meeting_turn",
                    license=LICENSE,
                    tags=["real_speech", "style_raw", f"ami:{turn['meeting']}", f"ami_split:{split}"],
                )
            )
    return rows
