"""Step 9: a real-dictation test set from the local history.

This is personal data. It is written under `data/real/` (gitignored), the build
never reads it, and `publish` never uploads it. The database is opened read-only.
The teacher's draft goes in `draft`; the hand-written reference goes in `reference`.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from pathlib import Path

from . import paths
from .llm import LlmClient
from .schema import example_id, read_jsonl
from .teacher import relabel

OUT = paths.REAL / "real.jsonl"


def default_history() -> Path:
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    return data_home / "WhisperNow" / "history.sqlite3"


def read_history(path: Path) -> list[dict]:
    """Every non-empty raw transcript, newest first, from either history schema."""
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if "entries" in tables:
            query = "SELECT raw_text, asr_model_id FROM entries ORDER BY created_at_ms DESC, id DESC"
        else:
            # The schema before single-entry history: one row per attempt, several per recording.
            query = (
                "SELECT a.raw_text, a.asr_model_id FROM recordings r "
                "JOIN attempts a ON a.id = COALESCE(r.preferred_attempt_id, "
                "(SELECT id FROM attempts WHERE recording_id = r.id ORDER BY ordinal DESC LIMIT 1)) "
                "ORDER BY r.created_at_ms DESC, r.id DESC"
            )
        rows = connection.execute(query).fetchall()
    finally:
        connection.close()
    seen, transcripts = set(), []
    for raw_text, model in rows:
        text = (raw_text or "").strip()
        if text and text not in seen:
            seen.add(text)
            transcripts.append({"input": text, "asr_model_id": model or ""})
    return transcripts


async def export_real(llm: LlmClient | None, history: Path, limit: int) -> int:
    """Adds new transcripts to `data/real/real.jsonl`, keeping any references already written."""
    existing = {row["input"]: row for row in read_jsonl(OUT)} if OUT.exists() else {}
    rows = list(existing.values())
    new = [t for t in read_history(history) if t["input"] not in existing][: max(0, limit - len(rows))]
    drafts = await asyncio.gather(*(relabel(llm, t["input"]) for t in new)) if llm else [""] * len(new)
    for transcript, draft in zip(new, drafts, strict=True):
        rows.append(
            {
                "id": example_id("real", transcript["input"]),
                **transcript,
                "draft": draft,
                "reference": "",
                "source": "real",
                "license": "personal, never upload",
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return len(rows)
