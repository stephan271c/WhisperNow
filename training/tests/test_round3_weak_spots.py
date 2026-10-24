import asyncio
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from whispernow_training.datagen.dataset import round3_weak_spots
from whispernow_training.datagen.schema import ROUND2_TAG, ROUND3_TAG, Example, append_jsonl, read_examples
from whispernow_training.datagen.speech import heard, repair
from whispernow_training.datagen.synthetic import generate, second_sample


def test_a_repair_row_takes_the_heard_text_and_the_new_reference():
    origin = Example(
        id="handy-1", input="book a table at that thai place", output="Book a table at that Thai place.",
        source="owner/handy", category="dictation", license="MIT", split="train", tags=["style_raw", "short"],
    )
    record = {"id": "handy-1", "recognizer": "parakeet", "heard": "Book a table at that tie place.",
              "input": "Book a table at that thai place.", "fixes": [["tie", "thai"]]}
    settings = SimpleNamespace(model="t", reasoning="high", judge_model="j", judge_reasoning="high")

    row = repair.repair_row(SimpleNamespace(settings=settings), origin, record, "Book a table at that Thai place.", attempt=1)
    assert row.input == record["heard"] and row.output == "Book a table at that Thai place."
    assert heard.origin_id(row) == "handy-1" and row.source == heard.SOURCE
    assert {repair.TAG, ROUND3_TAG, "relabeled", "recognizer:parakeet"} <= set(row.tags)
    assert ROUND2_TAG not in row.tags and "style_raw" not in row.tags
    assert row.teacher_model == "t" and row.judge_model == "j"


def test_repair_labels_the_target_plus_spare_and_the_build_keeps_exactly_the_target(tmp_path, monkeypatch):
    pool = [(Example(id=f"o{i:02d}", input=f"said {i}", output=f"Said {i}.", source="s", category="c", license="MIT"),
             {"id": f"o{i:02d}", "heard": f"heard {i}"}) for i in range(40)]
    monkeypatch.setattr(repair, "ROWS_FILE", tmp_path / "repair.jsonl")
    monkeypatch.setattr(repair, "REJECTED_FILE", tmp_path / "repair.rejected.jsonl")
    monkeypatch.setattr(repair, "GROUP", 4)
    monkeypatch.setattr(repair, "ROWS", 6)
    monkeypatch.setattr(repair, "pool", lambda: pool)

    async def every_other_kept(llm, group):
        kept = [replace(origin, id=f"h{origin.id}", input=record["heard"], tags=[f"heard_from:{origin.id}", repair.TAG])
                for origin, record in group if int(origin.id[1:]) % 2 == 0]
        rejected = [{"id": origin.id, "reason": "gates: x"} for origin, _ in group if int(origin.id[1:]) % 2]
        append_jsonl(repair.ROWS_FILE, [e.to_json() for e in kept])
        append_jsonl(repair.REJECTED_FILE, [json.dumps(row) for row in rejected])
        return Counter(kept=len(kept), gates=len(rejected))

    monkeypatch.setattr(repair, "_label_group", every_other_kept)
    asyncio.run(repair.label(None))
    kept, done = repair._done()
    assert len(kept) >= repair.with_spare(6) == 7 and len(done) < len(pool)
    rows = read_examples(repair.ROWS_FILE)
    assert [heard.origin_id(e) for e in repair.capped(rows[::-1])] == ["o00", "o02", "o04", "o06", "o08", "o10"]


def test_top_up_replaces_the_rows_the_check_drops(monkeypatch):
    targets = {"list_edits": 5, "mixed_message": 3}
    generated = {"list_edits": 5, "mixed_message": 3}
    drops = {"list_edits": 2, "mixed_message": 0}
    asked = []

    async def check_rows(llm, rows, checks, second):
        return {}

    async def generate_category(llm, category, target, run=""):
        asked.append((category, target))
        generated[category] = target
        return target

    monkeypatch.setitem(generate.RUN_TARGETS, ROUND3_TAG, targets)
    monkeypatch.setattr(generate, "generate_category", generate_category)
    monkeypatch.setattr(generate, "output_file", lambda category, run="": Path(f"{category}.{run}.jsonl"))
    monkeypatch.setattr(second_sample, "run_rows", lambda run, categories: [])
    monkeypatch.setattr(second_sample, "check_rows", check_rows)
    monkeypatch.setattr(second_sample, "checked_run",
                        lambda run, categories, checks, step: {c: [None] * (generated[c] - drops[c]) for c in categories})
    monkeypatch.setattr(second_sample, "read_examples", lambda path: [None] * generated[path.name.split(".")[0]])

    # Targets 5 and 3 with a 2% spare are 6 and 4.
    assert asyncio.run(round3_weak_spots.top_up(None)) == {"list_edits": 6, "mixed_message": 4}
    assert asked == [("list_edits", 8), ("mixed_message", 4)]
