import asyncio
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from whispernow_training.datagen import heard, v3
from whispernow_training.datagen.schema import V2_TAG, V3_TAG, Example, append_jsonl


def test_a_repair_row_takes_the_heard_text_and_the_new_reference():
    origin = Example(
        id="handy-1", input="book a table at that thai place", output="Book a table at that Thai place.",
        source="owner/handy", category="dictation", license="MIT", split="train", tags=["style_raw", "short"],
    )
    record = {"id": "handy-1", "recognizer": "parakeet", "heard": "Book a table at that tie place.",
              "input": "Book a table at that thai place.", "fixes": [["tie", "thai"]]}
    settings = SimpleNamespace(model="t", reasoning="high", judge_model="j", judge_reasoning="high")

    row = v3.repair_row(SimpleNamespace(settings=settings), origin, record, "Book a table at that Thai place.", attempt=1)
    assert row.input == record["heard"] and row.output == "Book a table at that Thai place."
    assert heard.origin_id(row) == "handy-1" and row.source == heard.SOURCE
    assert {v3.REPAIR_TAG, V3_TAG, "relabeled", "recognizer:parakeet"} <= set(row.tags)
    assert V2_TAG not in row.tags and "style_raw" not in row.tags
    assert row.teacher_model == "t" and row.judge_model == "j"


def test_repair_labels_the_target_plus_spare_and_the_build_keeps_exactly_the_target(tmp_path, monkeypatch):
    pool = [(Example(id=f"o{i:02d}", input=f"said {i}", output=f"Said {i}.", source="s", category="c", license="MIT"),
             {"id": f"o{i:02d}", "heard": f"heard {i}"}) for i in range(40)]
    monkeypatch.setattr(v3, "DIR", tmp_path)
    monkeypatch.setattr(v3, "GROUP", 4)
    monkeypatch.setattr(v3, "REPAIR_ROWS", 6)
    monkeypatch.setattr(v3, "repair_pool", lambda: pool)

    async def every_other_kept(llm, group):
        kept = [replace(origin, id=f"h{origin.id}", input=record["heard"], tags=[f"heard_from:{origin.id}", v3.REPAIR_TAG])
                for origin, record in group if int(origin.id[1:]) % 2 == 0]
        rejected = [{"id": origin.id, "reason": "gates: x"} for origin, _ in group if int(origin.id[1:]) % 2]
        append_jsonl(v3.rows_file(v3.REPAIR), [e.to_json() for e in kept])
        append_jsonl(v3.rejected_file(v3.REPAIR), [json.dumps(row) for row in rejected])
        return Counter(kept=len(kept), gates=len(rejected))

    monkeypatch.setattr(v3, "_repair_group", every_other_kept)
    asyncio.run(v3.repair(None))
    kept, done = v3._repair_done()
    assert len(kept) >= v3.with_spare(6) == 7 and len(done) < len(pool)
    rows = v3.read_examples(v3.rows_file(v3.REPAIR))
    assert [heard.origin_id(e) for e in v3.capped(rows[::-1])] == ["o00", "o02", "o04", "o06", "o08", "o10"]


def test_top_up_replaces_the_rows_the_check_drops(monkeypatch):
    targets = {"list_edits": 5, "mixed_message": 3}
    generated = {"list_edits": 5, "mixed_message": 3}
    drops = {"list_edits": 2, "mixed_message": 0}
    asked = []

    async def check(llm):
        return {}

    async def generate_category(llm, category, target, run=""):
        asked.append((category, target))
        generated[category] = target
        return target

    monkeypatch.setattr(v3.generate, "RUN_TARGETS", {V3_TAG: targets})
    monkeypatch.setattr(v3, "check", check)
    monkeypatch.setattr(v3.generate, "generate_category", generate_category)
    monkeypatch.setattr(v3, "checked_stage3", lambda: {c: [None] * (n - drops[c]) for c, n in generated.items()})
    monkeypatch.setattr(v3, "read_examples", lambda path: [None] * generated[path.name.split(".")[0]])
    monkeypatch.setattr(v3.generate, "output_file", lambda category, run="": Path(f"{category}.{run}.jsonl"))

    # Targets 5 and 3 with a 2% spare are 6 and 4.
    assert asyncio.run(v3.top_up(None)) == {"list_edits": 6, "mixed_message": 4}
    assert asked == [("list_edits", 8), ("mixed_message", 4)]
