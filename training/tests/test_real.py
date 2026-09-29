import sqlite3

from whispernow_training.datagen import real


def test_reads_the_single_entry_schema_newest_first(tmp_path):
    path = tmp_path / "history.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE entries (id INTEGER PRIMARY KEY, created_at_ms INTEGER, raw_text TEXT, asr_model_id TEXT)")
        db.executemany(
            "INSERT INTO entries (created_at_ms, raw_text, asr_model_id) VALUES (?, ?, ?)",
            [(1, "Old one.", "m"), (2, "  ", "m"), (3, "New one.", None), (4, "Old one.", "m")],
        )
    assert real.read_history(path) == [
        {"input": "Old one.", "asr_model_id": "m"},
        {"input": "New one.", "asr_model_id": ""},
    ]


def test_reads_the_preferred_attempt_of_the_older_schema(tmp_path):
    path = tmp_path / "history.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE recordings (id INTEGER PRIMARY KEY, created_at_ms INTEGER, preferred_attempt_id INTEGER)")
        db.execute("CREATE TABLE attempts (id INTEGER PRIMARY KEY, recording_id INTEGER, ordinal INTEGER, raw_text TEXT, asr_model_id TEXT)")
        db.executemany("INSERT INTO recordings VALUES (?, ?, ?)", [(1, 10, None), (2, 20, 3)])
        db.executemany(
            "INSERT INTO attempts VALUES (?, ?, ?, ?, ?)",
            [(1, 1, 0, "First try.", "a"), (2, 1, 1, "Retry.", "b"), (3, 2, 0, "Preferred.", "c"), (4, 2, 1, "Other.", "c")],
        )
    assert [row["input"] for row in real.read_history(path)] == ["Preferred.", "Retry."]
