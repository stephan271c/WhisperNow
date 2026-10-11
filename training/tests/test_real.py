import sqlite3

from whispernow_training.datagen import real


def test_reads_the_single_entry_schema_newest_first(tmp_path):
    path = tmp_path / "history.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE entries (id INTEGER PRIMARY KEY, created_at_ms INTEGER, raw_text TEXT, asr_model_id TEXT)")
        db.executemany(
            "INSERT INTO entries (created_at_ms, raw_text, asr_model_id) VALUES (?, ?, ?)",
            [(1, "Old one.", "m"), (2, "  ", "m"), (2, None, "m"), (3, "New one.", None), (4, "Old one.", "m")],
        )
    assert real.read_history(path) == [
        {"input": "Old one.", "asr_model_id": "m"},
        {"input": "New one.", "asr_model_id": ""},
    ]
