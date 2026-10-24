import json

from whispernow_training.datagen.references import rejudge
from whispernow_training.datagen.schema import Example


def _row(id_, output, tags=()):
    return Example(
        id=id_, input="shopping eggs and milk", output=output, source="generated", category="c",
        license="x", tags=list(tags), teacher_model="original-model",
    )


def _write_pass(directory, name, records):
    rejudge.overrides_file(directory, name).write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )


def test_apply_overrides_in_order_without_restoring_dropped_rows(tmp_path):
    rows = [
        _row("fix", "Eggs and milk.", ["style_raw", "relabeled"]),
        _row("keep", "Eggs and milk."),
        _row("drop_later", "Eggs and milk."),
        _row("drop_first", "Eggs and milk."),
        _row("untouched", "Eggs and milk."),
    ]
    _write_pass(tmp_path, "rejudged", [
        {"id": "fix", "passed": True, "output": "- Eggs\n- Milk", "relabel_model": "first-model"},
        {"id": "keep", "passed": True, "output": "Eggs and milk.", "relabel_model": "first-model"},
        {"id": "drop_later", "passed": True, "output": "Eggs and milk."},
        {"id": "drop_first", "passed": False, "output": "Eggs and milk."},
    ])
    _write_pass(tmp_path, "strict", [
        {"id": "fix", "passed": True, "output": "Shopping:\n- Eggs\n- Milk", "relabel_model": "second-model"},
        {"id": "drop_later", "passed": False, "output": "Eggs and milk."},
        {"id": "drop_first", "passed": True, "output": "Restored."},
    ])

    applied = {e.id: e for e in rejudge.apply(rows, tmp_path)}
    assert set(applied) == {"fix", "keep", "untouched"}
    fixed = applied["fix"]
    assert fixed.output == "Shopping:\n- Eggs\n- Milk" and fixed.layout == "list"
    assert fixed.teacher_model == "second-model"
    assert {"style_raw", "rejudged", "strict", "relabeled"} <= set(fixed.tags)
    assert fixed.tags.count("relabeled") == 1
    assert applied["keep"].teacher_model == "original-model"
    assert applied["untouched"] is rows[-1]
    header_only = {e.id: e for e in rejudge.apply(rows, tmp_path, passes=("rejudged",))}
    assert set(header_only) == {"fix", "keep", "drop_later", "untouched"}
    assert header_only["fix"].output == "- Eggs\n- Milk"
    assert header_only["fix"].teacher_model == "first-model"


def test_latest_record_wins_and_judge_reference_keeps_its_model_credit(tmp_path):
    row = _row("rescued", "Eggs and milk.")
    _write_pass(tmp_path, "strict", [
        {"id": "rescued", "passed": False, "output": "Eggs and milk."},
        {"id": "rescued", "passed": True, "output": "Shopping:\n- Eggs\n- Milk", "teacher_model": "judge-model"},
    ])
    (applied,) = rejudge.apply([row], tmp_path)
    assert applied.output == "Shopping:\n- Eggs\n- Milk"
    assert applied.teacher_model == "judge-model"
    assert {"strict", "relabeled", "judge_model_reference"} <= set(applied.tags)
    assert rejudge.apply([row], tmp_path, passes=()) == [row]
