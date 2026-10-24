import json
from pathlib import Path

from whispernow_training.datagen.sources import aawaaz

FIXTURES = [json.loads(line) for line in (Path(__file__).parent / "fixtures" / "aawaaz_rows.jsonl").open()]


def fixture(category: str) -> dict:
    return next(row for row in FIXTURES if row["category"] == category)


def test_real_row_loses_backticks_but_keeps_the_code():
    output = aawaaz.normalize_output(fixture("technical_code")["output"])
    assert "`" not in output
    assert "I did pip install requests but" in output
    assert "maybe pip3 instead of just pip." in output


def test_real_row_keeps_asterisks_that_are_content():
    output = aawaaz.normalize_output(fixture("academic_research")["output"])
    assert "6-31G** basis set" in output


def test_real_meeting_notes_lose_bold_and_keep_structure():
    raw = fixture("meeting_notes")["output"]
    output = aawaaz.normalize_output(raw)
    assert "**" not in output
    assert output.startswith("Standup Meeting Notes\n\nAttendees:")
    assert "\nLogin Flow Redesign\n" in output
    assert "- Action item: James to have API endpoints ready by Thursday." in output


def test_real_list_passes_through_unchanged():
    raw = fixture("shopping_lists")["output"]
    assert aawaaz.normalize_output(raw) == raw


def test_headings_numbering_and_bullets_become_plain():
    raw = "## Banana Bread\n\n*Ingredients:*\n* 3 bananas\n• 1 cup flour\n\n1) Mash the bananas.\n2) Mix."
    assert aawaaz.normalize_output(raw) == (
        "Banana Bread\n\nIngredients:\n- 3 bananas\n- 1 cup flour\n\n1. Mash the bananas.\n2. Mix."
    )


def test_unfixable_rows_are_dropped():
    assert aawaaz.normalize_output("| a | b |\n|---|---|") is None
    assert aawaaz.normalize_output("See [the docs](https://example.com).") is None
    assert aawaaz.normalize_output('"A quoted reply."') is None


def test_import_drops_duplicate_inputs_and_holds_out_test_rows():
    rows = {"casual_conversation": [{"input": f"row {i}", "output": f"Row {i}."} for i in range(200)]}
    rows["casual_conversation"].append({"input": "row 0", "output": "Again."})
    examples, dropped = aawaaz.import_rows(rows)
    assert len(examples) == 200
    assert dropped["duplicate input"] == 1
    assert all(e.source == aawaaz.REPO and e.license == "MIT" for e in examples)
    assert 0 < sum(e.split == "test" for e in examples) < 20
