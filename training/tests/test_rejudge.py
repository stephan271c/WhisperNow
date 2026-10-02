import asyncio
import json

from conftest import judge_items

from whispernow_training.datagen import gates, rejudge
from whispernow_training.datagen.schema import Example, read_jsonl


def _row(id_, input_, output, tags=()):
    return Example(id=id_, input=input_, output=output, source="generated", category="c", license="x", tags=list(tags))


def test_affected_rows_have_a_list_a_label_or_a_spoken_header():
    assert rejudge.affected(_row("a", "we need some stuff", "We need:\n- eggs\n- milk"))
    assert rejudge.affected(_row("b", "okay grocery list eggs and milk", "Eggs and milk."))
    assert not rejudge.affected(_row("c", "tell Sam the meeting moved to 3", "Tell Sam the meeting moved to 3."))


def test_rejudge_keeps_relabels_or_drops_and_apply_uses_the_verdicts(fake_llm, tmp_path):
    rows = [
        _row("keep", "shopping eggs and milk", "Shopping:\n- Eggs\n- Milk", ["style_raw"]),
        _row("fix", "grocery list eggs and milk", "To-dos:\n- Eggs\n- Milk", ["style_raw"]),
        _row("drop", "errands bank and post office", "Pharmacy:\n- Bank\n- Post office", ["style_raw", "relabeled"]),
        _row("prose", "tell Sam hi", "Tell Sam hi.", ["style_raw"]),
    ]

    def reply(messages, role):
        if role == "judge":
            results = []
            for item in judge_items(messages):
                wrong = item["reference"].startswith(("To-dos", "Pharmacy"))
                results.append({"i": item["i"], **{c: True for c in gates.COMPOSE_CRITERIA}, "nothing_invented": not wrong, "note": "misnamed header" if wrong else ""})
            return json.dumps({"results": results})
        return "Pharmacy:\n- Bank\n- Post office" if "errands" in messages[-1]["content"] else "Grocery list:\n- Eggs\n- Milk"

    llm = fake_llm(reply)
    judge = gates.Judge(gates.COMPOSE, gates.COMPOSE_CRITERIA)
    counts = asyncio.run(rejudge.rejudge(llm, tmp_path, rows, judge))
    assert counts["kept"] == 1 and counts["relabeled"] == 1 and counts["dropped"] == 1 and counts["skipped"] == 1
    # The row relabelled before gets a fresh sample: teacher attempt 2.
    assert 2 in llm.attempts
    applied = {e.id: e for e in rejudge.apply(rows, tmp_path)}
    assert set(applied) == {"keep", "fix", "prose"}
    assert applied["fix"].output.startswith("Grocery list:") and {"rejudged", "relabeled"} <= set(applied["fix"].tags)
    assert "rejudged" not in applied["prose"].tags
    # A relabelled row is credited to the model that wrote its new reference; a kept one keeps its own.
    assert applied["fix"].teacher_model == "teacher-model" and applied["keep"].teacher_model is None
    calls = llm.calls
    asyncio.run(rejudge.rejudge(llm, tmp_path, rows, judge))
    assert llm.calls == calls and len(list(read_jsonl(rejudge.overrides_file(tmp_path)))) == 3


def test_strict_pass_checks_every_row_alone_relabels_freshly_and_applies_after_the_header_pass(fake_llm, tmp_path):
    rows = [
        _row("phone", "call me on oh seven double three", "Call me on 0733."),
        _row("late", "the drive is three hours, actually more like four", "The drive is 3 hours."),
    ]
    sizes = []

    def reply(messages, role):
        if role == "judge":
            items = judge_items(messages)
            sizes.append(len(items))
            results = []
            for item in items:
                wrong = item["reference"] == "The drive is 3 hours."
                results.append({"i": item["i"], **{c: True for c in gates.COMPLETENESS_CRITERIA}, "corrections_right": not wrong, "note": ""})
            return json.dumps({"results": results})
        return "The drive is more like 4 hours."

    llm = fake_llm(reply)
    judge = gates.Judge(max_items=1)
    counts = asyncio.run(
        rejudge.rejudge(llm, tmp_path, rows, judge, name="strict", select=lambda _: True, first_attempt=rejudge.STRICT_FIRST_ATTEMPT)
    )
    assert counts["kept"] == 1 and counts["relabeled"] == 1 and set(sizes) == {1}
    # Judge requests are attempt 0; the one relabel is teacher attempt 3.
    assert [a for a in llm.attempts if a] == [3]
    applied = {e.id: e for e in rejudge.apply(rows, tmp_path)}
    assert applied["late"].output == "The drive is more like 4 hours." and {"strict", "relabeled"} <= set(applied["late"].tags)
    assert "rejudged" not in applied["phone"].tags and "strict" in applied["phone"].tags


def test_rescue_has_the_judge_model_label_a_row_the_teacher_failed_twice(fake_llm, tmp_path):
    rows = [_row("late", "BP one twenty two over seventy eight sorry one thirty two", "BP 122/78, sorry, 132/78.")]

    def reply(messages, role):
        if "REFERENCE" in messages[0]["content"]:
            wrong = all(item["reference"] != "BP 132/78." for item in judge_items(messages))
            results = [{"i": 0, **{c: True for c in gates.COMPLETENESS_CRITERIA}, "corrections_right": not wrong, "note": ""}]
            return json.dumps({"results": results})
        # The teacher keeps both sides; the judge's model writes the corrected value.
        return "BP 132/78." if role == "judge" else "BP 122/78, sorry, 132/78."

    llm = fake_llm(reply)
    kwargs = dict(name="strict", select=lambda _: True, first_attempt=rejudge.STRICT_FIRST_ATTEMPT)
    counts = asyncio.run(rejudge.rejudge(llm, tmp_path, rows, gates.Judge(max_items=1), **kwargs))
    assert counts["dropped"] == 1
    assert rejudge.apply(rows, tmp_path) == []
    # A later run with rescue labels the dropped row with the judge's model.
    counts = asyncio.run(rejudge.rejudge(llm, tmp_path, rows, gates.Judge(max_items=1), rescue=True, **kwargs))
    assert counts["rescued"] == 1
    (applied,) = rejudge.apply(rows, tmp_path)
    assert applied.output == "BP 132/78." and applied.teacher_model == "judge-model"
    assert "judge_model_reference" in applied.tags
    calls = llm.calls
    asyncio.run(rejudge.rejudge(llm, tmp_path, rows, gates.Judge(max_items=1), rescue=True, **kwargs))
    assert llm.calls == calls
