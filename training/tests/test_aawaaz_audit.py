import asyncio
import json

from conftest import judge_items

from whispernow_training.datagen.sources import aawaaz_audit
from whispernow_training.datagen.schema import Example, read_examples, write_examples


def _row(name, category, output, source_input=None):
    return Example(
        id=name,
        input=source_input or f"so um {name}",
        output=output,
        source="aawaaz",
        category=category,
        license="MIT",
    )


def _reply(messages, role):
    """Teacher: a proper sentence, except for hopeless rows. Judges: fail notes and 'BAD' outputs."""
    if role == "teacher":
        return "Still BAD." if "hopeless" in messages[-1]["content"] else "I went to the store."
    register = "register_kept" in messages[0]["content"]
    results = []
    for item in judge_items(messages):
        if register:
            results.append({"i": item["i"], "register_kept": not item["output"].startswith("Went"), "note": "notes"})
        else:
            ok = "BAD" not in item["output"]
            results.append(
                {"i": item["i"], **{c: ok for c in aawaaz_audit.CRITERIA}, "note": "" if ok else "bad"}
            )
    return json.dumps({"results": results})


def test_fix_flagged_combines_the_audit_and_register_judges(tmp_path, monkeypatch, fake_llm):
    for name in ("IMPORTED", "JUDGMENTS", "RELABEL_JUDGMENTS", "RELABELED", "CLEAN", "REPORT"):
        monkeypatch.setattr(aawaaz_audit, name, tmp_path / getattr(aawaaz_audit, name).name)
    monkeypatch.setattr(aawaaz_audit, "REGISTER_JUDGMENTS", tmp_path / "register.jsonl")
    monkeypatch.setattr(aawaaz_audit, "RELABEL_REGISTER_JUDGMENTS", tmp_path / "relabel_register.jsonl")
    rows = [
        _row("notes", "casual_conversation", "Went to the store."),  # register fails, relabel fixes it
        _row("fine", "legal_contract", "The Defendant agrees."),  # passes both
        _row("broken", "technical_code", "BAD output", "hopeless case"),  # relabel still fails
        _row("code", "technical_code", "Run the tests."),  # no register check
    ]
    write_examples(aawaaz_audit.IMPORTED, rows)
    llm = fake_llm(_reply)

    async def run():
        await aawaaz_audit.judge_aawaaz(llm, rows)
        register = await aawaaz_audit.judge_register(llm, rows)
        assert set(register) == {"notes", "fine"}
        return await aawaaz_audit.fix_flagged(llm)

    report = asyncio.run(run())
    kept = {e.id: e for e in read_examples(aawaaz_audit.CLEAN)}
    assert set(kept) == {"notes", "fine", "code"}
    assert kept["notes"].output == "I went to the store." and "relabeled" in kept["notes"].tags
    assert kept["fine"].output == "The Defendant agrees."
    assert (report["flagged"], report["fixed"], report["dropped"]) == (2, 1, 1)
    assert report["failed_criteria"]["register_kept"] == 1
