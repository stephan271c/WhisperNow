import asyncio

from whispernow_training.datagen import generate, v3, v5
from whispernow_training.datagen.schema import Example


def generated(id_, text, output):
    return Example(id=id_, input=text, output=output, source=generate.SOURCE, category="literal_cue", license="MIT",
                   split="train", tags=["v5"])


def test_every_v5_row_gets_a_second_sample_and_one_that_keeps_other_words_is_dropped(tmp_path, monkeypatch):
    rows = [
        generated("a", "hold on to the receipts until thursday", "Hold on to the receipts until Thursday."),
        generated("b", "scratch the surface and it's the same bug", "Scratch the surface and it's the same bug."),
    ]
    seconds = {"a": "Hold on to the receipts until Thursday.", "b": "It's the same bug."}

    async def relabel(llm, transcript, attempt=0):
        return seconds[next(e.id for e in rows if e.input == transcript)]

    async def check(llm, kind, cases):
        return []

    monkeypatch.setattr(v5, "CHECKS", tmp_path / "checks.jsonl")
    monkeypatch.setattr(v5, "DIR", tmp_path)
    monkeypatch.setattr(v5, "stage3_rows", lambda: rows)
    monkeypatch.setattr(v3, "relabel", relabel)
    monkeypatch.setattr(v3.review, "check", check)

    result = asyncio.run(v5.check(None))
    assert result["two_sample_checked"] == 2 and result["drop"] == 1
    monkeypatch.setattr(generate, "output_file", lambda category, run="": tmp_path / "rows.jsonl")
    (tmp_path / "rows.jsonl").write_text("".join(e.to_json() + "\n" for e in rows))
    assert [e.id for e in v5.load()] == ["a"]
