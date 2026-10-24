import asyncio
from dataclasses import replace

import pytest

from whispernow_training.datagen.dataset import round4_cutoffs_cues
from whispernow_training.datagen.schema import ROUND4_TAG, Example, read_examples, write_examples
from whispernow_training.datagen.speech import cutoff
from whispernow_training.datagen.synthetic import generate, second_sample


def row(id_, text, output, split="train", tags=()):
    return Example(id=id_, input=text, output=output, source="owner/src", category="c", license="MIT",
                   split=split, tags=list(tags))


def generated(id_, text, output):
    return replace(row(id_, text, output, tags=[ROUND4_TAG]), source=generate.SOURCE, category="literal_cue")


def test_every_row_gets_a_second_sample_and_one_that_keeps_other_words_is_dropped(tmp_path, monkeypatch):
    rows = [
        generated("a", "hold on to the receipts until thursday", "Hold on to the receipts until Thursday."),
        generated("b", "scratch the surface and it's the same bug", "Scratch the surface and it's the same bug."),
    ]
    seconds = {"a": "Hold on to the receipts until Thursday.", "b": "It's the same bug."}

    async def relabel(llm, transcript, attempt=0):
        return seconds[next(e.id for e in rows if e.input == transcript)]

    async def check(llm, kind, cases):
        return []

    monkeypatch.setattr(round4_cutoffs_cues, "CHECKS", tmp_path / "checks.jsonl")
    monkeypatch.setattr(generate, "output_file", lambda category, run="": tmp_path / "rows.jsonl")
    (tmp_path / "rows.jsonl").write_text("".join(e.to_json() + "\n" for e in rows))
    monkeypatch.setattr(second_sample, "relabel", relabel)
    monkeypatch.setattr(second_sample.review, "check", check)

    result = asyncio.run(round4_cutoffs_cues.check(None))
    assert result["two_sample_checked"] == 2 and result["drop"] == 1
    assert [e.id for e in round4_cutoffs_cues.load()] == ["a"]


def test_round4_swaps_cut_inputs_and_adds_each_categorys_first_target_of_new_rows(tmp_path, monkeypatch):
    base_dir, out_dir = tmp_path / "round3", tmp_path / "build"
    write_examples(base_dir / "train.jsonl", [
        row("a", "um we need to book the big room", "We need to book the big room."),
        row("b", "call the plumber", "Call the plumber."),
    ])
    write_examples(base_dir / "val.jsonl", [row("val", "so the plan", "The plan.", split="val")])
    held_out = "please hold on to the receipts until thursday so we can check"
    write_examples(base_dir / "test.jsonl", [row("test", held_out, "Hold on to the receipts.", split="test")])
    monkeypatch.setattr(round4_cutoffs_cues, "_benchmark_rows", lambda: [])
    record = {"id": "a", "kept": True, "tts": "phonon", "input": "um we need to book the big ro room",
              "cuts": [{"word": "room", "prefix": "ro", "piece": ["ro"]}]}
    texts = ["hold on to the receipts until Thursday", "never mind the parking we'll take the train",
             "can you scratch off the lottery ticket for me"]
    new = [generated(f"t{i}", text, text.capitalize() + ".") for i, text in enumerate(texts)]
    monkeypatch.setattr(cutoff, "ROWS", 1)
    monkeypatch.setattr(cutoff, "load", lambda: [record])
    monkeypatch.setattr(round4_cutoffs_cues, "load", lambda: new)
    monkeypatch.setattr(second_sample, "run_rows", lambda run, categories: new)
    monkeypatch.setitem(generate.RUN_TARGETS, ROUND4_TAG, {"literal_cue": 2})

    report = round4_cutoffs_cues.build(base_dir, out_dir)
    train = {e.id: e for e in read_examples(out_dir / "train.jsonl")}
    assert train["a"].input == record["input"] and train["a"].output == "We need to book the big room."
    assert {cutoff.TAG, "tts:phonon"} <= set(train["a"].tags) and train["b"].input == "call the plumber"
    assert {"t1", "t2"} <= set(train) and "t0" not in train
    assert report["round4"]["added"] == {"literal_cue": "2/2"} and report["round4"]["cutoff"]["cutoff_rows"] == "1/1"
    assert report["round4"]["dropped"] == {"shares 6 words with a held-out input": 1}
    assert (out_dir / "test.jsonl").read_text() == (base_dir / "test.jsonl").read_text()

    monkeypatch.setattr(cutoff, "ROWS", 2)
    with pytest.raises(SystemExit):
        round4_cutoffs_cues.build(base_dir, out_dir)
