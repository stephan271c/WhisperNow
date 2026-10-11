import asyncio
import json
import re

from conftest import judge_items

from whispernow_training.datagen import generate, paths
from whispernow_training.datagen.schema import read_examples, read_jsonl


def reply(messages, role):
    user = messages[-1]["content"]
    if role == "judge":
        results = []
        for item in judge_items(messages):
            ok = "item 3 " not in item["transcript"]
            results.append({"i": item["i"], **{c: ok for c in generate.CRITERIA}, "note": "" if ok else "sounds written"})
        return json.dumps({"results": results})
    if role == "generator":
        batch = int(re.search(r"Batch (\d+)\.", user).group(1))
        count = int(re.search(r"Write (\d+) transcripts", user).group(1))
        items = [
            {"input": f"batch {batch} item {i} um we need milk and eggs", "layout": "list" if i % 2 == 0 else "prose", "tags": ["t"]}
            for i in range(count)
        ]
        items.append({"input": items[0]["input"], "layout": "list"})
        return json.dumps({"items": items})
    assert role == "teacher"
    index = int(re.search(r"item (\d+)", user).group(1))
    # The teacher lists every row, so odd rows (meant as prose) fail the layout check.
    return "We need:\n- Milk\n- Eggs" if index % 2 == 0 or index == 1 else "We need milk and eggs."


def test_generation_writes_passing_rows_logs_batches_and_resumes(fake_llm, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "STAGE3", tmp_path)
    llm = fake_llm(reply)
    kept = asyncio.run(generate.generate_category(llm, "mixed_message", 6, batch_size=5, progress=lambda _: None))
    assert kept == 6
    rows = read_examples(tmp_path / "mixed_message.jsonl")
    assert len(rows) == 6 and len({r.input for r in rows}) == 6
    assert all(r.source == "teacher" and r.teacher_model == "teacher-model" and r.license == "zai-output" for r in rows)
    assert all(r.layout == "list" for r in rows)
    log = list(read_jsonl(tmp_path / "mixed_message.batches.jsonl"))
    styles = [b["style"] for b in log]
    assert set(styles) <= set(generate.STYLES)
    rejected = list(read_jsonl(tmp_path / "mixed_message.rejected.jsonl"))
    assert any(r["failed"] == ["layout_intent"] for r in rejected)
    # The rejection notes reach the next batch's request.
    assert log[0]["notes"]
    second_request = generate.generator_messages("mixed_message", "raw", 5, 1, log[0]["notes"])
    assert log[0]["notes"][0] in second_request[1]["content"]

    calls = llm.calls
    assert asyncio.run(generate.generate_category(llm, "mixed_message", 6, batch_size=5, progress=lambda _: None)) == 6
    assert llm.calls == calls
    assert asyncio.run(generate.generate_category(llm, "mixed_message", 8, batch_size=5, progress=lambda _: None)) == 8
    assert len(read_examples(tmp_path / "mixed_message.jsonl")) == 8

    repeat = tmp_path / "repeat"
    monkeypatch.setattr(paths, "STAGE3", repeat)
    asyncio.run(generate.generate_category(fake_llm(reply), "mixed_message", 6, batch_size=5, progress=lambda _: None))
    assert [b["style"] for b in read_jsonl(repeat / "mixed_message.batches.jsonl")] == styles


def test_held_out_rows_are_pinned_to_test_and_new(fake_llm, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "STAGE3", tmp_path)
    llm = fake_llm(reply)
    asyncio.run(generate.generate_category(llm, "mixed_message", 4, batch_size=5, progress=lambda _: None))
    trained = read_examples(tmp_path / "mixed_message.jsonl")
    kept = asyncio.run(
        generate.generate_category(llm, "mixed_message", 4, batch_size=5, held_out=True, progress=lambda _: None)
    )
    assert kept == 4
    held_out = read_examples(tmp_path / "mixed_message.test.jsonl")
    assert all(r.split == "test" for r in held_out)
    assert not {r.input for r in held_out} & {r.input for r in trained}
    # Held-out batches are numbered after the training batches.
    first = len(list(read_jsonl(tmp_path / "mixed_message.batches.jsonl")))
    assert all(int(r.input.split()[1]) >= first for r in held_out)
    assert (tmp_path / "mixed_message.test.batches.jsonl").exists()


def test_generator_requests_are_stable_per_batch():
    first = generate.generator_messages("list_edits", "raw", 50, 3, [])
    assert first == generate.generator_messages("list_edits", "raw", 50, 3, [])
    assert first != generate.generator_messages("list_edits", "raw", 50, 4, [])


def test_every_category_has_guidance():
    for category in generate.TARGETS:
        assert (paths.PROMPTS / "categories" / f"{category}.md").exists()


def test_items_survive_one_malformed_item():
    reply = (
        '{"items": [{"input": "get milk and \\"eggs\\"", "layout": "list", "tags": ["a"]}, '
        '{"input": "hey kofi", "layout": "mixed", "tags": "x".split() + ["y"]}]}'
    )
    assert generate.parse_items(reply, 5) == [
        {"input": 'get milk and "eggs"', "layout": "list", "tags": ["a"], "evidence": []},
        {"input": "hey kofi", "layout": "mixed", "tags": [], "evidence": []},
    ]


def test_items_salvage_the_well_formed_objects_and_evidence_of_a_broken_reply():
    reply = '{"items": [{"input": "we uh um go with {blue}", "evidence": ["uh um"], "layout": "prose"}, {"input": "bad "quote" here", "evidence": []}]}'
    parsed = generate.parse_items(reply, 10)
    assert [item["input"] for item in parsed] == ["we uh um go with {blue}"]
    assert parsed[0]["evidence"] == ["uh um"]
