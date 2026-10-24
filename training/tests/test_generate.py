import asyncio
import json
import re

import pytest

from conftest import judge_items

from whispernow_training.datagen.synthetic import generate
from whispernow_training.datagen import paths
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
    monkeypatch.setattr(paths, "GENERATED", tmp_path)
    generator_requests = []

    def recorded_reply(messages, role):
        if role == "generator":
            generator_requests.append(messages[-1]["content"])
        return reply(messages, role)

    llm = fake_llm(recorded_reply)
    kept = asyncio.run(generate.generate_category(llm, "mixed_message", 6, batch_size=5, progress=lambda _: None))
    assert kept == 6
    rows = read_examples(tmp_path / "mixed_message.jsonl")
    assert len(rows) == 6 and len({r.input for r in rows}) == 6
    assert all(r.source == "teacher" and r.teacher_model == "teacher-model" and r.license == "MIT" for r in rows)
    assert all(r.layout == "list" for r in rows)
    log = list(read_jsonl(tmp_path / "mixed_message.batches.jsonl"))
    styles = [b["style"] for b in log]
    assert set(styles) <= set(generate.STYLES)
    rejected = list(read_jsonl(tmp_path / "mixed_message.rejected.jsonl"))
    assert any(r["failed"] == ["layout_intent"] for r in rejected)
    assert log[0]["notes"]
    assert log[0]["notes"][0] in generator_requests[1]

    calls = llm.calls
    assert asyncio.run(generate.generate_category(llm, "mixed_message", 6, batch_size=5, progress=lambda _: None)) == 6
    assert llm.calls == calls
    assert asyncio.run(generate.generate_category(llm, "mixed_message", 8, batch_size=5, progress=lambda _: None)) == 8
    assert len(read_examples(tmp_path / "mixed_message.jsonl")) == 8

    repeat = tmp_path / "repeat"
    monkeypatch.setattr(paths, "GENERATED", repeat)
    asyncio.run(generate.generate_category(fake_llm(reply), "mixed_message", 6, batch_size=5, progress=lambda _: None))
    assert [b["style"] for b in read_jsonl(repeat / "mixed_message.batches.jsonl")] == styles


def test_held_out_rows_are_pinned_to_test_and_new(fake_llm, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "GENERATED", tmp_path)
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


def test_run_rows_are_tagged_new_and_numbered_after_train_and_test(fake_llm, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "GENERATED", tmp_path)
    llm = fake_llm(reply)
    asyncio.run(generate.generate_category(llm, "mixed_message", 4, batch_size=5, progress=lambda _: None))
    asyncio.run(generate.generate_category(llm, "mixed_message", 4, batch_size=5, held_out=True, progress=lambda _: None))
    earlier = [*read_examples(tmp_path / "mixed_message.jsonl"), *read_examples(tmp_path / "mixed_message.test.jsonl")]
    kept = asyncio.run(generate.generate_category(llm, "mixed_message", 4, batch_size=5, run="round3", progress=lambda _: None))
    assert kept == 4
    rows = read_examples(tmp_path / "mixed_message.round3.jsonl")
    assert all("round3" in r.tags and r.split != "test" for r in rows)
    assert not {r.input for r in rows} & {r.input for r in earlier}
    first = sum(len(list(read_jsonl(tmp_path / f"mixed_message{stem}.batches.jsonl"))) for stem in ("", ".test"))
    assert all(int(r.input.split()[1]) >= first for r in rows)


def test_client_configuration_failure_preserves_batches_and_resumes_after_repair(fake_llm, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "GENERATED", tmp_path)
    generator_requests = []

    def malformed_then_valid(messages, role):
        if role == "generator":
            generator_requests.append(messages)
            if len(generator_requests) == 1:
                return '{"items": null}'
        return reply(messages, role)

    llm = fake_llm(malformed_then_valid)
    assert asyncio.run(generate.generate_category(llm, "mixed_message", 1, batch_size=1, progress=lambda _: None)) == 1
    log_path = tmp_path / "mixed_message.batches.jsonl"
    log = list(read_jsonl(log_path))
    assert len(log) == 2 and log[0]["error"].startswith("unreadable generator reply:")
    assert log[0]["returned"] == log[0]["passed"] == 0
    prior_files = {path: path.read_bytes() for path in tmp_path.glob("*.jsonl")}
    prior_rows = read_examples(tmp_path / "mixed_message.jsonl")
    failure = KeyError("LLM_API_KEY")
    failed_requests = []

    def unavailable(messages, role):
        failed_requests.append(messages)
        raise failure

    llm.reply = unavailable
    with pytest.raises(KeyError) as error:
        asyncio.run(generate.generate_category(llm, "mixed_message", 2, batch_size=1, progress=lambda _: None))
    assert error.value is failure and len(failed_requests) == 1
    assert {path: path.read_bytes() for path in tmp_path.glob("*.jsonl")} == prior_files

    llm.reply = malformed_then_valid
    assert asyncio.run(generate.generate_category(llm, "mixed_message", 2, batch_size=1, progress=lambda _: None)) == 2
    assert generator_requests[-1] == failed_requests[0]
    assert read_examples(tmp_path / "mixed_message.jsonl")[:len(prior_rows)] == prior_rows
    resumed_log = list(read_jsonl(log_path))
    assert resumed_log[:-1] == log and not resumed_log[-1]["error"]


def test_unreadable_model_replies_remain_bounded_completed_batches(fake_llm, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "GENERATED", tmp_path)
    llm = fake_llm(lambda messages, role: "no json here")
    assert asyncio.run(generate.generate_category(llm, "mixed_message", 1, batch_size=1, progress=lambda _: None)) == 0
    log = list(read_jsonl(tmp_path / "mixed_message.batches.jsonl"))
    assert llm.calls == len(log) == 5
    assert all(record["error"].startswith("unreadable generator reply:") for record in log)
    assert all(record["returned"] == record["passed"] == 0 for record in log)
    assert not read_examples(tmp_path / "mixed_message.jsonl")
    assert not list(read_jsonl(tmp_path / "mixed_message.rejected.jsonl"))
    assert asyncio.run(generate.generate_category(llm, "mixed_message", 1, batch_size=1, progress=lambda _: None)) == 0
    assert llm.calls == 5


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
