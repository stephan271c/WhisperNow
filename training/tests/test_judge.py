import asyncio
import json
from dataclasses import replace

import pytest

from conftest import judge_items

from whispernow_training.datagen.judge import judge_all, judge_batch, size_batches
from whispernow_training.datagen.llm import LlmError
from whispernow_training.datagen.schema import read_jsonl

CRITERIA = ("content_preserved", "not_answered")


def verdict_reply(messages, role):
    assert role == "judge"
    results = []
    for item in judge_items(messages):
        ok = "bad" not in item["output"]
        results.append({"i": item["i"], "content_preserved": ok, "not_answered": True, "note": "" if ok else "dropped"})
    return json.dumps({"results": results})


def test_judge_batch_maps_verdicts_to_ids(fake_llm):
    llm = fake_llm(verdict_reply)
    pairs = [("a", {"transcript": "t", "output": "fine"}), ("b", {"transcript": "t", "output": "bad"})]
    verdicts = asyncio.run(judge_batch(llm, "rules", CRITERIA, pairs))
    assert [(v.id, v.passed, v.failed) for v in verdicts] == [("a", True, []), ("b", False, ["content_preserved"])]
    assert verdicts[1].judge_model == "judge-model"


def test_judge_batch_splits_a_malformed_reply(fake_llm):
    def reply(messages, role):
        items = judge_items(messages)
        return "garbage" if len(items) > 1 else verdict_reply(messages, role)

    llm = fake_llm(reply)
    pairs = [(str(i), {"transcript": "t", "output": "fine"}) for i in range(4)]
    verdicts = asyncio.run(judge_batch(llm, "rules", CRITERIA, pairs))
    assert all(v.passed for v in verdicts) and len(verdicts) == 4
    assert llm.calls == 7


def test_judge_all_resumes_from_its_file(fake_llm, tmp_path):
    rules = tmp_path / "rules.md"
    rules.write_text("rules")
    out = tmp_path / "verdicts.jsonl"
    items = [(str(i), {"transcript": "t", "output": "fine"}) for i in range(5)]
    llm = fake_llm(verdict_reply)
    asyncio.run(judge_all(llm, items[:3], out, instructions_file=rules, criteria=CRITERIA, batch_size=2, progress=lambda _: None))
    calls = llm.calls
    verdicts = asyncio.run(judge_all(llm, items, out, instructions_file=rules, criteria=CRITERIA, batch_size=2, progress=lambda _: None))
    assert len(verdicts) == 5
    assert llm.calls == calls + 1


@pytest.mark.parametrize("failure", [
    ValueError("invalid client configuration"),
    LlmError("transport retries exhausted", retryable=True),
], ids=["configuration", "exhausted-transport"])
def test_client_failures_preserve_completed_judgments_and_leave_the_rest_pending(failure, fake_llm, tmp_path):
    rules = tmp_path / "rules.md"
    rules.write_text("rules")
    out = tmp_path / "verdicts.jsonl"
    items = [(id_, {"transcript": id_, "output": "fine"}) for id_ in ("a", "b", "c", "d")]

    def reply(messages, role):
        if any(item["transcript"] == "c" for item in judge_items(messages)):
            raise failure
        return verdict_reply(messages, role)

    llm = fake_llm(reply)
    with pytest.raises(type(failure)) as error:
        asyncio.run(judge_all(llm, items, out, instructions_file=rules, criteria=CRITERIA, batch_size=2))
    assert error.value is failure and llm.calls == 2
    saved = list(read_jsonl(out))
    assert [row["id"] for row in saved] == ["a", "b"] and all(row["passed"] for row in saved)

    sent = []

    def resumed_reply(messages, role):
        sent.extend(judge_items(messages))
        return verdict_reply(messages, role)

    resumed = fake_llm(resumed_reply)
    current = asyncio.run(judge_all(resumed, items, out, instructions_file=rules, criteria=CRITERIA, batch_size=2))
    assert [item["transcript"] for item in sent] == ["c", "d"] and resumed.calls == 1
    assert all(verdict.passed for verdict in current.values())
    assert list(read_jsonl(out))[:2] == saved and len(list(read_jsonl(out))) == 4


def test_judge_all_rejudges_only_changed_same_id_content(fake_llm, tmp_path):
    rules = tmp_path / "rules.md"
    rules.write_text("rules")
    out = tmp_path / "verdicts.jsonl"
    items = [(id_, {"transcript": "original transcript", "reference": "original reference", "output": "fine"})
             for id_ in ("a", "b")]
    first = asyncio.run(judge_all(fake_llm(verdict_reply), items, out, instructions_file=rules, criteria=CRITERIA))
    items[0][1]["output"] = "bad new content"
    sent = []

    def reply(messages, role):
        sent.extend(judge_items(messages))
        return verdict_reply(messages, role)

    llm = fake_llm(reply)
    current = asyncio.run(judge_all(llm, items, out, instructions_file=rules, criteria=CRITERIA))
    assert len(sent) == 1 and sent[0]["output"] == "bad new content"
    assert current["b"] == first["b"] and current["a"].request_key != first["a"].request_key
    assert not current["a"].passed
    assert len(out.read_text().splitlines()) == 3
    calls = llm.calls
    assert asyncio.run(judge_all(llm, items, out, instructions_file=rules, criteria=CRITERIA)) == current
    assert llm.calls == calls


@pytest.mark.parametrize("change", ["instructions", "criteria", "fault_kinds", "model"])
def test_judge_all_invalidates_changed_contract(change, fake_llm, tmp_path):
    rules = tmp_path / "rules.md"
    rules.write_text("rules")
    out = tmp_path / "verdicts.jsonl"
    items = [("a", {"transcript": "a transcript", "reference": "a reference", "output": "fine"})]
    llm = fake_llm(lambda messages, role: json.dumps({"results": [
        {"i": 0, "content_preserved": True, "not_answered": True, "quality": False,
         "faults": [{"kind": "dropped", "fact": "a missing fact"}]},
    ]}))
    original = asyncio.run(judge_all(llm, items, out, instructions_file=rules, criteria=CRITERIA))["a"]
    criteria, faults = CRITERIA, None
    if change == "instructions":
        rules.write_text("updated rules")
    elif change == "criteria":
        criteria = (*CRITERIA, "quality")
    elif change == "fault_kinds":
        faults = {"dropped": "content_preserved"}
    else:
        llm.settings = replace(llm.settings, judge_model="new-model")
    current = asyncio.run(judge_all(llm, items, out, instructions_file=rules, criteria=criteria, fault_kinds=faults))["a"]
    assert llm.calls == 2 and current.request_key != original.request_key
    if change in ("criteria", "fault_kinds"):
        assert not current.passed and original.passed


def test_judge_errors_remain_pending(fake_llm, tmp_path):
    rules = tmp_path / "rules.md"
    rules.write_text("rules")
    out = tmp_path / "verdicts.jsonl"
    items = [("a", {"transcript": "a transcript", "output": "bad"})]
    malformed = fake_llm(lambda messages, role: "malformed")
    errors = asyncio.run(judge_all(malformed, items, out, instructions_file=rules, criteria=CRITERIA))
    assert errors["a"].failed == ["judge_error"] and malformed.calls == 3
    current = asyncio.run(judge_all(fake_llm(verdict_reply), items, out, instructions_file=rules, criteria=CRITERIA))
    assert current["a"].failed == ["content_preserved"]


def test_size_batches_caps_items_and_characters():
    items = [(str(i), {"transcript": "x" * n, "output": ""}) for i, n in enumerate([10, 10, 10, 50, 5, 5])]
    assert [[id_ for id_, _ in b] for b in size_batches(items, 3)] == [["0", "1", "2"], ["3", "4", "5"]]
    # A row longer than the cap still gets a batch of its own.
    assert [[id_ for id_, _ in b] for b in size_batches(items, 10, max_chars=25)] == [["0", "1"], ["2"], ["3"], ["4", "5"]]
