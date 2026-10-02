import asyncio
import json

from conftest import judge_items

from whispernow_training.datagen.judge import judge_all, judge_batch, size_batches

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


def test_size_batches_caps_items_and_characters():
    items = [(str(i), {"transcript": "x" * n, "output": ""}) for i, n in enumerate([10, 10, 10, 50, 5, 5])]
    assert [[id_ for id_, _ in b] for b in size_batches(items, 3)] == [["0", "1", "2"], ["3", "4", "5"]]
    # A row longer than the cap still gets a batch of its own.
    assert [[id_ for id_, _ in b] for b in size_batches(items, 10, max_chars=25)] == [["0", "1"], ["2"], ["3"], ["4", "5"]]
