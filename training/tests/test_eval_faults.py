import asyncio
import json

from conftest import judge_items

from whispernow_training.datagen.judge import Verdict, judge_all, load_verdicts
from whispernow_training.datagen.schema import Example
from whispernow_training.model import evaluate

REPLIES = {
    "clean": [],
    "unresolved": [{"kind": "unresolved", "fact": "kept 'I said 9:45 earlier' before 10:15"}],
    "lossy": [
        {"kind": "dropped", "fact": "'I think'"},
        {"kind": "wrong_side", "fact": "kept 6:30 beside 5:30"},
        {"kind": "invented", "fact": "'4.5 hours'"},
    ],
}


def _reply(faults_by_id):
    def reply(messages, role):
        items = judge_items(messages)
        return json.dumps({"results": [
            {"i": item["i"], "not_answered": True, "faults": faults_by_id[item["output"]], "note": ""} for item in items
        ]})

    return reply


def _judge(fake_llm, tmp_path, faults_by_id):
    items = [(id_, {"transcript": "t", "reference": "r", "output": id_}) for id_ in faults_by_id]
    return asyncio.run(judge_all(
        fake_llm(_reply(faults_by_id)), items, tmp_path / "v.jsonl", instructions_file=evaluate.JUDGE_INSTRUCTIONS,
        criteria=evaluate.JUDGE_CRITERIA, fault_kinds=evaluate.FAULT_KINDS, progress=lambda line: None,
    ))


def test_criteria_follow_the_faults_and_an_unresolved_correction_fails_none(fake_llm, tmp_path):
    verdicts = _judge(fake_llm, tmp_path, REPLIES)
    assert verdicts["clean"].passed and verdicts["clean"].faults == []
    assert verdicts["unresolved"].passed and len(verdicts["unresolved"].faults) == 1
    assert verdicts["lossy"].failed == ["facts_kept", "nothing_invented"]
    assert "wrong_side: kept 6:30 beside 5:30" in verdicts["lossy"].note
    assert load_verdicts(tmp_path / "v.jsonl")["lossy"].faults == verdicts["lossy"].faults


def test_an_unreadable_reply_is_asked_again_before_it_is_a_judge_error(fake_llm, tmp_path):
    verdicts = _judge(fake_llm, tmp_path, {"odd": [{"kind": "nitpick", "fact": "x"}]})
    assert verdicts["odd"].failed == ["judge_error"]

    replies = iter(['{"results": [{"i": 0, "faults": [{"kind": "dropped", "fact": "kept "x""}]}]}', json.dumps(
        {"results": [{"i": 0, "not_answered": True, "faults": [{"kind": "dropped", "fact": "'x'"}]}]}
    )])
    llm = fake_llm(lambda messages, role: next(replies))
    verdicts = asyncio.run(judge_all(
        llm, [("a", {"transcript": "t", "output": "o"})], tmp_path / "w.jsonl",
        instructions_file=evaluate.JUDGE_INSTRUCTIONS, criteria=evaluate.JUDGE_CRITERIA,
        fault_kinds=evaluate.FAULT_KINDS, progress=lambda line: None,
    ))
    assert verdicts["a"].failed == ["facts_kept"] and llm.attempts == [0, 1]


def _row(id_, words):
    text = " ".join(["word"] * words)
    return Example(id=id_, input=text, output=text, source="s", category="c", license="l")


def test_scores_count_faults_per_row_and_per_thousand_words():
    rows = [_row("a", 100), _row("b", 300), _row("c", 100)]
    outputs = {row.id: {"output": row.output, "hit_limit": False} for row in rows}
    judged = {
        "a": Verdict("a", True, [], "", "judge-model", REPLIES["unresolved"]),
        "b": Verdict("b", False, ["facts_kept", "nothing_invented"], "", "judge-model", REPLIES["lossy"]),
        "c": Verdict("c", False, ["judge_error"], "", "judge-model"),
    }
    faults = evaluate.score(rows, outputs, judged)["overall"]["faults"]
    # The unreadable row counts in none; the other two hold 400 reference words.
    assert faults["facts_lost"] == {"per_row": 1.0, "per_1000_words": 5.0, "rows": 50.0, "n": 2}
    assert faults["factual_errors"] == {"per_row": 1.5, "per_1000_words": 7.5, "rows": 50.0, "n": 2}
    assert faults["unresolved"]["rows"] == 50.0 and faults["invented"]["per_row"] == 0.5
    assert "faults" not in evaluate.score(rows, outputs)["overall"]


def test_judged_scores_get_their_own_files(tmp_path, monkeypatch):
    rows = [_row("a", 5)]
    monkeypatch.setattr(evaluate, "load_rows", lambda split, limit: rows)

    async def judge(rows, outputs, out, rewritten):
        assert out.name == "test.faults.judge.jsonl"
        return {"a": Verdict("a", True, [], "", "judge-model")}

    monkeypatch.setattr(evaluate, "judge", judge)
    produce = lambda missing: [{"id": row.id, "output": row.output, "hit_limit": False} for row in missing]  # noqa: E731
    evaluate.evaluate(produce, tmp_path, "test", with_judge=True, generation={"source": "reference"})
    evaluate.evaluate(produce, tmp_path, "test", generation={"source": "reference"})
    judged = json.loads((tmp_path / "test.faults.scores.json").read_text())
    assert judged["judge"] == "faults" and judged["metrics_version"] == evaluate.METRICS_VERSION
    assert not json.loads((tmp_path / "test.scores.json").read_text())["judged"]
