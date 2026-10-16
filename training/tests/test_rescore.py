import argparse
import json
from dataclasses import replace

import pytest
from conftest import SETTINGS, judge_items

from whispernow_training.datagen import llm
from whispernow_training.datagen.judge import Verdict, load_verdicts
from whispernow_training.datagen.schema import Example, write_examples
from whispernow_training.model import cli, evaluate

LOSSY = [{"kind": "dropped", "fact": "'urgent'"}, {"kind": "changed", "fact": "5 became 15"}]


@pytest.fixture
def eval_dir(tmp_path, monkeypatch):
    ids = tmp_path / "ids.txt"
    ids.write_text("")
    monkeypatch.setattr(evaluate, "TEST_IDS_109", ids)
    rows = [Example(id=id_, input=text, output=text, source="s", category="c", license="l")
            for id_, text in {"a": "Order 5 boxes.", "b": "Open app.py now."}.items()]
    monkeypatch.setattr(evaluate, "load_rows", lambda split, limit: rows)

    async def judge(*args):
        raise AssertionError("rescore judged")

    monkeypatch.setattr(evaluate, "judge", judge)
    out = tmp_path / "eval"
    evaluate.write_outputs(out / "test.outputs.jsonl", [{"id": "a", "output": "Order 15 boxes.", "hit_limit": False},
                                                         {"id": "b", "output": "Open app.py now.", "hit_limit": False}])
    (out / "test.faults.scores.json").write_text('{"original": true}\n')
    return out


def _write_verdicts(out, verdicts):
    with (out / "test.faults.judge.jsonl").open("w") as sink:
        sink.writelines(json.dumps(verdict.record()) + "\n" for verdict in verdicts)


def test_rescore_scores_saved_outputs_and_verdicts_into_a_new_file(eval_dir):
    _write_verdicts(eval_dir, [Verdict("a", False, ["facts_kept"], "", "judge-model", LOSSY),
                               Verdict("b", True, [], "", "judge-model")])
    path = evaluate.rescore(eval_dir, "test")

    assert path.name == f"test.faults.metrics{evaluate.METRICS_VERSION}.scores.json"
    assert json.loads((eval_dir / "test.faults.scores.json").read_text()) == {"original": True}
    scores = json.loads(path.read_text())
    assert scores["metrics_version"] == evaluate.METRICS_VERSION and scores["judge"] == "faults"
    # "15" no longer keeps the reference's "5".
    assert scores["overall"]["protected_kept"] == {"rate": 50.0, "n": 2}
    assert scores["overall"]["faults"]["facts_lost"]["per_row"] == 1.0


def test_rescore_without_verdicts_scores_unjudged_and_refuses_partial_verdicts(eval_dir):
    path = evaluate.rescore(eval_dir, "test")
    assert path.name == f"test.metrics{evaluate.METRICS_VERSION}.scores.json"
    assert not json.loads(path.read_text())["judged"]

    _write_verdicts(eval_dir, [Verdict("a", True, [], "", "judge-model")])
    with pytest.raises(ValueError, match="1 of 2 test rows have no verdict"):
        evaluate.rescore(eval_dir, "test")


def test_a_sample_split_is_scored_from_the_whole_split_files(tmp_path, monkeypatch):
    rows = [Example(id=id_, input=text, output=text, source="s", category="c", license="l", split="test")
            for id_, text in {"a": "Order 5 boxes.", "b": "Open app.py now."}.items()]
    write_examples(tmp_path / "test.jsonl", rows)
    sample = tmp_path / "sample.txt"
    sample.write_text("b\n")
    monkeypatch.setattr(evaluate, "BUILD", tmp_path)
    monkeypatch.setattr(evaluate, "SAMPLES", {"test-sample": ("test", sample)})
    monkeypatch.setattr(evaluate, "TEST_IDS_109", sample)
    out = tmp_path / "eval"
    evaluate.write_outputs(out / "test.outputs.jsonl", [{"id": "a", "output": "Order 15 boxes.", "hit_limit": False},
                                                         {"id": "b", "output": "Open app.py now.", "hit_limit": False}])
    _write_verdicts(out, [Verdict("a", False, ["facts_kept"], "", "judge-model", LOSSY),
                          Verdict("b", True, [], "", "judge-model")])

    assert [row.id for row in evaluate.load_rows("test-sample")] == ["b"]
    path = evaluate.rescore(out, "test-sample")
    assert path.name == f"test-sample.faults.metrics{evaluate.METRICS_VERSION}.scores.json"
    overall = json.loads(path.read_text())["overall"]
    assert overall["rows"] == 1 and overall["faults"]["facts_lost"]["per_row"] == 0.0


def _scores(protected_kept, version=None):
    scores = {"overall": {"rows": 1, "protected_kept": {"rate": protected_kept, "n": 1}}}
    return scores if version is None else scores | {"metrics_version": version}


def test_compare_reads_rescored_scores_and_refuses_mixed_metric_versions(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "RUNS", tmp_path)
    for run, scores in {"old": _scores(80.0), "new": _scores(60.0, evaluate.METRICS_VERSION)}.items():
        (tmp_path / run / "eval").mkdir(parents=True)
        (tmp_path / run / "eval" / "test.scores.json").write_text(json.dumps(scores))
    args = argparse.Namespace(candidates=["old=old/eval", "new=new/eval"], stem="test", group=None)

    with pytest.raises(SystemExit, match=r"different metric versions \(old 1, new 2\)"):
        cli.cmd_compare(args)

    evaluate.rescored_path(tmp_path / "old" / "eval", "test").write_text(
        json.dumps(_scores(70.0, evaluate.METRICS_VERSION)))
    cli.cmd_compare(args)
    table = capsys.readouterr().out
    assert "| old | 1 |" in table and "70.0" in table and "80.0" not in table


def test_rejudge_judges_only_the_chosen_rows_and_keeps_the_other_verdicts(fake_llm, tmp_path, monkeypatch):
    ids = tmp_path / "ids.txt"
    ids.write_text("")
    monkeypatch.setattr(evaluate, "TEST_IDS_109", ids)
    rows = [Example(id=id_, input=f"Order {id_}.", output=f"Order {id_}.", source="s", category="c", license="l",
                    teacher_model="gpt-6.1-sol", teacher_reasoning="high") for id_ in ("a", "b")]
    monkeypatch.setattr(evaluate, "load_rows", lambda split, limit=None: rows)
    out = tmp_path / "eval"
    evaluate.write_outputs(out / "test.outputs.jsonl", [{"id": row.id, "output": row.output, "hit_limit": False} for row in rows])
    verdicts = out / "test.r112.faults.judge.jsonl"
    with verdicts.open("w") as sink:
        sink.writelines(json.dumps(Verdict(id_, False, ["facts_kept"], "", "judge-model", LOSSY[:1]).record()) + "\n"
                        for id_ in ("a", "b"))
    judged = []

    def reply(messages, role):
        items = judge_items(messages)
        judged.extend(item["output"] for item in items)
        return json.dumps({"results": [{"i": item["i"], "not_answered": True, "faults": []} for item in items]})

    class Client(fake_llm):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

    client = Client(reply)
    client.settings = replace(SETTINGS, judge_backend="codex", judge_model="gpt-6.1-sol", judge_reasoning="high")
    monkeypatch.setattr(llm.LlmSettings, "from_env", lambda: client.settings)
    monkeypatch.setattr(llm, "LlmClient", lambda settings: client)

    path = evaluate.rejudge(out, "test", ["a"])
    assert judged == ["Order a."]
    current = load_verdicts(verdicts)
    assert current["a"].passed and not current["b"].passed
    assert len(verdicts.read_text().splitlines()) == 3
    assert json.loads(path.read_text())["overall"]["faults"]["facts_lost"]["per_row"] == 0.5

    evaluate.rejudge(out, "test", ["a"])
    assert judged == ["Order a."]
    with pytest.raises(ValueError, match="1 ids are not test rows, such as c"):
        evaluate.rejudge(out, "test", ["c"])
