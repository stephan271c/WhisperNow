import json
import os
from dataclasses import replace

import httpx
from conftest import SETTINGS, judge_items

from whispernow_training.datagen import llm
from whispernow_training.datagen.schema import Example
from whispernow_training.model import evaluate


def _row(id_, transcript, reference):
    return Example(id=id_, input=transcript, output=reference, source="s", category="c", license="x")


def test_evaluation_resumes_only_matching_generation_requests(tmp_path, monkeypatch):
    rows = [_row("a", "first transcript", "Reference A.")]
    monkeypatch.setattr(evaluate, "load_rows", lambda split, limit: rows)
    requests = []
    replies = {"first transcript": "Independent output A.", "second transcript": "Independent output B.",
               "changed transcript": "Fresh output A."}

    def produce(pending):
        requests.append([row.id for row in pending])
        return [{"id": row.id, "output": replies[row.input], "hit_limit": False} for row in pending]

    generation = {"source": "hf", "revision": "weights-1", "repeat_penalty": 1.05}
    evaluate.evaluate(produce, tmp_path, "test", generation=generation)
    rows.append(_row("b", "second transcript", "Reference B."))
    evaluate.evaluate(produce, tmp_path, "test", generation=generation)
    evaluate.evaluate(produce, tmp_path, "test", generation=generation)
    rows[0] = replace(rows[0], input="changed transcript")
    evaluate.evaluate(produce, tmp_path, "test", generation=generation)
    outputs = evaluate.read_outputs(tmp_path / "test.outputs.jsonl")
    assert requests == [["a"], ["b"], ["a"]]
    assert outputs["a"]["output"] == "Fresh output A." and outputs["b"]["output"] == "Independent output B."
    old = [json.loads(line) for line in (tmp_path / "test.outputs.jsonl").read_text().splitlines()]
    assert old[0]["output"] == "Independent output A." and len(old) == 3
    for fields in ({"repeat_penalty": 1.1}, {"revision": "weights-2"}):
        generation = {**generation, **fields}
        evaluate.evaluate(produce, tmp_path, "test", generation=generation)
    assert requests[-2:] == [["a", "b"], ["a", "b"]]


def test_unknown_model_revisions_stay_pending(tmp_path, monkeypatch):
    row = _row("a", "a transcript", "Reference.")
    monkeypatch.setattr(evaluate, "load_rows", lambda split, limit: [row])
    requests = []

    def produce(pending):
        requests.extend(pending)
        return [{"id": "a", "output": "Current output.", "hit_limit": False}]

    generation = {"source": "server", "revision": None, "model_path": "/remote/model.gguf"}
    evaluate.evaluate(produce, tmp_path, "test", generation=generation)
    evaluate.evaluate(produce, tmp_path, "test", generation=generation)
    assert len(requests) == 2


def test_reference_generation_detects_changes_under_the_same_provenance(tmp_path, monkeypatch):
    rows = [replace(_row("a", "a transcript", "First reference."), teacher_model="gpt-6.1-sol", teacher_reasoning="high")]
    monkeypatch.setattr(evaluate, "load_rows", lambda split, limit: rows)
    requests = []

    def produce(pending):
        requests.extend(pending)
        return [{"id": row.id, "output": row.output, "hit_limit": False} for row in pending]

    generation = {"source": "reference"}
    evaluate.evaluate(produce, tmp_path, "test", generation=generation)
    rows[0] = replace(rows[0], output="Changed reference.")
    evaluate.evaluate(produce, tmp_path, "test", generation=generation)
    assert len(requests) == 2 and evaluate.reference_tag(rows) == "r112"
    assert evaluate.read_outputs(tmp_path / "test.outputs.jsonl")["a"]["output"] == "Changed reference."


def test_evaluation_rejudges_changed_references(fake_llm, tmp_path, monkeypatch):
    rows = [replace(_row(id_, f"transcript {id_}", f"Reference {id_}."),
                    teacher_model="gpt-6.1-sol", teacher_reasoning="high") for id_ in ("a", "b")]
    monkeypatch.setattr(evaluate, "load_rows", lambda split, limit: rows)
    rules = tmp_path / "rules.md"
    rules.write_text("judge rules")
    monkeypatch.setattr(evaluate, "JUDGE_INSTRUCTIONS", rules)
    generated, judged = [], []

    def produce(pending):
        generated.extend(row.id for row in pending)
        return [{"id": row.id, "output": f"Reference {row.id}.", "hit_limit": False} for row in pending]

    def reply(messages, role):
        items = judge_items(messages)
        judged.append(items)
        return json.dumps({"results": [
            {"i": item["i"], "not_answered": True, "faults": [] if item["output"] == item["reference"] else
             [{"kind": "changed", "fact": "a different reference"}]} for item in items
        ]})

    class Client(fake_llm):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

    client = Client(reply)
    client.settings = replace(SETTINGS, judge_backend="codex", judge_model="gpt-6.1-sol", judge_reasoning="high")
    monkeypatch.setattr(llm.LlmSettings, "from_env", lambda: client.settings)
    monkeypatch.setattr(llm, "LlmClient", lambda settings: client)
    generation = {"source": "hf", "revision": "weights"}
    first = evaluate.evaluate(produce, tmp_path, "test", with_judge=True, generation=generation)
    assert first["overall"]["facts_dropped"]["rate"] == 0.0
    evaluate.evaluate(produce, tmp_path, "test", with_judge=True, generation=generation)
    assert len(judged) == 1
    rows[0] = replace(rows[0], output="New reference a.")
    current = evaluate.evaluate(produce, tmp_path, "test", with_judge=True, generation=generation)
    assert generated == ["a", "b"] and len(judged[-1]) == 1
    assert judged[-1][0]["reference"] == "New reference a."
    assert current["overall"]["facts_dropped"]["rate"] == 50.0


def test_local_and_server_model_revisions_include_file_content(tmp_path, monkeypatch):
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "fixture-version")
    model = tmp_path / "model"
    model.mkdir()
    weights = model / "model.safetensors"
    weights.write_bytes(b"first weights")
    (model / "config.json").write_text('{"fixture": true}')
    first = evaluate.hf_generation(model)
    stat = weights.stat()
    weights.write_bytes(b"other weights")
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    changed = evaluate.hf_generation(model)
    assert first["revision"] != changed["revision"]
    assert first["chat_template"] == changed["chat_template"]
    props = {"model_path": str(weights), "chat_template": "served template", "build_info": "test-build",
             "default_generation_settings": {"n_ctx": 8192, "params": {"top_k": 40}}}
    requests = []

    def get(url, **kwargs):
        requests.append(url)
        return httpx.Response(200, json=props, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    first = evaluate.server_generation("http://localhost:8080/v1", repeat_last_n=2048)
    weights.write_bytes(b"newer weights")
    changed = evaluate.server_generation("http://localhost:8080/v1", repeat_last_n=2048)
    assert requests == ["http://localhost:8080/props"] * 2
    assert first["revision"] != changed["revision"] and first["repeat_last_n"] == 2048
