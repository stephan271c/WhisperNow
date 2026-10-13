import asyncio
import json
import re
import sys
from dataclasses import asdict, replace

import httpx
import pytest

from conftest import SETTINGS, judge_items

from whispernow_training.datagen import cli, compose, gates, generate, llm, noise, paths
from whispernow_training.datagen.schema import BatchRecord, read_jsonl


def test_concurrent_cli_failure_cleans_up_peer_sessions_before_closing_and_resumes(tmp_path, monkeypatch):
    run = replace(compose.BENCH, out_dir=tmp_path / "rows", shares={"150-300": 1.0})
    settings = replace(SETTINGS, backend="opencode", judge_backend="opencode", generator_backend="opencode",
                       concurrency=2, opencode_provider="local")
    monkeypatch.setattr(compose, "RUNS", {"bench": run})
    monkeypatch.setattr(compose, "LANES", {"150-300": 2})
    monkeypatch.setattr(llm, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(llm.LlmSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(sys, "argv", ["datagen", "compose", "--run", "bench", "--total", "1",
                                     "--category", "revised_choice"])
    peer_entered = asyncio.Event()
    release = asyncio.Event()
    events = []
    sessions = []
    clients = []
    generator_requests = []
    peer_session = ""
    failing = True

    async def handle(request):
        nonlocal peer_session
        path = request.url.path
        if request.method == "DELETE":
            if failing and path == f"/session/{peer_session}":
                await asyncio.sleep(0)
                events.append("peer deleted")
            return httpx.Response(200, json=True)
        if path == "/session":
            session_id = str(len(sessions))
            sessions.append(session_id)
            return httpx.Response(200, json={"id": session_id})
        assert request.method == "POST" and path.endswith("/message")
        user = json.loads(request.content)["parts"][0]["text"]
        if batch_match := re.search(r"Batch (\d+)\.", user):
            generator_requests.append(user)
            batch = int(batch_match.group(1))
            if failing:
                if batch == 0:
                    await asyncio.wait_for(peer_entered.wait(), timeout=5)
                    return httpx.Response(400, text="invalid generator request")
                peer_session = path.split("/")[2]
                peer_entered.set()
                try:
                    await asyncio.wait_for(release.wait(), timeout=5)
                except asyncio.CancelledError:
                    events.append("peer cancelled")
                    raise
                pytest.fail("the held peer completed after the failed lane returned")
            text = " ".join(f"b{batch}w{i}" for i in range(200))
            content = json.dumps({"items": [{"input": text, "layout": "prose"}]})
        elif "<transcript>" in user:
            content = re.search(r"<transcript>\n(.*)\n</transcript>", user, re.S).group(1)
        else:
            content = json.dumps({"results": [
                {"i": item["i"], **dict.fromkeys(gates.COMPOSE_CRITERIA, True)}
                for item in judge_items([{"content": user}])
            ]})
        return httpx.Response(200, json={"info": {}, "parts": [{"type": "text", "text": content}]})

    class HttpClient(httpx.AsyncClient):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(handle), **kwargs)
            clients.append(self)

        async def aclose(self):
            events.append("client closed")
            await super().aclose()

    async def start(server):
        server.url = "http://opencode.local"

    monkeypatch.setattr(llm.httpx, "AsyncClient", HttpClient)
    monkeypatch.setattr(llm._OpencodeServer, "start", start)
    monkeypatch.setattr(llm._OpencodeServer, "stop", lambda server: events.append("server stopped"))

    with pytest.raises(llm.LlmError, match="opencode HTTP 400: invalid generator request"):
        cli.main()
    assert peer_entered.is_set()
    assert events == ["peer cancelled", "peer deleted", "client closed", "server stopped"]
    assert clients[0].is_closed
    assert not list(run.out_dir.glob("*.jsonl"))
    assert not list((tmp_path / "cache").rglob("*.json"))

    failing = False
    monkeypatch.setattr(compose, "LANES", {"150-300": 1})
    cli.main()
    assert generator_requests[-1] == generator_requests[0]
    assert len(compose.load(run)) == 1
    records = list(read_jsonl(run.out_dir / "revised_choice.150-300.batches.jsonl"))
    assert len(records) == 1 and records[0]["batch"] == 0 and not records[0]["error"]


@pytest.mark.parametrize("workflow", ["generate", "compose", "noise"])
def test_legacy_missing_key_records_restore_only_infrastructure_budget(workflow, fake_llm, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "STAGE3", tmp_path)
    monkeypatch.setattr(noise, "OUT_DIR", tmp_path)
    monkeypatch.setattr(compose, "LANES", dict.fromkeys(compose.BUCKETS, 1))
    run = replace(compose.BENCH, out_dir=tmp_path)
    generator_requests = []
    malformed = False

    def reply(messages, role):
        user = messages[-1]["content"]
        if role == "judge":
            criteria = {*generate.CRITERIA, *gates.COMPOSE_CRITERIA}
            return json.dumps({"results": [{"i": item["i"], **dict.fromkeys(criteria, True)} for item in judge_items(messages)]})
        if role == "teacher":
            return re.search(r"<transcript>\n(.*)\n</transcript>", user, re.S).group(1)
        generator_requests.append(user)
        if malformed:
            return '{"items": null}'
        batch = int(re.search(r"Batch (\d+)\.", user).group(1))
        text = " ".join(f"b{batch}w{i}" for i in range(200)) if workflow == "compose" else f"batch {batch} uh um send the update"
        return json.dumps({"items": [{"input": text, "layout": "prose", "evidence": ["uh um"]}]})

    llm = fake_llm(reply)

    def resume(target):
        if workflow == "generate":
            return asyncio.run(generate.generate_category(llm, "mixed_message", target, batch_size=1, progress=lambda _: None))
        if workflow == "compose":
            known = {row.input for row in compose.load(run)}
            return asyncio.run(compose.generate_cell(llm, run, "revised_choice", "150-300", target, known, compose.counts_by_cell(run), progress=lambda _: None))
        return asyncio.run(noise.generate_kind(llm, "filler_cluster", target, {row.input for row in noise.load()}, progress=lambda _: None))

    assert resume(1) == 1
    log_path, = tmp_path.glob("*.batches.jsonl")
    output_path, = (path for path in tmp_path.glob("*.jsonl") if not path.name.endswith((".batches.jsonl", ".rejected.jsonl")))
    prior_rows = output_path.read_bytes()
    budget = 8 if workflow == "generate" else 6
    records = list(read_jsonl(log_path))
    records += [
        asdict(BatchRecord(batch=batch, style="raw", requested=1, error="unreadable generator reply: 'LLM_API_KEY'"))
        for batch in range(1, budget)
    ]
    records[0]["error"] = "unreadable generator reply: 'LLM_API_KEY'"
    records[2]["error"] = "unreadable generator reply: 'items'"
    records[-1]["notes"] = ["keep this content feedback"]
    log_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    credited = budget - 3

    assert resume(2) == 2
    resumed_log = list(read_jsonl(log_path))
    assert resumed_log[:-1] == records and resumed_log[-1]["batch"] == budget
    assert output_path.read_bytes().startswith(prior_rows)
    assert "keep this content feedback" in generator_requests[-1]

    malformed = True
    calls = llm.calls
    assert resume(3) == 2
    next_budget = 11 if workflow == "generate" else 6
    assert llm.calls - calls == next_budget + credited - len(resumed_log)
    assert len(list(read_jsonl(log_path))) == next_budget + credited
    calls = llm.calls
    assert resume(3) == 2 and llm.calls == calls
