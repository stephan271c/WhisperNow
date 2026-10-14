import asyncio
import json
from dataclasses import replace

import httpx
import pytest

from conftest import SETTINGS

from whispernow_training.datagen import llm, teacher
from whispernow_training.datagen.schema import Example
from whispernow_training.model import draft, evaluate, prompt


def _row(text):
    return Example(id=text, input=text, output=text, source="s", category="c", license="x")


def _local_boundary(monkeypatch, tmp_path, handle, events):
    settings = replace(SETTINGS, backend="opencode", concurrency=2, opencode_provider="local")
    monkeypatch.setattr(llm, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(llm.LlmSettings, "from_env", classmethod(lambda cls: settings))
    clients = []

    class HttpTransport(httpx.MockTransport):
        async def aclose(self):
            events.append("client closed")
            await super().aclose()

    class HttpClient(httpx.AsyncClient):
        def __init__(self, **kwargs):
            super().__init__(transport=HttpTransport(handle), **kwargs)
            clients.append(self)

    async def start(server):
        server.url = "http://opencode.local"

    monkeypatch.setattr(httpx, "AsyncClient", HttpClient)
    monkeypatch.setattr(llm._OpencodeServer, "start", start)
    monkeypatch.setattr(llm._OpencodeServer, "stop", lambda server: events.append("server stopped"))
    return clients


@pytest.mark.parametrize("source", ["teacher", "server"])
def test_failed_evaluation_joins_peer_cleanup_before_closing(source, tmp_path, monkeypatch):
    async def exercise():
        peer_entered = asyncio.Event()
        release = asyncio.Event()
        events = []
        sessions = []
        peer_session = ""
        peer_task = None
        failure = llm.LlmError("invalid evaluation request", retryable=False)

        async def handle(request):
            nonlocal peer_session, peer_task
            path = request.url.path
            if path == "/session":
                session = str(len(sessions))
                sessions.append(session)
                return httpx.Response(200, json={"id": session})
            if request.method == "DELETE":
                if path == f"/session/{peer_session}":
                    await asyncio.sleep(0)
                    events.append("peer deleted")
                return httpx.Response(200, json=True)
            body = json.loads(request.content)
            text = body["parts"][0]["text"] if source == "teacher" else body["messages"][-1]["content"]
            if "failed transcript" in text:
                await peer_entered.wait()
                raise failure
            assert "held transcript" in text
            peer_session = path.split("/")[2] if source == "teacher" else ""
            peer_task = asyncio.current_task()
            peer_entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                events.append("peer cancelled")
                raise
            pytest.fail("the held peer completed after the failed evaluation returned")

        clients = _local_boundary(monkeypatch, tmp_path, handle, events)
        rows = [_row("failed transcript"), _row("held transcript")]
        try:
            with pytest.raises(llm.LlmError) as error:
                if source == "teacher":
                    await evaluate.generate_teacher(rows)
                else:
                    await evaluate.generate_server("http://llama.local/v1", rows, concurrency=2)
            events_on_return = events.copy()
            pending_on_return = not peer_task.done()
        finally:
            if peer_task is not None and not peer_task.done():
                peer_task.cancel()
                cleanup = await asyncio.gather(peer_task, return_exceptions=True)
            else:
                cleanup = []

        assert error.value is failure
        assert not pending_on_return, f"evaluator closed its client while the peer was active; cleanup: {cleanup}"
        expected = ["peer cancelled", "peer deleted", "client closed", "server stopped"] if source == "teacher" else [
            "peer cancelled", "client closed",
        ]
        assert events_on_return == expected
        assert peer_task.cancelled() and clients[0].is_closed
        assert not list((tmp_path / "cache").rglob("*.json"))

    asyncio.run(asyncio.wait_for(exercise(), timeout=5))


@pytest.mark.parametrize("source", ["teacher", "server"])
def test_successful_evaluation_preserves_input_order_requests_limits_and_teacher_cache(source, tmp_path, monkeypatch):
    async def exercise():
        release_first = asyncio.Event()
        release_second = asyncio.Event()
        two_entered = asyncio.Event()
        third_entered = asyncio.Event()
        events = []
        requests = []
        finished = []
        active = peak = 0

        async def handle(request):
            nonlocal active, peak
            if request.url.path == "/session":
                return httpx.Response(200, json={"id": str(len(requests))})
            if request.method == "DELETE":
                return httpx.Response(200, json=True)
            body = json.loads(request.content)
            requests.append(body)
            text = body["parts"][0]["text"] if source == "teacher" else body["messages"][-1]["content"]
            name = next(name for name in ("first", "second", "third") if f"{name} transcript" in text)
            active += 1
            peak = max(peak, active)
            if active == 2:
                two_entered.set()
            if name == "first":
                await release_first.wait()
            elif name == "second":
                await release_second.wait()
            else:
                third_entered.set()
            active -= 1
            finished.append(name)
            output = f" {name} output "
            reply = {"info": {}, "parts": [{"type": "text", "text": output}]} if source == "teacher" else {
                "choices": [{"message": {"content": output}, "finish_reason": "length" if name == "third" else "stop"}],
                "usage": {"completion_tokens": 3},
            }
            return httpx.Response(200, json=reply)

        clients = _local_boundary(monkeypatch, tmp_path, handle, events)
        rows = [_row(f"{name} transcript") for name in ("first", "second", "third")]
        if source == "teacher":
            generation = asyncio.create_task(evaluate.generate_teacher(rows))
        else:
            generation = asyncio.create_task(evaluate.generate_server(
                "http://llama.local/v1/", rows, concurrency=2, repeat_penalty=1.2, repeat_last_n=1024,
            ))
        try:
            await two_entered.wait()
            scheduled = asyncio.Event()
            asyncio.get_running_loop().call_soon(scheduled.set)
            await scheduled.wait()
            assert not third_entered.is_set(), "third request entered while both concurrency permits were occupied"
            release_second.set()
            await third_entered.wait()
            release_first.set()
            outputs = await generation
        finally:
            if not generation.done():
                generation.cancel()
            await asyncio.gather(generation, return_exceptions=True)
        if source == "teacher":
            assert await evaluate.generate_teacher(rows) == outputs
        assert [output["id"] for output in outputs] == [row.id for row in rows]
        assert [output["output"] for output in outputs] == [f"{name} output" for name in ("first", "second", "third")]
        assert finished[0] == "second" and peak == 2 and len(requests) == len(rows)
        assert len(clients) == 1 and clients[0].is_closed
        for row, body, output in zip(rows, requests, outputs, strict=True):
            if source == "teacher":
                messages = teacher.teacher_messages(row.input)
                assert body == {"agent": "datagen", "model": {"providerID": "local", "modelID": SETTINGS.model},
                                "parts": [{"type": "text", "text": messages[-1]["content"]}],
                                "system": messages[0]["content"], "variant": SETTINGS.reasoning}
                assert output["tokens"] is None and not output["hit_limit"]
            else:
                assert body == {"messages": prompt.messages(row.input), "max_tokens": prompt.max_tokens(row.input),
                                "temperature": prompt.TEMPERATURE, "repeat_penalty": 1.2, "repeat_last_n": 1024,
                                "draft_text": draft.preclean(row.input)}
                assert output["tokens"] == 3 and output["hit_limit"] == (row.id == "third transcript")

    asyncio.run(asyncio.wait_for(exercise(), timeout=5))


@pytest.mark.parametrize("source", ["teacher", "server"])
def test_evaluation_reports_each_simultaneous_client_failure(source, tmp_path, monkeypatch):
    failures = [llm.LlmError(f"failed transcript {index}", retryable=False) for index in range(2)]

    async def handle(request):
        if request.url.path == "/session":
            return httpx.Response(200, json={"id": "local"})
        if request.method == "DELETE":
            return httpx.Response(200, json=True)
        body = json.loads(request.content)
        text = body["parts"][0]["text"] if source == "teacher" else body["messages"][-1]["content"]
        raise next(failure for failure in failures if str(failure) in text)

    _local_boundary(monkeypatch, tmp_path, handle, [])
    rows = [_row(str(failure)) for failure in failures]
    with pytest.raises(ExceptionGroup) as error:
        if source == "teacher":
            asyncio.run(evaluate.generate_teacher(rows))
        else:
            asyncio.run(evaluate.generate_server("http://llama.local/v1", rows, concurrency=2))
    assert error.value.exceptions == tuple(failures)
