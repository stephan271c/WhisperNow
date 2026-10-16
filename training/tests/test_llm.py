import asyncio
import json
import os
import shlex
import time
from dataclasses import replace

import httpx
import pytest
from conftest import SETTINGS

from whispernow_training.datagen import llm
from whispernow_training.datagen.llm import LlmSettings, ResponseCache, codex_command, parse_json_reply, strip_think


def test_response_cache_separates_model_reasoning_messages_and_attempt(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "CACHE", tmp_path)
    client = llm.LlmClient(SETTINGS)
    messages = [{"role": "user", "content": "hi"}]
    requests = 0

    async def connect(backend):
        pass

    async def request(backend, model, reasoning, sent):
        nonlocal requests
        requests += 1
        return f"reply {requests}"

    monkeypatch.setattr(client, "_connect", connect)
    monkeypatch.setattr(client, "_request", request)

    async def run():
        assert await client.complete(messages) == "reply 1"
        assert await client.complete([dict(messages[0])]) == "reply 1"
        assert await client.complete(messages, attempt=0) == "reply 1"
        assert await client.complete(messages, reasoning="high") == "reply 2"
        client.settings = replace(SETTINGS, model="another-model")
        assert await client.complete(messages) == "reply 3"
        assert await client.complete([{"role": "user", "content": "different"}]) == "reply 4"
        client.settings = SETTINGS
        assert await client.complete(messages, attempt=1) == "reply 5"
        assert await client.complete(messages, attempt=1) == "reply 5"
        assert await client.complete(messages, attempt=2) == "reply 6"

    asyncio.run(run())
    assert client.calls == requests == 6 and client.cache_hits == 3


def test_response_cache_round_trip(tmp_path):
    cache = ResponseCache(tmp_path)
    assert cache.get("ab12") is None
    cache.put("ab12", "reply", {"model": "m", "seconds": 1.0})
    assert cache.get("ab12") == "reply"
    assert not list(tmp_path.rglob("*.tmp"))


def test_parse_json_reply_tolerates_fences_and_prose():
    assert parse_json_reply('{"a": 1}') == {"a": 1}
    assert parse_json_reply('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_reply('Here you go: {"a": [1, 2]} done') == {"a": [1, 2]}
    assert parse_json_reply('Here: [{"a": 1}] done') == [{"a": 1}]
    # A long reply that dropped the closing brace still yields its items.
    assert parse_json_reply('{"items": [{"input": "x"}, {"input": "y"}]') == {"items": [{"input": "x"}, {"input": "y"}]}
    with pytest.raises(ValueError):
        parse_json_reply("no json here")


def test_parse_json_reply_closes_brackets_a_reply_left_open():
    # The judge's one-row replies sometimes stop before their last "]}".
    reply = '{"results":[{"i":0,"faults":[{"kind":"dropped","fact":"\'a [b] \\"c\\" {d}\'"}],"note":""}'
    assert parse_json_reply(reply) == {"results": [{"i": 0, "faults": [{"kind": "dropped", "fact": "'a [b] \"c\" {d}'"}], "note": ""}]}
    # A reply cut off inside a string, or with a mismatched bracket, is not guessed at.
    with pytest.raises(ValueError):
        parse_json_reply('{"results": [{"i": 0, "note": "cut')
    with pytest.raises(ValueError):
        parse_json_reply('{"results": [{"i": 0}}')


def test_strip_think_keeps_only_the_final_text():
    assert strip_think("<think>plan</think>\nHello.") == "Hello."


def test_codex_command_sends_our_instructions_and_ignores_the_users_setup(tmp_path):
    command = codex_command("codex", "gpt-6.1-sol", "low", tmp_path / "system.md", tmp_path / "reply.txt")
    assert command[:2] == ["codex", "exec"] and command[-1] == "-"
    assert 'cli_auth_credentials_store="file"' in command
    for flag in ("--ephemeral", "--ignore-user-config", "--ignore-rules"):
        assert flag in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert command[command.index("--model") + 1] == "gpt-6.1-sol"
    configs = [command[i + 1] for i, part in enumerate(command) if part == "-c"]
    assert f'model_instructions_file="{tmp_path / "system.md"}"' in configs
    assert 'model_reasoning_effort="low"' in configs and "project_doc_max_bytes=0" in configs
    assert not any("reasoning" in part for part in codex_command("codex", "m", "", tmp_path / "s", tmp_path / "r"))


def test_each_role_can_use_its_own_backend_and_model(monkeypatch):
    monkeypatch.setattr(llm, "load_dotenv", lambda _: None)
    for name in ("LLM_JUDGE_BACKEND", "LLM_JUDGE_MODEL", "LLM_JUDGE_REASONING", "LLM_GENERATOR_MODEL", "LLM_GENERATOR_REASONING"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_BACKEND", "opencode")
    monkeypatch.setenv("LLM_MODEL", "glm-5.3")
    monkeypatch.setenv("LLM_REASONING", "low")
    monkeypatch.setenv("LLM_GENERATOR_BACKEND", "codex")
    settings = LlmSettings.from_env()
    # Unset roles use the teacher's settings.
    assert settings.target("teacher") == settings.target("judge") == ("opencode", "glm-5.3", "low")
    assert settings.target("generator") == ("codex", "glm-5.3", "low")
    monkeypatch.setenv("LLM_JUDGE_BACKEND", "codex")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "gpt-6.1-sol")
    monkeypatch.setenv("LLM_JUDGE_REASONING", "high")
    assert LlmSettings.from_env().target("judge") == ("codex", "gpt-6.1-sol", "high")
    monkeypatch.setenv("LLM_JUDGE_BACKEND", "anthropic")
    with pytest.raises(SystemExit):
        LlmSettings.from_env()


@pytest.mark.parametrize("messages", [
    [],
    [{"role": "user", "content": "first"}, {"role": "user", "content": "second"}],
    [{"role": "assistant", "content": "reply"}, {"role": "user", "content": "hi"}],
    [{"role": "system", "content": "first"}, {"role": "system", "content": "second"}, {"role": "user", "content": "hi"}],
])
def test_cli_backend_rejects_unsupported_messages_before_connecting(messages, monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "CACHE", tmp_path)
    client = llm.LlmClient(replace(SETTINGS, backend="codex"))

    async def connect(backend):
        pytest.fail("unsupported messages reached the backend")

    monkeypatch.setattr(client, "_connect", connect)
    with pytest.raises(ValueError, match="one optional system message and one user message"):
        asyncio.run(client.complete(messages))


@pytest.mark.parametrize("backend", ["codex", "openai"])
def test_supported_messages_reach_the_backend(backend, monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "CACHE", tmp_path)
    client = llm.LlmClient(replace(SETTINGS, backend=backend))
    messages = [{"role": "system", "content": "instructions"}, {"role": "user", "content": "hi"}]
    if backend == "openai":
        messages += [{"role": "assistant", "content": "reply"}, {"role": "user", "content": "again"}]

    async def connect(backend):
        pass

    async def request(backend, model, reasoning, sent):
        assert sent == (messages if backend == "openai" else ("instructions", "hi"))
        return "done"

    monkeypatch.setattr(client, "_connect", connect)
    monkeypatch.setattr(client, "_request", request)
    assert asyncio.run(client.complete(messages)) == "done"


def _fake_router(monkeypatch, tmp_path, fail_first: bool = False, expire: bool = False) -> list[dict]:
    """An OpenRouter batch API that answers each request with its user message reversed. Returns the
    batches sent; with `fail_first`, the first request of the first batch fails; with `expire`, every batch
    expires with only its first request answered."""
    monkeypatch.setattr(llm, "CACHE", tmp_path)
    monkeypatch.setattr(llm, "_BATCH_POLL_SECONDS", 0.0)
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    sent: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-key"
        if request.method == "POST":
            body = json.loads(request.content)
            assert list(body)[:2] == ["endpoint", "model"]
            sent.append(body)
            return httpx.Response(202, json={"id": f"b{len(sent)}", "status": "validating"})
        number = int(request.url.path.rsplit("/b", 1)[1])
        results = [
            {"custom_id": item["custom_id"], "error": {"message": "overloaded"}, "response": None}
            if fail_first and number == 1 and index == 0 else
            {"custom_id": item["custom_id"], "error": None, "response": {"status_code": 200, "body": {
                "choices": [{"message": {"content": item["body"]["messages"][-1]["content"][::-1]}}]}}}
            for index, item in enumerate(sent[number - 1]["requests"])
        ]
        if expire:
            return httpx.Response(200, json={"id": f"b{number}", "status": "expired", "results": results[:1]})
        return httpx.Response(200, json={"id": f"b{number}", "status": "completed", "results": results})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs))
    return sent


def test_complete_batch_caches_each_reply_and_sends_failures_again(monkeypatch, tmp_path):
    sent = _fake_router(monkeypatch, tmp_path, fail_first=True)
    client = llm.LlmClient(replace(SETTINGS, model="vendor/model:batch", base_url="https://router.test/api/v1"))
    requests = [[{"role": "system", "content": "clean"}, {"role": "user", "content": text}] for text in ("ab", "cd", "ef")]

    async def no_request(*_args):
        raise AssertionError("complete should answer from the cache")

    monkeypatch.setattr(client, "_request", no_request)

    async def run():
        await client.complete_batch(requests)
        return [await client.complete(messages) for messages in requests]

    assert asyncio.run(run()) == ["ba", "dc", "fe"]
    assert [len(batch["requests"]) for batch in sent] == [3, 1]
    assert sent[0]["requests"][0]["body"]["reasoning_effort"] == "low"
    asyncio.run(client.complete_batch(requests))
    assert len(sent) == 2


def test_complete_batch_waits_for_the_batch_a_killed_run_sent(monkeypatch, tmp_path):
    sent = _fake_router(monkeypatch, tmp_path)
    settings = replace(SETTINGS, model="vendor/model:batch", base_url="https://router.test/api/v1")
    requests = [[{"role": "user", "content": "ab"}]]
    asyncio.run(llm.LlmClient(settings).complete_batch(requests))
    for reply in tmp_path.glob("llm/*/*.json"):
        reply.unlink()
    asyncio.run(llm.LlmClient(settings).complete_batch(requests))
    assert len(sent) == 1 and len(list(tmp_path.glob("llm/*/*.json"))) == 1


def test_an_expired_batch_keeps_its_answers_and_is_not_sent_again(monkeypatch, tmp_path):
    sent = _fake_router(monkeypatch, tmp_path, expire=True)
    settings = replace(SETTINGS, model="vendor/model:batch", base_url="https://router.test/api/v1")
    requests = [[{"role": "user", "content": text}] for text in ("ab", "cd")]
    with pytest.raises(llm.LlmError, match="expired with 1 of 2"):
        asyncio.run(llm.LlmClient(settings).complete_batch(requests))
    assert len(sent) == 1 and len(list(tmp_path.glob("llm/*/*.json"))) == 1


def test_response_cache_separates_backends_and_endpoints(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "CACHE", tmp_path)
    messages = [{"role": "user", "content": "hi"}]

    async def connect(backend):
        pass

    async def request(backend, model, reasoning, sent):
        return f"{backend}:{client.settings.base_url}"

    client = llm.LlmClient(SETTINGS)
    monkeypatch.setattr(client, "_connect", connect)
    monkeypatch.setattr(client, "_request", request)
    assert asyncio.run(client.complete(messages)) == "openai:None"
    assert asyncio.run(client.complete(messages)) == "openai:None" and client.cache_hits == 1
    client.settings = replace(SETTINGS, base_url="http://other-provider/v1")
    assert asyncio.run(client.complete(messages)) == "openai:http://other-provider/v1"
    client.settings = replace(SETTINGS, backend="codex")
    assert asyncio.run(client.complete(messages)) == "codex:None" and client.calls == 3


@pytest.mark.parametrize("retryable", [False, True])
def test_terminal_client_failure_uses_only_the_client_retry_budget(retryable, monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "CACHE", tmp_path)
    monkeypatch.setattr(llm, "_BACKOFF_BASE_SECONDS", 0.0)
    monkeypatch.setattr(llm, "_BACKOFF_CAP_SECONDS", 0.0)
    client = llm.LlmClient(SETTINGS)
    failure = llm.LlmError("transport failed", retryable=retryable)
    requests = 0

    async def connect(backend):
        pass

    async def request(backend, model, reasoning, messages):
        nonlocal requests
        requests += 1
        raise failure

    monkeypatch.setattr(client, "_connect", connect)
    monkeypatch.setattr(client, "_request", request)

    async def run():
        async with client:
            await client.complete([{"role": "user", "content": "bring milk"}])

    with pytest.raises(llm.LlmError) as error:
        asyncio.run(run())
    assert error.value is failure and requests == (llm._MAX_ATTEMPTS if retryable else 1)
    assert not list(tmp_path.rglob("*.json"))


def test_a_usage_limit_waits_without_using_up_retries(monkeypatch, tmp_path):
    import asyncio

    from whispernow_training.datagen import llm as llm_module

    monkeypatch.setattr(llm_module, "CACHE", tmp_path)
    monkeypatch.setattr(llm_module, "_LIMIT_WAIT_SECONDS", 0.0)
    settings = replace(SETTINGS, backend="codex", model="m", judge_model="m")
    client = llm_module.LlmClient(settings)
    refusals = llm_module._MAX_ATTEMPTS + 2
    expected_waits = refusals

    async def request(backend, model, reasoning, messages):
        nonlocal refusals
        if refusals:
            refusals -= 1
            raise llm_module.LlmError("codex exit 1: You've hit your usage limit. Try again at 9:00 PM.", retryable=True, limited=True)
        return "done"

    async def connect(backend):
        pass

    monkeypatch.setattr(client, "_request", request)
    monkeypatch.setattr(client, "_connect", connect)
    assert asyncio.run(client.complete([{"role": "user", "content": "hi"}])) == "done"
    assert client.limit_waits == expected_waits and client.retries == 0


def test_a_request_sent_before_the_limit_does_not_report_the_account_back(monkeypatch, tmp_path, capsys):
    import asyncio

    from whispernow_training.datagen import llm as llm_module

    monkeypatch.setattr(llm_module, "CACHE", tmp_path)
    monkeypatch.setattr(llm_module, "_LIMIT_WAIT_SECONDS", 0.05)
    settings = replace(SETTINGS, backend="codex", model="m", judge_model="m", concurrency=2)
    client = llm_module.LlmClient(settings)

    async def request(backend, model, reasoning, messages):
        if messages[1] == "early":
            await asyncio.sleep(0.01)
            return "done"
        raise llm_module.LlmError("You've hit your usage limit.", retryable=True, limited=True)

    async def connect(backend):
        pass

    async def run():
        refused = asyncio.ensure_future(client.complete([{"role": "user", "content": "refused"}]))
        await client.complete([{"role": "user", "content": "early"}])
        await asyncio.sleep(0.1)
        refused.cancel()

    monkeypatch.setattr(client, "_request", request)
    monkeypatch.setattr(client, "_connect", connect)
    asyncio.run(run())
    log = capsys.readouterr().err
    assert "USAGE LIMIT on codex" in log and "ACCOUNT OK" not in log


@pytest.mark.parametrize("exit_code, log, limited", [
    (0, "ERROR codex_api::endpoint::responses_websocket: failed to connect to websocket: HTTP error: 403 Forbidden", False),
    (1, "ERROR: Selected model is at capacity. Please try a different model.", False),
    (1, "You've hit your usage limit. Try again at 9:00 PM.", True),
])
def test_codex_refusals_classify_throttling_and_usage_limits(tmp_path, monkeypatch, exit_code, log, limited):
    monkeypatch.setattr(llm, "CACHE", tmp_path)
    fake = tmp_path / "fake-codex"
    fake.write_text(f"#!/bin/sh\ncat > /dev/null\necho {shlex.quote(log)} >&2\nexit {exit_code}\n")
    fake.chmod(0o755)
    monkeypatch.setattr(llm.shutil, "which", lambda _: str(fake))
    runner = llm._CodexRunner()
    try:
        with pytest.raises(llm.LlmError) as caught:
            asyncio.run(runner.complete("gpt-6.1-sol", "high", ("", "hi")))
        error = caught.value
        assert error.retryable and error.limited is limited and error.throttled is not limited
    finally:
        runner.stop()


@pytest.mark.parametrize("reply, exit_code, expected", [
    ('{"type": "result", "is_error": false, "result": "Cleaned text."}', 0, "Cleaned text."),
    ('{"type": "result", "is_error": true, "result": "Claude AI usage limit reached|1760000000"}', 1, "limited"),
    ('{"type": "result", "is_error": true, "result": "API Error: 529 Overloaded"}', 1, "retryable"),
])
def test_claude_runner_returns_the_result_and_classifies_failures(tmp_path, monkeypatch, reply, exit_code, expected):
    fake = tmp_path / "fake-claude"
    seen = tmp_path / "seen"
    fake.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > {seen}\ncat > /dev/null\necho {shlex.quote(reply)}\nexit {exit_code}\n')
    fake.chmod(0o755)
    monkeypatch.setattr(llm.shutil, "which", lambda _: str(fake))
    runner = llm._ClaudeRunner()
    try:
        if expected == "Cleaned text.":
            assert asyncio.run(runner.complete("claude-haiku-5-5", "low", ("Our rules.", "hi"))) == expected
            args = seen.read_text().splitlines()
            assert args[args.index("--system-prompt") + 1] == "Our rules." and args[args.index("--effort") + 1] == "low"
            assert args[args.index("--tools") + 1] == "" and "--safe-mode" in args
        else:
            with pytest.raises(llm.LlmError) as caught:
                asyncio.run(runner.complete("claude-haiku-5-5", "low", ("", "hi")))
            assert caught.value.limited is (expected == "limited") and caught.value.retryable is (expected == "retryable")
    finally:
        runner.stop()


def _alive(pid_file) -> bool:
    try:
        os.kill(int(pid_file.read_text()), 0)
    except (ProcessLookupError, ValueError, FileNotFoundError):
        return False
    return True


def test_a_subprocess_that_leaves_a_child_holding_its_pipe_returns_and_the_child_is_killed(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "_ORPHAN_GRACE_SECONDS", 0.3)
    pid_file = tmp_path / "child.pid"
    script = f"cat >/dev/null; echo oops >&2; sleep 60 & echo $! > {pid_file}; exit 3"
    started = time.monotonic()
    code, stderr = asyncio.run(llm.run_process_group(["sh", "-c", script], b"hi", tmp_path, 30))
    assert code == 3 and b"oops" in stderr
    assert time.monotonic() - started < 10
    time.sleep(0.2)
    assert not _alive(pid_file)


def test_a_subprocess_that_runs_over_its_timeout_is_killed_with_its_group(tmp_path):
    pid_file = tmp_path / "child.pid"
    script = f"sleep 60 & echo $! > {pid_file}; wait"
    with pytest.raises(TimeoutError):
        asyncio.run(llm.run_process_group(["sh", "-c", script], b"", tmp_path, 0.5))
    time.sleep(0.2)
    assert not _alive(pid_file)
