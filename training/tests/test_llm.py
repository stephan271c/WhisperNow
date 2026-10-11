import asyncio
import os
import shlex
import time
from dataclasses import replace

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
    monkeypatch.setenv("LLM_JUDGE_BACKEND", "claude")
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
