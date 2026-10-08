import asyncio
import os
import time

import pytest

from whispernow_training.datagen import llm
from whispernow_training.datagen.llm import LlmSettings, ResponseCache, cache_key, codex_command, parse_json_reply, strip_think


def test_cache_key_depends_on_model_reasoning_and_messages():
    messages = [{"role": "user", "content": "hi"}]
    assert cache_key("m", "low", messages) == cache_key("m", "low", [dict(messages[0])])
    assert cache_key("m", "low", messages) != cache_key("m", "high", messages)
    assert cache_key("m", "low", messages) != cache_key("n", "low", messages)
    # A second attempt is cached apart; the first keeps the key it always had.
    assert cache_key("m", "low", messages, 0) == cache_key("m", "low", messages)
    assert cache_key("m", "low", messages, 1) not in (cache_key("m", "low", messages), cache_key("m", "low", messages, 2))


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
    for name in ("LLM_JUDGE_BACKEND", "LLM_JUDGE_MODEL", "LLM_JUDGE_REASONING", "LLM_GENERATOR_MODEL", "LLM_GENERATOR_REASONING"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_BACKEND", "opencode")
    monkeypatch.setenv("LLM_MODEL", "glm-5.3")
    monkeypatch.setenv("LLM_REASONING", "low")
    monkeypatch.setenv("LLM_GENERATOR_BACKEND", "codex")
    settings = LlmSettings.from_env()
    # Unset roles keep the teacher's settings, so their cached requests keep their keys.
    assert settings.target("teacher") == settings.target("judge") == ("opencode", "glm-5.3", "low")
    assert settings.target("generator") == ("codex", "glm-5.3", "low")
    monkeypatch.setenv("LLM_JUDGE_BACKEND", "codex")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "gpt-6.1-sol")
    monkeypatch.setenv("LLM_JUDGE_REASONING", "high")
    assert LlmSettings.from_env().target("judge") == ("codex", "gpt-6.1-sol", "high")
    monkeypatch.setenv("LLM_JUDGE_BACKEND", "claude")
    with pytest.raises(SystemExit):
        LlmSettings.from_env()


def test_a_usage_limit_waits_without_using_up_retries_and_is_logged(monkeypatch, tmp_path, capsys):
    import asyncio

    from whispernow_training.datagen import llm as llm_module

    monkeypatch.setattr(llm_module, "CACHE", tmp_path)
    monkeypatch.setattr(llm_module, "_LIMIT_WAIT_SECONDS", 0.0)
    settings = LlmSettings(
        backend="codex", model="m", judge_model="m", concurrency=1, reasoning="low", judge_reasoning="low",
        opencode_provider="", base_url=None,
    )
    client = llm_module.LlmClient(settings)
    refusals = llm_module._MAX_ATTEMPTS + 2

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
    log = capsys.readouterr().err
    assert log.count("USAGE LIMIT on codex") == 1 and "ACCOUNT OK again" in log
    assert llm_module._CODEX_LIMITED.search("Error: You've hit your usage limit. Upgrade to Pro")
    assert llm_module._CODEX_SIGNED_OUT.search("401 Unauthorized: please run codex login")


def test_a_request_sent_before_the_limit_does_not_report_the_account_back(monkeypatch, tmp_path, capsys):
    import asyncio

    from whispernow_training.datagen import llm as llm_module

    monkeypatch.setattr(llm_module, "CACHE", tmp_path)
    monkeypatch.setattr(llm_module, "_LIMIT_WAIT_SECONDS", 0.05)
    settings = LlmSettings(
        backend="codex", model="m", judge_model="m", concurrency=2, reasoning="low", judge_reasoning="low",
        opencode_provider="", base_url=None,
    )
    client = llm_module.LlmClient(settings)

    async def request(backend, model, reasoning, messages):
        if messages[0]["content"] == "early":
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
