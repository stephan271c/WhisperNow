import pytest

from whispernow_training.datagen.llm import ResponseCache, cache_key, parse_json_reply, strip_think


def test_cache_key_depends_on_model_reasoning_and_messages():
    messages = [{"role": "user", "content": "hi"}]
    assert cache_key("m", "low", messages) == cache_key("m", "low", [dict(messages[0])])
    assert cache_key("m", "low", messages) != cache_key("m", "high", messages)
    assert cache_key("m", "low", messages) != cache_key("n", "low", messages)


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
    assert parse_json_reply('{"items": [{"input": "x"}, {"input": "y"}]') == [{"input": "x"}, {"input": "y"}]
    with pytest.raises(ValueError):
        parse_json_reply("no json here")


def test_strip_think_keeps_only_the_final_text():
    assert strip_think("<think>plan</think>\nHello.") == "Hello."
