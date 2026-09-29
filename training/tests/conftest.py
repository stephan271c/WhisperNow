import json

import pytest

from whispernow_training.datagen.llm import LlmSettings


class FakeLlm:
    """Answers each request with `reply(messages, role)`; counts calls."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0
        self.cache_hits = 0
        self.settings = LlmSettings(
            backend="openai",
            model="teacher-model",
            judge_model="judge-model",
            concurrency=1,
            reasoning="low",
            judge_reasoning="low",
            opencode_provider="",
            base_url=None,
        )

    async def complete(self, messages, *, role="teacher"):
        self.calls += 1
        return self.reply(messages, role)


def judge_items(messages):
    """The item dicts a judge request carries."""
    lines = messages[-1]["content"].split("\n")[1:]
    return [json.loads(line) for line in lines]


@pytest.fixture
def fake_llm():
    return FakeLlm
