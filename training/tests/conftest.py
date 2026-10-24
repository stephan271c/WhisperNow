import json
from dataclasses import replace

import pytest

from whispernow_training.datagen.llm import Generated, LlmSettings, Route

SETTINGS = LlmSettings(
    backend="openai",
    model="teacher-model",
    judge_model="judge-model",
    concurrency=1,
    reasoning="low",
    judge_reasoning="low",
    opencode_provider="",
    base_url=None,
    judge_backend="openai",
    generator_backend="openai",
    generator_model="teacher-model",
    generator_reasoning="low",
)


class FakeLlm:
    """Answers each request with `reply(messages, role)`; counts calls."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0
        self.cache_hits = 0
        self.tokens = 0
        self.request_seconds = 0.0
        self.limit_waits = 0
        self.attempts = []
        # (role, reasoning override) per request.
        self.reasonings = []
        self.settings = replace(SETTINGS)

    async def complete(self, messages, *, role="teacher", attempt=0, reasoning=None):
        self.calls += 1
        self.attempts.append(attempt)
        self.reasonings.append((role, reasoning))
        return self.reply(messages, role)

    async def generate(self, messages, *, cached=True, attempt=0):
        settings = self.settings
        route = Route(settings.generator_backend, settings.generator_model, settings.generator_reasoning)
        return Generated(await self.complete(messages, role="generator", attempt=attempt), route)


def judge_items(messages):
    """The item dicts a judge request carries."""
    lines = messages[-1]["content"].split("\n")[1:]
    return [json.loads(line) for line in lines]


@pytest.fixture(autouse=True)
def _relabel_records(tmp_path, monkeypatch):
    """No test reads the real relabel records, which a running relabel may leave partial."""
    from whispernow_training.datagen import relabel

    monkeypatch.setattr(relabel, "DIR", tmp_path / "relabel")
    monkeypatch.setattr(relabel, "INPUTS", tmp_path / "relabel" / "inputs")


@pytest.fixture
def fake_llm():
    return FakeLlm
