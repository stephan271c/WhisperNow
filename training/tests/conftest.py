import json
from dataclasses import replace

import pytest

from whispernow_training.datagen.llm import LlmSettings

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


@pytest.fixture
def seed_families():
    from whispernow_training.datagen.inject import injected_rows
    from whispernow_training.datagen.schema import Example

    clean = " ".join(["Meet on Wednesday at 4 by the red door. Bring 12 chairs and 3 tables."] * 11)
    families = []
    # Hashing these example IDs put each injected derivative opposite its parent.
    for id_, split in (("parent-13", "train"), ("parent-51", "val")):
        parent = Example(id=id_, input=clean, output=clean, source="teacher", category="multi_topic", license="x",
                         split="train")
        derivative, = injected_rows([parent], count=1, seed=114)
        assert derivative.id != parent.id and derivative.input != parent.input
        assert derivative.output == parent.output and derivative.seed_group == parent.seed_group
        families.append((parent, derivative, split))
    return families
