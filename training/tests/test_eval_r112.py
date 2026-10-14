import asyncio
import json

from conftest import judge_items

from whispernow_training.datagen.judge import judge_all
from whispernow_training.datagen.schema import Example
from whispernow_training.model import evaluate


def _row(id_, teacher_model=None, teacher_reasoning=None):
    return Example(id=id_, input="hi", output="Hi.", source="s", category="c", license="x",
                   teacher_model=teacher_model, teacher_reasoning=teacher_reasoning)


def test_reference_tag_marks_only_uniform_r112_provenance():
    new = [_row("a", "gpt-6.1-sol", "high"), _row("b", "gpt-6.1-sol", "high")]
    assert evaluate.reference_tag(new) == "r112"
    assert evaluate.reference_tag([*new, _row("c", "glm-5.3-flash")]) == ""
    assert evaluate.reference_tag([_row("d", "gpt-6.1-sol")]) == ""


def test_judge_all_sends_long_rows_alone(fake_llm, tmp_path):
    sizes = []

    def reply(messages, role):
        items = judge_items(messages)
        sizes.append(len(items))
        return json.dumps({"results": [{"i": item["i"], "ok": True, "note": ""} for item in items]})

    items = [(f"s{i}", {"transcript": "short one", "output": "x"}) for i in range(4)]
    items += [(f"l{i}", {"transcript": " ".join(["word"] * 301), "output": "x"}) for i in range(2)]
    rules = tmp_path / "rules.md"
    rules.write_text("rules")
    verdicts = asyncio.run(judge_all(fake_llm(reply), items, tmp_path / "v.jsonl", instructions_file=rules, criteria=("ok",),
                                     batch_size=15, solo_over_words=300, progress=lambda line: None))
    assert len(verdicts) == 6 and sorted(sizes) == [1, 1, 4]
