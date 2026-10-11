from whispernow_training.datagen.schema import Example
from whispernow_training.model import evaluate


def _row(id_, words, split="test", tags=()):
    text = " ".join(["word"] * words)
    return Example(id=id_, input=text, output=text, source="s", category="c", license="l", tags=list(tags), split=split)


def test_scores_group_by_length_pattern_and_test_generation(tmp_path, monkeypatch):
    ids = tmp_path / "ids.txt"
    ids.write_text("old\n")
    monkeypatch.setattr(evaluate, "TEST_IDS_109", ids)
    rows = [_row("old", 20), _row("new", 200, tags=["pattern:restart"]), _row("bench", 700, split="", tags=["pattern:restart"])]
    outputs = {r.id: {"output": r.output, "hit_limit": False} for r in rows}
    scores = evaluate.score(rows, outputs)
    assert set(scores["length"]) == {"<50", "150-300", "600-1000"}
    assert {k: v["rows"] for k, v in scores["test_group"].items()} == {"todo109": 1, "new": 1}
    assert scores["pattern"]["restart"]["rows"] == 2


def test_length_ratio_uses_total_output_and_reference_words():
    rows = [_row("a", 4), _row("b", 2)]
    outputs = {"a": {"output": "word word", "hit_limit": False},
               "b": {"output": "word word word word", "hit_limit": False}}
    assert evaluate.score(rows, outputs)["overall"]["length_ratio"] == 1.0
