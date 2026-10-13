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


def test_protected_token_rates_use_complete_reference_token_counts(tmp_path, monkeypatch):
    ids = tmp_path / "ids.txt"
    ids.write_text("")
    monkeypatch.setattr(evaluate, "TEST_IDS_109", ids)
    references = {"number": "Order 5 boxes.", "path": "Open app.py.",
                  "kept": "Order 5 boxes via app.py.", "plain": "Hello."}
    rows = [Example(id=id_, input=text, output=text, source="s", category="c", license="l")
            for id_, text in references.items()]
    outputs = {"number": {"output": "Order 15 boxes.", "hit_limit": False},
               "path": {"output": "Open xapp.py.", "hit_limit": False},
               "kept": {"output": references["kept"], "hit_limit": False},
               "plain": {"output": references["plain"], "hit_limit": False}}
    scores = evaluate.score(rows, outputs)
    for summary in (scores["overall"], scores["category"]["c"]):
        assert summary["rows"] == 4
        assert summary["protected_kept"] == {"rate": 33.33, "n": 3}
        assert summary["protected_token_recall"] == 50.0


def test_chrf_summary_averages_unequal_reference_scores(tmp_path, monkeypatch):
    ids = tmp_path / "ids.txt"
    ids.write_text("")
    monkeypatch.setattr(evaluate, "TEST_IDS_109", ids)
    rows = [Example(id="partial", input="ab", output="ab", source="s", category="partial", license="l"),
            Example(id="disjoint", input="xyz", output="xyz", source="s", category="disjoint", license="l")]
    outputs = {"partial": {"output": "aab", "hit_limit": False},
               "disjoint": {"output": "abc", "hit_limit": False}}
    scores = evaluate.score(rows, outputs)
    # The partial match's precisions are 2/3 and 1/2, both recalls 1; disjoint text scores 0.
    assert scores["category"]["partial"]["chrf"] == 87.5
    assert scores["category"]["disjoint"]["chrf"] == 0.0
    assert scores["overall"]["chrf"] == 43.75
