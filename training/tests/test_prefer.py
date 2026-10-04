from whispernow_training.datagen.schema import EMPTY, Example
from whispernow_training.model import bench, evaluate, prefer
from whispernow_training.model.metrics import check

LONG = " ".join(["we need to finish the report"] * 40)


def _row(id_: str, words: str = LONG, tags: list[str] | None = None, output: str = "Finish the report.") -> Example:
    return Example(id=id_, input=words, output=output, source="teacher", category="multi_topic", license="x",
                   tags=tags or [], seed_group=f"g-{id_}", split="train")


def test_only_insertions_allows_added_words_and_nothing_else():
    assert prefer.only_insertions("Buy milk. Call Dana at 3.", "Buy milk and 2 eggs.\nCall Dana at 3 about the lease.")
    # Case and punctuation may change where a fact goes in.
    assert prefer.only_insertions("Buy milk. Then call Dana.", "Buy milk, then call Dana and Sam.")
    assert not prefer.only_insertions("Buy milk.", "Get milk and eggs.")
    assert not prefer.only_insertions("Buy milk and eggs.", "Buy milk and 2 eggs, not bread.".replace("milk and ", ""))
    assert not prefer.only_insertions("Buy milk.", "Buy milk.")


def test_inputs_are_long_or_corrections_and_never_held_out():
    train = [_row(f"long-{i}") for i in range(10)]
    train += [_row(f"fix-{i}", "short one", ["reasoned_correction"]) for i in range(5)]
    train += [_row("short", "short one"), _row("empty", output=EMPTY)]
    held = _row("bench-1")
    train.append(Example(**{**held.__dict__, "id": "same-group", "seed_group": held.seed_group}))
    picked = {row.id for row in prefer.select_inputs(train, [held])}
    assert {f"long-{i}" for i in range(10)} | {f"fix-{i}" for i in range(5)} == picked


def _verdict(id_, k, output, failed=(), note=""):
    return {"id": id_, "k": k, "output": output, "failed": list(failed), "note": note, "passed": not failed}


def test_pairs_put_the_minimal_pair_first_and_match_lengths():
    rows = {"a": _row("a"), "b": _row("b"), "c": _row("c")}
    verdicts = [
        _verdict("a", 0, "Finish.", ["facts_complete"], "dropped 'the report'"),
        _verdict("a", 1, "Finish the report, um.", ["leftover_filler"]),
        _verdict("a", 2, "Finish the whole report now, please.", []),
        _verdict("a", 3, "Finish the report.", []),
        # Nothing passes: a failure is paired with the gated reference.
        _verdict("b", 0, "Finish the memo.", ["nothing_invented"]),
        # Only passes: no pair, one RFT row.
        _verdict("c", 0, "Finish the report.", []),
    ]
    assert prefer.restore_candidates(verdicts) == [verdicts[0]]
    restored = [{"id": "a", "k": 0, "rejected": "Finish.", "restored": "Finish the report.", "inserted_only": True,
                 "failed": [], "passed": True}]
    pairs, rft = prefer.build_pairs(rows, verdicts, restored)
    by_id = [(p["id"], p["kind"], p["chosen_from"], p["chosen"], p["rejected"]) for p in pairs]
    assert by_id == [
        ("a", "minimal", "restored", "Finish the report.", "Finish."),
        ("a", "filler", "sample", "Finish the report.", "Finish the report, um."),
        ("b", "invented", "reference", "Finish the report.", "Finish the memo."),
    ]
    assert {row["id"] for row in rft} <= {"a", "c"} and all(row["output"] for row in rft)


def test_code_failures_catch_fillers_and_layout():
    row = _row("a")
    assert prefer.code_failures(row, "Um, finish the report.", False) == ["leftover_filler"]
    assert "formatting_added" in prefer.code_failures(row, "- Finish the report.\n- Today.", False)
    assert prefer.code_failures(row, "Finish the report.", True) == ["loop"]
    assert prefer.kind_of(["leftover_filler", "facts_complete"]) == "dropped"


def test_length_ratio_and_latency_cases():
    row = _row("a", output="one two three four")
    summary = evaluate._summarize([(check(row, "one two"), None), (check(row, "one two three four five six"), None)])
    assert summary["length_ratio"] == 1.0
    names = list(bench.cases())
    assert names == ["short", "words_100", "words_200", "long", "words_500", "words_1000"]
