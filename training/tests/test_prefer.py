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


def _verdict(id_, k, output, failed=(), note="", faults=()):
    return {"id": id_, "k": k, "output": output, "failed": list(failed), "faults": list(faults), "note": note,
            "passed": not failed}


def test_pairs_put_the_minimal_pair_first_and_match_lengths():
    rows = {"a": _row("a"), "b": _row("b"), "c": _row("c")}
    verdicts = [
        _verdict("a", 0, "Finish.", ["dropped"], "- dropped: 'the report'"),
        _verdict("a", 1, "Finish the report, um.", ["leftover_filler"]),
        _verdict("a", 2, "Finish the whole report now, please.", []),
        _verdict("a", 3, "Finish the report.", []),
        # Nothing passes: a failure is paired with the gated reference.
        _verdict("b", 0, "Finish the memo.", ["invented"]),
        # Only passes: no pair, one RFT row.
        _verdict("c", 0, "Finish the report.", []),
    ]
    assert prefer.restore_candidates(verdicts) == [verdicts[0]]
    restored = [{"id": "a", "k": 0, "rejected": "Finish.", "restored": "Finish the report.", "edit_ok": True,
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
    assert prefer.kind_of(["leftover_filler", "dropped"]) == "dropped"
    assert prefer.kind_of(["dropped", "wrong_side"]) == "wrong_side"


def test_length_ratio_and_latency_cases():
    row = _row("a", output="one two three four")
    summary = evaluate._summarize([(check(row, "one two"), None), (check(row, "one two three four five six"), None)])
    assert summary["length_ratio"] == 1.0
    names = list(bench.cases())
    assert names == ["short", "words_100", "words_200", "long", "words_500", "words_1000"]


def _judged(failed=(), faults=()):
    from whispernow_training.datagen.judge import Verdict

    return Verdict(id="x", passed=not failed, failed=list(failed), note="", judge_model="j", faults=list(faults))


def test_judge_failures_follow_the_eval_judges_fault_kinds():
    faults = [{"kind": "unresolved", "fact": "kept 'Tuesday' beside 'Wednesday'"},
              {"kind": "dropped", "fact": "'by Friday'"}, {"kind": "wrong_side", "fact": "'4' for '5'"}]
    failed, kept = prefer.judge_failures(_judged(["facts_kept"], faults))
    assert failed == ["wrong_side", "dropped"] and [f["kind"] for f in kept] == ["dropped", "wrong_side"]
    # An unresolved correction alone loses nothing, so the output passes.
    assert prefer.judge_failures(_judged([], faults[:1])) == ([], [])
    assert prefer.judge_failures(_judged(["not_answered"]))[0] == ["answered"]
    assert prefer.judge_failures(_judged(["judge_error"]))[0] == ["judge_error"]
    assert prefer.fault_note(kept) == "- dropped: 'by Friday'\n- wrong_side: '4' for '5'"


def test_only_fact_edits_replace_the_quoted_value_and_nothing_else():
    faults = [{"kind": "wrong_side", "fact": "kept 'Tuesday' though the speaker settled on 'Wednesday'"}]
    assert prefer.only_fact_edits("Meet Tuesday at 4.", "Meet Wednesday at 4.", faults)
    assert prefer.only_fact_edits("Meet Tuesday at 4.", "Meet Wednesday at 4 by the door.", faults)
    assert not prefer.only_fact_edits("Meet Tuesday at 4.", "See you Wednesday at 4.", faults)
    assert not prefer.only_fact_edits("Meet Tuesday at 4.", "Meet Tuesday at 4.", faults)


def test_restore_candidates_take_lost_facts_and_check_the_edit_by_kind():
    dropped = _verdict("a", 0, "Finish.", ["dropped"], faults=[{"kind": "dropped", "fact": "'the report'"}])
    changed = _verdict("b", 0, "Pay 30 dollars.", ["changed"], faults=[{"kind": "changed", "fact": "'30' for '40'"}])
    invented = _verdict("c", 0, "Pay 40 dollars now.", ["invented", "dropped"])
    filler = _verdict("d", 0, "Um, pay.", ["dropped", "leftover_filler"])
    assert prefer.restore_candidates([dropped, changed, invented, filler]) == [dropped, changed]
    assert prefer.edit_ok(dropped, "Finish the report.")
    assert not prefer.edit_ok(dropped, "Complete the report.")
    assert prefer.edit_ok(changed, "Pay 40 dollars.")
    assert not prefer.edit_ok(changed, "Send 40 dollars.")
