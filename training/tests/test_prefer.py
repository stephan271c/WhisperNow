import json
from dataclasses import replace

import httpx
from conftest import SETTINGS, judge_items

from whispernow_training.datagen import llm
from whispernow_training.datagen.schema import EMPTY, Example, read_jsonl
from whispernow_training.model import evaluate, prefer

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


def test_pairs_and_rft_keep_injected_families_together(seed_families, tmp_path):
    train = [row for parent, derivative, _ in seed_families for row in (parent, derivative)]
    rows = {row.id: row for row in prefer.select_inputs(train, [])}
    assert set(rows) == {row.id for row in train}
    verdicts = [verdict for row in rows.values() for verdict in (
        _verdict(row.id, 0, "Meet on Wednesday.", ["dropped"]),
        _verdict(row.id, 1, row.output),
    )]
    pairs, rft = prefer.build_pairs(rows, verdicts, [])
    assert (pairs, rft) == prefer.build_pairs(rows, verdicts, [])
    prefer._write(tmp_path / "pairs.jsonl", pairs)
    saved = list(read_jsonl(tmp_path / "pairs.jsonl"))
    for parent, derivative, split in seed_families:
        family = [pair for pair in saved if pair["seed_group"] == parent.seed_group]
        assert {pair["id"] for pair in family} == {parent.id, derivative.id}
        assert {pair["split"] for pair in family} == {split}
    train_parent, train_derivative, _ = seed_families[0]
    assert {row["id"] for row in rft} == {train_parent.id, train_derivative.id}
    assert {row["seed_group"] for row in rft} == {train_parent.seed_group}
    assert all(row["split"] == "train" for row in rft)
    held = [derivative for _, derivative, _ in seed_families]
    assert prefer.select_inputs(train, held) == []


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


def test_run_validates_samples_verdicts_and_restored_judgments(tmp_path, monkeypatch):
    out = tmp_path / "prefer"
    out.mkdir()
    rows = [_row("parent-13", "Bring uh 12 chairs.", output="Bring 12 chairs.")]
    prefer._write(out / "inputs.jsonl", [row.__dict__ for row in rows])
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fixture weights")
    generation = {"source": "server", "revision": evaluate.model_revision(model), "model_path": str(model)}
    monkeypatch.setattr(prefer, "server_generation", lambda server: generation)
    rules, restore_rules = tmp_path / "judge.md", tmp_path / "restore.md"
    rules.write_text("judge-contract")
    restore_rules.write_text("restore-contract")
    monkeypatch.setattr(evaluate, "JUDGE_INSTRUCTIONS", rules)
    monkeypatch.setattr(prefer, "RESTORE", restore_rules)
    monkeypatch.setattr(llm, "CACHE", tmp_path / "cache")
    settings = replace(SETTINGS, backend="codex", model="gpt-6.1-sol", reasoning="high",
                       judge_backend="codex", judge_model="gpt-6.1-sol", judge_reasoning="high")
    monkeypatch.setattr(llm.LlmSettings, "from_env", lambda: settings)
    samples, requests = [], []

    async def sample(server, pending, **kwargs):
        samples.append([(row.input, k) for row, k in pending])
        return [{"id": row.id, "k": k, "output": "Bring chairs.", "hit_limit": False}
                for row, k in pending]

    async def connect(self, backend):
        pass

    async def request(self, backend, model, reasoning, messages):
        requests.append(messages)
        system, user = messages
        if system == "restore-contract":
            return rows[0].output
        items = judge_items([{"content": user}])
        return json.dumps({"results": [
            {"i": item["i"], "not_answered": True, "faults": [] if item["output"] == item["reference"] else
             [{"kind": "dropped", "fact": repr(rows[0].output.split()[1])}]} for item in items
        ]})

    monkeypatch.setattr(prefer, "sample", sample)
    monkeypatch.setattr(llm.LlmClient, "_connect", connect)
    monkeypatch.setattr(llm.LlmClient, "_request", request)
    prefer.run(out, "http://fixture/v1")
    assert len(requests) == 3
    prefer.run(out, None)
    assert len(requests) == 3 and len(samples) == 1
    rows[0] = replace(rows[0], input="Bring um 14 chairs.", output="Bring 14 chairs.")
    prefer._write(out / "inputs.jsonl", [row.__dict__ for row in rows])
    prefer.run(out, "http://fixture/v1")
    assert len(requests) == 6 and samples[-1] == [("Bring um 14 chairs.", k) for k in range(prefer.SAMPLES + 1)]
    restored = list(read_jsonl(out / "restored.jsonl"))
    assert [row["restored"] for row in restored] == ["Bring 12 chairs.", "Bring 14 chairs."]
    assert len(list(read_jsonl(out / "samples.jsonl"))) == 2 * (prefer.SAMPLES + 1)
    rules.write_text("changed-judge-contract")
    prefer.run(out, None)
    assert len(requests) == 8 and len(samples) == 2


def test_run_resumes_partial_samples_without_replacing_completed_outputs(tmp_path, monkeypatch):
    out = tmp_path / "prefer"
    out.mkdir()
    row = _row("a", "Bring uh 12 chairs.", output="Bring 12 chairs.")
    prefer._write(out / "inputs.jsonl", [row.__dict__])
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fixture weights")
    generation = {"source": "server", "revision": evaluate.model_revision(model), "model_path": str(model)}
    monkeypatch.setattr(prefer, "server_generation", lambda server: generation)
    requested = []

    def reply(request):
        requested.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": f"Provider completion {len(requested)}."},
                                                       "finish_reason": "stop"}]})

    client = httpx.AsyncClient
    transport = httpx.MockTransport(reply)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client(transport=transport, **kwargs))

    async def verdicts_for(rows, samples, judged):
        return [_verdict(item["id"], item["k"], item["output"]) for item in samples]

    async def restore(*args):
        return []

    monkeypatch.setattr(prefer, "verdicts_for", verdicts_for)
    monkeypatch.setattr(prefer, "restore", restore)
    prefer.run(out, "http://fixture/v1")
    original = list(read_jsonl(out / "samples.jsonl"))
    completed = [item for item in original if item["k"] != 2]
    prefer._write(out / "samples.jsonl", completed)
    requested.clear()
    prefer.run(out, "http://fixture/v1")
    saved = list(read_jsonl(out / "samples.jsonl"))
    assert len(requested) == 1 and requested[0]["seed"] == 2
    assert saved[:len(completed)] == completed and saved[-1]["k"] == 2
    assert saved[-1]["output"] != original[2]["output"]
    prefer.run(out, None)
    assert len(requested) == 1 and list(read_jsonl(out / "samples.jsonl")) == saved
