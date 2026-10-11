import asyncio
import json
from dataclasses import replace

import pytest
from conftest import SETTINGS, judge_items

from whispernow_training.datagen import gates, inject, relabel
from whispernow_training.datagen.llm import LlmError, require_r112
from whispernow_training.datagen.schema import EMPTY, Example, write_examples

SOL = replace(
    SETTINGS, backend="codex", model="gpt-6.1-sol", judge_model="gpt-6.1-sol", reasoning="high",
    judge_reasoning="high", judge_backend="codex", generator_backend="codex",
    generator_model="gpt-6.1-sol", generator_reasoning="high",
)


def _row(id_, input_, output, source="shantanugoel/aawaaz-transcript-cleanup-dataset", license="MIT", **fields):
    return Example(id=id_, input=input_, output=output, source=source, category="c", license=license, **fields)


def _passing(messages, role, fail=lambda item: ""):
    if role != "judge":
        return "unused"
    results = []
    for item in judge_items(messages):
        note = fail(item)
        results.append({"i": item["i"], **{c: True for c in gates.COMPLETENESS_CRITERIA}, "clean": not note, "note": note})
    return json.dumps({"results": results})


def test_the_guard_refuses_any_role_off_gpt_6_1_sol_high(fake_llm):
    require_r112(SOL, "teacher", "judge", "generator")
    with pytest.raises(SystemExit, match="judge: codex/gpt-6.1-sol/low"):
        require_r112(replace(SOL, judge_reasoning="low"), "teacher", "judge", "generator")
    with pytest.raises(SystemExit, match="teacher: opencode/glm-5.3-flash/low"):
        require_r112(replace(SOL, backend="opencode", model="glm-5.3-flash", reasoning="low"), "teacher", "judge", "generator")


@pytest.fixture
def relabel_dirs(tmp_path, monkeypatch):
    built = tmp_path / "train.jsonl"
    monkeypatch.setitem(relabel.SOURCES, "build", (built,))
    return tmp_path, built


def test_run_resumes_records_provenance_and_apply_replaces_or_drops(fake_llm, relabel_dirs):
    _, built = relabel_dirs
    rows = [
        _row("a", "uh two bags of ice", "Two bags of ice — please."),
        _row("b", "send it to Ana no wait to Sam", "Send it to Ana, no wait, to Sam.", source="teacher", license="zai-output", teacher_model="glm-5.3-flash"),
        _row("e", "um uh", EMPTY, source="injected", license="MIT"),
        Example(
            id="i", input="two bags of uh ice", output="x", source="injected", category="filler_injection", license="MIT"
        ),
    ]
    write_examples(built, rows)

    def reply(messages, role):
        if role == "judge":
            return _passing(messages, role, lambda item: "kept the abandoned Ana" if "Ana" in item["reference"] else "")
        user = messages[-1]["content"]
        if "um uh" in user:
            return EMPTY
        return "Send it to Ana." if "Sam" in user else "2 bags of ice, please."

    llm = fake_llm(reply)
    llm.settings = SOL
    scope_rows = relabel.inputs("build")
    assert [e.id for e in scope_rows] == ["a", "b", "e"] and len(relabel.old_injected()) == 1
    summary = asyncio.run(relabel.run(llm, "build", scope_rows))
    assert summary["rows"] == 3 and summary["passed"] == 2
    records = relabel.load_records("build")
    assert records["a"]["teacher_model"] == "gpt-6.1-sol" and records["a"]["teacher_reasoning"] == "high"
    assert not records["b"]["passed"] and records["b"]["first_note"] == "kept the abandoned Ana"
    calls = llm.calls
    asyncio.run(relabel.run(llm, "build", scope_rows))
    assert llm.calls == calls
    relabel.require_complete("build", records)

    kept, dropped = relabel.apply([*rows[:3], _row("new", "hello", "Hello.")], records)
    assert [e.id for e in kept] == ["a", "e"] and {e.id: r for e, r in dropped} == {"b": "failed the relabel gate", "new": "not relabelled"}
    a = kept[0]
    assert a.output == "2 bags of ice, please." and a.teacher_reasoning == "high" and relabel.TAG in a.tags and a.license == "MIT"
    # Injected rows made from a relabelled parent carry its provenance.
    parent = replace(a, id="p", output="Bring 2 bags of ice and the blue cooler to the dock by noon.", split="train")
    injected = inject.injected_rows([parent], 1, 1)
    assert injected and injected[0].teacher_model == "gpt-6.1-sol" and injected[0].teacher_reasoning == "high"

    report = relabel.compare(relabel.inputs("build"), kept, records)
    assert report["all"]["old"]["em_dash"] == 0.5 and report["all"]["new"]["em_dash"] == 0.0
    assert report["by_kind"]["generated"]["dropped"] == 1
    assert "| aawaaz-transcript-cleanup-dataset |" in relabel.compare_markdown(report)


def test_run_rechecks_changed_content_and_contract(fake_llm, tmp_path, monkeypatch):
    rows = [_row("a", "hello there", "Old reference."), _row("b", "bye now", "Other old reference.")]
    rules = tmp_path / "rules.md"
    rules.write_text("first judge rules")
    monkeypatch.setitem(relabel.JUDGES, "build", replace(relabel.JUDGES["build"], instructions=rules))
    teachers = []

    def reply(messages, role):
        if role == "judge":
            return _passing(messages, role)
        transcript = messages[-1]["content"].split("<transcript>\n", 1)[1].split("\n</transcript>", 1)[0]
        teachers.append(transcript)
        return {"hello there": "Hello there.", "hi again": "Hi again.", "bye now": "Bye now."}[transcript]

    llm = fake_llm(reply)
    llm.settings = SOL
    asyncio.run(relabel.run(llm, "build", rows))
    assert len(teachers) == 2 and set(teachers) == {"bye now", "hello there"}
    before = relabel.load_records("build")
    calls = llm.calls
    assert asyncio.run(relabel.run(llm, "build", rows))["rows"] == 0 and llm.calls == calls
    rows[0] = replace(rows[0], input="hi again")
    assert asyncio.run(relabel.run(llm, "build", rows))["rows"] == 1
    current = relabel.load_records("build")
    assert current["a"]["output"] == "Hi again." and current["b"] == before["b"]
    rules.write_text("changed judge rules")
    assert asyncio.run(relabel.run(llm, "build", rows))["rows"] == 2
    assert set(teachers[-2:]) == {"hi again", "bye now"}


def test_a_run_with_no_finished_group_stops_and_shows_where_it_waits(fake_llm, relabel_dirs, capsys):
    _, built = relabel_dirs
    write_examples(built, [_row("a", "hi there", "Hi there.")])

    class Stuck(fake_llm):
        async def complete(self, messages, **_):
            await asyncio.Event().wait()

    llm = Stuck(None)
    llm.settings = SOL
    with pytest.raises(relabel.Stalled):
        asyncio.run(relabel.run(llm, "build", relabel.inputs("build"), stall_seconds=0.05))
    assert "STALLED" in capsys.readouterr().err and not relabel.load_records("build")


def test_retry_checks_dropped_rows_again_and_rewrites_the_ones_that_still_fail(fake_llm, relabel_dirs):
    _, built = relabel_dirs
    write_examples(built, [_row("a", "hi there", "Hi there."), _row("b", "bye now", "Bye now."), _row("c", "ok then", "Ok.")])
    # The first run drops a and b; c passes.
    first = fake_llm(lambda m, role: _passing(m, role, lambda item: "" if item["reference"] == "Ok then." else "bad")
                     if role == "judge" else {"hi": "Hi.", "by": "Bye.", "ok": "Ok then."}[_transcript_start(m)])
    first.settings = SOL
    asyncio.run(relabel.run(first, "build", relabel.inputs("build")))
    assert sorted(id_ for id_, r in relabel.load_records("build").items() if not r["passed"]) == ["a", "b"]

    # Now "Bye." passes as it is, and a's rewrite carrying the note "short", and the earlier "bad", passes.
    def reply(messages, role):
        if role == "judge":
            return _passing(messages, role, lambda item: "short" if item["reference"] == "Hi." else "")
        assert "short" in messages[-1]["content"] and "bad" in messages[-1]["content"]
        return "Hi there."

    llm = fake_llm(reply)
    llm.settings = SOL
    assert asyncio.run(relabel.retry(llm, "build", relabel.inputs("build"))) == {"scope": "build", "round": 1, "retried": 2, "rechecked": 0, "passed": 2}
    records = relabel.load_records("build")
    assert records["a"]["output"] == "Hi there." and records["a"]["attempt"] == 2 and records["a"]["retry"] == 1
    assert records["b"]["output"] == "Bye." and records["b"]["passed"] and records["c"]["output"] == "Ok then."
    calls = llm.calls
    assert asyncio.run(relabel.retry(llm, "build", relabel.inputs("build")))["retried"] == 0 and llm.calls == calls
    # A row the user ruled dropped stays dropped.
    with relabel.records_path("build").open("a", encoding="utf-8") as sink:
        sink.write(json.dumps({**records["a"], "passed": False, "ruling": "too garbled"}) + "\n")
    assert asyncio.run(relabel.retry(llm, "build", relabel.inputs("build"), round_=2))["retried"] == 0


def test_retry_rechecks_settled_hedges_and_preserves_unchanged_records(fake_llm, relabel_dirs):
    _, built = relabel_dirs
    write_examples(built, [
        _row("a", "maybe rice actually yeah get it", "Rice."),
        _row("b", "bye now", "Bye now."),
        _row("c", "maybe soup yes get it", "Soup."),
    ])

    def first_reply(messages, role):
        if role == "judge":
            return _passing(messages, role, lambda item: "old wording" if "old rice" in item["reference"] else "")
        if _transcript_start(messages) == "ma":
            if "soup" in messages[-1]["content"]:
                return "Maybe soup."
            return "Maybe rice." if "old wording" in messages[-1]["content"] else "Maybe old rice."
        return "Bye now."

    first = fake_llm(first_reply)
    first.settings = SOL
    asyncio.run(relabel.run(first, "build", relabel.inputs("build")))
    before = relabel.load_records("build")
    assert all(r["passed"] for r in before.values())
    assert before["a"]["attempt"] == 1 and before["c"]["attempt"] == 0

    def reply(messages, role):
        if role == "judge":
            return _passing(messages, role, lambda item: "settled" if item["reference"] == "Maybe rice." else "")
        assert "settled" in messages[-1]["content"]
        return "Rice."

    llm = fake_llm(reply)
    llm.settings = SOL
    report = asyncio.run(relabel.retry(llm, "build", relabel.inputs("build"), round_=2, settled_hedges=True))
    assert report == {"scope": "build", "round": 2, "retried": 2, "rechecked": 2, "passed": 2}
    records = relabel.load_records("build")
    assert records["a"]["output"] == "Rice." and records["a"]["retry"] == 2 and records["a"]["attempt"] == 2
    assert records["b"] == before["b"]
    assert records["c"] == {**before["c"], "retry": 2}
    calls = llm.calls
    assert asyncio.run(relabel.retry(llm, "build", relabel.inputs("build"), round_=2, settled_hedges=True))["retried"] == 0
    assert llm.calls == calls
    assert asyncio.run(relabel.retry(llm, "build", relabel.inputs("build"), round_=3, settled_hedges=True))["rechecked"] == 1
    assert relabel.load_records("build")["c"] == {**before["c"], "retry": 3}


def test_retry_preserves_each_existing_attempt_when_dropped_rows_pass_without_a_rewrite(fake_llm, relabel_dirs):
    _, built = relabel_dirs
    write_examples(built, [_row("a", "hi there", "Hi there."), _row("b", "bye now", "Bye now.")])

    def failed_reply(messages, role):
        if role == "judge":
            return _passing(messages, role, lambda item: "bad")
        return "Hi." if _transcript_start(messages) == "hi" else "Bye."

    failed = fake_llm(failed_reply)
    failed.settings = SOL
    rows = relabel.inputs("build")
    asyncio.run(relabel.run(failed, "build", rows))
    asyncio.run(relabel.retry(failed, "build", rows[1:], round_=1))
    before = relabel.load_records("build")
    assert before["a"]["attempt"] == 1 and before["b"]["attempt"] == 2

    def passed_reply(messages, role):
        assert role == "judge"
        return _passing(messages, role)

    passed = fake_llm(passed_reply)
    passed.settings = SOL
    assert asyncio.run(relabel.retry(passed, "build", rows, round_=2))["passed"] == 2
    records = relabel.load_records("build")
    assert records["a"]["attempt"] == 1 and records["b"]["attempt"] == 2
    assert records["a"]["first_failed"] == records["b"]["first_failed"] == []


@pytest.mark.parametrize("change", ["reference", "judge_contract"])
def test_retry_same_round_rechecks_changed_requests_and_preserves_rulings(change, fake_llm, tmp_path, monkeypatch):
    rows = [_row("a", "hello there", "Old reference."), _row("b", "bye now", "Other reference."),
            _row("ruled", "okay then", "Ruled reference.")]
    references = {"hello there": "Hello there.", "hi again": "Hi again.", "bye now": "Bye now.", "okay then": "Okay then."}
    rules = tmp_path / "rules.md"
    rules.write_text("judge rules")
    monkeypatch.setitem(relabel.JUDGES, "build", replace(relabel.JUDGES["build"], instructions=rules))
    reject = True
    judged = []

    def reply(messages, role):
        if role == "judge":
            judged.extend(judge_items(messages))
            return _passing(messages, role, lambda item: "rejected" if reject or item["reference"] != references[item["transcript"]] else "")
        transcript = messages[-1]["content"].split("<transcript>\n", 1)[1].split("\n</transcript>", 1)[0]
        return references[transcript]

    llm = fake_llm(reply)
    llm.settings = SOL
    asyncio.run(relabel.run(llm, "build", rows))
    reject = False
    assert asyncio.run(relabel.retry(llm, "build", rows, round_=1))["passed"] == 3
    before = relabel.load_records("build")
    ruling = {**before["ruled"], "passed": False, "ruling": "too garbled"}
    path = relabel.records_path("build")
    with path.open("a", encoding="utf-8") as sink:
        sink.write(json.dumps(ruling) + "\n")
    calls = llm.calls
    assert asyncio.run(relabel.retry(llm, "build", rows, round_=1))["retried"] == 0 and llm.calls == calls
    if change == "judge_contract":
        rules.write_text("changed judge rules")
    else:
        altered = {**before["a"], "output": "Incorrect output."}
        with path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(altered) + "\n")
    history = path.read_text()
    judged.clear()
    retried = asyncio.run(relabel.retry(llm, "build", rows, round_=1))
    assert retried["retried"] == (2 if change == "judge_contract" else 1)
    assert retried["passed"] == retried["retried"]
    current = relabel.load_records("build")
    assert current["a"]["output"] == references[rows[0].input] and current["a"]["retry"] == 1
    assert current["ruled"] == ruling and all(item["transcript"] != "okay then" for item in judged)
    if change != "judge_contract":
        assert current["b"] == before["b"]
    assert path.read_text().startswith(history)
    calls = llm.calls
    assert asyncio.run(relabel.retry(llm, "build", rows, round_=1))["retried"] == 0 and llm.calls == calls


def _transcript_start(messages):
    return messages[-1]["content"].split("<transcript>\n", 1)[1][:2]


def test_a_partial_relabel_blocks_the_build(fake_llm, relabel_dirs):
    _, built = relabel_dirs
    write_examples(built, [_row("a", "hi there", "Hi there."), _row("b", "bye now", "Bye now.")])
    llm = fake_llm(lambda messages, role: _passing(messages, role) if role == "judge" else "Hi there.")
    llm.settings = SOL
    asyncio.run(relabel.run(llm, "build", relabel.inputs("build")[:1]))
    with pytest.raises(SystemExit, match="partial"):
        relabel.require_complete("build", relabel.load_records("build"))


def test_old_judge_errors_block_completeness_until_relabelled_or_ruled(fake_llm, relabel_dirs):
    _, built = relabel_dirs
    write_examples(built, [_row("a", "hi there", "Hi there.")])
    rows = relabel.inputs("build")
    malformed = fake_llm(lambda messages, role: "malformed" if role == "judge" else "Hi there.")
    malformed.settings = SOL
    asyncio.run(relabel.run(malformed, "build", rows))
    records = relabel.load_records("build")
    assert records["a"]["failed"] == ["judge_error"]
    with pytest.raises(SystemExit, match="partial"):
        relabel.require_complete("build", records)
    relabel.require_complete("build", {"a": {**records["a"], "ruling": "manually dropped"}})

    resumed = fake_llm(lambda messages, role: _passing(messages, role) if role == "judge" else "Hi there.")
    resumed.settings = SOL
    assert asyncio.run(relabel.run(resumed, "build", rows))["rows"] == 1
    current = relabel.load_records("build")
    relabel.require_complete("build", current)
    assert current["a"]["passed"] and not current["a"]["failed"]


def test_failed_group_joins_inflight_clients_before_returning_and_resumes_without_late_writes(
    fake_llm, relabel_dirs, monkeypatch,
):
    _, built = relabel_dirs
    rows = [_row(id_, f"bring {item}", f"Bring {item}.") for id_, item in (("a", "milk"), ("b", "eggs"), ("c", "bread"))]
    write_examples(built, rows)
    rows = relabel.inputs("build")
    monkeypatch.setattr(relabel, "GROUP", 1)
    finished = fake_llm(lambda messages, role: _passing(messages, role) if role == "judge" else "Bring milk.")
    finished.settings = SOL
    asyncio.run(relabel.run(finished, "build", rows[:1]))
    saved = relabel.records_path("build").read_text()
    failure = LlmError("transport retries exhausted", retryable=True)

    async def exercise():
        held = asyncio.Event()
        release = asyncio.Event()
        stopped = asyncio.Event()

        class Failing(fake_llm):
            async def complete(self, messages, *, role="teacher", **kwargs):
                if role == "teacher" and "bread" in messages[-1]["content"]:
                    held.set()
                    try:
                        await release.wait()
                        return "Bring bread."
                    finally:
                        await asyncio.sleep(0)
                        stopped.set()
                if role == "judge":
                    await held.wait()
                    raise failure
                return await super().complete(messages, role=role, **kwargs)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                assert stopped.is_set()

        async with Failing(lambda messages, role: "Bring eggs.") as llm:
            llm.settings = SOL
            with pytest.raises(LlmError) as error:
                await relabel.run(llm, "build", rows)
            assert error.value is failure and stopped.is_set()
            assert relabel.records_path("build").read_text() == saved
            release.set()
            await asyncio.sleep(0)
            assert relabel.records_path("build").read_text() == saved

    asyncio.run(asyncio.wait_for(exercise(), timeout=2))

    def reply(messages, role):
        assert "milk" not in messages[-1]["content"]
        if role == "judge":
            return _passing(messages, role)
        return "Bring eggs." if "eggs" in messages[-1]["content"] else "Bring bread."

    resumed = fake_llm(reply)
    resumed.settings = SOL
    assert asyncio.run(relabel.run(resumed, "build", rows))["rows"] == 2
    assert relabel.records_path("build").read_text().startswith(saved)
    assert all(record["passed"] for record in relabel.load_records("build").values())


def test_licenses_name_the_new_reference_writer_and_keep_the_transcripts():
    def license_of(license, teacher_model):
        return relabel.relabelled_license(_row("x", "a", "b", license=license, teacher_model=teacher_model))

    assert license_of("MIT", None) == "MIT"
    assert license_of("openai-output", "gpt-6.1-sol") == "openai-output"
    assert license_of("openai-output+zai-output", "gpt-6.1-sol") == "openai-output+zai-output"
    assert license_of("zai-output", "glm-5.3-flash") == "openai-output+zai-output"
    assert license_of("openai-output+zai-output", "glm-5.3") == "openai-output"
