import asyncio
import json
from dataclasses import replace

import pytest
from conftest import judge_items

from whispernow_training.datagen import gates, inject, relabel, teacher
from whispernow_training.datagen.llm import LlmSettings
from whispernow_training.datagen.schema import EMPTY, Example, write_examples

SOL = LlmSettings(
    backend="codex", model="gpt-6.1-sol", judge_model="gpt-6.1-sol", concurrency=1, reasoning="high",
    judge_reasoning="high", opencode_provider="", base_url=None, judge_backend="codex",
    generator_backend="codex", generator_model="gpt-6.1-sol", generator_reasoning="high",
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
    relabel.check_settings(SOL)
    with pytest.raises(SystemExit, match="judge: codex/gpt-6.1-sol/low"):
        relabel.check_settings(replace(SOL, judge_reasoning="low"))
    with pytest.raises(SystemExit, match="teacher: opencode/glm-5.3-flash/low"):
        relabel.check_settings(fake_llm(None).settings.__class__(
            backend="opencode", model="glm-5.3-flash", judge_model="gpt-6.1-sol", concurrency=1, reasoning="low",
            judge_reasoning="high", opencode_provider="", base_url=None, judge_backend="codex",
            generator_backend="codex", generator_model="gpt-6.1-sol", generator_reasoning="high",
        ))


def test_code_checks_fail_em_dashes_and_kept_cues_and_flag_counts_and_dropped_hedges():
    failed, note = gates.code_check("so the 2019, oh wait. the 2020 one", "So the 2020 one — our best.")
    assert failed == ["em_dash"] and "em dash" in note
    failed, _ = gates.code_check("call Sam. wait, call Ana", "Call Sam. Wait, call Ana.")
    assert failed == ["correction_cue"]
    assert gates.code_check("wait for me at the gate", "Wait for me at the gate.") == ([], "")
    flags = gates.code_flags("I think two bags of ice", "Two bags of ice.")
    assert "spelled-out number 'Two bags'" in flags and "'i think'" in flags


def test_the_judge_sees_flags_long_rows_go_alone_and_a_relabel_carries_the_judges_note(fake_llm):
    long_input = " ".join(["we need the boxes"] * 80)
    requests = []

    def reply(messages, role):
        if role == "judge":
            requests.append(judge_items(messages))
            return _passing(messages, role, lambda item: "dropped 'maybe'" if item["reference"] == "Two bags." else "")
        user = messages[-1]["content"]
        if "boxes" in user:
            return "We need the boxes."
        return "Maybe 2 bags." if "dropped 'maybe'" in user else "Two bags."

    llm = fake_llm(reply)
    judge = gates.Judge(solo_over_words=300)
    results = asyncio.run(gates.gate(llm, [("short", "maybe two bags"), ("long", long_input)], judge=judge, relabel_note=True))
    assert results["short"].passed and results["short"].attempt == 1 and results["short"].output == "Maybe 2 bags."
    assert results["short"].first_note == "dropped 'maybe'"
    first = next(items for items in requests if any(item["reference"] == "Two bags." for item in items))
    assert len(first) == 1 and "spelled-out number 'Two bags'" in first[0]["flags"]
    assert any(len(items) == 1 and items[0]["transcript"] == long_input for items in requests)
    # Without the note, the relabel is the app's request unchanged.
    plain = teacher.teacher_messages("maybe two bags")
    assert teacher.REVIEW_NOTE.split("{")[0] not in plain[-1]["content"]
    assert "dropped 'maybe'" in teacher.teacher_messages("maybe two bags", note="dropped 'maybe'")[-1]["content"]


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
        _row("i", "two bags of uh ice", "Two bags of ice — please.", source="injected").__class__(
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
    relabel.require_complete("build")

    kept, dropped = relabel.apply([*rows[:3], _row("new", "hello", "Hello.")], "build")
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

    # Now "Bye." passes as it is, and a's rewrite carrying the note "short" passes.
    def reply(messages, role):
        if role == "judge":
            return _passing(messages, role, lambda item: "short" if item["reference"] == "Hi." else "")
        assert "short" in messages[-1]["content"]
        return "Hi there."

    llm = fake_llm(reply)
    llm.settings = SOL
    assert asyncio.run(relabel.retry(llm, "build", relabel.inputs("build"))) == {"scope": "build", "round": 1, "retried": 2, "passed": 2}
    records = relabel.load_records("build")
    assert records["a"]["output"] == "Hi there." and records["a"]["attempt"] == 2 and records["a"]["retry"] == 1
    assert records["b"]["output"] == "Bye." and records["b"]["passed"] and records["c"]["output"] == "Ok then."
    calls = llm.calls
    assert asyncio.run(relabel.retry(llm, "build", relabel.inputs("build")))["retried"] == 0 and llm.calls == calls


def _transcript_start(messages):
    return messages[-1]["content"].split("<transcript>\n", 1)[1][:2]


def test_a_partial_relabel_blocks_the_build(fake_llm, relabel_dirs):
    _, built = relabel_dirs
    write_examples(built, [_row("a", "hi there", "Hi there."), _row("b", "bye now", "Bye now.")])
    llm = fake_llm(lambda messages, role: _passing(messages, role) if role == "judge" else "Hi there.")
    llm.settings = SOL
    asyncio.run(relabel.run(llm, "build", relabel.inputs("build")[:1]))
    with pytest.raises(SystemExit, match="partial"):
        relabel.require_complete("build")


def test_licenses_name_the_new_reference_writer_and_keep_the_transcripts():
    def license_of(license, teacher_model):
        return relabel.relabelled_license(_row("x", "a", "b", license=license, teacher_model=teacher_model))

    assert license_of("MIT", None) == "MIT"
    assert license_of("openai-output", "gpt-6.1-sol") == "openai-output"
    assert license_of("openai-output+zai-output", "gpt-6.1-sol") == "openai-output+zai-output"
    assert license_of("zai-output", "glm-5.3-flash") == "openai-output+zai-output"
    assert license_of("openai-output+zai-output", "glm-5.3") == "openai-output"


def test_the_review_page_marks_the_change(relabel_dirs):
    old = _row("a", "two bags of ice", "Two bags of ice — please.")
    record = {"output": "2 bags of ice, please.", "passed": True, "attempt": 0, "failed": [], "note": "", "first_failed": [], "first_note": ""}
    report = {"scopes": {"build": {"rows": 1, "pass_rate": 1.0, "first_try": 1.0}}, "estimate": {}}
    page = relabel.review_page([("build", old, record)], report, relabel_dirs[0] / "review.html").read_text()
    assert "<del>Two</del><ins>2</ins>" in page and 'name="v1" value="fault"' in page


def test_a_reason_counts_as_kept_in_parentheses_but_not_when_dropped():
    from whispernow_training.model.metrics import kept_counts

    transcript = "two bags of carrots no make it three because Joaquin is bringing his kids"
    assert kept_counts(transcript, "- 3 bags of carrots (Joaquin is bringing his kids)") == {"because": (1, 1)}
    assert kept_counts(transcript, "- 3 bags of carrots") == {"because": (1, 0)}


def test_the_keep_rate_excuses_repeats_closing_asides_and_corrections_but_not_plain_drops():
    from whispernow_training.model.metrics import kept_counts

    def excused(transcript, text):
        return kept_counts(transcript, text, excuse=True)

    assert excused("bnp was 320 but i think that's um i think that's reactive", "BNP was 320, but I think that's reactive.") == {"i think": (1, 1)}
    assert excused("that's the list. i think that's everything", "That's the list.") == {}
    assert excused("that's it, i think", "That's it.") == {}
    assert excused("marcus i think or maybe mateo, no, marcus", "Marcus.") == {}
    assert excused("maybe 3 bags of ice", "3 bags of ice.") == {"maybe": (1, 0)}
    assert kept_counts("maybe 3 bags of ice", "3 bags of ice.") == {"maybe": (1, 0)}
    # A stutter inside a reason says its words once.
    assert kept_counts("leave them out because i don't i don't remember", "Leave them out because I don't remember.") == {"because": (1, 1)}


def test_acronyms_are_not_hesitation_fillers():
    assert not relabel._HESITATION.search("the ER visit at UM hospital")
    assert relabel._HESITATION.search("Um, the ER visit")


def test_a_kept_casual_contraction_fails_the_code_check():
    failed, note = gates.code_check("we wanna go", "We wanna go.")
    assert failed == ["contraction"] and "'wanna'" in note
    assert gates.code_check("we wanna go", "We want to go.") == ([], "")


def test_code_settles_empty_without_the_judge(fake_llm):
    llm = fake_llm(lambda messages, role: pytest.fail("the judge saw an EMPTY row"))
    verdicts = asyncio.run(gates.check_references(llm, [
        ("a", "um, uh... hmm?", EMPTY), ("b", "Erm. Mm", "Mm."), ("c", "buy milk", EMPTY),
    ]))
    assert verdicts["a"].passed and verdicts["a"].judge_model == "code"
    assert verdicts["b"].failed == ["empty"] and verdicts["c"].failed == ["empty"]


@pytest.mark.parametrize("exit_code, log", [
    (0, "ERROR codex_api::endpoint::responses_websocket: failed to connect to websocket: HTTP error: 403 Forbidden"),
    (1, "ERROR: Selected model is at capacity. Please try a different model."),
])
def test_a_refused_or_full_codex_is_throttling_and_retries_past_the_attempt_cap(tmp_path, monkeypatch, exit_code, log):
    from whispernow_training.datagen import llm as llm_module

    monkeypatch.setattr(llm_module, "CACHE", tmp_path)
    fake = tmp_path / "fake-codex"
    fake.write_text(f"#!/bin/sh\ncat > /dev/null\necho '{log}' >&2\nexit {exit_code}\n")
    fake.chmod(0o755)
    monkeypatch.setattr(llm_module.shutil, "which", lambda _: str(fake))
    runner = llm_module._CodexRunner()
    with pytest.raises(llm_module.LlmError) as caught:
        asyncio.run(runner.complete("gpt-6.1-sol", "high", [{"role": "user", "content": "hi"}]))
    error = caught.value
    assert error.retryable and error.throttled and not error.limited
    assert not llm_module._CODEX_THROTTLED.search("codex exit 1: model not found")
