import dataclasses
import asyncio
import json
import re
from collections import Counter

import pytest

from conftest import judge_items

from whispernow_training.datagen import gates
from whispernow_training.datagen.llm import LlmError
from whispernow_training.datagen.schema import EMPTY


def _transcript(messages):
    return re.search(r"<transcript>\n(.*)\n</transcript>", messages[-1]["content"], re.S).group(1)


def test_missing_protected_finds_dropped_tokens_but_not_corrected_ones():
    assert gates.missing_protected("send it to ops@example.com by 5", "Send it to ops@example.com by 5.") == []
    assert gates.missing_protected("the file is in src/app.py", "The file is in the source folder.") == ["src/app.py"]
    # "12" is the discarded side of a correction.
    assert gates.missing_protected("order 12 no sorry 14 boxes", "Order 14 boxes.") == []
    # Numbers match as numbers: "23rd" as "23", but "20" is not inside "2026".
    assert gates.missing_protected("on the 23rd", "On the 23.") == []
    assert gates.missing_protected("about 20 people", "About 2026 people.") == ["20"]


def test_a_value_a_late_correction_may_take_back_goes_to_the_judge_not_the_code_check():
    padding = " and it looks great in the light" * 6
    transcript = f"I like the cork at $14 a square foot{padding} actually forget the cork go with bamboo"
    assert gates.code_check(transcript, "Go with bamboo.") == ([], "")
    assert "drops '14'" in gates.code_flags(transcript, "Go with bamboo.")
    # With no correction after it, a dropped value still fails in code.
    assert gates.code_check(f"go with bamboo{padding} at $14 a square foot", "Go with bamboo.")[0] == ["protected_tokens"]


def test_a_restarted_or_merged_value_goes_to_the_judge_not_the_code_check():
    # Restarted: the speaker replaces "55" and "30" with the value after the pause.
    transcript = "keep them at 55... 58 degrees in precinct 30— 31"
    assert gates.code_check(transcript, "Keep them at 58 degrees in precinct 31.") == ([], "")
    assert "drops '55', which is right only if the speaker restarted" in gates.code_flags(transcript, "Keep them at 58 degrees in precinct 31.")
    # Merged: "27 six" is written as one number.
    transcript = "at 27 six per placement, about one sixty six thousand at risk"
    assert gates.code_check(transcript, "At $27,600 per placement, about $166,000 at risk.") == ([], "")
    assert "part of a larger number" in gates.code_flags(transcript, "At $27,600 per placement, about $166,000 at risk.")
    # ... or spoken around it: "point 1.8" is 0.18, "oh 600" is 0600, "17/2 61, 4 08" is 17/261,408.
    for transcript, reference in [
        ("an indirect effect of point 1.8", "An indirect effect of 0.18."),
        ("we leave at oh 600 tomorrow", "We leave at 0600 tomorrow."),
        ("application 17/2 61, 4 08", "Application 17/261,408."),
    ]:
        assert gates.code_check(transcript, reference) == ([], ""), transcript
    # Misspoke: the speaker says which value was wrong.
    transcript = "I paid her $18. Oh and the payment, I said $18, it was $28 because of the longer walk"
    assert gates.code_check(transcript, "I paid her $28 because of the longer walk.") == ([], "")
    assert "said they misspoke" in gates.code_flags(transcript, "I paid her $28 because of the longer walk.")
    # A value followed by an ordinary word still fails in code.
    assert gates.code_check("keep them at 55 degrees", "Keep them cold.")[0] == ["protected_tokens"]


@pytest.mark.parametrize("transcript, reference, phrase", [
    ("uh rice, maybe some jasmine rice, actually yeah get it, we're out", "- Rice, maybe jasmine (we're out)", "maybe some jasmine rice"),
    ("get the blue one maybe, yeah the blue one", "Get the blue one, maybe.", "the blue one maybe"),
    ("Maybe, maybe the jasmine rice, yeah", "Maybe jasmine rice.", "maybe the jasmine rice"),
    ("maybe, I think the jasmine rice, yeah", "Maybe jasmine rice.", "maybe, I think the jasmine rice"),
    ("probably, maybe the jasmine rice, yeah", "Probably jasmine rice.", "probably, maybe the jasmine rice"),
])
def test_a_kept_hedge_the_speaker_settles_soon_after_goes_to_the_judge(transcript, reference, phrase):
    found = gates.settled_hedges(transcript, reference)
    assert len(found) == 1 and phrase.casefold() in found[0].casefold()
    assert phrase.casefold() in gates.code_flags(transcript, reference).casefold()


def test_a_hedge_left_open_or_kept_on_another_item_is_not_flagged():
    assert gates.settled_hedges("maybe a bag of basmati just in case", "- Maybe a bag of basmati (just in case)") == []
    transcript = "maybe some jasmine rice, actually yeah get it, we're out"
    assert gates.settled_hedges(transcript, "- Jasmine rice (we're out)") == []
    assert gates.settled_hedges(transcript + ", maybe pears", "- Jasmine rice\n- Maybe pears") == []


def _gate_llm(fake_llm, *, bad_first=(), bad_always=(), unrealistic=()):
    def reply(messages, role):
        if role == "judge":
            results = []
            for item in judge_items(messages):
                ok = "WRONG" not in item["reference"]
                real = not any(word in item["transcript"] for word in unrealistic)
                results.append({"i": item["i"], **{c: ok for c in gates.COMPOSE_CRITERIA}, "input_realistic": real, "note": ""})
            return json.dumps({"results": results})
        return _transcript(messages).upper()

    llm = fake_llm(reply)
    original = llm.complete

    async def complete(messages, *, role="teacher", attempt=0, reasoning=None):
        text = await original(messages, role=role, attempt=attempt, reasoning=reasoning)
        if role == "teacher":
            transcript = _transcript(messages)
            if any(w in transcript for w in bad_always) or (attempt == 0 and any(w in transcript for w in bad_first)):
                return "WRONG"
        return text

    llm.complete = complete
    return llm


def test_gate_relabels_a_failure_once_then_drops_it(fake_llm):
    llm = _gate_llm(fake_llm, bad_first=("flaky",), bad_always=("broken",))
    rows = [("a", "fine one"), ("b", "flaky one"), ("c", "broken one")]
    judge = gates.Judge(gates.COMPOSE, gates.COMPOSE_CRITERIA, {"a": "raw", "b": "raw", "c": "raw"})
    results = asyncio.run(gates.gate(llm, rows, judge=judge))
    assert results["a"].passed and results["a"].attempt == 0 and results["a"].output == "FINE ONE"
    assert results["b"].passed and results["b"].attempt == 1 and results["b"].first_failed
    assert not results["c"].passed and results["c"].attempt == 1
    assert 1 in llm.attempts


def test_gate_drops_an_unrealistic_transcript_without_a_relabel(fake_llm):
    llm = _gate_llm(fake_llm, unrealistic=("polished",))
    judge = gates.Judge(gates.COMPOSE, gates.COMPOSE_CRITERIA, {"a": "recognizer"})
    results = asyncio.run(gates.gate(llm, [("a", "a polished text")], judge=judge))
    assert not results["a"].passed and results["a"].attempt == 0 and "input_realistic" in results["a"].failed
    assert 1 not in llm.attempts


def test_gate_catches_a_missing_protected_token_in_code(fake_llm):
    def reply(messages, role):
        if role == "judge":
            return json.dumps({"results": [{"i": i["i"], **{c: True for c in gates.COMPLETENESS_CRITERIA}} for i in judge_items(messages)]})
        return "Call the team."

    results = asyncio.run(gates.gate(fake_llm(reply), [("a", "call the team at 555 0199 about ticket ABC-123")]))
    assert not results["a"].passed and results["a"].failed == ["protected_tokens"]


def test_missing_protected_forgives_every_mention_of_a_revised_value():
    transcript = (
        "Hot dogs, 100? Last year we did 80 and ran out around 4. Buns, same count, and um condiments, mustard, relish, "
        "ketchup, napkins, plates, cups, forks, a grill brush, charcoal, lighter fluid, two folding tables and chairs "
        "for the older folks, oh wait, back to the hot dogs, I said 100, um actually make it 120"
    )
    assert gates.missing_protected(transcript, "Hot dogs: 120 (last year 80 ran out around 4).") == []
    assert gates.missing_protected("Bandanas for the kids, 15 of them? There's 14 in the class, get 16", "Bandanas: 16 (14 in the class).") == []


@pytest.mark.parametrize("reasoning", [None, "low"])
@pytest.mark.parametrize("teacher_reasoning", [None, "high"])
def test_gate_keeps_teacher_reasoning_for_both_attempts_and_judge_reasoning_for_checks(
    fake_llm, reasoning, teacher_reasoning,
):
    written = 0

    def reply(messages, role):
        nonlocal written
        if "<transcript>" in messages[-1]["content"]:
            assert role == "teacher"
            written += 1
            return "WRONG" if written == 1 else "BP 132/78."
        results = [
            {"i": item["i"], **{c: "WRONG" not in item["reference"] for c in gates.COMPLETENESS_CRITERIA}, "note": ""}
            for item in judge_items(messages)
        ]
        return json.dumps({"results": results})

    llm = fake_llm(reply)
    judge = dataclasses.replace(gates.COMPLETENESS_JUDGE, reasoning=reasoning)
    results = asyncio.run(gates.gate(
        llm, [("a", "BP one thirty two over seventy eight")], judge=judge, teacher_reasoning=teacher_reasoning,
    ))
    assert results["a"].passed and results["a"].attempt == 1 and results["a"].output == "BP 132/78."
    assert Counter(llm.reasonings) == Counter({("teacher", teacher_reasoning): 2, ("judge", reasoning): 2})


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
            return json.dumps({"results": [
                {"i": item["i"], **{c: True for c in gates.COMPLETENESS_CRITERIA},
                 "clean": item["reference"] != "Two bags.",
                 "note": "dropped 'maybe'" if item["reference"] == "Two bags." else ""}
                for item in judge_items(messages)
            ]})
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


@pytest.mark.parametrize("phase", ["teacher", "judge", "relabel"])
def test_client_failure_joins_sibling_calls_before_the_gate_returns(phase, fake_llm):
    failure = LlmError("transport retries exhausted", retryable=True)

    async def exercise():
        held = asyncio.Event()
        release = asyncio.Event()
        stopped = asyncio.Event()

        class Failing(fake_llm):
            async def complete(self, messages, *, role="teacher", **kwargs):
                if role == ("judge" if phase == "judge" else "teacher"):
                    transcript = judge_items(messages)[0]["transcript"] if role == "judge" else _transcript(messages)
                    if "eggs" in transcript:
                        await held.wait()
                        raise failure
                    held.set()
                    try:
                        await release.wait()
                        return "Bring bread."
                    finally:
                        await asyncio.sleep(0)
                        stopped.set()
                return await super().complete(messages, role=role, **kwargs)

        def reply(messages, role):
            return json.dumps({"results": [
                {"i": item["i"], **{criterion: False for criterion in gates.COMPLETENESS_CRITERIA}, "note": "bad"}
                for item in judge_items(messages)
            ]})

        llm = Failing(reply)
        rows = [("a", "bring eggs"), ("b", "bring bread")]
        references = {"a": "Bring eggs.", "b": "Bring bread."} if phase != "teacher" else None
        with pytest.raises(LlmError) as error:
            await gates.gate(llm, rows, references=references, judge=gates.Judge(max_items=1))
        assert error.value is failure and stopped.is_set()
        release.set()
        await asyncio.sleep(0)

    asyncio.run(asyncio.wait_for(exercise(), timeout=2))


def test_gate_reports_each_simultaneous_teacher_failure(fake_llm):
    failures = {text: LlmError(f"failed {text}", retryable=False) for text in ("bring eggs", "bring bread")}

    class Failing(fake_llm):
        async def complete(self, messages, **kwargs):
            raise failures[_transcript(messages)]

    llm = Failing(lambda messages, role: pytest.fail("a failed reference reached the judge"))
    with pytest.raises(ExceptionGroup) as error:
        asyncio.run(gates.gate(llm, [("a", "bring eggs"), ("b", "bring bread")]))
    assert error.value.exceptions == tuple(failures.values())
