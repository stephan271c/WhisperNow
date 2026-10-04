import dataclasses
import asyncio
import json
import re

from conftest import judge_items

from whispernow_training.datagen import gates


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


def test_gate_can_relabel_a_failure_with_the_judge_model(fake_llm):
    roles = []

    def reply(messages, role):
        if "<transcript>" in messages[-1]["content"]:
            roles.append(role)
            # The teacher keeps both sides of the correction; the judge's model gets it right.
            return "BP 132/78." if role == "judge" else "WRONG: BP 122/78, sorry, 132/78."
        results = [
            {"i": item["i"], **{c: "WRONG" not in item["reference"] for c in gates.COMPLETENESS_CRITERIA}, "note": ""}
            for item in judge_items(messages)
        ]
        return json.dumps({"results": results})

    llm = fake_llm(reply)
    results = asyncio.run(gates.gate(llm, [("a", "BP one twenty two over seventy eight sorry one thirty two")], relabel_role="judge"))
    assert results["a"].passed and results["a"].attempt == 1 and results["a"].output == "BP 132/78."
    assert roles == ["teacher", "judge"]
    # The teacher's first sample and the judge's relabel and checks at the judge's own setting.
    assert llm.reasonings == [("teacher", None), ("judge", None), ("judge", None), ("judge", None)]


def test_a_judge_reasoning_applies_to_its_checks_and_the_judge_models_relabel(fake_llm):
    def reply(messages, role):
        if "<transcript>" in messages[-1]["content"]:
            return "BP 132/78." if role == "judge" else "WRONG"
        results = [
            {"i": item["i"], **{c: "WRONG" not in item["reference"] for c in gates.COMPLETENESS_CRITERIA}, "note": ""}
            for item in judge_items(messages)
        ]
        return json.dumps({"results": results})

    llm = fake_llm(reply)
    judge = dataclasses.replace(gates.COMPLETENESS_JUDGE, reasoning="low")
    results = asyncio.run(gates.gate(llm, [("a", "BP one thirty two over seventy eight")], judge=judge, relabel_role="judge"))
    assert results["a"].passed and results["a"].attempt == 1
    assert llm.reasonings == [("teacher", None), ("judge", "low"), ("judge", "low"), ("judge", "low")]
