import json
from dataclasses import replace

from whispernow_training.datagen import build, heard, paths
from whispernow_training.datagen.schema import EMPTY, V2_TAG, Example, write_examples


def counts(*texts, rare=()):
    common = {unit.word: 100 for text in texts for unit in heard.units(text)[1]}
    return {word: n for word, n in common.items() if word not in rare}


def restore(spoken, heard_text, reference, rare=()):
    return heard.restore(spoken, heard_text, reference, counts(spoken, heard_text, rare=rare))


def words(text):
    return [unit.word for unit in heard.units(text)[1]]


def test_a_misheard_name_takes_the_reference_spelling_and_keeps_the_recognizer_punctuation():
    result = restore(
        "please send it to okafor by friday", "Please send it to Ockhaffer by Friday.",
        "Please send it to Okafor by Friday.", rare=("ockhaffer",),
    )
    assert result.text == "Please send it to Okafor by Friday."
    assert result.fixes == [("Ockhaffer", "Okafor")] and not result.reason


def test_numbers_match_by_value_however_they_are_written():
    assert words("The budget is $1,200.") == words("the budget is twelve hundred dollars")
    assert words("April 15, 2024") == words("april fifteenth twenty twenty four")
    assert words("175") == words("a hundred and seventy five")
    assert words("serial 96201") == words("serial nine six two oh one")
    assert words("a sublimit of$2 million") == words("a sublimit of two million dollars")
    assert restore("the budget is twelve hundred dollars", "The budget is $1,200.", "The budget is $1,200.").fixes == []


def test_a_changed_number_or_lost_negation_is_put_back():
    assert restore("we need fifteen chairs", "We need 50 chairs.", "We need 15 chairs.").text == "We need fifteen chairs."
    assert restore("we can't ship on monday", "We can ship on Monday.", "We can't ship on Monday.").text == (
        "We can't ship on Monday."
    )


def test_a_restored_word_replacing_a_glued_token_gets_its_space():
    result = restore("a sublimit of three million dollars", "A sublimit of$2 million.", "A sublimit of $3 million.")
    assert result.text == "A sublimit of three million dollars."


def test_article_and_filler_slips_stay_as_heard():
    result = restore("um so the fourth song should be free bird", "Um, so a fourth song should be, uh, Free Bird.", "The fourth song should be Free Bird.")
    assert result.text == "Um, so a fourth song should be, uh, Free Bird." and result.fixes == []


def test_a_filler_heard_as_a_word_or_a_swapped_preposition_is_put_back():
    assert restore("uh the fourth song", "Our fourth song.", "The fourth song.").text == "Uh the fourth song."
    assert restore("put the ice in the cooler", "Put the ice and the cooler.", "Put the ice in the cooler.").text == (
        "Put the ice in the cooler."
    )


def test_a_filler_glued_to_digits_is_not_left_as_a_code():
    assert restore("the code is uh four seven two one", "The code is A4721.", "The code is 4721.").text == (
        "The code is uh four seven two one."
    )


def test_a_row_that_lost_a_whole_phrase_is_rejected():
    spoken = "book the room for the offsite on the ninth floor near the big windows by the north stairs"
    result = restore(spoken, "Book the room for the offsite.", "Book the room for the offsite on the ninth floor near the big windows.")
    assert result.reason.startswith("restored run")


def test_a_heard_row_keeps_its_origin_reference_provenance_and_content_tags():
    origin = Example(
        id="o1", input="um so we need fifteen chairs", output="We need 15 chairs.", source="teacher", category="dense_values",
        license="MIT", tags=["composed", "style_raw", "asr_parakeet", "value_correction"], seed_group="g1", split="train",
        teacher_model="gpt-6.1-sol", generator_provider="codex", judge_model="gpt-6.1-sol",
    )
    row = heard.heard_row(origin, {"id": "o1", "input": "Um so we need 15 chairs.", "recognizer": "whisper"})
    assert (row.input, row.output, row.source, row.category, row.seed_group, row.split) == (
        "Um so we need 15 chairs.", "We need 15 chairs.", heard.SOURCE, "dense_values", "g1", "train",
    )
    assert row.tags == ["heard", "style_recognizer", "recognizer:whisper", "from:teacher", "heard_from:o1", V2_TAG, "value_correction"]
    assert (row.teacher_model, row.generator_provider, row.judge_model) == ("gpt-6.1-sol", "codex", "gpt-6.1-sol")
    assert heard.origin_id(row) == "o1"


def test_a_larger_sample_extends_a_smaller_one_and_skips_empty_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "BUILD", tmp_path)
    rows = [
        Example(id=f"r{i}", input=f"call the plumber about sink {i}", output=f"Call the plumber about sink {i}.", source="s",
                category="c", license="MIT", split="train")
        for i in range(8)
    ]
    rows.append(Example(id="empty", input="um uh", output=EMPTY, source="s", category="c", license="MIT", split="train"))
    write_examples(tmp_path / "train.jsonl", rows)
    assert [e.id for e in heard.sample_origins(3)] == [e.id for e in heard.sample_origins(6)][:3]
    assert "empty" not in {e.id for e in heard.sample_origins(9)}


def test_build_v2_swaps_in_heard_inputs_the_meaning_check_passed_for_their_origins(tmp_path, monkeypatch):
    monkeypatch.setattr(build.compose, "TARGETED", replace(build.compose.TARGETED, out_dir=tmp_path / "stage4-targeted"))
    monkeypatch.setattr(heard, "OUT_DIR", tmp_path / "heard")
    monkeypatch.setattr(heard, "MEANING", tmp_path / "heard" / "meaning.jsonl")
    v1_dir, out_dir = tmp_path / "build", tmp_path / "build-v2"
    monkeypatch.setattr(paths, "BUILD", v1_dir)
    origins = [
        Example(id=f"o{i}", input=f"um we need to book room {i} for tuesday", output=f"We need to book room {i} for Tuesday.",
                source="owner/src", category="c", license="MIT", split="train")
        for i in range(4)
    ]
    write_examples(v1_dir / "train.jsonl", origins)
    for split in ("val", "test"):
        write_examples(v1_dir / f"{split}.jsonl", [])
    (tmp_path / "heard").mkdir()
    heard_inputs = [f"Um, we need to book room {i} for Tuesday." for i in range(4)]
    records = [
        {"id": f"o{i}", "recognizer": "parakeet", "input": text, "kept": i != 3, "reason": "restored run of 7" if i == 3 else "",
         "fixes": [], "attempt": 1, "seconds": 2.0}
        for i, text in enumerate(heard_inputs)
    ]
    (tmp_path / "heard" / "heard.parakeet.0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    verdicts = [
        {"id": "o0", "input": heard_inputs[0], "p": 0.01},
        {"id": "o1", "input": heard_inputs[1], "p": 0.9},
        {"id": "o2", "input": "an input restored by older rules", "p": 0.01},
        {"id": "o3", "input": heard_inputs[3], "p": 0.01},
    ]
    heard.MEANING.write_text("".join(json.dumps(v) + "\n" for v in verdicts))

    report = build.build_v2(v1_dir, out_dir)
    train = {e.input: e for e in build.read_examples(out_dir / "train.jsonl")}
    assert set(train) == {e.input for e in origins[1:]} | {heard_inputs[0]}
    assert train[heard_inputs[0]].seed_group == origins[0].seed_group
    assert (report["v2"]["added"], report["v2"]["heard_swapped"]) == (0, 1)
    assert heard.report()["meaning_check"] == {"passed": 1, "flagged": 1, "not_checked": 1}

    build.build_v2(v1_dir, tmp_path / "build-v2-no-heard", with_heard=False)
    assert (tmp_path / "build-v2-no-heard" / "train.jsonl").read_bytes() == (v1_dir / "train.jsonl").read_bytes()


def test_a_sign_the_comparison_drops_only_matches_the_same_sign():
    assert words("about ±15%") != words("about fifteen percent")
    assert restore("about plus or minus fifteen percent", "About ±15%.", "About ±15%.").text == "About plus or minus fifteen percent."


def test_a_word_opening_a_quote_is_not_a_name():
    assert heard.names('He kept saying, "The shuttle leaves." Then the bus came for you and Pat. Ask GrainLab.') == {
        "pat": "Pat", "grainlab": "GrainLab",
    }


def test_a_filler_heard_as_a_capital_a_mid_sentence_is_put_back():
    assert restore("it's a garden uh type three", "It's a Garden A type 3.", "It's a Garden type 3.").text == (
        "It's a Garden uh type 3."
    )
    assert restore("uh so we start", "A so we start.", "So we start.").text == "A so we start."
