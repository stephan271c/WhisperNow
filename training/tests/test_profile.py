from whispernow_training.datagen import profile


def test_profile_counts_filler_formatting_and_correction_forms():
    texts = [
        "Um so we need the uh the green one, um, and and the bread.",
        "Get carrots or actually no, let's go with peas instead. Friday, no wait, Saturday.",
    ]
    p = profile.profile(texts)
    assert p["rows"] == 2 and p["words"] == 28
    # Three fillers: "Um" capitalized and bare, "uh" bare, "um," with a comma after.
    assert p["comma_after_pct"] == 33.3 and p["capitalized_bare_pct"] == 33.3
    assert p["filler_in_repeat_per_100"] > 0 and p["doubled_word_per_100"] > 0
    assert p["redecisions_per_100"] > 0 and p["instead_per_100"] > 0 and p["stock_cues_per_100"] > 0
    assert "| a |" in profile.table({"a": p})
