from whispernow_training.datagen import restyle


def test_punctuation_casing_and_written_numbers_keep_the_same_words():
    spoken = "um so she's fifty one and uh we started her on point five milligrams at three thirty p m"
    written = "Um, so she's 51, and, uh, we started her on 0.5 milligrams at 3:30 PM."
    assert restyle.same_words(spoken, written)
    assert restyle.same_words("it costs two hundred and thirty dollars", "It costs $230.")
    assert restyle.same_words("about twenty percent more", "About 20% more.")
    assert restyle.same_words("a hundred people on the twenty fifth", "100 people on the 25th.")
    assert restyle.same_words("start h r t today", "Start HRT today.")
    assert restyle.same_words("that means its alive in nextjs", "That means it's alive in Next.js.")
    assert restyle.same_words("two and a half cups on june sixteenth twenty twenty five", "2 1/2 cups on June 16th, 2025.")


def test_acronyms_and_letters_with_digits_keep_the_same_words():
    assert restyle.same_words("the florist the a v company", "The florist, the AV company.")
    assert restyle.same_words("her b m i is high and his p s a", "Her BMI is high and his PSA")
    assert restyle.same_words("by q three on e c two", "By Q3 on EC2.")
    assert restyle.same_words("so like fifty k back in the seventies", "So like 50k back in the '70s.")
    assert restyle.same_words("the sixteenth", "The 16th.")
    assert restyle.same_words("from the m and a due diligence at a small r and d lab", "From the M&A due diligence at a small R&D lab.")


def test_a_changed_dropped_or_contracted_word_fails():
    assert not restyle.same_words("I am going to the store", "I'm going to the store.")
    assert not restyle.same_words("um we need milk", "We need milk.")
    assert not restyle.same_words("pool dot map", "pool.map")
    assert not restyle.same_words("the the report", "The report.")
    assert not restyle.same_words("about fifty bucks", "About $50.")
    assert not restyle.same_words("orders at breadlab supply dot com", "orders@breadlabsupply.com")
    assert not restyle.same_words("uh so ethan", "Um, so Ethan.")
    assert not restyle.same_words("some concerns period the", "Some concerns. The")


def test_the_restyled_share_is_fixed_by_id():
    from whispernow_training.datagen.schema import Example

    rows = [Example(id=f"r{i}", input="x", output="x", source="s", category="c", license="MIT") for i in range(2000)]
    share = sum(map(restyle.chosen, rows)) / len(rows)
    assert 0.45 < share < 0.55
    assert [restyle.chosen(r) for r in rows] == [restyle.chosen(r) for r in rows]


def test_recognizer_quirks_follow_the_row_style():
    spoken = " ".join(f"and then um we moved box {n} to the shelf and it costs five dollars" for n in ("one", "two", "three") * 12)
    text = " ".join(f"And then, um, we moved box {n} to the shelf, and it costs $5." for n in (1, 2, 3) * 12)
    styled = {name: restyle.recognizer_quirks(text, "row-1", name, spoken) for name in restyle.ASR_STYLES}
    assert styled["parakeet"] == restyle.recognizer_quirks(text, "row-1", "parakeet", spoken)
    assert all(restyle.same_words(spoken, t) for t in styled.values())
    assert styled["plain"] == text
    assert "costs$5" in styled["parakeet"] and "costs $5" in styled["whisper"]
    assert "five dollars" in styled["qwen3"] and "$" not in styled["qwen3"]
    assert styled["parakeet"].count(",") < text.count(",") / 2
    short = "Um so we need milk."
    assert restyle.recognizer_quirks(short, "row-2", "whisper") in (short, "um so we need milk")


def test_spoken_numbers_come_back_from_the_transcript():
    spoken = "grant cut by thirty one percent for twenty twenty five so about two hundred and thirty dollars"
    text = "Grant cut by 31% for 2025. So about $230."
    rng = __import__("random").Random(1)
    assert restyle.spoken_numbers(text, spoken, rng) == (
        "Grant cut by thirty one percent for twenty twenty five. So about two hundred and thirty dollars."
    )
    assert restyle.spoken_numbers("By Q3.", "by q three", rng) == "By Q3."  # a number inside a word stays


def test_asr_styles_split_the_rows_in_their_shares():
    styles = [restyle.asr_style(f"r{i}") for i in range(3000)]
    assert sum(style.share for style in restyle.ASR_STYLES.values()) == 100
    for name, style in restyle.ASR_STYLES.items():
        assert abs(styles.count(name) / len(styles) - style.share / 100) < 0.04


def test_filler_commas_are_mostly_dropped_mid_sentence():
    text = " ".join(f"We need, um, box {i}, uh, today." for i in range(200))
    thinned = restyle.filler_commas(text, "row-3")
    assert restyle.same_words(text, thinned)
    assert thinned.count(", um") + thinned.count("um,") < 0.3 * text.count(", um,") * 2
    assert "box 7, uh" in text and "Um, so" in restyle.filler_commas("Um, so we go.", "row-4")
