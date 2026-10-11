from whispernow_training.datagen.schema import Example, example_id, layout_of, read_examples, write_examples


def test_layout_of_plain_and_paragraphs():
    assert layout_of("One sentence. Two sentences.") == "plain"
    assert layout_of("First topic.\n\nSecond topic.") == "paragraphs"


def test_layout_of_list_with_lead_in_and_labels():
    assert layout_of("We need:\n- Milk\n- Eggs") == "list"
    assert layout_of("Grocery list:\n- Milk\n\nPharmacy:\n- Ibuprofen (200 mg)") == "list"
    assert layout_of("Deploy steps:\n1. Build.\n2. Copy the dist folder.") == "list"


def test_layout_of_mixed_when_sentences_surround_a_list():
    output = "Hey, on the way home can you grab:\n- Milk\n- Eggs\n\nOh, and did you call your mom about Sunday?"
    assert layout_of(output) == "mixed"


def test_example_recomputes_layout_and_defaults_seed_group():
    example = Example(id="x", input="i", output="We need:\n- Milk", source="s", category="c", license="MIT")
    assert example.layout == "list"
    assert example.seed_group == "x"


def test_example_id_is_stable_and_names_the_repo():
    assert example_id("owner/repo", "hello").startswith("repo-")
    assert example_id("owner/repo", "hello") != example_id("owner/repo", "hello!")


def test_jsonl_round_trip(tmp_path):
    rows = [Example(id="a", input="um hi", output="Hi.", source="s", category="c", license="MIT", tags=["t"])]
    write_examples(tmp_path / "rows.jsonl", rows)
    assert read_examples(tmp_path / "rows.jsonl") == rows
