import random

from whispernow_training.datagen.inject import inject
from whispernow_training.datagen.schema import EMPTY, Example
from whispernow_training.model import code_pairs, facts

REFERENCE = (
    "Packing list for Saturday:\n"
    "- Two tents and the blue tarp\n"
    "- The big cooler with ice packs\n"
    "- Maybe the camp stove if Noah brings fuel\n"
    "\n"
    "I think we leave at 7:30, because the road closes at noon. Dana pays the $40 site fee. "
    "Call the ranger in case the gate is locked."
)


def _row(id_: str, output: str = REFERENCE, input_: str = "spoken words", tags: list[str] | None = None) -> Example:
    return Example(id=id_, input=input_, output=output, source="teacher", category="multi_topic", license="x",
                   tags=tags or [], seed_group=f"g-{id_}", split="train")


def _spans(text: str, kind: str) -> list[str]:
    return [span.text(text) for span in facts.fact_spans(text) if span.kind == kind]


def test_fact_spans_find_values_names_hedges_reasons_and_unit_starts():
    assert {"7:30", "$40", "Saturday", "noon", "Two"} <= set(_spans(REFERENCE, "value"))
    assert "Noah" in _spans(REFERENCE, "name")
    # A capital that only opens a sentence or item is not a name; its word is weighted as a unit start.
    assert "Call" not in _spans(REFERENCE, "name") and "The" not in _spans(REFERENCE, "name")
    assert {"Maybe", "I think"} <= set(_spans(REFERENCE, "hedge"))
    assert "because the road closes at noon" in _spans(REFERENCE, "reason")
    assert "in case the gate is locked" in _spans(REFERENCE, "reason")
    # Each list item and sentence starts with a weighted word, where dropping it is decided.
    assert {"Two", "The", "Maybe", "I", "Dana", "Call"} <= set(_spans(REFERENCE, "unit_start"))
    assert "a" not in _spans("It costs $15,000 a month.", "value")


def test_fact_spans_find_negations():
    text = "Don't bring the tent. We can\u2019t park there, so no cooler and nothing glass. Not Friday, never mind."
    assert _spans(text, "negation") == ["Don't", "can\u2019t", "no", "nothing", "Not", "never"]
    # Words that only contain the letters are not negations.
    assert _spans("Note the north gate and the nonstop bus.", "negation") == []


def test_fact_spans_cover_a_corrected_value():
    clean = "Meet on Wednesday at 4 by the red door."
    spoken = inject(clean, "s", "recognizer", kinds=("correction",))
    assert spoken.corrections
    _, right, _ = spoken.corrections[0]
    # The value the speaker settled on is weighted wherever it lands in the reference.
    assert right in {span.text(clean) for span in facts.fact_spans(clean)}


def test_units_split_lines_into_sentences_and_items():
    kinds = [(u.kind, REFERENCE[u.start:u.end]) for u in facts.units(REFERENCE)]
    assert ("bullet", "- The big cooler with ice packs") in kinds
    assert ("sentence", "Dana pays the $40 site fee.") in kinds
    assert ("sentence", "Packing list for Saturday:") in kinds


def _removed_added(chosen: str, rejected: str) -> tuple[str, str]:
    return code_pairs._middles(chosen, rejected)


def test_each_breaker_changes_one_span_only():
    rng = random.Random(0)
    for _ in range(30):
        for name, breaker in (("sentence", code_pairs.drop_sentence), ("item", code_pairs.drop_item),
                              ("reason", code_pairs.drop_reason), ("hedge", code_pairs.drop_hedge),
                              ("changed", code_pairs.change_value)):
            rejected = breaker(REFERENCE, rng)
            assert rejected is not None, name
            assert code_pairs.one_span(REFERENCE, rejected), (name, rejected)
            removed, added = _removed_added(REFERENCE, rejected)
            if name == "changed":
                assert removed and added and " " not in removed + added
            else:
                assert not added or (len(added) == 1 and removed[-1].lower() == added.lower())


def test_drops_remove_whole_units_and_keep_the_text_plain():
    rng = random.Random(1)
    sentence = code_pairs.drop_sentence(REFERENCE, rng)
    assert "  " not in sentence and ". ." not in sentence
    item = code_pairs.drop_item(REFERENCE, rng)
    assert item.count("\n- ") == 2 and "\n\n\n" not in item
    reason = code_pairs.drop_reason("We leave at 7:30, because the road closes at noon. Bring water.", rng)
    assert reason == "We leave at 7:30. Bring water."
    hedge = code_pairs.drop_hedge("Maybe bring the stove.", rng)
    assert hedge == "Bring the stove."
    assert code_pairs.drop_hedge("I think that we should go.", rng) is None
    # A numbered item is dropped only at the end of its list, so the numbers stay in order.
    numbered = "Steps:\n1. Rinse the lentils\n2. Chop the onions\n3. Simmer for 20 minutes"
    assert code_pairs.drop_item(numbered, rng) == "Steps:\n1. Rinse the lentils\n2. Chop the onions"


def test_one_span_rejects_two_edits_and_identity():
    assert not code_pairs.one_span("Buy milk and eggs.", "Buy milk and eggs.")
    assert not code_pairs.one_span("Buy 2 eggs and 3 apples.", "Buy 4 eggs and 3 pears.")
    assert code_pairs.one_span("Buy 2 eggs.", "Buy 3 eggs.")
    assert code_pairs.one_span("Maybe buy eggs.", "Buy eggs.")


def test_wrong_side_puts_the_abandoned_value_back():
    clean = "Book the room for Thursday at 3 and order 12 chairs."
    spoken = inject(clean, "w", "recognizer", kinds=("correction",))
    row = _row("c", output=clean, input_=spoken.text, tags=spoken.tags)
    assert code_pairs.corrections(row) == [(w, r) for w, r, _ in spoken.corrections]
    rejected = code_pairs.wrong_side(row, random.Random(0))
    wrong, right, _ = spoken.corrections[0]
    assert rejected == clean.replace(right, wrong, 1)
    assert code_pairs.one_span(clean, rejected)


def test_corrections_need_the_tag_and_a_resolved_reference():
    row = _row("n", output="Meet on Wednesday.", input_="Meet on Tuesday no Wednesday.")
    assert code_pairs.corrections(row) == []
    tagged = _row("t", output="Meet on Wednesday.", input_="Meet on Tuesday no Wednesday.", tags=["correction:short"])
    assert code_pairs.corrections(tagged) == [("Tuesday", "Wednesday")]
    # Both sides kept, or the right one said twice, gives no known answer to swap.
    kept = _row("k", output="Meet on Tuesday or Wednesday.", input_="Meet on Tuesday no Wednesday.", tags=["correction:short"])
    assert code_pairs.corrections(kept) == []


def test_build_pairs_in_prefer_format_without_held_out_rows():
    train = [_row(f"r{i}") for i in range(40)] + [_row("empty", output=EMPTY)]
    held = _row("bench")
    train.append(Example(**{**held.__dict__, "id": "same-group"}))
    pairs = code_pairs.build(train, [held], count=30, long_share=0.0)
    assert len(pairs) == 30
    assert {p["id"] for p in pairs}.isdisjoint({"bench", "same-group", "empty"})
    assert set(pairs[0]) == {"id", "split", "input", "chosen", "rejected", "kind", "chosen_from", "bucket"}
    assert all(p["chosen"] == REFERENCE and code_pairs.one_span(p["chosen"], p["rejected"]) for p in pairs)
    assert len({p["kind"] for p in pairs}) >= 4
