import random
import re

import pytest

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


def _spans(text: str) -> list[str]:
    return [text[span.start:span.end] for span in facts.fact_spans(text)]


def test_fact_spans_find_values_names_hedges_reasons_and_unit_starts():
    assert {"7:30", "$40", "Saturday", "noon", "Two", "Noah", "Maybe", "I think",
            "because the road closes at noon", "in case the gate is locked", "The", "I", "Dana", "Call"} <= set(_spans(REFERENCE))
    text = "It costs $15,000 a month."
    assert "$15,000" in _spans(text)
    mask = facts.char_mask(text)
    assert all(mask[text.index("$15,000"):text.index("$15,000") + 7])
    assert not any(mask[text.index(" a "):text.index(" a ") + 3])


def test_fact_spans_find_negations():
    text = "Don't bring the tent. We can\u2019t park there, so no cooler and nothing glass. Not Friday, never mind."
    assert {"Don't", "can\u2019t", "no", "nothing", "Not", "never"} <= set(_spans(text))
    false_matches = "Note the north gate and the nonstop bus."
    mask = facts.char_mask(false_matches)
    assert not any(mask[false_matches.index("north"):])


def test_fact_spans_cover_a_corrected_value():
    clean = "Meet on Wednesday at 4 by the red door."
    spoken = inject(clean, "s", "recognizer", kinds=("correction",))
    assert spoken.corrections
    _, right, _ = spoken.corrections[0]
    assert right in _spans(clean)


def test_units_split_lines_into_sentences_and_items():
    kinds = [(u.kind, REFERENCE[u.start:u.end]) for u in facts.units(REFERENCE)]
    assert ("bullet", "- The big cooler with ice packs") in kinds
    assert ("sentence", "Dana pays the $40 site fee.") in kinds
    assert ("sentence", "Packing list for Saturday:") in kinds


@pytest.mark.parametrize(("text", "expected"), [
    ("Bring enough water for everyone. Fine.", "Fine."),
    ("Fine. Bring enough water for everyone. Okay.", "Fine. Okay."),
    ("Fine. Bring enough water for everyone.", "Fine."),
    (
        "Trip notes:\nFine. Bring enough water for everyone. Okay.\n\nKeep the blue tent dry.",
        "Trip notes:\nFine. Okay.\n\nKeep the blue tent dry.",
    ),
], ids=("first", "middle", "last", "other-lines"))
def test_sentence_deletion_removes_the_whole_eligible_sentence(text, expected):
    assert code_pairs.drop_sentence(text, random.Random(0)) == expected


def test_sentence_deletion_requires_an_eligible_unit_on_a_multi_sentence_line():
    assert code_pairs.drop_sentence("Bring enough water for everyone.", random.Random(0)) is None
    assert code_pairs.drop_sentence("Ready. Fine.", random.Random(0)) is None


@pytest.mark.parametrize(("text", "expected"), [
    (
        "List:\n- Water\n- Bring enough clean towels\n- Tent\n\nKeep the road map.",
        "List:\n- Water\n- Tent\n\nKeep the road map.",
    ),
    (
        "List:\n- Water\n- Tent\n- Bring enough clean towels",
        "List:\n- Water\n- Tent",
    ),
], ids=("middle", "last-without-newline"))
def test_item_deletion_removes_the_whole_eligible_line(text, expected):
    assert code_pairs.drop_item(text, random.Random(0)) == expected


def test_item_deletion_keeps_short_lists_and_numbering():
    assert code_pairs.drop_item("- Bring enough clean towels\n- Pack the blue tent", random.Random(0)) is None
    numbered = "Steps:\n1. Rinse the lentils\n2. Chop the onions\n3. Simmer for 20 minutes"
    assert code_pairs.drop_item(numbered, random.Random(0)) == "Steps:\n1. Rinse the lentils\n2. Chop the onions"


def test_reasons_and_hedges_remove_the_whole_modifier():
    rng = random.Random(1)
    reason = code_pairs.drop_reason("We leave at 7:30, because the road closes at noon. Bring water.", rng)
    assert reason == "We leave at 7:30. Bring water."
    hedge = code_pairs.drop_hedge("Maybe bring the stove.", rng)
    assert hedge == "Bring the stove."
    assert code_pairs.drop_hedge("I think that we should go.", rng) is None


def test_changing_an_amount_preserves_everything_around_it():
    rejected = code_pairs.change_value("Dana pays $40 for the campsite.", random.Random(0))
    assert rejected is not None
    assert re.fullmatch(r"Dana pays \$(?!40\b)[1-9]\d* for the campsite\.", rejected)


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
    assert {"split", "input", "chosen", "rejected"} <= set(pairs[0])
    assert all(p["chosen"] == REFERENCE and code_pairs.one_span(p["chosen"], p["rejected"]) for p in pairs)
    assert len({p["kind"] for p in pairs}) >= 4


def test_build_keeps_injected_families_together(seed_families):
    train = [row for parent, derivative, _ in seed_families for row in (parent, derivative)]
    pairs = code_pairs.build(train, [], count=len(train), long_share=1.0)
    assert pairs == code_pairs.build(train, [], count=len(train), long_share=1.0)
    for parent, derivative, split in seed_families:
        family = [pair for pair in pairs if pair["seed_group"] == parent.seed_group]
        assert {pair["id"] for pair in family} == {parent.id, derivative.id}
        assert {pair["split"] for pair in family} == {split}
    held = [derivative for _, derivative, _ in seed_families]
    assert code_pairs.build(train, held, count=len(train), long_share=1.0) == []
