from whispernow_training.datagen.schema import Example
from whispernow_training.model.metrics import check, kept_counts


def test_eval_opener_flags_assistant_replies_but_allows_dictated_i_can():
    row = Example(id="a", input="the list", output="The list.", source="s", category="c", license="x")
    assert check(row, "Sure, here's the list").assistant_opener
    assert not check(row, "I can deduct the mileage").assistant_opener


def test_a_reason_counts_as_kept_in_parentheses_but_not_when_dropped():
    transcript = "two bags of carrots no make it three because Joaquin is bringing his kids"
    assert kept_counts(transcript, "- 3 bags of carrots (Joaquin is bringing his kids)") == {"because": (1, 1)}
    assert kept_counts(transcript, "- 3 bags of carrots") == {"because": (1, 0)}


def test_the_keep_rate_excuses_repeats_closing_asides_and_corrections_but_not_plain_drops():
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
