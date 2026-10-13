import pytest

from whispernow_training.datagen.schema import Example
from whispernow_training.model.metrics import check, chrf, kept_counts


def test_chrf_uses_character_precision_and_recall():
    # Two shared characters give precision 2/3 and recall 1, hence F2 = 10/11.
    assert chrf("aab", "ab", n=1, beta=2) == pytest.approx(1000 / 11)


def test_chrf_is_zero_for_disjoint_text():
    assert chrf("abc", "xyz") == 0.0


def test_check_reports_chrf_for_unequal_text():
    row = Example(id="partial", input="ab", output="ab", source="s", category="c", license="x")
    # Orders 1 and 2 average precision (2/3 + 1/2)/2 = 7/12 and recall 1.
    assert check(row, "aab").chrf == pytest.approx(87.5)


def test_a_changed_number_is_not_a_kept_reference_token():
    row = Example(id="number", input="order 5 boxes", output="Order 5 boxes.", source="s", category="c", license="x")
    checks = check(row, "Order 15 boxes.")
    assert checks.protected_kept is False
    assert (checks.protected_tokens_kept, checks.protected_tokens_total) == (0, 1)


def test_a_changed_filename_is_not_a_kept_reference_token():
    row = Example(id="path", input="open app.py", output="Open app.py.", source="s", category="c", license="x")
    checks = check(row, "Open xapp.py.")
    assert checks.protected_kept is False
    assert (checks.protected_tokens_kept, checks.protected_tokens_total) == (0, 1)


def test_unchanged_protected_values_survive_surrounding_punctuation():
    row = Example(id="kept", input="order 5 boxes via app.py", output="Order 5 boxes via app.py.",
                  source="s", category="c", license="x")
    checks = check(row, "Order 5 boxes via (app.py).")
    assert checks.protected_kept is True
    assert (checks.protected_tokens_kept, checks.protected_tokens_total) == (2, 2)


def test_protected_token_counts_keep_only_the_output_occurrences():
    row = Example(id="repeated", input="run app.py then app.py", output="Run app.py, then app.py.",
                  source="s", category="c", license="x")
    checks = check(row, "Run app.py.")
    assert checks.protected_kept is False
    assert (checks.protected_tokens_kept, checks.protected_tokens_total) == (1, 2)


def test_protected_tokens_use_the_reference_after_a_spoken_correction():
    row = Example(id="correction", input="order 5 boxes no make it 15 boxes", output="Order 15 boxes.",
                  source="s", category="c", license="x")
    checks = check(row, "Order 5 boxes.")
    assert checks.protected_kept is False
    assert (checks.protected_tokens_kept, checks.protected_tokens_total) == (0, 1)


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
