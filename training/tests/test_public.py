from whispernow_training.datagen.sources import public
from whispernow_training.datagen.schema import Example


def test_americanize_keeps_case():
    assert public.americanize("My favourite Colour is GREY.") == "My favorite Color is GRAY."
    assert public.americanize("The colonel") == "The colonel"


def test_handy_tags_and_category():
    tags = public.handy_tags("send it to lisbon no wait munich", "Send it to Munich.")
    assert "self_correction" in tags and "short" in tags
    assert public.handy_category(tags) == "self_correction"
    assert public.handy_category(public.handy_tags("what time is it", "What time is it?")) == "dictated_question"
    assert public.handy_category(public.handy_tags("forward it to the team", "Forward it to the team.")) == "no_edit"


def test_handy_example_normalizes_british_spelling_ordinals_and_question_marks():
    example = public._handy_example(
        {"input": "what colour is the invoice due on the fifteenth", "output": "What colour is the invoice due on the fifteenth."},
        "",
    )
    assert example.output == "What color is the invoice due on the 15th?"
    assert example.input == "what color is the invoice due on the fifteenth"
    assert {"americanized", "normalized"} <= set(example.tags)
    assert example.license == "CC-BY-4.0"


def test_ordinals_and_question_marks():
    assert public.ordinals_to_digits("due on the twenty-first and the thirtieth") == "due on the 21st and the 30th"
    assert public.ordinals_to_digits("the first time") == "the first time"
    assert public.question_mark("Where is it.") == "Where is it?"
    assert public.question_mark("Where is it. It was here.") == "Where is it. It was here."
    assert public.question_mark("Send it.") == "Send it."


def test_modal_may_is_lowercased_and_the_month_kept():
    assert public.modal_may("Other factors May contribute, and May be why.") == "Other factors may contribute, and may be why."
    assert public.modal_may("We ship in May and the deadline is May 3.") == "We ship in May and the deadline is May 3."
    assert public.modal_may("May we ship in May once the tests pass?") == "May we ship in May once the tests pass?"


def test_spliced_handy_rows_are_detected():
    assert public.spliced("i think Thus, each such firm may offer itself")
    assert public.spliced("the contract renews in january. In Mainz, the Rhine leaves the valley")
    assert not public.spliced("at the London site, book it for nine people")
    assert not public.spliced("Send it to Yuki.")


def test_sotto_category():
    assert public.sotto_category("the plan is fine", "The plan is fine.") == "preserve_wording"
    assert public.sotto_category("i like the plan", "I like the plan.") == "adversarial"
    assert public.sotto_category("meet at noon no wait at one", "Meet at one.") == "self_correction"
    assert public.sotto_category("we should we should leave now", "We should leave now.") == "false_start"
    assert public.sotto_category("um we should leave", "We should leave.") is None


def test_disfl_qa_tidying_and_rewording():
    assert public.tidy_question("what network was designed by the french") == "What network was designed by the french?"
    assert public.tidy_question("Who else.") == "Who else?"
    assert public.tidy_question("Top-250\\t?") == ""
    assert public.reworded("what number of students are there", "How many students are there?")
    assert not public.reworded("How fast can no sorry how many types are there?", "How many types are there?")


def test_switchboard_index_matches_lines_and_shared_ngrams():
    index = public.SwitchboardIndex(["uh we used to live in a house with a big yard out back"])
    assert index.contains("we used to live in a house with a big yard out back", "")
    assert index.contains("so um we used to live in a house with a big yard", "")
    assert not index.contains("send the invoice to accounts payable today please", "")


def test_sample_by_category_is_seeded_and_caps_test_rows():
    rows = [
        Example(id=f"id{i}", input=f"i{i}", output="o", source="s", category="c", license="MIT", split="test" if i % 5 == 0 else "")
        for i in range(100)
    ]
    first = public.sample_by_category(rows, {"c": 20}, seed=7)
    second = public.sample_by_category(list(reversed(rows)), {"c": 20}, seed=7)
    assert [e.id for e in first] == [e.id for e in second]
    assert sum(e.split == "test" for e in first) == 2
    assert sum(e.split == "" for e in first) == 20
    assert first != public.sample_by_category(rows, {"c": 20}, seed=8)
