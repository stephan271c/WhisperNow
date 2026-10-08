import pytest
from jinja2.exceptions import TemplateError

from whispernow_training.model.draft import preclean
from whispernow_training.model.prompt import decoding, golden, golden_cases, max_tokens, messages, prompt_completion, render


@pytest.mark.parametrize("case", golden_cases(), ids=lambda case: case["name"])
def test_messages_and_rendered_prompt_match_the_golden_fixture(case):
    assert messages(case["transcript"]) == case["messages"]
    assert render(case["messages"]) == case["rendered"]
    assert max_tokens(case["transcript"]) == case["max_tokens"]
    assert preclean(case["transcript"]) == case["draft_text"]


def test_decoding_matches_the_golden_fixture():
    assert decoding() == golden()["decoding"]


def test_a_training_row_renders_as_the_prompt_then_the_reply_and_end_of_turn():
    row = prompt_completion(" um hi ", "Hi.")
    full = render(row["prompt"] + row["completion"], add_generation_prompt=False)
    assert full == render(row["prompt"]) + "Hi.<|im_end|>\n"


def test_the_template_refuses_a_system_turn():
    with pytest.raises(TemplateError, match="system"):
        render([{"role": "system", "content": "rules"}, *messages("hi")])
