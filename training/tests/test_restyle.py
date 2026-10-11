import asyncio
import json

import pytest

from whispernow_training.datagen import restyle
from whispernow_training.datagen.llm import LlmError
from whispernow_training.datagen.schema import Example, read_jsonl, write_examples


@pytest.fixture
def restyle_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(restyle, "OUT", tmp_path / "restyle.jsonl")
    monkeypatch.setattr(restyle, "REJECTED", tmp_path / "rejected.jsonl")
    monkeypatch.setattr(restyle.stage1, "CLEAN", tmp_path / "clean.jsonl")
    monkeypatch.setattr(restyle, "chosen", lambda example: True)
    rows = [Example(id=id_, input=f"bring {item}", output=f"Bring {item}.", source="s", category="c", license="MIT")
            for id_, item in (("a", "milk"), ("b", "eggs"), ("c", "bread"))]
    write_examples(restyle.stage1.CLEAN, rows)
    return rows


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


@pytest.mark.parametrize("failure", [
    ValueError("invalid client configuration"),
    LlmError("transport retries exhausted", retryable=True),
], ids=["configuration", "exhausted-transport"])
def test_client_failures_preserve_finished_batches_and_resume_the_rest(failure, fake_llm, restyle_rows, monkeypatch):
    monkeypatch.setattr(restyle, "MAX_BATCH", 1)
    sent = []

    def reply(messages, role):
        assert role == "teacher"
        sent.append(messages[-1]["content"])
        if "eggs" in messages[-1]["content"]:
            raise failure
        return json.dumps({"items": [{"n": 0, "text": "Bring milk."}]})

    llm = fake_llm(reply)
    with pytest.raises(type(failure)) as error:
        asyncio.run(restyle.restyle_all(llm, limit=2))
    assert error.value is failure and llm.calls == 2
    saved = list(read_jsonl(restyle.OUT))
    assert saved == [{"id": "a", "input": "Bring milk."}] and not list(read_jsonl(restyle.REJECTED))

    def resumed_reply(messages, role):
        assert "milk" not in messages[-1]["content"]
        sent.append(messages[-1]["content"])
        return json.dumps({"items": [{"n": 0, "text": "Bring eggs." if "eggs" in messages[-1]["content"] else "Bring bread."}]})

    resumed = fake_llm(resumed_reply)
    assert asyncio.run(restyle.restyle_all(resumed)) == {"restyled": 2, "failed": 0, "remaining_batches": 0}
    assert resumed.calls == 2 and [row["id"] for row in read_jsonl(restyle.OUT)] == ["a", "b", "c"]
    assert list(read_jsonl(restyle.OUT))[0] == saved[0] and not list(read_jsonl(restyle.REJECTED))


@pytest.mark.parametrize("reply", [
    "malformed",
    '{"items": []}',
    '{"items": [{"n": 0}]}',
    '{"items": [{"n": 0, "text": 123}]}',
    '{"items": [{"n": 0, "text": "Bring milk."}, {"n": 0, "text": "Bring milk."}]}',
])
def test_malformed_restyle_replies_are_diagnostics_that_remain_pending(reply, fake_llm, restyle_rows):
    malformed = fake_llm(lambda messages, role: reply)
    assert asyncio.run(restyle.restyle_all(malformed, limit=1)) == {"restyled": 0, "failed": 1, "remaining_batches": 0}
    diagnostic, = read_jsonl(restyle.REJECTED)
    assert diagnostic["id"] == "a" and diagnostic["why"].startswith("reply:")
    assert malformed.calls == 1 and not list(read_jsonl(restyle.OUT))
    resumed = fake_llm(lambda messages, role: '{"items": [{"n": 0, "text": "Bring milk."}]}')
    assert asyncio.run(restyle.restyle_all(resumed, limit=1))["restyled"] == 1
    assert resumed.calls == 1 and list(read_jsonl(restyle.OUT)) == [{"id": "a", "input": "Bring milk."}]
    assert list(read_jsonl(restyle.REJECTED)) == [diagnostic]


def test_legacy_infrastructure_diagnostics_resume_but_content_rejections_stay_done(fake_llm, restyle_rows):
    rejected = [
        {"id": "a", "why": "reply: transport retries exhausted", "text": ""},
        {"id": "b", "why": "eggs != milk", "text": "Bring milk."},
    ]
    restyle.REJECTED.write_text("".join(json.dumps(row) + "\n" for row in rejected))

    def reply(messages, role):
        user = messages[-1]["content"]
        assert "eggs" not in user and "milk" in user and "bread" in user
        return '{"items": [{"n": 0, "text": "Bring milk."}, {"n": 1, "text": "Bring bread."}]}'

    llm = fake_llm(reply)
    assert asyncio.run(restyle.restyle_all(llm))["restyled"] == 2 and llm.calls == 1
    assert [row["id"] for row in read_jsonl(restyle.OUT)] == ["a", "c"]
    assert list(read_jsonl(restyle.REJECTED)) == rejected
    assert asyncio.run(restyle.restyle_all(llm))["restyled"] == 0 and llm.calls == 1
