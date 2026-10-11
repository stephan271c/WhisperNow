import re
from collections import Counter

import pytest

from whispernow_training.datagen import inject, sources
from whispernow_training.datagen.schema import EMPTY, Example
from whispernow_training.model.metrics import protected_tokens
from whispernow_training.text_rules import FILLER

CLEAN = (
    "Hi Sarah, the meeting moved to Thursday at 3 PM in room B2. Can you send the deck to "
    "jane.doe@example.com and run npm install --save-dev first? I'm sure the U.S. team will join."
)
TEXTS = (
    CLEAN,
    "We're meeting on Tuesday at 3:30 with the design team. Bring the blue folder, the budget sheet, the "
    "old contracts, and two printed copies. I think we need 12 chairs and a projector, so please book room 4 "
    "before Friday.",
    "Can you order the green one for my sister? She wants it by March, and the total should stay under $80. "
    "It's a gift, so we should wrap it and we can send it from the office.",
    "Right now it's too cold to paint the fence, but we could do it in the spring when it warms up. I don't "
    "know if the hardware store still has the stain we liked, so I need to call them about the order and ask "
    "them to hold it for us until then.",
)
FILLER_WORDS = {"um", "uh", "er", "hmm"}
CORRECTION_WORDS = {
    "no", "sorry", "wait", "i", "mean", "or", "actually", "let's", "make", "it", "go", "with", "except", "well",
    "that's", "not", "right", "wrong", "got", "that", "mixed", "up", "had", "so", "just", "instead",
} | FILLER_WORDS
AMBIGUOUS_WORDS = {"like", "actually", "basically", "so", "well", "okay", "right"}
_WORD = re.compile(r"[\w'$%:.@/-]+")


def _words(text: str) -> list[str]:
    return [w.strip(".:").casefold() for w in _WORD.findall(text) if w.strip(".:")]


def _added(clean: str, spoken: str) -> Counter:
    return Counter(_words(spoken)) - Counter(_words(clean))


def _is_subsequence(clean: str, spoken: str) -> bool:
    it = iter(_words(spoken))
    return all(word in it for word in _words(clean))


def _cases(kind: str):
    styles = ("recognizer",) if kind == "pause" else inject.STYLES
    for text in TEXTS:
        for style in styles:
            for seed in range(25):
                yield text, style, seed, inject.inject(text, seed, style, kinds=(kind,), level="heavy")


@pytest.mark.parametrize("kind", inject.KINDS)
def test_each_kind_applies_and_is_deterministic(kind):
    applied = 0
    for text, style, seed, result in _cases(kind):
        assert result == inject.inject(text, seed, style, kinds=(kind,), level="heavy")
        assert set(result.kinds) <= {kind}
        applied += bool(result.kinds)
    assert applied > 50
    different = {inject.inject(TEXTS[1], seed, "recognizer", kinds=(kind,)).text for seed in range(10)}
    assert len(different) > 1


@pytest.mark.parametrize("kind", inject.KINDS)
def test_each_kind_keeps_every_clean_word_and_protected_token(kind):
    for text, style, seed, result in _cases(kind):
        spoken = result.text
        assert _is_subsequence(text, spoken), (kind, seed, spoken)
        for token in protected_tokens(text):
            assert token in spoken, (kind, seed, token, spoken)


@pytest.mark.parametrize("kind", [k for k in inject.KINDS if k != "correction"])
def test_random_noise_adds_no_ambiguous_word(kind):
    for text, _style, _seed, result in _cases(kind):
        added = _added(text, result.text)
        assert not set(added) & AMBIGUOUS_WORDS, (kind, result.text)
        assert "you know" not in result.text.lower() and "i mean" not in result.text.lower()
        new_words = set(added) - set(_words(text)) - {w.split("'")[0] for w in _words(text)}
        assert new_words <= FILLER_WORDS, (kind, new_words, result.text)


def test_corrections_insert_a_wrong_value_the_target_never_has():
    forms = Counter()
    for text, _style, _seed, result in _cases("correction"):
        assert result.corrections or not result.kinds
        spoken = result.text.casefold()
        for wrong, right, form in result.corrections:
            forms[form] += 1
            digits = re.escape(wrong.lstrip("$").rstrip("%"))
            assert not re.search(rf"(?<![\d.]){digits}(?!\d)" if wrong[:1] in "$0123456789" else rf"\b{digits}\b", text, re.I)
            assert right in text
            assert spoken.index(wrong.casefold()) < spoken.rindex(right.casefold())
        new_words = set(_added(text, result.text)) - {w.casefold() for w, _, _ in result.corrections}
        assert new_words <= CORRECTION_WORDS, (new_words, result.text)
    assert set(forms) == set(inject.CORRECTION_FORMS)
    # Stacked cues stay rare.
    assert forms["short"] > forms["reasoned"]


def test_recognizer_fillers_are_mostly_bare_and_capitalized_at_a_sentence_start():
    results = [inject.inject(text, seed, "recognizer", kinds=("filler", "cluster")) for text in TEXTS for seed in range(60)]
    bare = [r for r in results if r.fillers == "bare"]
    assert 0.7 < len(bare) / len(results) < 0.95
    for result in bare:
        assert not re.search(r"\b(?:um|uh|er|hmm),", result.text, re.I), result.text
    wrapped = [r for r in results if r.fillers == "commas"]
    assert any(re.search(r"\b(?:um|uh|er|hmm),", r.text) for r in wrapped)
    assert any(re.search(r"(?:^|[.?!] )(?:Um|Uh|Er|Hmm) [a-z]", r.text) for r in bare)


def test_fillers_land_where_speakers_plan_not_inside_a_phrase_name_or_spoken_symbol():
    text = (
        "Okay so the car is at Bell Street Motors and Mateo drove me home because the rear brakes were nearly gone. "
        "Then type cd space slash home slash bakery and run python3 space dash dash version, then ask Dennis about "
        "the big green tarp."
    )
    before = Counter()
    for seed in range(300):
        tokens = inject.inject(text, seed, "recognizer", kinds=("filler",), level="heavy").text.split()
        for i, token in enumerate(tokens):
            if token.lower() not in FILLER_WORDS:
                continue
            previous = tokens[i - 1].strip(",.").lower() if i else ""
            following = tokens[i + 1].strip(",.").lower() if i + 1 < len(tokens) else ""
            assert previous not in {"bell", "street", "space", "slash", "dash"}, " ".join(tokens)
            assert following not in {"street", "motors", "space", "slash", "dash"}, " ".join(tokens)
            before[previous] += 1
    planned = sum(before[w] for w in ("", "so", "and", "because", "then", "at", "about", "type", "run", "ask", "the"))
    inside = sum(before[w] for w in ("rear", "nearly", "big", "green", "brakes", "car"))
    assert planned > 4 * inside


def test_densify_adds_hesitation_to_a_spoken_text_and_keeps_its_words():
    spoken = (
        "So we need eggs, a dozen of the brown ones, and milk. Actually no, let's do oat milk instead because Grace "
        "is off dairy now. Then bread, the sourdough from Fred Meyer, two loaves since Xavier eats a lot of "
        "sandwiches, and chicken thighs, like 3 pounds, we can freeze half. Oh and maybe some rice, I think we're low."
    )
    for style, text in (("recognizer", spoken), ("raw", spoken.lower().replace(",", "").replace(".", ""))):
        rates = []
        for seed in range(40):
            result = inject.densify(text, seed, style)
            assert result == inject.densify(text, seed, style)
            assert _is_subsequence(text, result.text), result.text
            assert set(_added(text, result.text)) <= FILLER_WORDS | set(_words(text)), result.text
            for token in protected_tokens(text):
                assert token in result.text
            assert "Fred Meyer" in result.text or style == "raw"
            if style == "raw":
                assert not re.search(r"[A-Z]", result.text.replace("Um", "").replace("Uh", "")), result.text
            rates.append(100 * inject.filler_count(result.text) / len(result.text.split()))
        assert 2.5 < sum(rates) / len(rates) < 6.5
    already_dense = "um so uh we need um eggs and uh milk and um uh bread"
    assert inject.densify(already_dense, 1, "raw").text == already_dense


def test_pause_punctuation_splits_a_sentence_without_touching_words():
    seen_period = seen_question = False
    for text, _style, _seed, result in _cases("pause"):
        assert _words(result.text) == _words(text)
        seen_period |= bool(re.search(r"\b[a-z]+\. [A-Za-z]", result.text)) and result.text.count(".") > text.count(".")
        seen_question |= result.text.count("?") > text.count("?")
    assert seen_period
    assert seen_question


def test_raw_style_drops_case_and_edge_punctuation_but_not_code():
    spoken = inject.inject(CLEAN + " There's a cache at ./data:/app/data, and NASA will use it.", 1, "raw").text
    assert "," not in spoken and "?" not in spoken
    assert "sarah" in spoken and "Sarah" not in spoken
    for token in ("B2", "jane.doe@example.com", "--save-dev", "U.S.", "I'm", "there's", "./data:/app/data", "NASA"):
        assert token in spoken


def test_levels_and_kinds_are_tagged():
    levels = Counter()
    for seed in range(200):
        result = inject.inject(TEXTS[1], seed, "recognizer")
        levels[result.level] += 1
        assert f"noise_level:{result.level}" in result.tags
        assert all(f"noise:{kind}" in result.tags for kind in result.kinds)
    assert set(levels) == set(inject.LEVELS)


def test_injected_rows_keep_origin_target_group_and_license():
    origins = [
        Example(id=f"a{i}", input=f"raw {i}", output=f"We met on day {i} and it went well, I think.", source="owner/aawaaz", category="c", license="MIT")
        for i in range(10)
    ]
    origins.append(Example(id="list", input="x", output="We need:\n- Milk", source="owner/aawaaz", category="c", license="MIT"))
    origins.append(Example(id="t", input="y", output="Held out, as a test row.", source="owner/aawaaz", category="c", license="MIT", split="test"))
    rows = inject.injected_rows(origins, 20, seed=3)
    assert len(rows) == 10
    by_group = {e.id: e for e in origins}
    for row in rows:
        origin = by_group[row.seed_group]
        assert row.output == origin.output and row.input != origin.output
        assert row.source == "injected" and row.license == "MIT"
        assert row.split == ""
        assert any(tag.startswith("noise_level:") for tag in row.tags)
    assert {"style_recognizer", "style_raw"} <= {t for row in rows for t in row.tags}
    assert [r.input for r in rows] == [r.input for r in inject.injected_rows(origins, 20, seed=3)]


def test_injected_rows_take_paragraph_origins_as_one_spoken_block():
    text = "We met on Friday about the launch, and it went well.\n\nThe venue holds 200 people and costs 5000 dollars."
    origin = Example(id="p", input="x", output=text, source="teacher", category="c", license="zai-output")
    assert origin.layout == "paragraphs"
    assert inject.injected_rows([origin], 5, seed=1) == []
    (row,) = inject.injected_rows([origin], 5, seed=1, layouts=("plain", "paragraphs"))
    assert row.output == text and "\n" not in row.input
    assert {"200", "5000"} <= set(re.findall(r"\d+", row.input))


def test_empty_rows_are_unique_filler_only_inputs():
    rows = inject.empty_rows(50, seed=9)
    assert len(rows) == 50
    assert len({r.input for r in rows}) == 50
    for row in rows:
        assert row.output == EMPTY
        assert all(FILLER.fullmatch(word) for word in sources.words(row.input))
    assert [r.input for r in rows] == [r.input for r in inject.empty_rows(50, seed=9)]
