import random

from whispernow_training.datagen import inject, sources
from whispernow_training.datagen.schema import EMPTY, Example

CLEAN = (
    "Hi Sarah, the meeting moved to Thursday at 3 PM in room B2. Can you send the deck to "
    "jane.doe@example.com and run npm install --save-dev first? I'm sure the U.S. team will join."
)
PROTECTED = ("B2", "jane.doe@example.com", "--save-dev", "npm", "U.S.", "3", "PM")


def test_injection_is_deterministic_for_a_seed():
    assert inject.inject(CLEAN, random.Random(4), "recognizer") == inject.inject(CLEAN, random.Random(4), "recognizer")
    assert inject.inject(CLEAN, random.Random(4), "raw") != inject.inject(CLEAN, random.Random(5), "raw")


def test_recognizer_style_keeps_every_token_and_adds_speech():
    for seed in range(20):
        spoken = inject.inject(CLEAN, random.Random(seed), "recognizer")
        assert spoken != CLEAN
        for token in PROTECTED:
            assert token in spoken
        # Removing what was added gives back the clean words in order.
        added = set(sources.FILLERS) | {"you", "know", "i", "mean", "like", "so", "okay", "yeah", "alright"}
        clean_words = sources.words(CLEAN)
        spoken_words = [w for w in sources.words(spoken)]
        position = 0
        for word in spoken_words:
            if position < len(clean_words) and word == clean_words[position]:
                position += 1
            else:
                assert word in added or word == clean_words[position - 1]
        assert position == len(clean_words)


def test_raw_style_drops_case_and_edge_punctuation_but_not_code():
    spoken = inject.inject(CLEAN, random.Random(1), "raw")
    assert "," not in spoken and "?" not in spoken
    assert "sarah" in spoken and "Sarah" not in spoken
    for token in ("B2", "jane.doe@example.com", "--save-dev", "U.S.", "I'm"):
        assert token in spoken



def test_raw_style_keeps_leading_punctuation_and_lowercases_contractions():
    assert inject._raw_token("./data:/app/data,") == "./data:/app/data"
    assert inject._raw_token("There's") == "there's"
    assert inject._raw_token("I'm") == "I'm"
    assert inject._raw_token("NASA,") == "NASA"


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
    assert {"style_recognizer", "style_raw"} <= {t for row in rows for t in row.tags}
    assert [r.input for r in rows] == [r.input for r in inject.injected_rows(origins, 20, seed=3)]


def test_empty_rows_are_unique_filler_only_inputs():
    rows = inject.empty_rows(50, seed=9)
    assert len(rows) == 50
    assert len({r.input for r in rows}) == 50
    for row in rows:
        assert row.output == EMPTY
        assert all(word in sources.FILLERS | {"umm", "uhh"} for word in sources.words(row.input))
    assert [r.input for r in rows] == [r.input for r in inject.empty_rows(50, seed=9)]
