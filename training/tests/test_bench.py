from whispernow_training.datagen.sources import ami
from whispernow_training.datagen.bench import assemble
from whispernow_training.datagen.schema import Example


def _example(id_, text, tags=()):
    return Example(id=id_, input=text, output=text, source="s", category="c", license="l", tags=list(tags))


def test_ami_turns_read_through_backchannels_and_break_on_long_gaps():
    segments = [
        {"meeting_id": "m1", "speaker_id": "A", "begin_time": 0.0, "end_time": 2.0, "text": "SO I THINK WE"},
        {"meeting_id": "m1", "speaker_id": "B", "begin_time": 2.0, "end_time": 2.5, "text": "YEAH"},
        {"meeting_id": "m1", "speaker_id": "A", "begin_time": 2.5, "end_time": 4.0, "text": "SHOULD GO I'M SURE"},
        {"meeting_id": "m1", "speaker_id": "A", "begin_time": 10.0, "end_time": 11.0, "text": "ANYWAY"},
        {"meeting_id": "m1", "speaker_id": "B", "begin_time": 11.0, "end_time": 13.0, "text": "OKAY LET'S MOVE ON NOW"},
    ]
    turns = ami.turns(segments)
    assert [t["text"] for t in turns] == ["so I think we should go I'm sure", "anyway", "okay let's move on now"]
    assert turns[0]["segments"] == 2


def test_pick_ami_spreads_turns_over_buckets_and_meetings():
    turns = [
        _example(f"t{i}", " ".join(["w"] * (160 + i)), [f"ami:m{i % 3}"]) for i in range(20)
    ] + [_example("long", " ".join(["w"] * 700), ["ami:m9"])]
    picked = assemble.pick_ami(turns)
    assert len([t for t in picked if t.id != "long"]) == 6  # two per meeting
    assert any(t.id == "long" for t in picked)


def test_overlap_flags_near_copies_of_build_inputs():
    build = [_example("b", "we need to book the ferry for nine people on friday and pay the deposit today")]
    bench_rows = [
        _example("copy", "um we need to book the ferry for nine people on friday and pay the deposit today ok"),
        _example("new", "the orchard delivery is late again so call the driver after lunch and ask about the pears"),
    ]
    assert [row["id"] for row in assemble.overlap(bench_rows, build)] == ["copy"]


def test_ami_raw_text_lowercases_and_joins_spelled_letters():
    assert ami.raw_text("I THINK THE T. V. REMOTE IS FINE I'M SURE") == "I think the tv remote is fine I'm sure"
    assert ami.raw_text("IT'S AN L. C. D.") == "it's an lcd"
