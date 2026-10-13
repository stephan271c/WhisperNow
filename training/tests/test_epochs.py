import os
from dataclasses import replace

import pytest

from whispernow_training.datagen import build, epochs, inject
from whispernow_training.datagen.schema import EMPTY, Example, read_examples, write_examples
from whispernow_training.model import config

CLEAN = (
    "Move the standup to Thursday at 10:30 and tell Priya the budget is $450. "
    "We still need the blue folders, two boxes of pens and the projector cable from room 4."
)
SPOKEN = (
    "Okay so for the trip we need the tent, the green one, and two sleeping bags. Then snacks for the drive and "
    "a cooler, and Rosa said she'd bring the stove, so we can skip it."
)


@pytest.fixture
def epoch_build(tmp_path, monkeypatch):
    monkeypatch.setattr(epochs.paths, "BUILD", tmp_path)
    monkeypatch.setattr(epochs, "EPOCHS", tmp_path / "epochs")
    return tmp_path


def _injected(i: int, style: str = "recognizer") -> Example:
    noisy = inject.inject(CLEAN, f"build-{i}", style)
    return Example(
        id=f"injected-{i}", input=noisy.text, output=CLEAN, source=inject.SOURCE, category="filler_injection",
        license="MIT", tags=["injected", f"style_{style}", "from:owner/src", *noisy.tags, "long_origin"],
        seed_group=f"g{i}", split="train",
    )


def test_injected_rows_get_new_noise_each_epoch_around_the_same_target():
    row = _injected(1)
    assert epochs.reinjected(row, 1) is row
    second, third = epochs.reinjected(row, 2), epochs.reinjected(row, 3)
    assert second.input != row.input and third.input != second.input
    assert epochs.reinjected(row, 2) == second
    for variant in (second, third):
        assert (variant.id, variant.output, variant.seed_group, variant.split) == (row.id, CLEAN, "g1", "train")
        assert {"injected", "style_recognizer", "from:owner/src", "long_origin"} <= set(variant.tags)
        assert sum(tag.startswith("noise_level:") for tag in variant.tags) == 1


def test_raw_and_empty_injected_rows():
    raw = epochs.reinjected(_injected(2, "raw"), 2)
    assert raw.input == raw.input.lower() or "PM" in raw.input
    assert "style_raw" in raw.tags
    empty = Example(id="injected-e", input="um uh", output=EMPTY, source=inject.SOURCE, category="empty",
                    license="MIT", tags=["empty"], split="train")
    assert epochs.reinjected(empty, 4) is empty


def _composed(i: int) -> tuple[Example, Example]:
    """A composed train row before the variant step, and as the build wrote it (plus a later tag)."""
    before = Example(id=f"teacher-{i}", input=SPOKEN, output="Trip list", source="teacher", category="list_corrections",
                     license="x", tags=["style_recognizer", "composed"], seed_group=f"g{i}", split="train")
    built = build.spoken_variant(before, "")
    return before, replace(built, tags=[*built.tags, "later_tag"])


def test_composed_rows_draw_their_variant_again_from_the_row_before_it(monkeypatch):
    before, built = _composed(1)
    monkeypatch.setattr(build, "gather_with_drops", lambda: ([before], [], None))
    base = epochs._prepare_variants([built])[built.id]
    assert epochs.respoken(built, base, 1) is built
    fresh = epochs.respoken(built, base, 2)
    expected = build.spoken_variant(before, "epoch2")
    assert fresh == replace(built, input=expected.input, tags=[*expected.tags, "later_tag"])
    assert epochs.respoken(built, base, 2) == fresh


def test_epoch_rows_renoise_only_programmatic_rows_and_keep_the_order(monkeypatch):
    before, built = _composed(7)
    other = Example(id="aawaaz-1", input="um so the meeting is at three", output="The meeting is at 3.",
                    source=build.aawaaz.REPO, category="meeting_notes", license="MIT", split="train")
    injected = _injected(3)
    train = [other, built, injected]
    monkeypatch.setattr(build, "gather_with_drops", lambda: ([before], [], None))
    bases = epochs._prepare_variants(train)
    rows = epochs.epoch_rows(train, bases, 3)
    assert [r.id for r in rows] == [r.id for r in train]
    assert rows[0] is other
    assert [r.output for r in rows] == [r.output for r in train]
    assert rows[2].input != injected.input
    injection = inject.inject(CLEAN, "injected-3-epoch3", "recognizer")
    assert rows[2] == replace(
        injected, input=injection.text,
        tags=["injected", "style_recognizer", "from:owner/src", "long_origin", *injection.tags],
    )
    assert rows[1].input != built.input
    spoken = build.spoken_variant(before, "epoch3")
    assert rows[1] == replace(built, input=spoken.input, tags=[*spoken.tags, "later_tag"])
    assert all(original is first for original, first in zip(train, epochs.epoch_rows(train, bases, 1), strict=True))


@pytest.mark.parametrize("changed", ["source", "build"])
def test_materialization_rejects_source_and_build_drift(epoch_build, monkeypatch, changed):
    before, built = _composed(1)
    if changed == "source":
        before = replace(before, input="This source row no longer matches the build.")
    else:
        built = replace(built, input="This build row no longer matches the source.")
    monkeypatch.setattr(build, "gather_with_drops", lambda: ([before], [], None))
    write_examples(epoch_build / "train.jsonl", [built])
    with pytest.raises(ValueError, match=f"{built.id}: the build's variant does not come out of the rebuilt row"):
        epochs.materialize(3)
    assert not epochs.EPOCHS.exists()


def test_materialization_rejects_missing_reconstructed_rows(epoch_build, monkeypatch):
    _, built = _composed(1)
    monkeypatch.setattr(build, "gather_with_drops", lambda: ([], [], None))
    write_examples(epoch_build / "train.jsonl", [built])
    with pytest.raises(ValueError, match="1 train rows are not in the rebuilt rows.*teacher-1"):
        epochs.materialize(2)
    assert not epochs.EPOCHS.exists()


def test_materialized_epochs_keep_short_source_variants_and_reference_metadata(epoch_build, monkeypatch):
    gathered = [
        Example(id=f"short-{name}", input="Okay so, um, bring the green tent and two sleeping bags for the trip.",
                output="Bring the tent and two sleeping bags.", source=source, category="notes", license="MIT",
                tags=["style_recognizer"], teacher_model="teacher", teacher_reasoning="low")
        for name, source in (("bare", build.generate.SOURCE), ("handy", build.sources.HANDY),
                             ("disfl", build.sources.DISFL_QA))
    ]
    before = build.bare_filler_share([replace(row, split="train") for row in gathered])
    assert "fillers_stripped" in before[0].tags
    train = []
    for row in before:
        built = build.spoken_variant(row)
        train.append(replace(built, tags=[*built.tags, "later_tag", "later_tag"]))
    monkeypatch.setattr(build, "gather_with_drops", lambda: (gathered, [], None))
    write_examples(epoch_build / "train.jsonl", train)
    files = epochs.materialize(4)
    assert files[0] == epoch_build / "train.jsonl"
    assert read_examples(files[0]) == train
    for epoch, path in enumerate(files[1:], 2):
        expected = []
        for row, original in zip(train, before, strict=True):
            variant = build.spoken_variant(original, f"epoch{epoch}")
            expected.append(replace(row, input=variant.input, tags=[*variant.tags, "later_tag", "later_tag"]))
        assert read_examples(path) == expected


def test_cached_epochs_do_not_require_reconstructing_sources(epoch_build, monkeypatch):
    before, built = _composed(1)
    monkeypatch.setattr(build, "gather_with_drops", lambda: ([before], [], None))
    write_examples(epoch_build / "train.jsonl", [built])
    files = epochs.materialize(2)
    contents = [path.read_bytes() for path in files]
    modified = [path.stat().st_mtime_ns for path in files]
    monkeypatch.setattr(build, "gather_with_drops", lambda: ([], [], None))
    assert epochs.materialize(2) == files
    assert [path.read_bytes() for path in files] == contents
    assert [path.stat().st_mtime_ns for path in files] == modified
    with pytest.raises(ValueError, match="not in the rebuilt rows"):
        epochs.materialize(3)


def test_noise_per_epoch_needs_whole_epochs_over_the_build(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('base_model = "m"\nepochs = 3\n')
    assert config.load(path, ["noise_per_epoch=true"]).noise_per_epoch
    with pytest.raises(SystemExit):
        config.load(path, ["noise_per_epoch=true", "epochs=2.5"])


def test_a_rebuild_rewrites_the_epoch_files(epoch_build, monkeypatch):
    monkeypatch.setattr(epochs.build, "gather_with_drops", lambda: ([], [], None))
    write_examples(epoch_build / "train.jsonl", [_injected(1)])
    epochs.materialize(2)
    assert [e.id for e in read_examples(epochs.epoch_file(2))] == ["injected-1"]
    write_examples(epoch_build / "train.jsonl", [_injected(2)])
    later = epochs.epoch_file(2).stat().st_mtime + 10
    os.utime(epoch_build / "train.jsonl", (later, later))
    epochs.materialize(2)
    assert [e.id for e in read_examples(epochs.epoch_file(2))] == ["injected-2"]
