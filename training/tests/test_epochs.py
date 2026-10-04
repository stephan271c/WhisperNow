import re
from dataclasses import replace

import pytest

from whispernow_training.datagen import build, epochs, inject
from whispernow_training.datagen.schema import EMPTY, Example
from whispernow_training.model import config, report

CLEAN = (
    "Move the standup to Thursday at 10:30 and tell Priya the budget is $450. "
    "We still need the blue folders, two boxes of pens and the projector cable from room 4."
)
SPOKEN = (
    "Okay so for the trip we need the tent, the green one, and two sleeping bags. Then snacks for the drive and "
    "a cooler, and Rosa said she'd bring the stove, so we can skip it."
)


def _injected(i: int, style: str = "recognizer") -> Example:
    noisy = inject.inject(CLEAN, f"build-{i}", style)
    return Example(
        id=f"injected-{i}", input=noisy.text, output=CLEAN, source=inject.SOURCE, category="filler_injection",
        license="MIT", tags=["injected", f"style_{style}", "from:owner/src", *noisy.tags, "long_origin"],
        seed_group=f"g{i}", split="train",
    )


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


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
        # The clean words are all said, in order; what is added is noise the target drops.
        said = iter(_words(variant.input))
        assert all(word in said for word in _words(CLEAN)), variant.input
        assert not re.search(r"\b(?:like|i mean|you know|basically)\b", variant.input.lower())


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
    return before, replace(built, tags=[*built.tags, "audit_relabeled"])


def test_composed_rows_draw_their_variant_again_from_the_row_before_it():
    before, built = _composed(1)
    assert epochs.respoken(built, before, 1) is built
    fresh = epochs.respoken(built, before, 2)
    expected = build.spoken_variant(before, "epoch2")
    assert fresh.input == expected.input and fresh.output == built.output
    assert fresh.tags == [*expected.tags, "audit_relabeled"]
    with pytest.raises(ValueError):
        epochs.respoken(replace(built, input="something else"), before, 2)


def test_each_epoch_keeps_the_build_share_of_clean_inputs():
    pairs = [_composed(i) for i in range(400)]
    built_dense = sum("densified" in built.tags for _, built in pairs)
    for epoch in (2, 3, 4, 5):
        dense = sum("densified" in epochs.respoken(built, before, epoch).tags for before, built in pairs)
        assert abs(dense - built_dense) < 50, (epoch, dense, built_dense)


def test_epoch_rows_renoise_only_programmatic_rows_and_keep_the_order():
    before, built = _composed(7)
    other = Example(id="aawaaz-1", input="um so the meeting is at three", output="The meeting is at 3.",
                    source=build.aawaaz.REPO, category="meeting_notes", license="MIT", split="train")
    train = [other, built, _injected(3)]
    rows = epochs.epoch_rows(train, {before.id: before}, 3)
    assert [r.id for r in rows] == [r.id for r in train]
    assert rows[0] is other
    assert [r.output for r in rows] == [r.output for r in train]
    assert epochs.epoch_rows(train, {before.id: before}, 1) == train


def test_noise_per_epoch_needs_whole_epochs_over_the_build(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('base_model = "m"\nepochs = 3\n')
    assert config.load(path, ["noise_per_epoch=true"]).noise_per_epoch
    with pytest.raises(SystemExit):
        config.load(path, ["noise_per_epoch=true", "epochs=2.5"])
    with pytest.raises(SystemExit):
        config.load(path, ["objective='dpo'"])


def test_val_loss_per_epoch_is_counted_in_steps():
    # A noise run is one pass: its trainer's epoch ends at 1, but 3 epochs' worth of steps.
    train = {
        "config": {"epochs": 3},
        "steps": 300,
        "final_eval_loss": 0.30,
        "log_history": [
            {"step": 50, "epoch": 0.17, "eval_loss": 0.5},
            {"step": 100, "epoch": 0.33, "eval_loss": 0.4},
            {"step": 200, "epoch": 0.67, "eval_loss": 0.35},
            {"step": 250, "epoch": 0.83, "eval_loss": 0.33},
        ],
    }
    assert report.val_loss_by_epoch(train) == [0.4, 0.35, 0.30]


def test_a_rebuild_rewrites_the_epoch_files(tmp_path, monkeypatch):
    import os

    from whispernow_training.datagen.schema import read_examples, write_examples

    monkeypatch.setattr(epochs.paths, "BUILD", tmp_path)
    monkeypatch.setattr(epochs, "EPOCHS", tmp_path / "epochs")
    write_examples(tmp_path / "train.jsonl", [_injected(1)])
    epochs.materialize(2)
    assert [e.id for e in read_examples(epochs.epoch_file(2))] == ["injected-1"]
    write_examples(tmp_path / "train.jsonl", [_injected(2)])
    later = epochs.epoch_file(2).stat().st_mtime + 10
    os.utime(tmp_path / "train.jsonl", (later, later))
    epochs.materialize(2)
    assert [e.id for e in read_examples(epochs.epoch_file(2))] == ["injected-2"]
