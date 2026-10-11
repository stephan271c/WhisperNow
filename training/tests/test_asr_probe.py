"""The Python port of the app's segmenter and join, checked against the Rust tests' cases."""

import json
import random
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from whispernow_training.datagen import asr_probe
from whispernow_training.datagen.schema import Example

SECOND = asr_probe.SAMPLE_RATE


def energy(frame):
    return float(np.mean(frame**2)) > 0.01 * 0.01


def tone(seconds, amplitude=0.25):
    count = int(seconds * SECOND)
    return np.where(np.arange(count) % 2 == 0, amplitude, -amplitude).astype(np.float32)


def silence(seconds):
    return tone(seconds, 0.0)


def segment(*parts):
    return asr_probe.segment(np.concatenate(parts), energy)


def test_silence_and_a_click_produce_no_segments():
    assert segment(silence(5.0)) == []
    click = silence(2.0)
    click[SECOND : SECOND + 512] = tone(0.032)
    assert asr_probe.segment(click, energy) == []


def test_a_long_pause_splits_speech_into_padded_segments():
    segments = segment(silence(1.0), tone(2.0), silence(1.0), tone(1.5), silence(1.0))
    assert len(segments) == 2
    for (start, end), (speech_start, speech_end) in zip(segments, [(SECOND, 3 * SECOND), (4 * SECOND, 11 * SECOND // 2)]):
        assert speech_start - asr_probe.PADDING - asr_probe.FRAME <= start <= speech_start - asr_probe.PADDING
        assert speech_end + asr_probe.PADDING - asr_probe.FRAME <= end <= speech_end + asr_probe.PADDING + asr_probe.FRAME


def test_short_pauses_do_not_split_and_short_segments_merge():
    audio = np.concatenate([tone(1.5), silence(0.4), tone(1.5)])
    assert asr_probe.segment(audio, energy) == [(0, len(audio))]
    merged = segment(tone(0.5), silence(1.0), tone(1.5), silence(1.0))
    assert len(merged) == 1 and merged[0][0] == 0 and merged[0][1] > 3 * SECOND


def test_a_long_segment_is_cut_below_the_maximum():
    segments = segment(tone(60.0))
    assert len(segments) >= 3
    assert all(end - start <= asr_probe.MAX_SEGMENT for start, end in segments)


def test_join_lowercases_a_continuing_segment_but_keeps_i_and_acronyms():
    assert asr_probe.join_segment_texts(["So we went", "To the store.", "Then home"]) == "So we went to the store. Then home"
    assert asr_probe.join_segment_texts(["He said \"stop.\"", "Nobody moved"]) == "He said \"stop.\" Nobody moved"
    assert (
        asr_probe.join_segment_texts(["and then", "I left", "I'm back", "NASA called", "A plan"])
        == "and then I left I'm back NASA called a plan"
    )
    assert asr_probe.join_segment_texts(["  one.  ", "", " ", "Two."]) == "one. Two."


@pytest.mark.parametrize("keep_audio", [True, False])
def test_run_preserves_shard_resume_metadata_audio_and_progress(tmp_path, monkeypatch, capsys, keep_audio):
    out = tmp_path / "probe"
    out.mkdir()
    (out / "parakeet.0.jsonl").write_text(json.dumps({"id": "row-1"}) + "\n")
    audio = tmp_path / "audio"
    rows = [Example(f"row-{i}", "first two three", "Fixed.", "dataset", "prose", "CC0", ["tag"]) for i in range(4)]
    monkeypatch.setattr(asr_probe, "OUT_DIR", out)
    monkeypatch.setattr(asr_probe, "AUDIO_DIR", audio)
    monkeypatch.setattr(asr_probe, "sample_rows", lambda n, seed: rows)
    spoken_ids = []

    def speak(text, rng):
        spoken_ids.append(text)
        return asr_probe.Spoken(tone(1.0), rng.choice(asr_probe.ENGLISH_VOICES), 1.05, ["first two", "three"], [0.25])

    monkeypatch.setattr(asr_probe, "Speaker", lambda: SimpleNamespace(speak=speak))
    monkeypatch.setattr(asr_probe, "Recognizer", lambda: SimpleNamespace(transcribe=lambda samples: ("First two three.", ["First two three."])))

    assert asr_probe.run(4, 7, keep_audio=keep_audio, shard=1, shards=2) == 1
    record = json.loads((out / "parakeet.1.jsonl").read_text())
    assert record == {
        "id": "row-3", "spoken": "first two three", "model": "parakeet", "heard": "First two three.",
        "segments": ["First two three."], "source": "dataset", "category": "prose", "layout": "plain",
        "tags": ["tag"], "chunk_ends": [2, 3], "pauses": [0.25],
        "voice": random.Random("7-row-3").choice(asr_probe.ENGLISH_VOICES), "speed": 1.05, "seconds": 1.0,
    }
    assert capsys.readouterr().out == "shard 1: 1/1 row-3 1.0s\n"
    assert (audio / "row-3.wav").exists() == keep_audio
    if keep_audio:
        with wave.open(str(audio / "row-3.wav")) as saved:
            assert saved.getframerate() == SECOND
            assert saved.getnframes() == SECOND

    assert asr_probe.run(4, 7, keep_audio=keep_audio, shard=1, shards=2) == 0
    assert spoken_ids == ["first two three"]
    assert len((out / "parakeet.1.jsonl").read_text().splitlines()) == 1
