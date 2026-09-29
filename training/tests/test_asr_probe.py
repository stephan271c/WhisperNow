"""The Python port of the app's segmenter and join, checked against the Rust tests' cases."""

import numpy as np

from whispernow_training.datagen import asr_probe

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
