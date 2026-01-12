"""Tests for AudioProcessor and audio chunking functionality."""

import numpy as np
import pytest

from src.transcribe.core.audio.audio_processor import (
    MAX_DURATION_SECONDS,
    AudioChunkInfo,
    AudioPreview,
    AudioProcessor,
    needs_chunking,
)


class TestNeedsChunking:
    def test_short_audio_does_not_need_chunking(self):
        sample_rate = 16000
        # 10 seconds of audio
        audio_data = np.zeros(10 * sample_rate, dtype=np.float32)

        assert needs_chunking(audio_data, sample_rate) is False

    def test_long_audio_needs_chunking(self):
        sample_rate = 16000
        # 45 seconds of audio
        audio_data = np.zeros(45 * sample_rate, dtype=np.float32)

        assert needs_chunking(audio_data, sample_rate) is True

    def test_exactly_threshold_does_not_need_chunking(self):
        sample_rate = 16000
        # Exactly 30 seconds
        audio_data = np.zeros(int(MAX_DURATION_SECONDS * sample_rate), dtype=np.float32)

        assert needs_chunking(audio_data, sample_rate) is False


class TestAudioChunkInfo:
    def test_chunk_info_creation(self):
        chunk = AudioChunkInfo(start_sample=0, end_sample=16000, duration_seconds=1.0)
        assert chunk.start_sample == 0
        assert chunk.end_sample == 16000
        assert chunk.duration_seconds == 1.0

    def test_sample_count_property(self):
        chunk = AudioChunkInfo(
            start_sample=1000, end_sample=5000, duration_seconds=0.25
        )
        assert chunk.sample_count == 4000


class TestAudioPreview:
    def test_preview_creation(self):
        preview = AudioPreview(
            duration_seconds=45.5,
            sample_rate=16000,
            sample_count=728000,
            needs_chunking=True,
            estimated_chunks=2,
        )
        assert preview.duration_seconds == 45.5
        assert preview.needs_chunking is True
        assert preview.estimated_chunks == 2

    def test_duration_formatted_seconds(self):
        preview = AudioPreview(
            duration_seconds=25.0,
            sample_rate=16000,
            sample_count=400000,
            needs_chunking=False,
            estimated_chunks=1,
        )
        assert preview.duration_formatted == "25s"

    def test_duration_formatted_minutes(self):
        preview = AudioPreview(
            duration_seconds=95.0,
            sample_rate=16000,
            sample_count=1520000,
            needs_chunking=True,
            estimated_chunks=4,
        )
        assert preview.duration_formatted == "1m 35s"


class TestAudioProcessorSplitting:
    def test_short_audio_returns_single_chunk(self):
        processor = AudioProcessor()
        sample_rate = 16000
        # 10 seconds of silence
        audio_data = np.zeros(10 * sample_rate, dtype=np.float32)

        chunks = processor.split_audio(audio_data, sample_rate)

        assert len(chunks) == 1
        assert len(chunks[0]) == len(audio_data)

    def test_long_silence_creates_multiple_chunks(self):
        processor = AudioProcessor(max_duration=10.0, min_chunk_duration=2.0)
        sample_rate = 16000
        # 25 seconds of silence (should create ~3 chunks with 10s max)
        audio_data = np.zeros(25 * sample_rate, dtype=np.float32)

        chunks = processor.split_audio(audio_data, sample_rate)

        assert len(chunks) >= 2
        # Verify all original audio is covered (approximately, due to overlap)
        total_samples = sum(len(c) for c in chunks)
        assert total_samples >= len(audio_data)

    def test_split_at_silence_points(self):
        processor = AudioProcessor(
            max_duration=5.0, min_chunk_duration=1.0, silence_threshold=0.1
        )
        sample_rate = 16000

        # Create audio: loud-silence-loud-silence-loud pattern
        # 3s loud, 1s silence, 3s loud, 1s silence, 3s loud = 11s
        loud = np.sin(np.linspace(0, 1000, 3 * sample_rate)) * 0.5
        silence = np.zeros(sample_rate, dtype=np.float64)

        audio_data = np.concatenate([loud, silence, loud, silence, loud]).astype(
            np.float32
        )

        chunks = processor.split_audio(audio_data, sample_rate)

        # Should create multiple chunks
        assert len(chunks) >= 2

    def test_fallback_to_time_based_splitting(self):
        processor = AudioProcessor(
            max_duration=5.0,
            min_chunk_duration=1.0,
            silence_threshold=0.001,  # Very low threshold - hard to find silence
        )
        sample_rate = 16000

        # Continuous loud audio (no silence)
        audio_data = (np.sin(np.linspace(0, 5000, 15 * sample_rate)) * 0.5).astype(
            np.float32
        )

        chunks = processor.split_audio(audio_data, sample_rate)

        # Should still create chunks, using time-based fallback
        assert len(chunks) >= 2


class TestAudioProcessorPreview:
    def test_preview_short_audio(self):
        processor = AudioProcessor()
        sample_rate = 16000
        audio_data = np.zeros(10 * sample_rate, dtype=np.float32)

        preview = processor.preview(audio_data, sample_rate)

        assert preview.needs_chunking is False
        assert preview.estimated_chunks == 1
        assert preview.duration_seconds == 10.0

    def test_preview_long_audio(self):
        processor = AudioProcessor()
        sample_rate = 16000
        audio_data = np.zeros(60 * sample_rate, dtype=np.float32)

        preview = processor.preview(audio_data, sample_rate)

        assert preview.needs_chunking is True
        assert preview.estimated_chunks >= 2


class TestTranscriptionCombining:
    def test_combine_single_transcription(self):
        processor = AudioProcessor()

        result = processor.combine_transcriptions(["Hello world"])

        assert result == "Hello world"

    def test_combine_multiple_transcriptions(self):
        processor = AudioProcessor()

        result = processor.combine_transcriptions(["Hello", "world", "how are you"])

        assert result == "Hello world how are you"

    def test_combine_handles_trailing_spaces(self):
        processor = AudioProcessor()

        result = processor.combine_transcriptions(["Hello ", " world"])

        assert result == "Hello world"

    def test_combine_removes_double_spaces(self):
        processor = AudioProcessor()

        result = processor.combine_transcriptions(["Hello  ", "  world"])

        assert result == "Hello world"

    def test_combine_empty_list(self):
        processor = AudioProcessor()

        result = processor.combine_transcriptions([])

        assert result == ""

    def test_combine_filters_empty_transcriptions(self):
        processor = AudioProcessor()

        result = processor.combine_transcriptions(["Hello", "", "world", "   ", "!"])

        assert result == "Hello world !"

    def test_combine_none_in_list(self):
        processor = AudioProcessor()

        # Filter out None before passing (as the real code does)
        transcriptions = ["Hello", None, "world"]
        valid = [t for t in transcriptions if t]

        result = processor.combine_transcriptions(valid)

        assert result == "Hello world"


class TestAudioProcessorConfiguration:
    def test_default_configuration(self):
        processor = AudioProcessor()

        assert processor.max_duration == 30.0
        assert processor.min_chunk_duration == 5.0
        assert processor.silence_threshold == 0.02

    def test_custom_configuration(self):
        processor = AudioProcessor(
            max_duration=60.0, min_chunk_duration=10.0, silence_threshold=0.05
        )

        assert processor.max_duration == 60.0
        assert processor.min_chunk_duration == 10.0
        assert processor.silence_threshold == 0.05
