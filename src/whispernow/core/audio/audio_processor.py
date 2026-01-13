"""
Audio processing utilities for handling long audio recordings.

Provides automatic splitting of long audio using silence detection
for optimal split points, without interrupting speech.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ...utils.logger import get_logger

logger = get_logger(__name__)


MAX_DURATION_SECONDS = 30.0
MIN_CHUNK_DURATION_SECONDS = 5.0
SILENCE_THRESHOLD = 0.02
SILENCE_DURATION_SECONDS = 0.3
OVERLAP_DURATION_SECONDS = 0.1


@dataclass
class AudioChunkInfo:
    start_sample: int
    end_sample: int
    duration_seconds: float

    @property
    def sample_count(self) -> int:
        return self.end_sample - self.start_sample


@dataclass
class AudioPreview:
    duration_seconds: float
    sample_rate: int
    sample_count: int
    needs_chunking: bool
    estimated_chunks: int
    chunk_infos: List[AudioChunkInfo] = field(default_factory=list)

    @property
    def duration_formatted(self) -> str:
        minutes = int(self.duration_seconds // 60)
        seconds = int(self.duration_seconds % 60)
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"


def needs_chunking(audio_data: np.ndarray, sample_rate: int) -> bool:
    duration = len(audio_data) / sample_rate
    return duration > MAX_DURATION_SECONDS


class AudioProcessor:
    """Handles audio processing including smart splitting with silence detection."""

    def __init__(
        self,
        max_duration: float = MAX_DURATION_SECONDS,
        min_chunk_duration: float = MIN_CHUNK_DURATION_SECONDS,
        silence_threshold: float = SILENCE_THRESHOLD,
        silence_duration: float = SILENCE_DURATION_SECONDS,
        overlap_duration: float = OVERLAP_DURATION_SECONDS,
    ):
        self.max_duration = max_duration
        self.min_chunk_duration = min_chunk_duration
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.overlap_duration = overlap_duration

    def preview(self, audio_data: np.ndarray, sample_rate: int) -> AudioPreview:
        duration = len(audio_data) / sample_rate
        requires_chunking = duration > self.max_duration

        chunk_infos = []
        if requires_chunking:
            split_points = self._find_split_points(audio_data, sample_rate)

            if not split_points:
                split_points = self._generate_time_based_splits(
                    len(audio_data), sample_rate
                )

            start_idx = 0
            for end_idx in split_points + [len(audio_data)]:
                chunk_duration = (end_idx - start_idx) / sample_rate
                chunk_infos.append(
                    AudioChunkInfo(
                        start_sample=start_idx,
                        end_sample=end_idx,
                        duration_seconds=chunk_duration,
                    )
                )
                start_idx = end_idx

            estimated_chunks = len(chunk_infos)
        else:
            estimated_chunks = 1
            chunk_infos = [
                AudioChunkInfo(
                    start_sample=0,
                    end_sample=len(audio_data),
                    duration_seconds=duration,
                )
            ]

        logger.debug(f"Audio preview: {duration:.1f}s, {estimated_chunks} chunk(s)")

        return AudioPreview(
            duration_seconds=duration,
            sample_rate=sample_rate,
            sample_count=len(audio_data),
            needs_chunking=requires_chunking,
            estimated_chunks=estimated_chunks,
            chunk_infos=chunk_infos,
        )

    def split_audio(self, audio_data: np.ndarray, sample_rate: int) -> List[np.ndarray]:
        duration = len(audio_data) / sample_rate

        if duration <= self.max_duration:
            logger.debug(
                f"Audio duration {duration:.1f}s under threshold, no splitting needed"
            )
            return [audio_data]

        logger.info(f"Splitting {duration:.1f}s audio into chunks...")

        split_points = self._find_split_points(audio_data, sample_rate)

        if not split_points:
            logger.warning(
                "No suitable silence points found, using time-based splitting"
            )
            split_points = self._generate_time_based_splits(
                len(audio_data), sample_rate
            )

        chunks = self._create_chunks(audio_data, sample_rate, split_points)

        logger.info(f"Split audio into {len(chunks)} chunks")
        return chunks

    def _find_split_points(self, audio_data: np.ndarray, sample_rate: int) -> List[int]:
        max_chunk_samples = int(self.max_duration * sample_rate)
        min_chunk_samples = int(self.min_chunk_duration * sample_rate)
        silence_samples = int(self.silence_duration * sample_rate)

        if audio_data.ndim > 1:
            analysis_audio = np.mean(audio_data, axis=1)
        else:
            analysis_audio = audio_data

        audio_float = analysis_audio.astype(np.float32)
        max_val = np.max(np.abs(audio_float))
        if max_val > 0:
            audio_normalized = audio_float / max_val
        else:
            audio_normalized = audio_float

        audio_abs = np.abs(audio_normalized)

        window_size = int(0.1 * sample_rate)
        if window_size > 1 and len(audio_abs) > window_size:
            audio_smooth = np.convolve(
                audio_abs, np.ones(window_size) / window_size, mode="same"
            )
        else:
            audio_smooth = audio_abs

        split_points = []
        last_split = 0

        search_start = min_chunk_samples
        while search_start < len(audio_data):
            search_end = min(
                search_start + max_chunk_samples - min_chunk_samples, len(audio_data)
            )

            best_split = self._find_best_silence(
                audio_smooth, search_start, search_end, silence_samples, sample_rate
            )

            if best_split is not None:
                split_points.append(best_split)
                last_split = best_split
                search_start = best_split + min_chunk_samples
            else:
                forced_split = min(last_split + max_chunk_samples, len(audio_data) - 1)
                if forced_split < len(audio_data) - min_chunk_samples:
                    split_points.append(forced_split)
                    last_split = forced_split
                    search_start = forced_split + min_chunk_samples
                else:
                    break

        return split_points

    def _find_best_silence(
        self,
        audio_smooth: np.ndarray,
        start: int,
        end: int,
        silence_samples: int,
        sample_rate: int,
    ) -> Optional[int]:
        step = int(0.05 * sample_rate)
        if step < 1:
            step = 1

        best_silence_start = None
        best_silence_quality = float("inf")

        for i in range(end - silence_samples, start, -step):
            if i < 0 or i + silence_samples >= len(audio_smooth):
                continue

            silence_region = audio_smooth[i : i + silence_samples]
            max_level = np.max(silence_region)
            avg_level = np.mean(silence_region)

            if max_level < self.silence_threshold:
                silence_quality = avg_level + (max_level * 0.1)

                if silence_quality < best_silence_quality:
                    best_silence_quality = silence_quality
                    best_silence_start = i + silence_samples // 2

        return best_silence_start

    def _generate_time_based_splits(
        self, total_samples: int, sample_rate: int
    ) -> List[int]:
        target_samples = int(self.max_duration * sample_rate * 0.9)

        split_points = []
        current_pos = target_samples

        while current_pos < total_samples - int(self.min_chunk_duration * sample_rate):
            split_points.append(current_pos)
            current_pos += target_samples

        return split_points

    def _create_chunks(
        self, audio_data: np.ndarray, sample_rate: int, split_points: List[int]
    ) -> List[np.ndarray]:
        chunks = []
        overlap_samples = int(self.overlap_duration * sample_rate)

        start_idx = 0
        for i, end_idx in enumerate(split_points + [len(audio_data)]):
            chunk_start = max(0, start_idx - (overlap_samples if i > 0 else 0))
            chunk_end = min(len(audio_data), end_idx + overlap_samples)

            chunk_data = audio_data[chunk_start:chunk_end]
            chunks.append(chunk_data)

            chunk_duration = len(chunk_data) / sample_rate
            logger.debug(f"Created chunk {i+1}: {chunk_duration:.1f}s")

            start_idx = end_idx

        return chunks

    def combine_transcriptions(self, transcriptions: List[str]) -> str:
        if not transcriptions:
            return ""

        valid_transcriptions = [t.strip() for t in transcriptions if t and t.strip()]

        if not valid_transcriptions:
            return ""

        combined = ""
        for i, transcription in enumerate(valid_transcriptions):
            if i > 0:
                if not combined.endswith(" ") and not transcription.startswith(" "):
                    combined += " "

            combined += transcription

        while "  " in combined:
            combined = combined.replace("  ", " ")

        return combined.strip()


_audio_processor: Optional[AudioProcessor] = None


def get_audio_processor() -> AudioProcessor:
    global _audio_processor
    if _audio_processor is None:
        _audio_processor = AudioProcessor()
    return _audio_processor
