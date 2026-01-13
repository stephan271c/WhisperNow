from datetime import datetime
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

from ...utils.logger import get_logger
from ..transcript_processor import (
    Enhancement,
    LLMProcessor,
    apply_vocabulary_replacements,
)
from .transcriber import TranscriptionEngine

logger = get_logger(__name__)


class TranscriptionWorkerThread(QThread):
    """
    Background thread for transcription + LLM enhancement.

    Performs the entire transcription pipeline in a background thread:
    1. Audio transcription via ASR engine
    2. Vocabulary replacements
    3. LLM enhancement (if configured)

    Signals:
        finished: Emitted when processing completes successfully
                  (final_text, raw_text, enhanced_text, enhancement_name, cost)
        error: Emitted when processing fails (error_message)
    """

    # (final_text, raw_text, enhanced_text, enhancement_name, cost)
    finished = Signal(str, str, object, object, object)
    error = Signal(str)

    def __init__(
        self,
        transcriber: TranscriptionEngine,
        audio_data: np.ndarray,
        sample_rate: int,
        vocabulary_replacements: dict,
        llm_processor: Optional[LLMProcessor],
        enhancement: Optional[Enhancement],
        parent=None,
    ):

        super().__init__(parent)
        self._transcriber = transcriber
        self._audio_data = audio_data
        self._sample_rate = sample_rate
        self._vocabulary_replacements = vocabulary_replacements
        self._llm_processor = llm_processor
        self._enhancement = enhancement

    def run(self):
        import time

        start_time = time.time()

        try:

            logger.info(
                f"Background transcription started: {len(self._audio_data)} samples"
            )
            raw_text = self._transcriber.transcribe_chunked(
                self._audio_data, self._sample_rate
            )

            if not raw_text:
                logger.warning("Transcription returned empty result")
                self.error.emit("Transcription returned empty result")
                return

            duration = time.time() - start_time
            logger.info(
                f"Transcription completed in {duration:.2f}s: '{raw_text[:50]}{'...' if len(raw_text) > 50 else ''}'"
            )

            processed_text = apply_vocabulary_replacements(
                raw_text, self._vocabulary_replacements
            )

            final_text = processed_text
            enhanced_text = None
            enhancement_name = None
            cost = None

            if self._llm_processor and self._enhancement:
                if self._llm_processor.is_configured():
                    logger.info(f"Applying LLM enhancement: {self._enhancement.title}")
                    response = self._llm_processor.process(
                        processed_text, self._enhancement
                    )
                    final_text = response.content
                    enhanced_text = response.content
                    enhancement_name = self._enhancement.title
                    cost = response.cost_usd
                else:
                    logger.warning("LLM processor not configured, skipping enhancement")

            if enhanced_text is None and processed_text != raw_text:
                enhanced_text = processed_text
                enhancement_name = "Vocabulary Replacement"

            total_duration = time.time() - start_time
            logger.info(f"Total processing completed in {total_duration:.2f}s")

            self.finished.emit(
                final_text, raw_text, enhanced_text, enhancement_name, cost
            )

        except Exception as e:
            logger.exception(f"Background transcription error: {e}")
            self.error.emit(str(e))
