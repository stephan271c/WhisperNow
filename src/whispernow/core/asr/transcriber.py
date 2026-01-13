"""
Transcription engine using pluggable ASR backends.

Handles model loading, downloading, and transcription with support for
multiple ASR backends (NeMo, HuggingFace Transformers).
Emits signals for progress updates (loading, ready, processing).
"""

import gc
import time
from enum import Enum, auto
from typing import Callable, Optional

import numpy as np

from ...utils.logger import get_logger
from ..audio.audio_processor import AudioProcessor, needs_chunking
from .backends import (
    ASRBackend,
    BackendType,
    TranscriptionResult,
    create_backend,
    detect_backend_type,
)


class EngineState(Enum):
    NOT_LOADED = auto()
    DOWNLOADING = auto()
    LOADING = auto()
    READY = auto()
    PROCESSING = auto()
    ERROR = auto()


class TranscriptionEngine:
    """
    Manages ASR model loading and transcription with pluggable backends.

    The model is downloaded on first use and cached locally.
    Emits callbacks for state changes and progress updates.

    Uses Sherpa-ONNX as the ASR backend with pre-trained models.

    Example:
        engine = TranscriptionEngine(
            model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
        )
    """

    def __init__(
        self,
        model_name: str = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
        backend_type: BackendType = BackendType.AUTO,
        use_gpu: bool = True,
        on_state_change: Optional[Callable[[EngineState, str], None]] = None,
        on_download_progress: Optional[Callable[[float], None]] = None,
    ):
        """
        Initialize the transcription engine.

        Args:
            model_name: HuggingFace model name or local path
            backend_type: ASR backend to use (AUTO, NEMO, or HUGGINGFACE)
            use_gpu: Whether to use GPU if available
            on_state_change: Callback for state changes (state, message)
            on_download_progress: Callback for download progress (0.0-1.0)
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.on_state_change = on_state_change
        self.on_download_progress = on_download_progress

        # Determine backend type
        if backend_type == BackendType.AUTO:
            self._backend_type = detect_backend_type(model_name)
        else:
            self._backend_type = backend_type

        self._backend: Optional[ASRBackend] = None
        self._state = EngineState.NOT_LOADED
        self._device = "cpu"  # Will be updated by backend load
        self._audio_processor = AudioProcessor()
        self.logger = get_logger(__name__)

    @property
    def state(self) -> EngineState:
        return self._state

    @property
    def is_ready(self) -> bool:
        return self._state == EngineState.READY

    @property
    def device(self) -> str:
        if self._backend and self._backend.is_loaded:
            return self._backend.device
        return self._device

    @property
    def backend_type(self) -> BackendType:
        return self._backend_type

    @property
    def backend_name(self) -> str:
        return self._backend_type.name

    def _set_state(self, state: EngineState, message: str = "") -> None:
        self._state = state
        if self.on_state_change:
            self.on_state_change(state, message)

    def load_model(self) -> bool:
        """
        Load the ASR model. Downloads if not cached.

        Returns:
            True if model loaded successfully, False otherwise.

        Raises:
            Exception: If model loading fails.
        """
        if self._backend is not None and self._backend.is_loaded:
            return True

        self._set_state(
            EngineState.LOADING,
            f"Loading {self.backend_name} model: {self.model_name}...",
        )
        self._backend = create_backend(self._backend_type)
        self._backend.load(
            model_path=self.model_name,
            use_gpu=self.use_gpu,
            on_progress=self.on_download_progress,
        )

        self._set_state(
            EngineState.READY,
            f"Model loaded on {self._backend.device.upper()} ({self.backend_name})",
        )
        return True

    def transcribe(
        self, audio_data: np.ndarray, sample_rate: int = 16000
    ) -> Optional[str]:
        """
        Transcribe audio data to text.

        Args:
            audio_data: Numpy array of audio samples
            sample_rate: Sample rate of the audio

        Returns:
            Transcribed text, or None if transcription failed.
        """
        if not self.is_ready:
            try:
                self.load_model()
            except Exception as e:
                self._set_state(EngineState.ERROR, f"Failed to load model: {e}")
                return None

        self._set_state(EngineState.PROCESSING, "Transcribing...")

        start_time = time.time()

        try:
            result: TranscriptionResult = self._backend.transcribe(
                audio_data=audio_data, sample_rate=sample_rate
            )

            processing_time = time.time() - start_time
            audio_duration = len(audio_data) / sample_rate

            if processing_time > 0:
                rtf = audio_duration / processing_time
                self.logger.debug(
                    f"Transcription finished: audio_len={audio_duration:.2f}s, "
                    f"time={processing_time:.2f}s, speed={rtf:.2f}x"
                )

                if result.text:
                    cps = len(result.text) / processing_time
                    self.logger.debug(f"Transcription speed: {cps:.2f} chars/sec")

            self._set_state(EngineState.READY, "Ready")
            return result.text

        except Exception as e:
            self._set_state(EngineState.ERROR, f"Transcription failed: {e}")
            return None

    def transcribe_chunked(
        self, audio_data: np.ndarray, sample_rate: int = 16000
    ) -> Optional[str]:
        """
        Transcribe audio data, automatically chunking if too long.

        For audio over 30 seconds, splits into smaller chunks using
        silence detection for optimal split points, transcribes each
        chunk, and combines the results.

        Args:
            audio_data: Numpy array of audio samples
            sample_rate: Sample rate of the audio

        Returns:
            Transcribed text, or None if transcription failed.
        """
        # Check if chunking is needed
        if not needs_chunking(audio_data, sample_rate):
            return self.transcribe(audio_data, sample_rate)

        from ...utils.logger import get_logger

        logger = get_logger(__name__)

        chunks = self._audio_processor.split_audio(audio_data, sample_rate)
        logger.info(f"Processing {len(chunks)} audio chunks")
        transcriptions = []
        for i, chunk in enumerate(chunks):
            logger.debug(f"Transcribing chunk {i+1}/{len(chunks)}")
            text = self.transcribe(chunk, sample_rate)
            if text:
                transcriptions.append(text)

        combined = self._audio_processor.combine_transcriptions(transcriptions)
        logger.info(f"Combined {len(transcriptions)} transcriptions")

        return combined

    def transcribe_with_metadata(
        self, audio_data: np.ndarray, sample_rate: int = 16000
    ) -> Optional[TranscriptionResult]:
        """
        Transcribe audio data and return full result with metadata.

        Args:
            audio_data: Numpy array of audio samples
            sample_rate: Sample rate of the audio

        Returns:
            TranscriptionResult with text, confidence, and timestamps,
            or None if transcription failed.
        """
        if not self.is_ready:
            try:
                self.load_model()
            except Exception as e:
                self._set_state(EngineState.ERROR, f"Failed to load model: {e}")
                return None

        self._set_state(EngineState.PROCESSING, "Transcribing...")

        start_time = time.time()

        try:
            result = self._backend.transcribe(
                audio_data=audio_data, sample_rate=sample_rate
            )

            processing_time = time.time() - start_time
            audio_duration = len(audio_data) / sample_rate

            if processing_time > 0:
                rtf = audio_duration / processing_time
                self.logger.debug(
                    f"Transcription finished: audio_len={audio_duration:.2f}s, "
                    f"time={processing_time:.2f}s, speed={rtf:.2f}x"
                )

            self._set_state(EngineState.READY, "Ready")
            return result

        except Exception as e:
            self._set_state(EngineState.ERROR, f"Transcription failed: {e}")
            return None

    def unload(self) -> None:
        if self._backend is not None:
            self._backend.unload()
            self._backend = None

        gc.collect()
        self._set_state(EngineState.NOT_LOADED, "Model unloaded")

    def switch_model(
        self, model_name: str, backend_type: BackendType = BackendType.AUTO
    ) -> bool:
        """
        Switch to a different model (unloads current model first).

        Args:
            model_name: New model to load
            backend_type: Backend type for the new model

        Returns:
            True if switch was successful, False otherwise.
        """
        self.unload()

        self.model_name = model_name
        if backend_type == BackendType.AUTO:
            self._backend_type = detect_backend_type(model_name)
        else:
            self._backend_type = backend_type

        return self.load_model()

    def is_model_cached(self) -> bool:
        """
        Check if the model is already downloaded/cached.

        Returns:
            True if model files exist locally.
        """
        backend = create_backend(self._backend_type, self.model_name)
        return backend.is_model_cached(self.model_name)
