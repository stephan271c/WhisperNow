"""
Transcription engine using pluggable ASR backends.

Handles model loading, downloading, and transcription with support for
multiple ASR backends (NeMo, HuggingFace Transformers).
Emits signals for progress updates (loading, ready, processing).
"""

import gc
import time
from enum import Enum, auto
from typing import Optional, Callable

import numpy as np
import torch

from .backends import (
    ASRBackend,
    BackendType,
    TranscriptionResult,
    create_backend,
    detect_backend_type,
)
from ..audio.audio_processor import AudioProcessor, needs_chunking
from ...utils.logger import get_logger


class EngineState(Enum):
    """State of the transcription engine."""
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
    
    Supports multiple ASR backends:
    - NeMo: nvidia/parakeet-*, nvidia/canary-*, etc.
    - HuggingFace: openai/whisper-*, facebook/wav2vec2-*, etc.
    
    Example:
        # Auto-detect backend based on model name
        engine = TranscriptionEngine(model_name="nvidia/parakeet-tdt-0.6b-v3")
        
        # Explicitly specify backend
        engine = TranscriptionEngine(
            model_name="openai/whisper-large-v3",
            backend_type=BackendType.HUGGINGFACE
        )
    """
    
    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
        backend_type: BackendType = BackendType.AUTO,
        use_gpu: bool = True,
        on_state_change: Optional[Callable[[EngineState, str], None]] = None,
        on_download_progress: Optional[Callable[[float], None]] = None
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
        self._device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self._audio_processor = AudioProcessor()
        self.logger = get_logger(__name__)
    
    @property
    def state(self) -> EngineState:
        """Current state of the engine."""
        return self._state
    
    @property
    def is_ready(self) -> bool:
        """Check if the engine is ready to transcribe."""
        return self._state == EngineState.READY
    
    @property
    def device(self) -> str:
        """The device the model is running on ('cuda' or 'cpu')."""
        if self._backend and self._backend.is_loaded:
            return self._backend.device
        return self._device
    
    @property
    def backend_type(self) -> BackendType:
        """The type of ASR backend being used."""
        return self._backend_type
    
    @property
    def backend_name(self) -> str:
        """Human-readable name of the current backend."""
        return self._backend_type.name
    
    def _set_state(self, state: EngineState, message: str = "") -> None:
        """Update state and notify callback."""
        self._state = state
        if self.on_state_change:
            self.on_state_change(state, message)
    
    def load_model(self) -> bool:
        """
        Load the ASR model. Downloads if not cached.
        
        Returns:
            True if model loaded successfully, False otherwise.
        """
        if self._backend is not None and self._backend.is_loaded:
            return True
        
        try:
            self._set_state(
                EngineState.LOADING,
                f"Loading {self.backend_name} model: {self.model_name}..."
            )
            
            # Create the appropriate backend
            self._backend = create_backend(self._backend_type)
            
            # Load the model
            self._backend.load(
                model_name=self.model_name,
                use_gpu=self.use_gpu,
                on_progress=self.on_download_progress
            )
            
            self._set_state(
                EngineState.READY,
                f"Model loaded on {self._backend.device.upper()} ({self.backend_name})"
            )
            return True
            
        except Exception as e:
            self._set_state(EngineState.ERROR, f"Failed to load model: {e}")
            return False
    
    def transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000
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
            if not self.load_model():
                return None
        
        self._set_state(EngineState.PROCESSING, "Transcribing...")
        
        start_time = time.time()
        
        try:
            result: TranscriptionResult = self._backend.transcribe(
                audio_data=audio_data,
                sample_rate=sample_rate
            )
            
            processing_time = time.time() - start_time
            audio_duration = len(audio_data) / sample_rate
            
            # Log performance metrics
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
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000
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
        
        # Split audio into chunks
        from ...utils.logger import get_logger
        logger = get_logger(__name__)
        
        chunks = self._audio_processor.split_audio(audio_data, sample_rate)
        logger.info(f"Processing {len(chunks)} audio chunks")
        
        # Transcribe each chunk
        transcriptions = []
        for i, chunk in enumerate(chunks):
            logger.debug(f"Transcribing chunk {i+1}/{len(chunks)}")
            text = self.transcribe(chunk, sample_rate)
            if text:
                transcriptions.append(text)
        
        # Combine transcriptions
        combined = self._audio_processor.combine_transcriptions(transcriptions)
        logger.info(f"Combined {len(transcriptions)} transcriptions")
        
        return combined
    
    def transcribe_with_metadata(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000
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
            if not self.load_model():
                return None
        
        self._set_state(EngineState.PROCESSING, "Transcribing...")
        
        start_time = time.time()
        
        try:
            result = self._backend.transcribe(
                audio_data=audio_data,
                sample_rate=sample_rate
            )
            
            processing_time = time.time() - start_time
            audio_duration = len(audio_data) / sample_rate
            
            # Log performance metrics
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
        """Unload the model and free GPU memory."""
        if self._backend is not None:
            self._backend.unload()
            self._backend = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        self._set_state(EngineState.NOT_LOADED, "Model unloaded")
    
    def switch_model(
        self,
        model_name: str,
        backend_type: BackendType = BackendType.AUTO
    ) -> bool:
        """
        Switch to a different model (unloads current model first).
        
        Args:
            model_name: New model to load
            backend_type: Backend type for the new model
            
        Returns:
            True if switch was successful, False otherwise.
        """
        # Unload current model
        self.unload()
        
        # Update model configuration
        self.model_name = model_name
        if backend_type == BackendType.AUTO:
            self._backend_type = detect_backend_type(model_name)
        else:
            self._backend_type = backend_type
        
        # Load new model
        return self.load_model()
    
    def is_model_cached(self) -> bool:
        """
        Check if the model is already downloaded/cached.
        
        Returns:
            True if model files exist locally.
        """
        # Create a temporary backend instance to check cache
        # This is lightweight as it doesn't load the model
        backend = create_backend(self._backend_type, self.model_name)
        return backend.is_model_cached(self.model_name)
