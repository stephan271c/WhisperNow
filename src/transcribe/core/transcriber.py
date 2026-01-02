"""
Transcription engine using NeMo ASR models.

Handles model loading, downloading, and transcription.
Emits signals for progress updates (loading, ready, processing).
"""

import tempfile
import os
import gc
from pathlib import Path
from typing import Optional, Callable
from enum import Enum, auto

import numpy as np
import scipy.io.wavfile as wav
import torch


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
    Manages ASR model loading and transcription.
    
    The model is downloaded on first use and cached locally.
    Emits callbacks for state changes and progress updates.
    """
    
    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
        use_gpu: bool = True,
        on_state_change: Optional[Callable[[EngineState, str], None]] = None,
        on_download_progress: Optional[Callable[[float], None]] = None
    ):
        """
        Initialize the transcription engine.
        
        Args:
            model_name: HuggingFace model name
            use_gpu: Whether to use GPU if available
            on_state_change: Callback for state changes (state, message)
            on_download_progress: Callback for download progress (0.0-1.0)
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.on_state_change = on_state_change
        self.on_download_progress = on_download_progress
        
        self._model = None
        self._state = EngineState.NOT_LOADED
        self._device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    
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
        return self._device
    
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
        if self._model is not None:
            return True
        
        try:
            self._set_state(EngineState.LOADING, "Loading ASR model...")
            
            # Import here to avoid slow startup
            import nemo.collections.asr as nemo_asr
            
            # TODO: Add download progress tracking
            # NeMo handles caching automatically
            self._model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.model_name
            )
            
            if self._device == "cuda":
                self._model = self._model.cuda()
            
            self._set_state(EngineState.READY, f"Model loaded on {self._device.upper()}")
            return True
            
        except Exception as e:
            self._set_state(EngineState.ERROR, f"Failed to load model: {e}")
            return False
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Optional[str]:
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
        
        try:
            # Save audio to temp file (NeMo requires file path)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                wav.write(tmp_file.name, sample_rate, audio_data)
                tmp_path = tmp_file.name
            
            try:
                transcriptions = self._model.transcribe([tmp_path])
                result = transcriptions[0]
                
                # Extract text from Hypothesis object
                if hasattr(result, 'text'):
                    text = result.text
                else:
                    text = str(result)
                
                self._set_state(EngineState.READY, "Ready")
                return text
                
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
        except Exception as e:
            self._set_state(EngineState.ERROR, f"Transcription failed: {e}")
            return None
    
    def unload(self) -> None:
        """Unload the model and free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        self._set_state(EngineState.NOT_LOADED, "Model unloaded")
    
    def is_model_cached(self) -> bool:
        """
        Check if the model is already downloaded/cached.
        
        Returns:
            True if model files exist locally.
        """
        # NeMo caches models in ~/.cache/huggingface or similar
        # This is a simplified check - actual implementation would check HF cache
        # TODO: Implement proper cache check
        return False
