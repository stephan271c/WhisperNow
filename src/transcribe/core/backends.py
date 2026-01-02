"""
ASR Backend Abstraction Layer.

Provides a unified interface for different ASR model backends:
- NeMo: NVIDIA's NeMo toolkit (Parakeet, Canary, etc.)
- HuggingFace: Transformers pipeline (Whisper, Wav2Vec2, etc.)

This allows the TranscriptionEngine to work with any supported ASR model
without changing the core transcription logic.
"""

import tempfile
import os
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Optional, Callable
from dataclasses import dataclass

import numpy as np
import scipy.io.wavfile as wav


class BackendType(Enum):
    """Supported ASR backend types."""
    NEMO = auto()
    HUGGINGFACE = auto()
    AUTO = auto()  # Auto-detect based on model name


@dataclass
class TranscriptionResult:
    """
    Result from ASR transcription.
    
    Attributes:
        text: The transcribed text
        confidence: Optional confidence score (0.0-1.0)
        timestamps: Optional word-level timestamps
    """
    text: str
    confidence: Optional[float] = None
    timestamps: Optional[list] = None


class ASRBackend(ABC):
    """
    Abstract base class for ASR backends.
    
    All backend implementations must provide load() and transcribe() methods
    with consistent signatures to allow swapping backends without code changes.
    """
    
    @abstractmethod
    def load(
        self,
        model_name: str,
        use_gpu: bool = True,
        on_progress: Optional[Callable[[float], None]] = None
    ) -> None:
        """
        Load the ASR model.
        
        Args:
            model_name: Model identifier (HuggingFace repo or local path)
            use_gpu: Whether to use GPU acceleration if available
            on_progress: Optional callback for download/load progress (0.0-1.0)
            
        Raises:
            RuntimeError: If model loading fails
        """
        pass
    
    @abstractmethod
    def transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000
    ) -> TranscriptionResult:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: NumPy array of audio samples (mono, float32 or int16)
            sample_rate: Sample rate of the audio in Hz
            
        Returns:
            TranscriptionResult containing the transcribed text and metadata
            
        Raises:
            RuntimeError: If transcription fails
        """
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload the model and free resources."""
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        pass
    
    @property
    @abstractmethod
    def device(self) -> str:
        """Return the device the model is running on ('cuda' or 'cpu')."""
        pass

    @abstractmethod
    def is_model_cached(self, model_name: str) -> bool:
        """
        Check if the model is cached locally.
        
        Args:
            model_name: Model identifier
            
        Returns:
            True if model files exist locally
        """
        pass


class NeMoBackend(ASRBackend):
    """
    NeMo ASR backend for NVIDIA models.
    
    Supports models like:
    - nvidia/parakeet-tdt-0.6b-v3
    - nvidia/canary-1b
    - nvidia/parakeet-ctc-1.1b
    """
    
    def __init__(self):
        self._model = None
        self._device = "cpu"
    
    def load(
        self,
        model_name: str,
        use_gpu: bool = True,
        on_progress: Optional[Callable[[float], None]] = None
    ) -> None:
        import torch
        # Import here to avoid slow startup when using other backends
        import nemo.collections.asr as nemo_asr
        
        self._device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        try:
            # TODO: Hook into download progress if possible
            self._model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=model_name
            )
            
            if self._device == "cuda":
                self._model = self._model.cuda()
                
        except Exception as e:
            self._model = None
            raise RuntimeError(f"Failed to load NeMo model '{model_name}': {e}") from e
    
    def transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000
    ) -> TranscriptionResult:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # NeMo requires file paths, so save to temp file
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
            
            # Extract confidence if available
            confidence = None
            if hasattr(result, 'score'):
                confidence = float(result.score)
            
            return TranscriptionResult(text=text, confidence=confidence)
            
        except Exception as e:
            raise RuntimeError(f"NeMo transcription failed: {e}") from e
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def unload(self) -> None:
        import torch
        import gc
        
        if self._model is not None:
            del self._model
            self._model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
    
    @property
    def is_loaded(self) -> bool:
        return self._model is not None
    
    @property
    def device(self) -> str:
        return self._device

    def is_model_cached(self, model_name: str) -> bool:
        from huggingface_hub import scan_cache_dir
        try:
            # NeMo models often only have a .nemo file, and the filename varies.
            # Using scan_cache_dir to check if the repo exists in cache is reliable.
            cache_info = scan_cache_dir()
            for repo in cache_info.repos:
                if repo.repo_id == model_name:
                    return True
            return False
        except Exception:
            return False


class HuggingFaceBackend(ASRBackend):
    """
    HuggingFace Transformers ASR backend.
    
    Supports models like:
    - openai/whisper-large-v3
    - openai/whisper-base
    - facebook/wav2vec2-large-960h
    - facebook/hubert-large-ls960-ft
    """
    
    def __init__(self):
        self._pipeline = None
        self._device = "cpu"
    
    def load(
        self,
        model_name: str,
        use_gpu: bool = True,
        on_progress: Optional[Callable[[float], None]] = None
    ) -> None:
        import torch
        from transformers import pipeline
        
        self._device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        device_arg = 0 if self._device == "cuda" else -1
        
        try:
            self._pipeline = pipeline(
                task="automatic-speech-recognition",
                model=model_name,
                device=device_arg,
                # chunk_length_s=30,  # For long audio files
            )
        except Exception as e:
            self._pipeline = None
            raise RuntimeError(f"Failed to load HuggingFace model '{model_name}': {e}") from e
    
    def transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000
    ) -> TranscriptionResult:
        if self._pipeline is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            # HuggingFace pipeline can accept raw audio arrays directly
            # Normalize audio to float32 if needed
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.float32:
                audio_float = audio_data
            else:
                audio_float = audio_data.astype(np.float32)
            
            # Flatten if needed (mono audio)
            if audio_float.ndim > 1:
                audio_float = audio_float.flatten()
            
            result = self._pipeline(
                {"raw": audio_float, "sampling_rate": sample_rate}
            )
            
            text = result.get("text", "")
            
            # Some models return chunks with timestamps
            timestamps = None
            if "chunks" in result:
                timestamps = result["chunks"]
            
            return TranscriptionResult(text=text, timestamps=timestamps)
            
        except Exception as e:
            raise RuntimeError(f"HuggingFace transcription failed: {e}") from e
    
    def unload(self) -> None:
        import torch
        import gc
        
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
    
    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None
    
    @property
    def device(self) -> str:
        return self._device

    def is_model_cached(self, model_name: str) -> bool:
        from huggingface_hub import try_to_load_from_cache
        # Transformers models always have a config.json
        return try_to_load_from_cache(model_name, "config.json") is not None


# Known model prefixes for auto-detection
_NEMO_PREFIXES = (
    "nvidia/",
    "nemo/",
    "parakeet",
    "canary",
    "conformer",
    "citrinet",
)

_HUGGINGFACE_ASR_PREFIXES = (
    "openai/whisper",
    "facebook/wav2vec2",
    "facebook/hubert",
    "microsoft/",
    "jonatasgrosman/",
)


def detect_backend_type(model_name: str) -> BackendType:
    """
    Auto-detect the appropriate backend based on model name.
    
    Args:
        model_name: Model identifier
        
    Returns:
        BackendType.NEMO or BackendType.HUGGINGFACE
    """
    model_lower = model_name.lower()
    
    # Check for NeMo models
    for prefix in _NEMO_PREFIXES:
        if model_lower.startswith(prefix) or prefix in model_lower:
            return BackendType.NEMO
    
    # Check for known HuggingFace ASR models
    for prefix in _HUGGINGFACE_ASR_PREFIXES:
        if model_lower.startswith(prefix):
            return BackendType.HUGGINGFACE
    
    # Default to HuggingFace for unknown models (broader compatibility)
    return BackendType.HUGGINGFACE


def create_backend(
    backend_type: BackendType = BackendType.AUTO,
    model_name: Optional[str] = None
) -> ASRBackend:
    """
    Factory function to create the appropriate ASR backend.
    
    Args:
        backend_type: The type of backend to create, or AUTO to detect
        model_name: Required if backend_type is AUTO for detection
        
    Returns:
        An instance of the appropriate ASRBackend subclass
        
    Raises:
        ValueError: If AUTO is specified without a model_name
    """
    if backend_type == BackendType.AUTO:
        if model_name is None:
            raise ValueError("model_name is required when using BackendType.AUTO")
        backend_type = detect_backend_type(model_name)
    
    if backend_type == BackendType.NEMO:
        return NeMoBackend()
    elif backend_type == BackendType.HUGGINGFACE:
        return HuggingFaceBackend()
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")
