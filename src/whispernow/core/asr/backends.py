"""
ASR Backend Abstraction Layer.

Provides a unified interface for different ASR model backends:
- NeMo: NVIDIA's NeMo toolkit (Parakeet, Canary, etc.)
- HuggingFace: Transformers pipeline (Whisper, Wav2Vec2, etc.)

This allows the TranscriptionEngine to work with any supported ASR model
without changing the core transcription logic.
"""

import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional

import numpy as np


class BackendType(Enum):
    NEMO = auto()
    HUGGINGFACE = auto()
    AUTO = auto()


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

    @abstractmethod
    def load(
        self,
        model_name: str,
        use_gpu: bool = True,
        on_progress: Optional[Callable[[float], None]] = None,
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
        self, audio_data: np.ndarray, sample_rate: int = 16000
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
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        pass

    @abstractmethod
    def is_model_cached(self, model_name: str) -> bool:
        pass


class NeMoBackend(ASRBackend):

    def __init__(self):
        self._model = None
        self._device = "cpu"

    def load(
        self,
        model_name: str,
        use_gpu: bool = True,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> None:
        import nemo.collections.asr as nemo_asr
        import torch

        self._device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        try:
            self._model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=model_name, map_location=self._device
            )

            if self._device == "cuda":
                self._model = self._model.cuda()
            else:
                self._model = self._model.cpu()

            self._model.eval()

        except Exception as e:
            self._model = None
            raise RuntimeError(f"Failed to load NeMo model '{model_name}': {e}") from e

    def transcribe(
        self, audio_data: np.ndarray, sample_rate: int = 16000
    ) -> TranscriptionResult:
        import scipy.io.wavfile as wav

        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            wav.write(tmp_file.name, sample_rate, audio_data)
            tmp_path = tmp_file.name

        try:
            transcriptions = self._model.transcribe([tmp_path])
            result = transcriptions[0]

            if hasattr(result, "text"):
                text = result.text
            else:
                text = str(result)

            confidence = None
            if hasattr(result, "score"):
                confidence = float(result.score)

            return TranscriptionResult(text=text, confidence=confidence)

        except Exception as e:
            raise RuntimeError(f"NeMo transcription failed: {e}") from e
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def unload(self) -> None:
        import gc

        import torch

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
            cache_info = scan_cache_dir()
            for repo in cache_info.repos:
                if repo.repo_id == model_name:
                    return True
            return False
        except Exception:
            return False


class HuggingFaceBackend(ASRBackend):

    def __init__(self):
        self._pipeline = None
        self._device = "cpu"

    def load(
        self,
        model_name: str,
        use_gpu: bool = True,
        on_progress: Optional[Callable[[float], None]] = None,
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
            raise RuntimeError(
                f"Failed to load HuggingFace model '{model_name}': {e}"
            ) from e

    def transcribe(
        self, audio_data: np.ndarray, sample_rate: int = 16000
    ) -> TranscriptionResult:
        if self._pipeline is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.float32:
                audio_float = audio_data
            else:
                audio_float = audio_data.astype(np.float32)

            if audio_float.ndim > 1:
                audio_float = audio_float.flatten()

            result = self._pipeline({"raw": audio_float, "sampling_rate": sample_rate})

            text = result.get("text", "")

            timestamps = None
            if "chunks" in result:
                timestamps = result["chunks"]

            return TranscriptionResult(text=text, timestamps=timestamps)

        except Exception as e:
            raise RuntimeError(f"HuggingFace transcription failed: {e}") from e

    def unload(self) -> None:
        import gc

        import torch

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

        return try_to_load_from_cache(model_name, "config.json") is not None


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

    for prefix in _NEMO_PREFIXES:
        if model_lower.startswith(prefix) or prefix in model_lower:
            return BackendType.NEMO

    for prefix in _HUGGINGFACE_ASR_PREFIXES:
        if model_lower.startswith(prefix):
            return BackendType.HUGGINGFACE

    return BackendType.HUGGINGFACE


def create_backend(
    backend_type: BackendType = BackendType.AUTO, model_name: Optional[str] = None
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
