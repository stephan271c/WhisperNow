"""
ASR Backend Abstraction Layer.

Provides a unified interface for ASR using Sherpa-ONNX.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional

import numpy as np
import platformdirs

logger = logging.getLogger(__name__)


class BackendType(Enum):
    SHERPA_ONNX = auto()
    AUTO = auto()


@dataclass
class TranscriptionResult:
    text: str
    confidence: Optional[float] = None
    timestamps: Optional[list] = None
    tokens: Optional[list] = None
    durations: Optional[list] = None


class ASRBackend(ABC):
    @abstractmethod
    def load(
        self,
        model_path: str,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> None:
        pass

    @abstractmethod
    def transcribe(
        self, audio_data: np.ndarray, sample_rate: int = 16000
    ) -> TranscriptionResult:
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
    def is_model_cached(self, model_path: str) -> bool:
        pass


def get_models_dir() -> str:
    return os.path.join(platformdirs.user_data_dir("WhisperNow"), "models")


class SherpaOnnxBackend(ASRBackend):
    def __init__(self):
        self._recognizer = None
        self._device = "cpu"

    def load(
        self,
        model_path: str,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> None:
        import sherpa_onnx

        # Resolve model path - if it's just a name, look in our models directory
        if not os.path.isabs(model_path):
            model_path = os.path.join(get_models_dir(), model_path)

        if not os.path.isdir(model_path):
            raise RuntimeError(
                f"Model directory not found: {model_path}. "
                f"Please download the model first."
            )

        # Detect model files
        encoder = self._find_file(
            model_path, ["encoder.onnx", "encoder.int8.onnx", "encoder.fp16.onnx"]
        )
        decoder = self._find_file(
            model_path, ["decoder.onnx", "decoder.int8.onnx", "decoder.fp16.onnx"]
        )
        joiner = self._find_file(
            model_path, ["joiner.onnx", "joiner.int8.onnx", "joiner.fp16.onnx"]
        )
        tokens = self._find_file(model_path, ["tokens.txt"])

        if not all([encoder, decoder, joiner, tokens]):
            missing = []
            if not encoder:
                missing.append("encoder")
            if not decoder:
                missing.append("decoder")
            if not joiner:
                missing.append("joiner")
            if not tokens:
                missing.append("tokens")
            raise RuntimeError(
                f"Missing model files in {model_path}: {', '.join(missing)}"
            )

        self._device = "cpu"
        logger.info("Loading model with CPU provider")

        try:
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                tokens=tokens,
                num_threads=4,
                provider="cpu",
                debug=False,
                decoding_method="greedy_search",
                model_type="nemo_transducer",
            )
        except Exception as e:
            self._recognizer = None
            raise RuntimeError(f"Failed to load model from '{model_path}': {e}") from e

    def _find_file(self, directory: str, candidates: list[str]) -> Optional[str]:
        for name in candidates:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                return path
        return None

    def transcribe(
        self, audio_data: np.ndarray, sample_rate: int = 16000
    ) -> TranscriptionResult:
        if self._recognizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Convert to float32 if needed
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32768.0
        else:
            audio_float = audio_data.astype(np.float32)

        # Flatten to mono if stereo
        if audio_float.ndim > 1:
            audio_float = (
                audio_float[:, 0] if audio_float.shape[1] > 1 else audio_float.flatten()
            )

        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio_float)
        self._recognizer.decode_stream(stream)

        result = stream.result

        timestamps = None
        tokens = None
        durations = None

        if hasattr(result, "timestamps") and hasattr(result, "tokens"):
            timestamps = list(result.timestamps) if result.timestamps else None
            tokens = list(result.tokens) if result.tokens else None
        if hasattr(result, "durations"):
            durations = list(result.durations) if result.durations else None

        return TranscriptionResult(
            text=result.text,
            timestamps=timestamps,
            tokens=tokens,
            durations=durations,
        )

    def unload(self) -> None:
        if self._recognizer is not None:
            del self._recognizer
            self._recognizer = None

    @property
    def is_loaded(self) -> bool:
        return self._recognizer is not None

    @property
    def device(self) -> str:
        return self._device

    def is_model_cached(self, model_path: str) -> bool:
        if not os.path.isabs(model_path):
            model_path = os.path.join(get_models_dir(), model_path)

        if not os.path.isdir(model_path):
            return False

        # Check for required files
        required = ["tokens.txt"]
        encoder_candidates = ["encoder.onnx", "encoder.int8.onnx", "encoder.fp16.onnx"]
        decoder_candidates = ["decoder.onnx", "decoder.int8.onnx", "decoder.fp16.onnx"]
        joiner_candidates = ["joiner.onnx", "joiner.int8.onnx", "joiner.fp16.onnx"]

        for f in required:
            if not os.path.exists(os.path.join(model_path, f)):
                return False

        has_encoder = any(
            os.path.exists(os.path.join(model_path, c)) for c in encoder_candidates
        )
        has_decoder = any(
            os.path.exists(os.path.join(model_path, c)) for c in decoder_candidates
        )
        has_joiner = any(
            os.path.exists(os.path.join(model_path, c)) for c in joiner_candidates
        )

        return has_encoder and has_decoder and has_joiner


def detect_backend_type(model_name: str) -> BackendType:
    return BackendType.SHERPA_ONNX


def create_backend(
    backend_type: BackendType = BackendType.AUTO, model_name: Optional[str] = None
) -> ASRBackend:
    return SherpaOnnxBackend()
