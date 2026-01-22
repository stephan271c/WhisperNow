import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .file_utils import (
    find_file_by_suffix,
    find_file_exact,
    get_models_dir,
    is_valid_transducer_model,
    is_valid_whisper_model,
)

logger = logging.getLogger(__name__)


def _load_models_json() -> list[dict]:
    models_path = Path(__file__).parent / "models" / "models.json"
    if models_path.exists():
        with open(models_path, "r") as f:
            return json.load(f)
    return []


def get_model_type(model_id: str) -> str:
    models = _load_models_json()
    for model in models:
        if model.get("id") == model_id:
            model_type = model.get("type")
            if not model_type:
                raise ValueError(
                    f"Model '{model_id}' in models.json is missing a 'type' field. "
                    f"Please add 'type': 'whisper' or 'type': 'transducer' to the model entry."
                )
            return model_type
    raise ValueError(
        f"Model '{model_id}' not found in models.json. "
        f"Please add an entry for this model with 'id', 'name', and 'type' fields."
    )


@dataclass
class TranscriptionResult:
    text: str
    confidence: Optional[float] = None
    timestamps: Optional[list] = None
    tokens: Optional[list] = None
    durations: Optional[list] = None


class SherpaOnnxBackend:
    def __init__(self):
        self._recognizer = None
        self._device = "cpu"

    def load(
        self,
        model_path: str,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> None:
        import sherpa_onnx

        if not os.path.isabs(model_path):
            full_model_path = os.path.join(get_models_dir(), model_path)
            model_id = model_path  # The original is the model ID
        else:
            full_model_path = model_path
            model_id = os.path.basename(model_path)

        if not os.path.isdir(full_model_path):
            raise RuntimeError(
                f"Model directory not found: {full_model_path}. "
                f"Please download the model first."
            )

        model_type = get_model_type(model_id)
        logger.info(f"Loading model '{model_id}' as type '{model_type}'")

        self._device = "cpu"
        logger.info("Loading model with CPU provider")

        try:
            if model_type == "whisper":
                self._load_whisper_model(sherpa_onnx, full_model_path)
            else:
                self._load_transducer_model(sherpa_onnx, full_model_path)
        except Exception as e:
            self._recognizer = None
            raise RuntimeError(
                f"Failed to load model from '{full_model_path}': {e}"
            ) from e

    def _load_whisper_model(self, sherpa_onnx, model_path: str) -> None:
        encoder = find_file_by_suffix(model_path, "-encoder.onnx", "-encoder.int8.onnx")
        decoder = find_file_by_suffix(model_path, "-decoder.onnx", "-decoder.int8.onnx")
        tokens = find_file_by_suffix(model_path, "-tokens", "tokens.txt")

        if not encoder or not decoder or not tokens:
            missing = []
            if not encoder:
                missing.append("encoder (*-encoder.onnx)")
            if not decoder:
                missing.append("decoder (*-decoder.onnx)")
            if not tokens:
                missing.append("tokens (*-tokens)")
            raise RuntimeError(
                f"Missing Whisper model files in {model_path}: {', '.join(missing)}"
            )

        logger.info(
            f"Loading Whisper model: encoder={encoder}, decoder={decoder}, tokens={tokens}"
        )

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=encoder,
            decoder=decoder,
            tokens=tokens,
            num_threads=4,
            provider="cpu",
            debug=False,
            decoding_method="greedy_search",
        )

    def _load_transducer_model(self, sherpa_onnx, model_path: str) -> None:
        encoder = find_file_exact(
            model_path, ["encoder.onnx", "encoder.int8.onnx", "encoder.fp16.onnx"]
        )
        decoder = find_file_exact(
            model_path, ["decoder.onnx", "decoder.int8.onnx", "decoder.fp16.onnx"]
        )
        joiner = find_file_exact(
            model_path, ["joiner.onnx", "joiner.int8.onnx", "joiner.fp16.onnx"]
        )
        tokens = find_file_exact(model_path, ["tokens.txt"])

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
                f"Missing Transducer model files in {model_path}: {', '.join(missing)}"
            )

        logger.info(
            f"Loading Transducer model: encoder={encoder}, decoder={decoder}, joiner={joiner}"
        )

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

    def transcribe(
        self, audio_data: np.ndarray, sample_rate: int = 16000
    ) -> TranscriptionResult:
        if self._recognizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32768.0
        else:
            audio_float = audio_data.astype(np.float32)

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
            full_model_path = os.path.join(get_models_dir(), model_path)
            model_id = model_path
        else:
            full_model_path = model_path
            model_id = os.path.basename(model_path)

        if not os.path.isdir(full_model_path):
            return False

        model_type = get_model_type(model_id)

        if model_type == "whisper":
            return is_valid_whisper_model(full_model_path)
        else:
            return is_valid_transducer_model(full_model_path)
