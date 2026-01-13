"""
Model registry for available sherpa-onnx ASR models.

Provides a curated list of models available for download from GitHub releases.
"""

import os
from dataclasses import dataclass
from typing import List, Literal

from .backends import get_models_dir

GITHUB_RELEASE_BASE = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
)


@dataclass
class ModelInfo:
    id: str
    name: str
    size_mb: int
    description: str

    @property
    def url(self) -> str:
        return f"{GITHUB_RELEASE_BASE}/{self.id}.tar.bz2"


# Curated list of models - add more as needed
AVAILABLE_MODELS: List[ModelInfo] = [
    ModelInfo(
        id="sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16",
        name="Parakeet TDT 0.6B (FP16)",
        size_mb=1200,
        description="High-quality English ASR model",
    ),
]


DownloadStatus = Literal["downloaded", "not_downloaded"]


def get_model_by_id(model_id: str) -> ModelInfo | None:
    for model in AVAILABLE_MODELS:
        if model.id == model_id:
            return model
    return None


def is_model_downloaded(model_id: str) -> bool:
    """Check if a model is downloaded and valid."""
    models_dir = get_models_dir()
    model_path = os.path.join(models_dir, model_id)

    if not os.path.isdir(model_path):
        return False

    # Check for required files
    has_tokens = os.path.exists(os.path.join(model_path, "tokens.txt"))
    has_encoder = (
        os.path.exists(os.path.join(model_path, "encoder.onnx"))
        or os.path.exists(os.path.join(model_path, "encoder.int8.onnx"))
        or os.path.exists(os.path.join(model_path, "encoder.fp16.onnx"))
    )

    return has_tokens and has_encoder


def get_model_download_status(model_id: str) -> DownloadStatus:
    return "downloaded" if is_model_downloaded(model_id) else "not_downloaded"


def get_all_models_with_status() -> List[tuple[ModelInfo, DownloadStatus]]:
    """Get all available models with their download status."""
    return [(model, get_model_download_status(model.id)) for model in AVAILABLE_MODELS]
