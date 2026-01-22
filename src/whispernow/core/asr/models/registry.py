import json
import os
from dataclasses import dataclass
from typing import List, Literal

from ..file_utils import get_models_dir, is_valid_model_dir

GITHUB_RELEASE_BASE = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
)


@dataclass
class ModelInfo:
    id: str
    name: str
    type: str

    @property
    def url(self) -> str:
        return f"{GITHUB_RELEASE_BASE}/{self.id}.tar.bz2"


def load_models() -> List[ModelInfo]:
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "models.json")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [ModelInfo(**item) for item in data]
    except Exception as e:
        print(f"Error loading models.json: {e}")
        return []


AVAILABLE_MODELS: List[ModelInfo] = load_models()


DownloadStatus = Literal["downloaded", "not_downloaded"]


def get_model_by_id(model_id: str) -> ModelInfo | None:
    for model in AVAILABLE_MODELS:
        if model.id == model_id:
            return model
    return None


def is_model_downloaded(model_id: str) -> bool:
    models_dir = get_models_dir()
    model_path = os.path.join(models_dir, model_id)

    if not os.path.isdir(model_path):
        return False

    return is_valid_model_dir(model_path)


def get_model_download_status(model_id: str) -> DownloadStatus:
    return "downloaded" if is_model_downloaded(model_id) else "not_downloaded"


def get_all_models_with_status() -> List[tuple[ModelInfo, DownloadStatus]]:
    return [(model, get_model_download_status(model.id)) for model in AVAILABLE_MODELS]


def get_installed_asr_models() -> List[str]:
    models_dir = get_models_dir()

    if not os.path.exists(models_dir):
        return []

    models = [
        name
        for name in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, name))
        and is_valid_model_dir(os.path.join(models_dir, name))
    ]
    models.sort()
    return models


def delete_asr_model(model_name: str) -> tuple[bool, str]:
    import shutil

    models_dir = get_models_dir()
    model_path = os.path.join(models_dir, model_name)

    if not os.path.exists(model_path):
        return False, f"Model '{model_name}' not found"

    if not os.path.isdir(model_path):
        return False, f"'{model_name}' is not a directory"

    try:
        shutil.rmtree(model_path)
        return True, f"Deleted '{model_name}'"
    except Exception as e:
        return False, f"Failed to delete model: {str(e)}"
