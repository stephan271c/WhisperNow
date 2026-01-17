import os
from typing import List

from .file_utils import get_models_dir, has_file_with_suffix


def get_installed_asr_models() -> List[str]:
    models_dir = get_models_dir()

    if not os.path.exists(models_dir):
        return []

    models = []
    for name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, name)
        if os.path.isdir(model_path):
            has_tokens = has_file_with_suffix(model_path, "-tokens.txt", "tokens.txt")
            has_encoder = has_file_with_suffix(
                model_path,
                "-encoder.onnx",
                "-encoder.int8.onnx",
                "-encoder.fp16.onnx",
                "encoder.onnx",
                "encoder.int8.onnx",
                "encoder.fp16.onnx",
            )
            if has_tokens and has_encoder:
                models.append(name)

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
