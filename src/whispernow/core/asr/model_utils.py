"""
Utility functions for ASR model management.

Provides functions to discover installed/cached ASR models.
"""

import os
from typing import List

from .backends import get_models_dir


def get_installed_asr_models() -> List[str]:
    """
    Get list of installed ASR models from the local models directory.
    """
    models_dir = get_models_dir()

    if not os.path.exists(models_dir):
        return []

    models = []
    for name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, name)
        if os.path.isdir(model_path):
            # Check if it looks like a valid sherpa-onnx model
            has_tokens = os.path.exists(os.path.join(model_path, "tokens.txt"))
            has_encoder = os.path.exists(
                os.path.join(model_path, "encoder.onnx")
            ) or os.path.exists(os.path.join(model_path, "encoder.int8.onnx"))
            if has_tokens and has_encoder:
                models.append(name)

    models.sort()
    return models


def delete_asr_model(model_name: str) -> tuple[bool, str]:
    """
    Delete an ASR model from the local models directory.
    """
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
