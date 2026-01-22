import os
from typing import Optional

import platformdirs


def get_models_dir() -> str:
    return os.path.join(
        platformdirs.user_data_dir("WhisperNow", appauthor=False), "models"
    )


def find_file_by_suffix(directory: str, *suffixes: str) -> Optional[str]:
    try:
        for filename in os.listdir(directory):
            for suffix in suffixes:
                if filename.endswith(suffix):
                    return os.path.join(directory, filename)
    except OSError:
        pass
    return None


def has_file_with_suffix(directory: str, *suffixes: str) -> bool:
    return find_file_by_suffix(directory, *suffixes) is not None


def find_file_exact(directory: str, candidates: list[str]) -> Optional[str]:
    for name in candidates:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return path
    return None


def is_valid_whisper_model(model_path: str) -> bool:
    has_encoder = has_file_with_suffix(
        model_path, "-encoder.onnx", "-encoder.int8.onnx"
    )
    has_decoder = has_file_with_suffix(
        model_path, "-decoder.onnx", "-decoder.int8.onnx"
    )
    has_tokens = has_file_with_suffix(model_path, "-tokens", "tokens.txt")
    return has_encoder and has_decoder and has_tokens


def is_valid_transducer_model(model_path: str) -> bool:
    has_encoder = (
        find_file_exact(
            model_path, ["encoder.onnx", "encoder.int8.onnx", "encoder.fp16.onnx"]
        )
        is not None
    )
    has_decoder = (
        find_file_exact(
            model_path, ["decoder.onnx", "decoder.int8.onnx", "decoder.fp16.onnx"]
        )
        is not None
    )
    has_joiner = (
        find_file_exact(
            model_path, ["joiner.onnx", "joiner.int8.onnx", "joiner.fp16.onnx"]
        )
        is not None
    )
    has_tokens = os.path.exists(os.path.join(model_path, "tokens.txt"))
    return has_encoder and has_decoder and has_joiner and has_tokens


def is_valid_model_dir(model_path: str) -> bool:
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
    return has_tokens and has_encoder
