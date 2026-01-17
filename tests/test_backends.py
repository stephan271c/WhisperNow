from pathlib import Path

import pytest

from src.whispernow.core.asr.backends import get_model_type


def test_get_model_type_valid():
    model_type = get_model_type("sherpa-onnx-whisper-tiny")
    assert model_type == "whisper"

    model_type = get_model_type("sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16")
    assert model_type == "transducer"


def test_get_model_type_invalid():
    with pytest.raises(ValueError, match="not found in models.json"):
        get_model_type("unknown-model-id")


def test_models_json_exists():
    import src.whispernow.core.asr.backends as backends_module

    expected_path = Path(backends_module.__file__).parent / "models" / "models.json"
    assert expected_path.exists(), f"models.json not found at {expected_path}"
