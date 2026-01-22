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


def test_is_model_cached_whisper():
    from unittest.mock import MagicMock, patch

    from src.whispernow.core.asr.backends import SherpaOnnxBackend

    backend = SherpaOnnxBackend()

    with (
        patch("src.whispernow.core.asr.backends.get_model_type") as mock_type,
        patch("src.whispernow.core.asr.backends.is_valid_whisper_model") as mock_valid,
        patch("src.whispernow.core.asr.backends.os.path.isdir") as mock_isdir,
        patch("src.whispernow.core.asr.backends.get_models_dir") as mock_dir,
    ):

        mock_type.return_value = "whisper"
        mock_valid.return_value = True
        mock_isdir.return_value = True
        mock_dir.return_value = "/models"

        assert backend.is_model_cached("test-whisper") is True
        mock_valid.assert_called_once()


def test_is_model_cached_transducer():
    from unittest.mock import patch

    from src.whispernow.core.asr.backends import SherpaOnnxBackend

    backend = SherpaOnnxBackend()

    with (
        patch("src.whispernow.core.asr.backends.get_model_type") as mock_type,
        patch(
            "src.whispernow.core.asr.backends.is_valid_transducer_model"
        ) as mock_valid,
        patch("src.whispernow.core.asr.backends.os.path.isdir") as mock_isdir,
        patch("src.whispernow.core.asr.backends.get_models_dir") as mock_dir,
    ):

        mock_type.return_value = "transducer"
        mock_valid.return_value = True
        mock_isdir.return_value = True
        mock_dir.return_value = "/models"

        assert backend.is_model_cached("test-transducer") is True
        mock_valid.assert_called_once()
