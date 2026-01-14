"""Tests for model registry and downloader."""

import pytest

from src.whispernow.core.asr.model_registry import (
    AVAILABLE_MODELS,
    get_all_models_with_status,
    get_model_by_id,
    is_model_downloaded,
)


class TestModelRegistry:
    def test_available_models_not_empty(self):
        assert len(AVAILABLE_MODELS) > 0

    def test_model_has_required_fields(self):
        for model in AVAILABLE_MODELS:
            assert model.id
            assert model.name
            assert model.url.startswith("https://")
            assert model.url.endswith(".tar.bz2")

    def test_get_model_by_id_found(self):
        model = AVAILABLE_MODELS[0]
        found = get_model_by_id(model.id)
        assert found == model

    def test_get_model_by_id_not_found(self):
        found = get_model_by_id("nonexistent-model")
        assert found is None

    def test_get_all_models_with_status(self):
        results = get_all_models_with_status()
        assert len(results) == len(AVAILABLE_MODELS)
        for model, status in results:
            assert status in ("downloaded", "not_downloaded")

    def test_is_model_downloaded_returns_false_for_nonexistent(self):
        assert is_model_downloaded("nonexistent-model") is False

    def test_default_model_url_valid(self):
        model = get_model_by_id("sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16")
        assert model is not None
        assert "github.com" in model.url
        assert "k2-fsa" in model.url
