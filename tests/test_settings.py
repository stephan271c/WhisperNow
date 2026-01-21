"""Tests for Settings persistence."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.whispernow.core.settings import (
    HotkeyConfig,
    Settings,
    get_config_dir,
    get_data_dir,
)


class TestHotkeyConfig:
    def test_default_values(self):
        hotkey = HotkeyConfig()
        assert hotkey.modifiers == ["ctrl"]
        assert hotkey.key == "space"

    def test_display_string(self):
        hotkey = HotkeyConfig(modifiers=["ctrl", "shift"], key="a")
        assert hotkey.to_display_string() == "Ctrl + Shift + A"

    def test_serialization(self):
        hotkey = HotkeyConfig(modifiers=["alt"], key="space")
        data = hotkey.model_dump()
        assert data == {"modifiers": ["alt"], "key": "space"}


class TestSettings:
    def test_default_values(self):
        settings = Settings()
        assert settings.sample_rate == 16000
        assert settings.input_device is None
        assert settings.model_id == "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16"

    def test_save_load_cycle(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        with patch(
            "src.whispernow.core.settings.settings.get_config_dir",
            return_value=config_dir,
        ):
            # Create settings with custom values
            original = Settings(
                sample_rate=44100,
                input_device="Test Mic",
                hotkey=HotkeyConfig(modifiers=["alt"], key="r"),
                model_id="openai/whisper-base",
            )
            original.save()

            config_file = config_dir / "settings.json"
            assert config_file.exists()

            loaded = Settings.load()
            assert loaded.sample_rate == 44100
            assert loaded.input_device == "Test Mic"
            assert loaded.hotkey.modifiers == ["alt"]
            assert loaded.hotkey.key == "r"
            assert loaded.model_id == "openai/whisper-base"

    def test_load_nonexistent_returns_defaults(self, tmp_path):
        config_dir = tmp_path / "empty_config"
        config_dir.mkdir()

        with patch(
            "src.whispernow.core.settings.settings.get_config_dir",
            return_value=config_dir,
        ):
            settings = Settings.load()
            default = Settings()

            assert settings.sample_rate == default.sample_rate
            assert settings.model_id == default.model_id

    def test_load_corrupted_json_returns_defaults(self, tmp_path):
        config_dir = tmp_path / "corrupt_config"
        config_dir.mkdir()
        config_file = config_dir / "settings.json"
        config_file.write_text("{ this is not valid json }")

        with patch(
            "src.whispernow.core.settings.settings.get_config_dir",
            return_value=config_dir,
        ):
            settings = Settings.load()
            default = Settings()
            assert settings.sample_rate == default.sample_rate

    def test_reset_to_defaults(self):
        settings = Settings(
            sample_rate=44100,
            model_id="custom/model",
        )
        settings.reset_to_defaults()

        assert settings.sample_rate == 16000
        assert settings.model_id == "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16"


class TestConfigPaths:
    def test_get_config_dir_returns_path(self):
        result = get_config_dir()
        assert isinstance(result, Path)
        assert "whispernow" in str(result)

    def test_get_data_dir_returns_path(self):
        result = get_data_dir()
        assert isinstance(result, Path)
        assert "whispernow" in str(result)

    def test_config_and_data_dirs_exist(self):
        config_dir = get_config_dir()
        data_dir = get_data_dir()

        assert config_dir.exists()
        assert data_dir.exists()


class TestSettingsValidation:
    def test_invalid_sample_rate_resets_to_default(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "settings.json"

        config_file.write_text(json.dumps({"sample_rate": -1000}))

        with patch(
            "src.whispernow.core.settings.settings.get_config_dir",
            return_value=config_dir,
        ):
            settings = Settings.load()
            assert settings.sample_rate == 16000
        config_file.write_text(json.dumps({"sample_rate": 999999}))

        with patch(
            "src.whispernow.core.settings.settings.get_config_dir",
            return_value=config_dir,
        ):
            settings = Settings.load()
            assert settings.sample_rate == 16000

        config_file.write_text(json.dumps({"sample_rate": 100}))

        with patch(
            "src.whispernow.core.settings.settings.get_config_dir",
            return_value=config_dir,
        ):
            settings = Settings.load()
            assert settings.sample_rate == 16000

    def test_empty_model_id_resets_to_default(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "settings.json"

        config_file.write_text(json.dumps({"model_id": ""}))

        with patch(
            "src.whispernow.core.settings.settings.get_config_dir",
            return_value=config_dir,
        ):
            settings = Settings.load()
            assert settings.model_id == "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16"

    def test_invalid_hotkey_modifiers_resets(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "settings.json"
        config_file.write_text(
            json.dumps({"hotkey": {"modifiers": [], "key": "space"}})
        )

        with patch(
            "src.whispernow.core.settings.settings.get_config_dir",
            return_value=config_dir,
        ):
            settings = Settings.load()
            assert settings.hotkey.modifiers == ["ctrl"]
            assert settings.hotkey.key == "space"

    def test_invalid_hotkey_key_resets(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "settings.json"
        config_file.write_text(
            json.dumps({"hotkey": {"modifiers": ["ctrl"], "key": ""}})
        )

        with patch(
            "src.whispernow.core.settings.settings.get_config_dir",
            return_value=config_dir,
        ):
            settings = Settings.load()
            assert settings.hotkey.modifiers == ["ctrl"]
            assert settings.hotkey.key == "space"

    def test_validation_logs_warnings(self, tmp_path, caplog):
        import logging

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "settings.json"

        config_file.write_text(json.dumps({"sample_rate": -1000, "model_id": ""}))

        logger = logging.getLogger("whispernow")
        logger.propagate = True

        with patch(
            "src.whispernow.core.settings.settings.get_config_dir",
            return_value=config_dir,
        ):
            with caplog.at_level(logging.WARNING, logger="whispernow"):
                settings = Settings.load()

                assert "Invalid sample_rate" in caplog.text
                assert "Invalid model_id" in caplog.text

    def test_valid_settings_pass_validation(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "settings.json"

        valid_settings = {
            "sample_rate": 44100,
            "model_id": "custom/model",
            "hotkey": {"modifiers": ["alt", "shift"], "key": "r"},
        }
        config_file.write_text(json.dumps(valid_settings))

        with patch(
            "src.whispernow.core.settings.settings.get_config_dir",
            return_value=config_dir,
        ):
            settings = Settings.load()
            assert settings.sample_rate == 44100
            assert settings.model_id == "custom/model"
            assert settings.hotkey.modifiers == ["alt", "shift"]
            assert settings.hotkey.key == "r"


class TestPerProviderSettings:
    def test_get_set_provider_settings(self):
        from src.whispernow.core.settings import LLMProviderSettings

        settings = Settings()

        openai_settings = settings.get_provider_settings("openai")
        assert openai_settings.model == ""
        assert openai_settings.api_key is None

        settings.set_provider_settings(
            "openai",
            LLMProviderSettings(model="gpt-4", api_key="sk-test-key", api_base=None),
        )

        openai_settings = settings.get_provider_settings("openai")
        assert openai_settings.model == "gpt-4"
        assert openai_settings.api_key == "sk-test-key"

    def test_provider_settings_isolation(self):
        from src.whispernow.core.settings import LLMProviderSettings

        settings = Settings()

        settings.set_provider_settings(
            "openai", LLMProviderSettings(model="gpt-4", api_key="openai-key")
        )
        settings.set_provider_settings(
            "ollama",
            LLMProviderSettings(model="llama3.2", api_base="http://localhost:11434"),
        )

        openai = settings.get_provider_settings("openai")
        ollama = settings.get_provider_settings("ollama")

        assert openai.model == "gpt-4"
        assert openai.api_key == "openai-key"
        assert ollama.model == "llama3.2"
        assert ollama.api_base == "http://localhost:11434"
        assert ollama.api_key is None

    def test_backward_compatible_properties(self):
        from src.whispernow.core.settings import LLMProviderSettings

        settings = Settings()
        settings.llm_provider = "anthropic"

        settings.llm_model = "claude-3-sonnet"
        settings.llm_api_key = "anthropic-key"

        assert settings.llm_model == "claude-3-sonnet"
        assert settings.llm_api_key == "anthropic-key"

        provider_settings = settings.get_provider_settings("anthropic")
        assert provider_settings.model == "claude-3-sonnet"
        assert provider_settings.api_key == "anthropic-key"

    def test_migration_from_old_format(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "settings.json"

        old_settings = {
            "llm_provider": "openrouter",
            "llm_model": "anthropic/claude-3-sonnet",
            "llm_api_key": "sk-or-test-key",
            "llm_api_base": None,
        }
        config_file.write_text(json.dumps(old_settings))

        with patch(
            "src.whispernow.core.settings.settings.get_config_dir",
            return_value=config_dir,
        ):
            settings = Settings.load()

            provider_settings = settings.get_provider_settings("openrouter")
            assert provider_settings.model == "anthropic/claude-3-sonnet"
            assert provider_settings.api_key == "sk-or-test-key"

            assert settings.llm_provider == "openrouter"

    def test_per_provider_save_load_cycle(self, tmp_path):
        from src.whispernow.core.settings import LLMProviderSettings

        config_dir = tmp_path / "config"
        config_dir.mkdir()

        with patch(
            "src.whispernow.core.settings.settings.get_config_dir",
            return_value=config_dir,
        ):
            settings = Settings()
            settings.llm_provider = "openai"
            settings.set_provider_settings(
                "openai", LLMProviderSettings(model="gpt-4", api_key="openai-key")
            )
            settings.set_provider_settings(
                "ollama",
                LLMProviderSettings(
                    model="llama3.2", api_base="http://localhost:11434"
                ),
            )
            settings.save()

            loaded = Settings.load()

            assert loaded.llm_provider == "openai"

            openai = loaded.get_provider_settings("openai")
            assert openai.model == "gpt-4"
            assert openai.api_key == "openai-key"

            ollama = loaded.get_provider_settings("ollama")
            assert ollama.model == "llama3.2"
            assert ollama.api_base == "http://localhost:11434"
