"""
Tests for Settings persistence.

Verifies save/load cycles, default values, and corrupted file handling.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.transcribe.core.settings import (
    Settings,
    HotkeyConfig,
    get_config_dir,
    get_data_dir,
)


class TestHotkeyConfig:
    """Tests for HotkeyConfig dataclass."""
    
    def test_default_values(self):
        """Test default hotkey is Ctrl+Space."""
        hotkey = HotkeyConfig()
        assert hotkey.modifiers == ["ctrl"]
        assert hotkey.key == "space"
    
    def test_display_string(self):
        """Test human-readable hotkey display."""
        hotkey = HotkeyConfig(modifiers=["ctrl", "shift"], key="a")
        assert hotkey.to_display_string() == "Ctrl + Shift + A"
    
    def test_serialization(self):
        """Test HotkeyConfig serializes to dict correctly."""
        from dataclasses import asdict
        hotkey = HotkeyConfig(modifiers=["alt"], key="space")
        data = asdict(hotkey)
        assert data == {"modifiers": ["alt"], "key": "space"}


class TestSettings:
    """Tests for Settings class."""
    
    def test_default_values(self):
        """Test default settings values."""
        settings = Settings()
        assert settings.sample_rate == 16000
        assert settings.input_device is None
        assert settings.model_name == "nvidia/parakeet-tdt-0.6b-v3"
        assert settings.use_gpu is True
    
    def test_save_load_cycle(self, tmp_path):
        """Test settings can be saved and loaded back."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        with patch("src.transcribe.core.settings.settings.get_config_dir", return_value=config_dir):
            # Create settings with custom values
            original = Settings(
                sample_rate=44100,
                input_device="Test Mic",
                characters_per_second=200,
                instant_type=True,
                hotkey=HotkeyConfig(modifiers=["alt"], key="r"),
                model_name="openai/whisper-base",
                use_gpu=False,
            )
            original.save()
            
            # Verify file was created
            config_file = config_dir / "settings.json"
            assert config_file.exists()
            
            # Load and compare
            loaded = Settings.load()
            assert loaded.sample_rate == 44100
            assert loaded.input_device == "Test Mic"
            assert loaded.characters_per_second == 200
            assert loaded.instant_type is True
            assert loaded.hotkey.modifiers == ["alt"]
            assert loaded.hotkey.key == "r"
            assert loaded.model_name == "openai/whisper-base"
            assert loaded.use_gpu is False
    
    def test_load_nonexistent_returns_defaults(self, tmp_path):
        """Test loading from non-existent file returns defaults."""
        config_dir = tmp_path / "empty_config"
        config_dir.mkdir()
        
        with patch("src.transcribe.core.settings.settings.get_config_dir", return_value=config_dir):
            settings = Settings.load()
            default = Settings()
            
            assert settings.sample_rate == default.sample_rate
            assert settings.model_name == default.model_name
    
    def test_load_corrupted_json_returns_defaults(self, tmp_path):
        """Test loading corrupted JSON falls back to defaults."""
        config_dir = tmp_path / "corrupt_config"
        config_dir.mkdir()
        config_file = config_dir / "settings.json"
        config_file.write_text("{ this is not valid json }")
        
        with patch("src.transcribe.core.settings.settings.get_config_dir", return_value=config_dir):
            settings = Settings.load()
            default = Settings()
            assert settings.sample_rate == default.sample_rate
    
    def test_reset_to_defaults(self):
        """Test reset_to_defaults restores all values."""
        settings = Settings(
            sample_rate=44100,
            model_name="custom/model",
        )
        settings.reset_to_defaults()
        
        assert settings.sample_rate == 16000
        assert settings.model_name == "nvidia/parakeet-tdt-0.6b-v3"


class TestConfigPaths:
    """Tests for configuration path functions."""
    
    def test_get_config_dir_returns_path(self):
        """Test get_config_dir returns a Path object."""
        result = get_config_dir()
        assert isinstance(result, Path)
        # Should contain 'transcribe' in path
        assert "transcribe" in str(result)
    
    def test_get_data_dir_returns_path(self):
        """Test get_data_dir returns a Path object."""
        result = get_data_dir()
        assert isinstance(result, Path)
        assert "transcribe" in str(result)
    
    def test_config_and_data_dirs_exist(self):
        """Test that directories are created when accessed."""
        config_dir = get_config_dir()
        data_dir = get_data_dir()
        
        assert config_dir.exists()
        assert data_dir.exists()


class TestSettingsValidation:
    """Tests for settings validation logic."""
    
    def test_invalid_sample_rate_resets_to_default(self, tmp_path):
        """Test invalid sample rates are reset to default."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "settings.json"
        
        # Test negative sample rate
        config_file.write_text(json.dumps({"sample_rate": -1000}))
        
        with patch("src.transcribe.core.settings.settings.get_config_dir", return_value=config_dir):
            settings = Settings.load()
            assert settings.sample_rate == 16000  # Should reset to default
        
        # Test sample rate too high
        config_file.write_text(json.dumps({"sample_rate": 999999}))
        
        with patch("src.transcribe.core.settings.settings.get_config_dir", return_value=config_dir):
            settings = Settings.load()
            assert settings.sample_rate == 16000
        
        # Test sample rate too low
        config_file.write_text(json.dumps({"sample_rate": 100}))
        
        with patch("src.transcribe.core.settings.settings.get_config_dir", return_value=config_dir):
            settings = Settings.load()
            assert settings.sample_rate == 16000
    
    def test_empty_model_name_resets_to_default(self, tmp_path):
        """Test empty model name is reset to default."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "settings.json"
        
        config_file.write_text(json.dumps({"model_name": ""}))
        
        with patch("src.transcribe.core.settings.settings.get_config_dir", return_value=config_dir):
            settings = Settings.load()
            assert settings.model_name == "nvidia/parakeet-tdt-0.6b-v3"
    
    def test_negative_characters_per_second_resets(self, tmp_path):
        """Test negative characters_per_second is reset to default."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "settings.json"
        
        config_file.write_text(json.dumps({"characters_per_second": -50}))
        
        with patch("src.transcribe.core.settings.settings.get_config_dir", return_value=config_dir):
            settings = Settings.load()
            assert settings.characters_per_second == 150
    
    def test_invalid_hotkey_modifiers_resets(self, tmp_path):
        """Test invalid hotkey modifiers are reset to default."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "settings.json"
        
        # Empty modifiers list
        config_file.write_text(json.dumps({"hotkey": {"modifiers": [], "key": "space"}}))
        
        with patch("src.transcribe.core.settings.settings.get_config_dir", return_value=config_dir):
            settings = Settings.load()
            assert settings.hotkey.modifiers == ["ctrl"]
            assert settings.hotkey.key == "space"
    
    def test_invalid_hotkey_key_resets(self, tmp_path):
        """Test invalid hotkey key is reset to default."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "settings.json"
        
        # Empty key
        config_file.write_text(json.dumps({"hotkey": {"modifiers": ["ctrl"], "key": ""}}))
        
        with patch("src.transcribe.core.settings.settings.get_config_dir", return_value=config_dir):
            settings = Settings.load()
            assert settings.hotkey.modifiers == ["ctrl"]
            assert settings.hotkey.key == "space"
    
    def test_validation_logs_warnings(self, tmp_path, caplog):
        """Test that validation issues are logged as warnings."""
        import logging
        
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "settings.json"
        
        config_file.write_text(json.dumps({
            "sample_rate": -1000,
            "model_name": "",
            "characters_per_second": -50
        }))
        
        # Ensure we capture from the transcribe logger even if propagation is off
        logger = logging.getLogger("transcribe")
        logger.propagate = True
        
        with patch("src.transcribe.core.settings.settings.get_config_dir", return_value=config_dir):
            with caplog.at_level(logging.WARNING, logger="transcribe"):
                settings = Settings.load()
                
                # Check that warnings were logged
                assert "Invalid sample_rate" in caplog.text
                assert "Invalid model_name" in caplog.text
                assert "Invalid characters_per_second" in caplog.text
    
    def test_valid_settings_pass_validation(self, tmp_path):
        """Test valid settings pass validation without modification."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "settings.json"
        
        valid_settings = {
            "sample_rate": 44100,
            "model_name": "custom/model",
            "characters_per_second": 200,
            "hotkey": {"modifiers": ["alt", "shift"], "key": "r"}
        }
        config_file.write_text(json.dumps(valid_settings))
        
        with patch("src.transcribe.core.settings.settings.get_config_dir", return_value=config_dir):
            settings = Settings.load()
            assert settings.sample_rate == 44100
            assert settings.model_name == "custom/model"
            assert settings.characters_per_second == 200
            assert settings.hotkey.modifiers == ["alt", "shift"]
            assert settings.hotkey.key == "r"


class TestPerProviderSettings:
    """Tests for per-provider LLM settings storage."""
    
    def test_get_set_provider_settings(self):
        """Test getting and setting provider-specific settings."""
        from src.transcribe.core.settings import LLMProviderSettings
        
        settings = Settings()
        
        # Initially, provider settings should be empty
        openai_settings = settings.get_provider_settings("openai")
        assert openai_settings.model == ""
        assert openai_settings.api_key is None
        
        # Set settings for OpenAI
        settings.set_provider_settings("openai", LLMProviderSettings(
            model="gpt-4",
            api_key="sk-test-key",
            api_base=None
        ))
        
        # Verify they were saved
        openai_settings = settings.get_provider_settings("openai")
        assert openai_settings.model == "gpt-4"
        assert openai_settings.api_key == "sk-test-key"
    
    def test_provider_settings_isolation(self):
        """Test that changing one provider's settings doesn't affect another."""
        from src.transcribe.core.settings import LLMProviderSettings
        
        settings = Settings()
        
        # Set settings for two different providers
        settings.set_provider_settings("openai", LLMProviderSettings(
            model="gpt-4",
            api_key="openai-key"
        ))
        settings.set_provider_settings("ollama", LLMProviderSettings(
            model="llama3.2",
            api_base="http://localhost:11434"
        ))
        
        # Verify they are independent
        openai = settings.get_provider_settings("openai")
        ollama = settings.get_provider_settings("ollama")
        
        assert openai.model == "gpt-4"
        assert openai.api_key == "openai-key"
        assert ollama.model == "llama3.2"
        assert ollama.api_base == "http://localhost:11434"
        assert ollama.api_key is None  # Not set for Ollama
    
    def test_backward_compatible_properties(self):
        """Test that llm_model, llm_api_key properties work with active provider."""
        from src.transcribe.core.settings import LLMProviderSettings
        
        settings = Settings()
        settings.llm_provider = "anthropic"
        
        # Set via property
        settings.llm_model = "claude-3-sonnet"
        settings.llm_api_key = "anthropic-key"
        
        # Verify via property
        assert settings.llm_model == "claude-3-sonnet"
        assert settings.llm_api_key == "anthropic-key"
        
        # Verify underlying storage
        provider_settings = settings.get_provider_settings("anthropic")
        assert provider_settings.model == "claude-3-sonnet"
        assert provider_settings.api_key == "anthropic-key"
    
    def test_migration_from_old_format(self, tmp_path):
        """Test that old flat LLM settings are migrated to per-provider format."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "settings.json"
        
        # Old format with flat LLM settings
        old_settings = {
            "llm_provider": "openrouter",
            "llm_model": "anthropic/claude-3-sonnet",
            "llm_api_key": "sk-or-test-key",
            "llm_api_base": None
        }
        config_file.write_text(json.dumps(old_settings))
        
        with patch("src.transcribe.core.settings.settings.get_config_dir", return_value=config_dir):
            settings = Settings.load()
            
            # Should have migrated to per-provider format
            provider_settings = settings.get_provider_settings("openrouter")
            assert provider_settings.model == "anthropic/claude-3-sonnet"
            assert provider_settings.api_key == "sk-or-test-key"
            
            # Active provider should be preserved
            assert settings.llm_provider == "openrouter"
    
    def test_per_provider_save_load_cycle(self, tmp_path):
        """Test that per-provider settings persist through save/load cycle."""
        from src.transcribe.core.settings import LLMProviderSettings
        
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        with patch("src.transcribe.core.settings.settings.get_config_dir", return_value=config_dir):
            # Create settings with multiple providers configured
            settings = Settings()
            settings.llm_provider = "openai"
            settings.set_provider_settings("openai", LLMProviderSettings(
                model="gpt-4",
                api_key="openai-key"
            ))
            settings.set_provider_settings("ollama", LLMProviderSettings(
                model="llama3.2",
                api_base="http://localhost:11434"
            ))
            settings.save()
            
            # Load and verify
            loaded = Settings.load()
            
            assert loaded.llm_provider == "openai"
            
            openai = loaded.get_provider_settings("openai")
            assert openai.model == "gpt-4"
            assert openai.api_key == "openai-key"
            
            ollama = loaded.get_provider_settings("ollama")
            assert ollama.model == "llama3.2"
            assert ollama.api_base == "http://localhost:11434"
