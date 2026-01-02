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
        
        with patch("src.transcribe.core.settings.get_config_dir", return_value=config_dir):
            # Create settings with custom values
            original = Settings(
                sample_rate=44100,
                input_device="Test Mic",
                characters_per_second=200,
                auto_type_result=False,
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
            assert loaded.auto_type_result is False
            assert loaded.hotkey.modifiers == ["alt"]
            assert loaded.hotkey.key == "r"
            assert loaded.model_name == "openai/whisper-base"
            assert loaded.use_gpu is False
    
    def test_load_nonexistent_returns_defaults(self, tmp_path):
        """Test loading from non-existent file returns defaults."""
        config_dir = tmp_path / "empty_config"
        config_dir.mkdir()
        
        with patch("src.transcribe.core.settings.get_config_dir", return_value=config_dir):
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
        
        with patch("src.transcribe.core.settings.get_config_dir", return_value=config_dir):
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
