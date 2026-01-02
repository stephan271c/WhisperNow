"""
Settings management with JSON persistence.

Handles loading, saving, and validating application settings.
Stores configuration in platform-appropriate locations:
- Linux: ~/.config/transcribe/settings.json
- macOS: ~/Library/Application Support/transcribe/settings.json
- Windows: %APPDATA%/transcribe/settings.json
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple
import json
import platform


def get_config_dir() -> Path:
    """Get the platform-appropriate configuration directory."""
    system = platform.system()
    
    if system == "Windows":
        base = Path.home() / "AppData" / "Roaming"
    elif system == "Darwin":  # macOS
        base = Path.home() / "Library" / "Application Support"
    else:  # Linux and others
        base = Path.home() / ".config"
    
    config_dir = base / "transcribe"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_data_dir() -> Path:
    """Get the platform-appropriate data directory (for models, etc.)."""
    system = platform.system()
    
    if system == "Windows":
        base = Path.home() / "AppData" / "Local"
    elif system == "Darwin":  # macOS
        base = Path.home() / "Library" / "Application Support"
    else:  # Linux and others
        base = Path.home() / ".local" / "share"
    
    data_dir = base / "transcribe"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@dataclass
class HotkeyConfig:
    """Configuration for a hotkey combination."""
    modifiers: list[str] = field(default_factory=lambda: ["ctrl"])
    key: str = "space"
    
    def to_display_string(self) -> str:
        """Return human-readable hotkey string like 'Ctrl + Space'."""
        parts = [mod.capitalize() for mod in self.modifiers]
        parts.append(self.key.capitalize())
        return " + ".join(parts)


@dataclass
class Settings:
    """Application settings with defaults."""
    
    # Audio settings
    sample_rate: int = 16000
    input_device: Optional[str] = None  # None = system default
    
    # Typing behavior
    characters_per_second: int = 150  # 0 = instant
    auto_type_result: bool = True
    
    # Hotkey
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    
    # Model
    model_name: str = "nvidia/parakeet-tdt-0.6b-v3"
    use_gpu: bool = True
    
    # App behavior
    start_minimized: bool = False
    auto_start_on_login: bool = False
    show_notifications: bool = True
    
    # Window state (internal)
    window_geometry: Optional[Tuple[int, int, int, int]] = None
    
    @classmethod
    def load(cls) -> "Settings":
        """Load settings from config file, or return defaults if not found."""
        config_file = get_config_dir() / "settings.json"
        
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    data = json.load(f)
                
                # Handle nested HotkeyConfig
                if "hotkey" in data and isinstance(data["hotkey"], dict):
                    data["hotkey"] = HotkeyConfig(**data["hotkey"])
                
                return cls(**data)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Could not load settings: {e}. Using defaults.")
                return cls()
        
        return cls()
    
    def save(self) -> None:
        """Save settings to config file."""
        config_file = get_config_dir() / "settings.json"
        
        data = asdict(self)
        
        with open(config_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def reset_to_defaults(self) -> None:
        """Reset all settings to their default values."""
        default = Settings()
        for key, value in asdict(default).items():
            setattr(self, key, value)


# Convenience function for quick access
_settings_instance: Optional[Settings] = None

def get_settings() -> Settings:
    """Get the global settings instance (loads from disk on first call)."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings.load()
    return _settings_instance
