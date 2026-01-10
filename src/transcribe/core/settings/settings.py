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
from typing import Optional, Tuple, List, TYPE_CHECKING
import json
import platform

from ...utils.logger import get_logger
from .config import MAX_HISTORY_ENTRIES

logger = get_logger(__name__)


def _get_default_enhancements() -> List[dict]:
    """Get default enhancement presets as dicts (lazy import to avoid circular dependency)."""
    from ..transcript_processor.llm_processor import DEFAULT_ENHANCEMENTS
    return [e.to_dict() for e in DEFAULT_ENHANCEMENTS]


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
class TranscriptionRecord:
    """A single transcription history entry."""
    timestamp: str  # ISO format datetime
    raw_text: str
    enhanced_text: Optional[str] = None
    enhancement_name: Optional[str] = None
    cost_usd: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "TranscriptionRecord":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class Settings:
    """Application settings with defaults."""
    
    # Audio settings
    sample_rate: int = 16000
    input_device: Optional[str] = None  # None = system default
    
    # Typing behavior
    characters_per_second: int = 150  # 0 = instant
    instant_type: bool = False
    
    # Hotkey
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    
    # Model
    model_name: str = "nvidia/parakeet-tdt-0.6b-v3"
    use_gpu: bool = True
    
    # App behavior
    start_minimized: bool = False
    auto_start_on_login: bool = False

    first_run_complete: bool = False
    
    # Platform permissions (macOS)
    accessibility_permissions_granted: bool = False
    
    # LLM Enhancement settings
    enhancements: List[dict] = field(default_factory=list)  # List of Enhancement dicts
    active_enhancement_id: Optional[str] = None  # ID of active enhancement, None = disabled
    llm_provider: str = "openai"  # Provider: openai, anthropic, openrouter, ollama, gemini, other
    llm_model: str = "gpt-4o-mini"  # LLM model name
    llm_api_key: Optional[str] = None  # API key (user responsible for security)
    llm_api_base: Optional[str] = None  # Custom API base URL (for openrouter, ollama, etc.)
    
    # Vocabulary replacements: List of (original, replacement) tuples
    vocabulary_replacements: List[Tuple[str, str]] = field(default_factory=list)
    
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
                
                # Filter out unknown keys (backward compatibility)
                valid_keys = cls.__dataclass_fields__.keys()
                filtered_data = {k: v for k, v in data.items() if k in valid_keys}
                
                # Handle nested HotkeyConfig
                if "hotkey" in filtered_data and isinstance(filtered_data["hotkey"], dict):
                    filtered_data["hotkey"] = HotkeyConfig(**filtered_data["hotkey"])
                
                # Handle enhancements list (keep as dicts, not Enhancement objects)
                if "enhancements" in filtered_data:
                    if not isinstance(filtered_data["enhancements"], list):
                        filtered_data["enhancements"] = []
                
                settings = cls(**filtered_data)
                settings._validate()
                
                # Auto-populate default enhancements if none exist
                if not settings.enhancements:
                    settings.enhancements = _get_default_enhancements()
                
                return settings
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Could not load settings: {e}. Using defaults.", exc_info=True)
                return cls()
        
        # New settings: populate with defaults
        settings = cls()
        settings.enhancements = _get_default_enhancements()
        return settings
    
    def _validate(self) -> None:
        """Validate settings and reset invalid values to defaults."""
        defaults = Settings()
        
        # Validate sample_rate
        if not isinstance(self.sample_rate, int) or not (8000 <= self.sample_rate <= 192000):
            logger.warning(f"Invalid sample_rate {self.sample_rate}, resetting to {defaults.sample_rate}")
            self.sample_rate = defaults.sample_rate
        
        # Validate model_name
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            logger.warning(f"Invalid model_name '{self.model_name}', resetting to {defaults.model_name}")
            self.model_name = defaults.model_name
        
        # Validate characters_per_second
        if not isinstance(self.characters_per_second, int) or self.characters_per_second < 0:
            logger.warning(f"Invalid characters_per_second {self.characters_per_second}, resetting to {defaults.characters_per_second}")
            self.characters_per_second = defaults.characters_per_second
        
        # Validate hotkey configuration
        if not isinstance(self.hotkey, HotkeyConfig):
            logger.warning("Invalid hotkey configuration, resetting to default")
            self.hotkey = defaults.hotkey
        elif not self.hotkey.modifiers or not all(isinstance(m, str) and m.strip() for m in self.hotkey.modifiers):
            logger.warning(f"Invalid hotkey modifiers {self.hotkey.modifiers}, resetting to default")
            self.hotkey = defaults.hotkey
        elif not isinstance(self.hotkey.key, str) or not self.hotkey.key.strip():
            logger.warning(f"Invalid hotkey key '{self.hotkey.key}', resetting to default")
            self.hotkey = defaults.hotkey
    
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
    
    def get_active_enhancement(self) -> Optional["Enhancement"]:
        """
        Get the currently active enhancement, if any.
        
        Returns:
            The active Enhancement object, or None if no enhancement is active
            or the active_enhancement_id doesn't match any enhancement.
        """
        if not self.active_enhancement_id:
            return None
        
        from ..transcript_processor.llm_processor import Enhancement
        
        for enh_dict in self.enhancements:
            if enh_dict.get("id") == self.active_enhancement_id:
                return Enhancement.from_dict(enh_dict)
        
        return None


# Convenience function for quick access
_settings_instance: Optional[Settings] = None

def get_settings() -> Settings:
    """Get the global settings instance (loads from disk on first call)."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings.load()
    return _settings_instance


# =============================================================================
# History Management (separate file)
# =============================================================================

def get_history_file() -> Path:
    """Get the path to the transcription history file."""
    return get_config_dir() / "history.json"


def load_history() -> List[TranscriptionRecord]:
    """Load transcription history from history.json."""
    history_file = get_history_file()
    
    if not history_file.exists():
        return []
    
    try:
        with open(history_file, "r") as f:
            data = json.load(f)
        
        records = [TranscriptionRecord.from_dict(item) for item in data]
        return records
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.warning(f"Could not load history: {e}. Starting fresh.")
        return []


def save_history(records: List[TranscriptionRecord]) -> None:
    """Save transcription history, limiting to MAX_HISTORY_ENTRIES."""
    history_file = get_history_file()
    
    # Keep only the most recent entries
    records = records[-MAX_HISTORY_ENTRIES:]
    
    data = [record.to_dict() for record in records]
    
    with open(history_file, "w") as f:
        json.dump(data, f, indent=2)


def add_history_record(record: TranscriptionRecord) -> None:
    """Add a single transcription record to history."""
    records = load_history()
    records.append(record)
    save_history(records)


def clear_history() -> None:
    """Clear all transcription history."""
    history_file = get_history_file()
    if history_file.exists():
        history_file.unlink()
