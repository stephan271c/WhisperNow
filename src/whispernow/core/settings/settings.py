"""
Settings management with JSON persistence.

Handles loading, saving, and validating application settings.
Uses platformdirs for cross-platform directory resolution.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from platformdirs import user_config_path, user_data_path
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...utils.logger import get_logger
from .config import MAX_HISTORY_ENTRIES

logger = get_logger(__name__)

APP_NAME = "whispernow"


def _get_default_enhancements() -> List[dict]:
    from ..transcript_processor.llm_processor import get_default_enhancements

    return [e.model_dump() for e in get_default_enhancements()]


def get_config_dir() -> Path:
    return user_config_path(APP_NAME, ensure_exists=True)


def get_data_dir() -> Path:
    return user_data_path(APP_NAME, ensure_exists=True)


class HotkeyConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=False)

    modifiers: list[str] = Field(default_factory=lambda: ["ctrl"])
    key: str = "space"

    @field_validator("modifiers")
    @classmethod
    def modifiers_not_empty(cls, v):
        if not v or not all(isinstance(m, str) and m.strip() for m in v):
            raise ValueError("modifiers must be a non-empty list of non-empty strings")
        return v

    @field_validator("key")
    @classmethod
    def key_not_empty(cls, v):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("key must be a non-empty string")
        return v

    def to_display_string(self) -> str:
        parts = [mod.capitalize() for mod in self.modifiers]
        parts.append(self.key.capitalize())
        return " + ".join(parts)


class TranscriptionRecord(BaseModel):
    model_config = ConfigDict(validate_assignment=False)

    timestamp: str  # ISO format datetime
    raw_text: str
    enhanced_text: Optional[str] = None
    enhancement_name: Optional[str] = None
    cost_usd: Optional[float] = None

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "TranscriptionRecord":
        return cls.model_validate(data)


class LLMProviderSettings(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=False)

    model: str = ""
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    saved_models: List[str] = Field(default_factory=list)

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "LLMProviderSettings":
        return cls.model_validate(data)


class Settings(BaseModel):
    model_config = ConfigDict(validate_assignment=False)

    sample_rate: int = Field(default=16000, ge=8000, le=192000)
    input_device: Optional[str] = None
    characters_per_second: int = Field(default=150, ge=0)
    instant_type: bool = False

    hotkey: HotkeyConfig = Field(default_factory=HotkeyConfig)
    model_id: str = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"

    start_minimized: bool = False
    auto_start_on_login: bool = False
    first_run_complete: bool = False
    accessibility_permissions_granted: bool = False

    enhancements: List[dict] = Field(default_factory=list)
    active_enhancement_id: Optional[str] = None
    llm_provider: str = "openai"
    llm_provider_settings: Dict[str, dict] = Field(default_factory=dict)
    vocabulary_replacements: List[Tuple[str, str]] = Field(default_factory=list)

    window_geometry: Optional[Tuple[int, int, int, int]] = None

    @field_validator("model_id")
    @classmethod
    def model_id_not_empty(cls, v):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("model_id must be a non-empty string")
        return v

    @classmethod
    def load(cls) -> "Settings":
        config_file = get_config_dir() / "settings.json"

        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    data = json.load(f)

                # Filter to valid keys only
                valid_keys = cls.model_fields.keys()
                filtered_data = {k: v for k, v in data.items() if k in valid_keys}

                # Handle nested HotkeyConfig
                if "hotkey" in filtered_data and isinstance(
                    filtered_data["hotkey"], dict
                ):
                    try:
                        filtered_data["hotkey"] = HotkeyConfig.model_validate(
                            filtered_data["hotkey"]
                        )
                    except Exception:
                        logger.warning(
                            "Invalid hotkey configuration, resetting to default"
                        )
                        filtered_data["hotkey"] = HotkeyConfig()

                if "enhancements" in filtered_data:
                    if not isinstance(filtered_data["enhancements"], list):
                        filtered_data["enhancements"] = []

                # Migration from old flat LLM settings
                old_llm_model = data.get("llm_model")
                old_llm_api_key = data.get("llm_api_key")
                old_llm_api_base = data.get("llm_api_base")
                old_provider = data.get("llm_provider", "openai")

                if old_llm_model or old_llm_api_key or old_llm_api_base:
                    if "llm_provider_settings" not in filtered_data:
                        filtered_data["llm_provider_settings"] = {}
                    if old_provider not in filtered_data["llm_provider_settings"]:
                        filtered_data["llm_provider_settings"][old_provider] = {
                            "model": old_llm_model or "",
                            "api_key": old_llm_api_key,
                            "api_base": old_llm_api_base,
                        }
                    logger.info(
                        f"Migrated old LLM settings to per-provider format for '{old_provider}'"
                    )

                # Validate each field individually, falling back to defaults on error
                settings = cls._load_with_fallbacks(filtered_data)

                if not settings.enhancements:
                    settings.enhancements = _get_default_enhancements()

                return settings
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(
                    f"Could not load settings: {e}. Using defaults.", exc_info=True
                )
                return cls()

        settings = cls()
        settings.enhancements = _get_default_enhancements()
        return settings

    @classmethod
    def _load_with_fallbacks(cls, data: dict) -> "Settings":
        """Load settings with field-level fallback to defaults on validation errors."""
        defaults = cls()
        result_data = {}

        for field_name, field_info in cls.model_fields.items():
            if field_name in data:
                try:
                    # Validate individual field by creating partial model
                    test_data = {field_name: data[field_name]}
                    # For nested models, they're already validated
                    if field_name == "hotkey" and isinstance(
                        data[field_name], HotkeyConfig
                    ):
                        result_data[field_name] = data[field_name]
                    else:
                        cls.model_validate({**defaults.model_dump(), **test_data})
                        result_data[field_name] = data[field_name]
                except Exception as e:
                    default_val = getattr(defaults, field_name)
                    logger.warning(
                        f"Invalid {field_name} {data[field_name]!r}, resetting to {default_val}"
                    )
                    result_data[field_name] = default_val
            else:
                result_data[field_name] = getattr(defaults, field_name)

        return cls.model_construct(**result_data)

    def save(self) -> None:
        config_file = get_config_dir() / "settings.json"

        data = self.model_dump()

        with open(config_file, "w") as f:
            json.dump(data, f, indent=2)

    def reset_to_defaults(self) -> None:
        default = Settings()
        for key, value in default.model_dump().items():
            setattr(self, key, value)

    def get_active_enhancement(self) -> Optional["Enhancement"]:
        if not self.active_enhancement_id:
            return None

        from ..transcript_processor.llm_processor import Enhancement

        for enh_dict in self.enhancements:
            if enh_dict.get("id") == self.active_enhancement_id:
                return Enhancement.model_validate(enh_dict)

        return None

    def get_provider_settings(self, provider_id: str) -> LLMProviderSettings:
        if provider_id in self.llm_provider_settings:
            return LLMProviderSettings.model_validate(
                self.llm_provider_settings[provider_id]
            )
        return LLMProviderSettings()

    def set_provider_settings(
        self, provider_id: str, settings: LLMProviderSettings
    ) -> None:
        self.llm_provider_settings[provider_id] = settings.model_dump()

    @property
    def llm_model(self) -> str:
        return self.get_provider_settings(self.llm_provider).model

    @llm_model.setter
    def llm_model(self, value: str) -> None:
        settings = self.get_provider_settings(self.llm_provider)
        settings.model = value
        self.set_provider_settings(self.llm_provider, settings)

    @property
    def llm_api_key(self) -> Optional[str]:
        return self.get_provider_settings(self.llm_provider).api_key

    @llm_api_key.setter
    def llm_api_key(self, value: Optional[str]) -> None:
        settings = self.get_provider_settings(self.llm_provider)
        settings.api_key = value
        self.set_provider_settings(self.llm_provider, settings)

    @property
    def llm_api_base(self) -> Optional[str]:
        return self.get_provider_settings(self.llm_provider).api_base

    @llm_api_base.setter
    def llm_api_base(self, value: Optional[str]) -> None:
        settings = self.get_provider_settings(self.llm_provider)
        settings.api_base = value
        self.set_provider_settings(self.llm_provider, settings)


_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings.load()
    return _settings_instance


def get_history_file() -> Path:
    return get_config_dir() / "history.json"


def load_history() -> List[TranscriptionRecord]:
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
    history_file = get_history_file()
    records = records[-MAX_HISTORY_ENTRIES:]

    data = [record.to_dict() for record in records]

    with open(history_file, "w") as f:
        json.dump(data, f, indent=2)


def add_history_record(record: TranscriptionRecord) -> None:
    records = load_history()
    records.append(record)
    save_history(records)


def clear_history() -> None:
    history_file = get_history_file()
    if history_file.exists():
        history_file.unlink()
