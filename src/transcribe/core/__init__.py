# Core module - Business logic

"""
Core functionality for the transcription app.
Contains settings, audio recording, transcription engine, and hotkey listener.
"""

from .asr.backends import (
    ASRBackend,
    BackendType,
    TranscriptionResult,
    NeMoBackend,
    HuggingFaceBackend,
    create_backend,
    detect_backend_type,
)
from .asr.transcriber import TranscriptionEngine, EngineState, transcribe_audio
from .input.hotkey import HotkeyListener

__all__ = [
    # Backend abstraction
    "ASRBackend",
    "BackendType",
    "TranscriptionResult",
    "NeMoBackend",
    "HuggingFaceBackend",
    "create_backend",
    "detect_backend_type",
    # Transcription engine
    "TranscriptionEngine",
    "EngineState",
    "transcribe_audio",
    # Hotkey listener
    "HotkeyListener",
]
