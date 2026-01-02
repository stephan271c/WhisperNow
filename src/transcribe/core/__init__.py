# Core module - Business logic

"""
Core functionality for the transcription app.
Contains settings, audio recording, and transcription engine.
"""

from .backends import (
    ASRBackend,
    BackendType,
    TranscriptionResult,
    NeMoBackend,
    HuggingFaceBackend,
    create_backend,
    detect_backend_type,
)
from .transcriber import TranscriptionEngine, EngineState, transcribe_audio

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
]
