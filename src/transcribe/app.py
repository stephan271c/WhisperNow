"""
Main application entry point.

Thin wrapper around the application runtime.
"""

from .application import TranscribeApp, main

__all__ = ["TranscribeApp", "main"]


if __name__ == "__main__":
    main()
