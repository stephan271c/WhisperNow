import os
from typing import Optional

import platformdirs


def get_models_dir() -> str:
    return os.path.join(
        platformdirs.user_data_dir("WhisperNow", appauthor=False), "models"
    )


def find_file_by_suffix(directory: str, *suffixes: str) -> Optional[str]:
    """Find the first file in directory matching any of the given suffixes."""
    try:
        for filename in os.listdir(directory):
            for suffix in suffixes:
                if filename.endswith(suffix):
                    return os.path.join(directory, filename)
    except OSError:
        pass
    return None


def has_file_with_suffix(directory: str, *suffixes: str) -> bool:
    """Check if any file in directory matches any of the given suffixes."""
    return find_file_by_suffix(directory, *suffixes) is not None


def find_file_exact(directory: str, candidates: list[str]) -> Optional[str]:
    """Find the first file in directory matching any of the exact filenames."""
    for name in candidates:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return path
    return None
