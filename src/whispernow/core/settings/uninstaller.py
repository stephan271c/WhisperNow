"""
Uninstaller module for removing WhisperNow user data.
"""

import shutil
from pathlib import Path
from typing import List, Tuple

from platformdirs import user_config_path, user_data_path

from ...utils.logger import get_logger

logger = get_logger(__name__)

APP_NAME_CONFIG = "whispernow"
APP_NAME_DATA = "WhisperNow"


def get_all_data_dirs() -> List[Path]:
    dirs = []
    config_dir = user_config_path(APP_NAME_CONFIG, appauthor=False)
    if config_dir.exists():
        dirs.append(config_dir)

    data_dir = user_data_path(APP_NAME_DATA, appauthor=False)
    if data_dir.exists():
        dirs.append(data_dir)

    return dirs


def uninstall_user_data(dry_run: bool = False) -> Tuple[bool, List[str]]:
    errors = []
    dirs_to_delete = get_all_data_dirs()

    if not dirs_to_delete:
        return True, []

    for dir_path in dirs_to_delete:
        try:
            if dry_run:
                logger.info(f"[DRY RUN] Would delete {dir_path}")
            else:
                logger.info(f"Deleting {dir_path}")
                shutil.rmtree(dir_path)
        except Exception as e:
            error_msg = f"Failed to delete {dir_path}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

    return len(errors) == 0, errors
