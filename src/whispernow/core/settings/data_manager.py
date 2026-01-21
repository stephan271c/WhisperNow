"""
Module for managing WhisperNow user data.
"""

import shutil
from pathlib import Path
from typing import List, Tuple

from platformdirs import user_config_path, user_data_path

from ...utils.cleanup import generate_cleanup_script, run_cleanup_script
from ...utils.logger import get_log_dir, get_logger

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

    log_dir = get_log_dir()
    if log_dir.exists():
        dirs.append(log_dir)

    return dirs


def clear_user_data(
    dry_run: bool = False, skip_logging: bool = False
) -> Tuple[bool, List[str]]:
    errors = []
    dirs_to_delete = get_all_data_dirs()

    if not dirs_to_delete:
        return True, []

    for dir_path in dirs_to_delete:
        try:
            if dry_run:
                if not skip_logging:
                    logger.info(f"[DRY RUN] Would delete {dir_path}")
            else:
                if not skip_logging:
                    logger.info(f"Deleting {dir_path}")
                shutil.rmtree(dir_path)
        except Exception as e:
            error_msg = f"Failed to delete {dir_path}: {e}"
            if not skip_logging:
                logger.error(error_msg)
            errors.append(error_msg)

    return len(errors) == 0, errors


def schedule_cleanup_and_exit() -> None:
    """
    Schedule the deletion of all data directories using an external script
    and exit the application.
    """
    dirs_to_delete = get_all_data_dirs()
    if not dirs_to_delete:
        return

    try:
        script_path = generate_cleanup_script(dirs_to_delete)
        run_cleanup_script(script_path)
    except Exception as e:
        logger.error(f"Failed to schedule cleanup: {e}")
        # If we can't schedule cleanup, we can't really do much else
        # The user will have to manually delete
        pass
