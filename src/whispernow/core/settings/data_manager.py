import shutil
from pathlib import Path
from typing import List, Tuple, Union

from platformdirs import user_config_path, user_data_path

from ...utils.cleanup import generate_cleanup_script, run_cleanup_script
from ...utils.logger import get_log_dir, get_logger

logger = get_logger(__name__)

APP_NAME = "WhisperNow"


def get_all_data_paths() -> List[Path]:
    """Return specific data paths to delete: models, logs, settings, history.

    Does NOT return entire parent directories to avoid accidentally deleting
    the app directory on Windows (where config_dir == data_dir).
    """
    paths = []

    config_dir = user_config_path(APP_NAME, appauthor=False)
    data_dir = user_data_path(APP_NAME, appauthor=False)

    # Settings file
    settings_file = config_dir / "settings.json"
    if settings_file.exists():
        paths.append(settings_file.resolve())

    # History file
    history_file = config_dir / "history.json"
    if history_file.exists():
        paths.append(history_file.resolve())

    # Models directory
    models_dir = data_dir / "models"
    if models_dir.exists():
        paths.append(models_dir.resolve())

    # Logs directory
    log_dir = get_log_dir()
    if log_dir.exists():
        paths.append(log_dir.resolve())

    return paths


def _delete_path(path: Path) -> None:
    """Delete a file or directory."""
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def clear_user_data(
    dry_run: bool = False, skip_logging: bool = False
) -> Tuple[bool, List[str]]:
    errors = []
    paths_to_delete = get_all_data_paths()

    if not paths_to_delete:
        return True, []

    for path in paths_to_delete:
        try:
            if dry_run:
                if not skip_logging:
                    logger.info(f"[DRY RUN] Would delete {path}")
            else:
                if not skip_logging:
                    logger.info(f"Deleting {path}")
                _delete_path(path)
        except Exception as e:
            error_msg = f"Failed to delete {path}: {e}"
            if not skip_logging:
                logger.error(error_msg)
            errors.append(error_msg)

    return len(errors) == 0, errors


def schedule_cleanup_and_exit() -> None:
    paths_to_delete = get_all_data_paths()
    if not paths_to_delete:
        return

    try:
        script_path = generate_cleanup_script(paths_to_delete)
        run_cleanup_script(script_path)
    except Exception as e:
        logger.error(f"Failed to schedule cleanup: {e}")
        pass
