import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def get_log_dir() -> Path:
    import platform as plat  # Use alias to avoid conflict with Path

    system = plat.system()

    if system == "Windows":
        base = Path.home() / "AppData" / "Roaming"
    elif system == "Darwin":  # macOS
        base = Path.home() / "Library" / "Application Support"
    else:  # Linux and others
        base = Path.home() / ".config"

    log_dir = base / "whispernow" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


_logger_instance: Optional[logging.Logger] = None


def get_logger(name: str = "whispernow") -> logging.Logger:
    global _logger_instance

    if name.startswith("src.whispernow."):
        name = name.replace("src.whispernow.", "whispernow.", 1)
    elif name == "src.whispernow":
        name = "whispernow"

    if _logger_instance is None:
        from ..core.settings.config import LOG_TO_CONSOLE, get_log_level

        root_logger = logging.getLogger("whispernow")

        if root_logger.handlers:
            _logger_instance = root_logger
        else:
            level = get_log_level()
            root_logger.setLevel(level)

            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            log_file = get_log_dir() / "app.log"
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

            if LOG_TO_CONSOLE:
                console_handler = logging.StreamHandler(sys.stderr)
                console_handler.setLevel(level)
                console_handler.setFormatter(formatter)
                root_logger.addHandler(console_handler)

            root_logger.propagate = False

            _logger_instance = root_logger

    if name == "whispernow":
        return _logger_instance

    return logging.getLogger(name)


def shutdown_logging() -> None:
    """Shutdown logging and close all file handlers to release file locks."""
    global _logger_instance
    root_logger = logging.getLogger("whispernow")
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)
    _logger_instance = None
