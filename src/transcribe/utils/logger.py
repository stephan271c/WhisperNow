"""
Centralized logging configuration.

Provides a configured logger instance with rotating file handlers.
Logs are written to ~/.config/transcribe/logs/
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def get_log_dir() -> Path:
    """Get the platform-appropriate log directory."""
    # Import here to avoid circular dependency
    from ..core.settings import get_config_dir
    
    log_dir = get_config_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


# Global logger instance
_logger_instance: Optional[logging.Logger] = None


def get_logger(name: str = "transcribe") -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (defaults to root "transcribe" logger)
        
    Returns:
        Configured logger instance with file handler
    """
    global _logger_instance
    
    # Return existing logger if already configured
    if _logger_instance is not None and name == "transcribe":
        return _logger_instance
    
    # Create or get logger
    logger = logging.getLogger(name)
    
    # Only configure the root logger once
    if name == "transcribe" and _logger_instance is None:
        logger.setLevel(logging.INFO)
        
        # Create rotating file handler
        log_file = get_log_dir() / "app.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        # Prevent propagation to root logger to avoid duplicate logs
        logger.propagate = False
        
        _logger_instance = logger
    
    return logger
