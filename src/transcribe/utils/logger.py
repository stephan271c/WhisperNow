"""
Centralized logging configuration.

Provides a configured logger instance with rotating file handlers.
Logs are written to ~/.config/transcribe/logs/

Set LOG_TO_CONSOLE = True in config.py to also output logs to terminal.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def get_log_dir() -> Path:
    """Get the platform-appropriate log directory."""
    import platform as plat  # Use alias to avoid conflict with Path
    
    system = plat.system()
    
    if system == "Windows":
        base = Path.home() / "AppData" / "Roaming"
    elif system == "Darwin":  # macOS
        base = Path.home() / "Library" / "Application Support"
    else:  # Linux and others
        base = Path.home() / ".config"
    
    log_dir = base / "transcribe" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


# Global logger instance
_logger_instance: Optional[logging.Logger] = None


def get_logger(name: str = "transcribe") -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (defaults to root "transcribe" logger).
              Module names like "src.transcribe.app" are transformed to
              "transcribe.app" to maintain proper logger hierarchy.
        
    Returns:
        Configured logger instance with file handler (and optional console handler)
    """
    global _logger_instance
    
    # Transform "src.transcribe.xxx" to "transcribe.xxx" for proper hierarchy
    if name.startswith("src.transcribe."):
        name = name.replace("src.transcribe.", "transcribe.", 1)
    elif name == "src.transcribe":
        name = "transcribe"
    
    # Ensure the root "transcribe" logger is configured first
    # This must happen before returning any child logger so they inherit handlers
    if _logger_instance is None:
        # Import config here to avoid circular dependency
        from ..core.config import get_log_level, LOG_TO_CONSOLE
        
        root_logger = logging.getLogger("transcribe")
        
        # Prevent duplicate handlers during complex import chains
        if root_logger.handlers:
            _logger_instance = root_logger
        else:
            level = get_log_level()
            root_logger.setLevel(level)
            
            # Create formatter (shared by all handlers)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            
            # Create rotating file handler
            log_file = get_log_dir() / "app.log"
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding="utf-8"
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
            # Add console handler for development
            if LOG_TO_CONSOLE:
                console_handler = logging.StreamHandler(sys.stderr)
                console_handler.setLevel(level)
                console_handler.setFormatter(formatter)
                root_logger.addHandler(console_handler)
            
            # Prevent propagation to Python's root logger to avoid duplicate logs
            root_logger.propagate = False
            
            _logger_instance = root_logger
    
    # Return the root logger or a child logger
    if name == "transcribe":
        return _logger_instance
    
    # For child loggers (e.g., "transcribe.core.settings"), 
    # they inherit handlers from the parent "transcribe" logger via propagation
    return logging.getLogger(name)
