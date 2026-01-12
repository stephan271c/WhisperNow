"""
Centralized application configuration.

Edit the variables below to configure development settings.
"""

import logging

# =============================================================================
# DEVELOPMENT SETTINGS - Edit these for local development
# =============================================================================
LOG_LEVEL = "DEBUG"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_TO_CONSOLE = True  # Set to True to output logs to terminal
# =============================================================================

# =============================================================================
# HISTORY SETTINGS
# =============================================================================
MAX_HISTORY_ENTRIES = 20  # Number of transcription history records to keep
# =============================================================================


def get_log_level() -> int:
    """Get the logging level as an integer."""
    return getattr(logging, LOG_LEVEL.upper(), logging.INFO)
