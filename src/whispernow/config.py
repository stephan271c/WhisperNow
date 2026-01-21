"""
Development configuration.

Edit these settings to configure logging and debugging behavior.
"""

import logging

# =============================================================================
# DEVELOPMENT SETTINGS - Edit these for local development
# =============================================================================
LOG_LEVEL = "DEBUG"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_TO_CONSOLE = True  # Set to True to output logs to terminal
# =============================================================================


def get_log_level() -> int:
    """Get the logging level as an integer."""
    return getattr(logging, LOG_LEVEL.upper(), logging.INFO)
