"""
Text output controller for outputting transcribed text.

Handles platform-specific text output via clipboard paste.
"""

import platform
import subprocess
import time
from typing import Optional

from pynput.keyboard import Controller as KeyboardController
from pynput.keyboard import Key

from ...utils.logger import get_logger
from ...utils.platform import get_subprocess_kwargs

logger = get_logger(__name__)


class TextOutputController:

    def __init__(self):
        self._keyboard = KeyboardController()

    def output_text(self, text: str) -> None:
        logger.debug(
            f"Pasting text via clipboard: '{text[:50]}{'...' if len(text) > 50 else ''}'"
        )

        system = platform.system()
        if system == "Linux":
            copy_cmd = ["xclip", "-selection", "clipboard"]
            paste_cmd = ["xclip", "-selection", "clipboard", "-o"]
        elif system == "Windows":
            copy_cmd = ["clip"]
            paste_cmd = ["powershell", "-command", "Get-Clipboard"]
        else:
            logger.error(f"Unsupported platform: {system}")
            return

        old_clipboard = self._get_clipboard(paste_cmd)

        if not self._set_clipboard(copy_cmd, text):
            logger.error("Failed to set clipboard, cannot paste text")
            return

        time.sleep(0.05)
        with self._keyboard.pressed(Key.ctrl):
            self._keyboard.tap("v")
        time.sleep(0.1)

        if old_clipboard:
            self._set_clipboard(copy_cmd, old_clipboard)

    def _get_clipboard(self, paste_cmd: list) -> str:
        try:
            result = subprocess.run(
                paste_cmd,
                **get_subprocess_kwargs(capture_output=True, text=True, timeout=1),
            )
            return result.stdout if result.returncode == 0 else ""
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug(f"Failed to get clipboard: {e}")
            return ""
        except Exception as e:
            logger.warning(f"Unexpected error getting clipboard: {e}")
            return ""

    def _set_clipboard(self, copy_cmd: list, text: str) -> bool:
        try:
            subprocess.run(
                copy_cmd,
                **get_subprocess_kwargs(input=text, text=True, timeout=1, check=True),
            )
            return True
        except (
            subprocess.TimeoutExpired,
            FileNotFoundError,
            subprocess.CalledProcessError,
        ) as e:
            logger.error(f"Failed to set clipboard: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error setting clipboard: {e}")
            return False
