import time

import pyperclip
from pynput.keyboard import Controller as KeyboardController
from pynput.keyboard import Key

from ...utils.logger import get_logger

logger = get_logger(__name__)


class TextOutputController:

    def __init__(self):
        self._keyboard = KeyboardController()

    def output_text(self, text: str) -> None:
        logger.debug(
            f"Pasting text via clipboard: '{text[:50]}{'...' if len(text) > 50 else ''}'"
        )

        try:
            # pyperclip handles the platform-specific clipboard operations
            # It generally supports Unicode well on Windows and Linux
            pyperclip.copy(text)
        except Exception as e:
            logger.error(f"Failed to copy text to clipboard: {e}")
            return

        time.sleep(0.05)
        with self._keyboard.pressed(Key.ctrl):
            self._keyboard.tap("v")
        time.sleep(0.1)
