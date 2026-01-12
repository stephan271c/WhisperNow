"""
Text output controller for typing transcribed text.

Handles platform-specific text output via clipboard paste or
character-by-character keyboard typing.
"""

import platform
import subprocess
import time
from typing import Callable, Optional

from pynput.keyboard import Controller as KeyboardController, Key

from ...utils.logger import get_logger

logger = get_logger(__name__)


class TextOutputController:
    """
    Controller for outputting transcribed text to the active application.
    
    Supports two modes:
    - Instant paste: Uses clipboard + Ctrl/Cmd+V for fast output
    - Character typing: Types one character at a time for natural appearance
    """
    
    def __init__(
        self,
        on_complete: Optional[Callable[[], None]] = None
    ):
        """
        Initialize the text output controller.
        
        Args:
            on_complete: Callback invoked when text output completes
        """
        self._keyboard = KeyboardController()
        self._on_complete = on_complete
    
    def output_text(self, text: str, instant: bool = True) -> None:
        """
        Output text to the active application.
        
        Args:
            text: The text to output
            instant: If True, use clipboard paste. If False, type character-by-character.
        """
        if instant:
            self._paste_text(text)
        else:
            # For character-by-character, caller should use Qt timer
            # This is a simple fallback that blocks
            self._type_text_blocking(text)
        
        if self._on_complete:
            self._on_complete()
    
    def type_character(self, char: str) -> None:
        self._keyboard.type(char)
    
    def _type_text_blocking(self, text: str) -> None:
        self._keyboard.type(text)
    
    def _paste_text(self, text: str) -> None:
        logger.debug(f"Pasting text via clipboard: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        system = platform.system()
        
        if system == "Linux":
            copy_cmd = ['xclip', '-selection', 'clipboard']
            paste_cmd = ['xclip', '-selection', 'clipboard', '-o']
            paste_key = Key.ctrl
        elif system == "Darwin":  # macOS
            copy_cmd = ['pbcopy']
            paste_cmd = ['pbpaste']
            paste_key = Key.cmd
        elif system == "Windows":
            copy_cmd = ['clip']
            paste_cmd = ['powershell', '-command', 'Get-Clipboard']
            paste_key = Key.ctrl
        else:
            logger.warning(f"Unknown platform {system}, falling back to direct typing")
            self._keyboard.type(text)
            return
        
        old_clipboard = self._get_clipboard(paste_cmd)
        
        if not self._set_clipboard(copy_cmd, text):
            # Fallback to direct typing
            self._keyboard.type(text)
            return
        
        time.sleep(0.05)
        
        with self._keyboard.pressed(paste_key):
            self._keyboard.tap('v')
        
        time.sleep(0.1)
        
        if old_clipboard:
            self._set_clipboard(copy_cmd, old_clipboard)
    
    def _get_clipboard(self, paste_cmd: list) -> str:
        try:
            result = subprocess.run(
                paste_cmd,
                capture_output=True, text=True, timeout=1
            )
            return result.stdout if result.returncode == 0 else ""
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return ""
    
    def _set_clipboard(self, copy_cmd: list, text: str) -> bool:
        try:
            subprocess.run(
                copy_cmd,
                input=text, text=True, timeout=1, check=True
            )
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError) as e:
            logger.error(f"Failed to set clipboard: {e}")
            return False
