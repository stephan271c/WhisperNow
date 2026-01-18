"""
Hotkey listener for global keyboard shortcuts.

Listens for configurable hotkey combinations.
On macOS, uses subprocess-based CGEventTap to avoid threading issues.
On other platforms, uses pynput.
"""

import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional, Set

from PySide6.QtCore import QObject, QTimer, Signal

from ...utils.logger import get_logger
from ..settings.settings import Settings, get_settings

logger = get_logger(__name__)


# Virtual key codes for macOS (from HIToolbox/Events.h)
# These are physical key positions on ANSI US keyboard layout
MAC_KEYCODE_MAP = {
    # Letter keys (ANSI US layout positions)
    "a": 0x00,
    "s": 0x01,
    "d": 0x02,
    "f": 0x03,
    "h": 0x04,
    "g": 0x05,
    "z": 0x06,
    "x": 0x07,
    "c": 0x08,
    "v": 0x09,
    "b": 0x0B,
    "q": 0x0C,
    "w": 0x0D,
    "e": 0x0E,
    "r": 0x0F,
    "y": 0x10,
    "t": 0x11,
    "o": 0x1F,
    "u": 0x20,
    "i": 0x22,
    "p": 0x23,
    "l": 0x25,
    "j": 0x26,
    "k": 0x28,
    "n": 0x2D,
    "m": 0x2E,
    # Number keys (top row)
    "1": 0x12,
    "2": 0x13,
    "3": 0x14,
    "4": 0x15,
    "5": 0x17,
    "6": 0x16,
    "7": 0x1A,
    "8": 0x1C,
    "9": 0x19,
    "0": 0x1D,
    # Special keys
    "space": 0x31,
    "return": 0x24,
    "tab": 0x30,
    "escape": 0x35,
    "delete": 0x33,
    "enter": 0x24,  # Same as return
    # Function keys
    "f1": 0x7A,
    "f2": 0x78,
    "f3": 0x63,
    "f4": 0x76,
    "f5": 0x60,
    "f6": 0x61,
    "f7": 0x62,
    "f8": 0x64,
    "f9": 0x65,
    "f10": 0x6D,
    "f11": 0x67,
    "f12": 0x6F,
    # Arrow keys
    "left": 0x7B,
    "right": 0x7C,
    "down": 0x7D,
    "up": 0x7E,
    # Other common keys
    "home": 0x73,
    "end": 0x77,
    "pageup": 0x74,
    "pagedown": 0x79,
}

# Modifier flags for macOS NSEvent
MAC_MODIFIER_FLAGS = {
    "ctrl": 1 << 18,  # NSEventModifierFlagControl
    "alt": 1 << 19,  # NSEventModifierFlagOption
    "shift": 1 << 17,  # NSEventModifierFlagShift
    "cmd": 1 << 20,  # NSEventModifierFlagCommand
    "meta": 1 << 20,  # Same as cmd
}


class HotkeyListener(QObject):
    """
    Listens for hotkey combinations.

    On macOS, uses native NSEvent API (main thread safe).
    On other platforms, uses pynput in a background thread.

    Signals:
        hotkey_pressed: Emitted when the hotkey is pressed
        hotkey_released: Emitted when the hotkey is released
    """

    hotkey_pressed = Signal()
    hotkey_released = Signal()

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)

        self._is_hotkey_active = False

        settings = get_settings()
        self._setup_hotkey(settings)

        if sys.platform == "darwin":
            self._impl = _MacHotkeyListenerImpl(self)
        else:
            self._impl = _PynputHotkeyListenerImpl(self)

    def update_settings(self, settings: Settings) -> None:
        self._setup_hotkey(settings)
        self._impl.update_config(
            self._trigger_key, self._required_modifier_types, self._trigger_keycode
        )

    def _setup_hotkey(self, settings: Settings) -> None:
        self._required_modifier_types: Set[str] = set(settings.hotkey.modifiers)
        self._trigger_key = settings.hotkey.key
        key_lower = settings.hotkey.key.lower()

        if key_lower in MAC_KEYCODE_MAP:
            self._trigger_keycode = MAC_KEYCODE_MAP[key_lower]
        else:
            # Unknown key - log warning and fall back to space
            logger.warning(
                f"Unknown hotkey '{key_lower}' not in MAC_KEYCODE_MAP, "
                f"falling back to 'space'"
            )
            self._trigger_keycode = MAC_KEYCODE_MAP.get("space", 0x31)

    def start(self) -> None:
        self._impl.start()

    def stop(self) -> None:
        self._impl.stop()

    def _on_hotkey_pressed(self) -> None:
        if not self._is_hotkey_active:
            self._is_hotkey_active = True
            self.hotkey_pressed.emit()

    def _on_hotkey_released(self) -> None:
        if self._is_hotkey_active:
            self._is_hotkey_active = False
            self.hotkey_released.emit()


class _MacHotkeyListenerImpl:
    """
    Native macOS hotkey listener using subprocess-based CGEventTap.

    Spawns a separate Python process that runs CGEventTap on its main thread,
    avoiding HIToolbox threading violations that occur when integrating
    CGEventTap with Qt's event loop. Events are received via stdout IPC.
    """

    # Modifier flag masks from Quartz (defined here to avoid importing Quartz in main process)
    _MODIFIER_CONTROL = 0x40000  # kCGEventFlagMaskControl
    _MODIFIER_ALTERNATE = 0x80000  # kCGEventFlagMaskAlternate
    _MODIFIER_SHIFT = 0x20000  # kCGEventFlagMaskShift
    _MODIFIER_COMMAND = 0x100000  # kCGEventFlagMaskCommand

    def __init__(self, listener: HotkeyListener):
        self._listener = listener
        self._process: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._trigger_keycode = listener._trigger_keycode
        self._required_modifiers = listener._required_modifier_types
        self._is_hotkey_down = False
        self._stopping = False

    def update_config(
        self, trigger_key: str, modifiers: Set[str], trigger_keycode: int
    ) -> None:
        self._trigger_keycode = trigger_keycode
        self._required_modifiers = modifiers

    def start(self) -> None:
        self._stopping = False

        listener_script = Path(__file__).parent / "mac_listener.py"
        if not listener_script.exists():
            logger.error(f"macOS listener script not found: {listener_script}")
            return

        try:
            self._process = subprocess.Popen(
                [sys.executable, str(listener_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            self._reader_thread = threading.Thread(
                target=self._read_events, daemon=True
            )
            self._reader_thread.start()

            logger.info(
                f"macOS subprocess hotkey listener started: expecting keycode={self._trigger_keycode}, "
                f"modifiers={self._required_modifiers}"
            )
        except Exception as e:
            logger.error(f"Failed to start macOS hotkey subprocess: {e}")

    def stop(self) -> None:
        self._stopping = True

        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
            except Exception as e:
                logger.debug(f"Error stopping hotkey subprocess: {e}")
            finally:
                self._process = None

        logger.info("macOS subprocess hotkey listener stopped")

    def _read_events(self) -> None:
        if not self._process or not self._process.stdout:
            return

        try:
            for line in self._process.stdout:
                if self._stopping:
                    break

                line = line.strip()
                if not line:
                    continue

                if line == "READY":
                    logger.info("macOS hotkey subprocess is ready")
                    continue

                if line.startswith("ERROR:"):
                    error_msg = line[6:]
                    logger.error(f"macOS hotkey subprocess error: {error_msg}")
                    continue

                if line.startswith("KEY_DOWN:"):
                    self._handle_key_down(line)
                elif line.startswith("KEY_UP:"):
                    self._handle_key_up(line)

        except Exception as e:
            if not self._stopping:
                logger.error(f"Error reading from hotkey subprocess: {e}")

    def _handle_key_down(self, line: str) -> None:
        try:
            parts = line.split(":")
            if len(parts) >= 3:
                keycode = int(parts[1])
                flags = int(parts[2])

                logger.debug(
                    f"key down: keycode={keycode} (expecting {self._trigger_keycode}), "
                    f"modifiers=0x{flags:08x}"
                )

                if keycode == self._trigger_keycode and self._check_modifiers(flags):
                    if not self._is_hotkey_down:
                        self._is_hotkey_down = True
                        logger.info("Hotkey pressed!")
                        QTimer.singleShot(0, self._listener._on_hotkey_pressed)
        except (ValueError, IndexError) as e:
            logger.debug(f"Failed to parse KEY_DOWN: {line}, error: {e}")

    def _handle_key_up(self, line: str) -> None:
        try:
            parts = line.split(":")
            if len(parts) >= 2:
                keycode = int(parts[1])

                if keycode == self._trigger_keycode:
                    if self._is_hotkey_down:
                        self._is_hotkey_down = False
                        logger.debug("Hotkey released")
                        QTimer.singleShot(0, self._listener._on_hotkey_released)
        except (ValueError, IndexError) as e:
            logger.debug(f"Failed to parse KEY_UP: {line}, error: {e}")

    def _check_modifiers(self, flags: int) -> bool:
        for mod in self._required_modifiers:
            if mod == "ctrl":
                if not (flags & self._MODIFIER_CONTROL):
                    return False
            elif mod == "alt":
                if not (flags & self._MODIFIER_ALTERNATE):
                    return False
            elif mod == "shift":
                if not (flags & self._MODIFIER_SHIFT):
                    return False
            elif mod in ("cmd", "meta"):
                if not (flags & self._MODIFIER_COMMAND):
                    return False
        return True


class _PynputHotkeyListenerImpl:
    """
    Pynput-based hotkey listener for Windows and Linux.
    """

    def __init__(self, listener: HotkeyListener):
        from pynput import keyboard

        self._listener = listener
        self._keyboard_listener: Optional[keyboard.Listener] = None
        self._pressed_keys: set = set()

        self._trigger_key = keyboard.Key.space
        self._required_modifier_types = listener._required_modifier_types
        self._update_trigger_key(listener._trigger_key)

    def _update_trigger_key(self, key_name: str) -> None:
        from pynput import keyboard

        if key_name != "space":
            try:
                self._trigger_key = getattr(keyboard.Key, key_name)
            except AttributeError:
                self._trigger_key = keyboard.KeyCode.from_char(key_name)
        else:
            self._trigger_key = keyboard.Key.space

    def update_config(
        self, trigger_key: str, modifiers: Set[str], trigger_keycode: int
    ) -> None:
        self._required_modifier_types = modifiers
        self._update_trigger_key(trigger_key)

    def _check_hotkey(self) -> bool:
        from pynput import keyboard

        if self._trigger_key not in self._pressed_keys:
            return False

        for mod_type in self._required_modifier_types:
            is_pressed = False
            if mod_type == "ctrl":
                is_pressed = (
                    keyboard.Key.ctrl_l in self._pressed_keys
                    or keyboard.Key.ctrl_r in self._pressed_keys
                )
            elif mod_type == "alt":
                is_pressed = (
                    keyboard.Key.alt_l in self._pressed_keys
                    or keyboard.Key.alt_r in self._pressed_keys
                )
            elif mod_type == "shift":
                is_pressed = (
                    keyboard.Key.shift_l in self._pressed_keys
                    or keyboard.Key.shift_r in self._pressed_keys
                )
            elif mod_type in ("cmd", "meta"):
                is_pressed = (
                    keyboard.Key.cmd_l in self._pressed_keys
                    or keyboard.Key.cmd_r in self._pressed_keys
                )

            if not is_pressed:
                return False

        return True

    def _on_press(self, key) -> None:
        self._pressed_keys.add(key)

        if self._check_hotkey():
            self._listener._on_hotkey_pressed()

    def _on_release(self, key) -> None:
        try:
            self._pressed_keys.remove(key)
        except KeyError:
            pass

        if not self._check_hotkey():
            self._listener._on_hotkey_released()

    def start(self) -> None:
        from pynput import keyboard

        self._keyboard_listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._keyboard_listener.start()

        if hasattr(self._keyboard_listener, "IS_TRUSTED"):
            logger.info(
                f"Keyboard listener IS_TRUSTED: {self._keyboard_listener.IS_TRUSTED}"
            )
            if not self._keyboard_listener.IS_TRUSTED:
                logger.warning(
                    "Hotkey listener is NOT TRUSTED. Accessibility permissions not granted. "
                    "Add WhisperNow.app to System Settings > Privacy & Security > Accessibility"
                )

    def stop(self) -> None:
        if self._keyboard_listener:
            self._keyboard_listener.stop()
            self._keyboard_listener = None
