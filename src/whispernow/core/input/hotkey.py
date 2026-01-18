"""
Hotkey listener for global keyboard shortcuts.

Listens for configurable hotkey combinations.
On macOS, uses native NSEvent API to avoid threading issues.
On other platforms, uses pynput.
"""

import sys
from typing import Optional, Set

from PySide6.QtCore import QObject, QTimer, Signal

from ...utils.logger import get_logger
from ..settings.settings import Settings, get_settings

logger = get_logger(__name__)


# Virtual key codes for macOS (from HIToolbox/Events.h)
MAC_KEYCODE_MAP = {
    "space": 49,
    "return": 36,
    "tab": 48,
    "escape": 53,
    "delete": 51,
    "f1": 122,
    "f2": 120,
    "f3": 99,
    "f4": 118,
    "f5": 96,
    "f6": 97,
    "f7": 98,
    "f8": 100,
    "f9": 101,
    "f10": 109,
    "f11": 103,
    "f12": 111,
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
        # Look up keycode: use map for named keys, ord() only for single characters
        if key_lower in MAC_KEYCODE_MAP:
            self._trigger_keycode = MAC_KEYCODE_MAP[key_lower]
        elif len(key_lower) == 1:
            self._trigger_keycode = ord(key_lower)
        else:
            # Fallback for unknown multi-character keys
            self._trigger_keycode = MAC_KEYCODE_MAP.get("space", 49)

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
    Native macOS hotkey listener using NSEvent.

    Runs on the main thread, avoiding the TSMGetInputSourceProperty crash.
    """

    def __init__(self, listener: HotkeyListener):
        self._listener = listener
        self._down_monitor = None
        self._up_monitor = None
        self._trigger_keycode = listener._trigger_keycode
        self._required_modifiers = listener._required_modifier_types

    def update_config(
        self, trigger_key: str, modifiers: Set[str], trigger_keycode: int
    ) -> None:
        self._trigger_keycode = trigger_keycode
        self._required_modifiers = modifiers

    def start(self) -> None:
        try:
            from AppKit import NSEvent, NSFlagsChangedMask, NSKeyDownMask, NSKeyUpMask

            self._down_monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
                NSKeyDownMask | NSFlagsChangedMask, self._on_key_down
            )
            self._up_monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
                NSKeyUpMask | NSFlagsChangedMask, self._on_key_up
            )
            logger.info("macOS native hotkey listener started")
        except ImportError as e:
            logger.error(f"Failed to import AppKit for macOS hotkey listener: {e}")
        except Exception as e:
            logger.error(f"Failed to start macOS hotkey listener: {e}")

    def stop(self) -> None:
        try:
            from AppKit import NSEvent

            if self._down_monitor:
                NSEvent.removeMonitor_(self._down_monitor)
                self._down_monitor = None
            if self._up_monitor:
                NSEvent.removeMonitor_(self._up_monitor)
                self._up_monitor = None
            logger.info("macOS native hotkey listener stopped")
        except Exception as e:
            logger.error(f"Failed to stop macOS hotkey listener: {e}")

    def _check_modifiers(self, modifier_flags: int) -> bool:
        for mod in self._required_modifiers:
            flag = MAC_MODIFIER_FLAGS.get(mod, 0)
            if not (modifier_flags & flag):
                return False
        return True

    def _on_key_down(self, event) -> None:
        try:
            keycode = event.keyCode()
            modifier_flags = event.modifierFlags()

            if keycode == self._trigger_keycode and self._check_modifiers(
                modifier_flags
            ):
                self._listener._on_hotkey_pressed()
        except Exception as e:
            logger.error(f"Error in macOS key down handler: {e}")

    def _on_key_up(self, event) -> None:
        try:
            keycode = event.keyCode()

            if keycode == self._trigger_keycode:
                self._listener._on_hotkey_released()
        except Exception as e:
            logger.error(f"Error in macOS key up handler: {e}")


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
