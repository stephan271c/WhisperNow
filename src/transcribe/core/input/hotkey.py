"""
Hotkey listener for global keyboard shortcuts.

Listens for configurable hotkey combinations using pynput.
Emits Qt signals when hotkeys are pressed/released.
"""

from typing import Optional

from pynput import keyboard
from PySide6.QtCore import QObject, Signal

from ..settings.settings import Settings, get_settings


class HotkeyListener(QObject):
    """
    Listens for hotkey combinations using pynput.

    Runs in a separate thread to avoid blocking the Qt event loop.

    Signals:
        hotkey_pressed: Emitted when the hotkey is pressed
        hotkey_released: Emitted when the hotkey is released
    """

    hotkey_pressed = Signal()
    hotkey_released = Signal()

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)

        self._listener: Optional[keyboard.Listener] = None
        self._pressed_keys: set = set()
        self._is_hotkey_active = False

        settings = get_settings()
        self._setup_hotkey(settings)

    def update_settings(self, settings: Settings) -> None:
        self._setup_hotkey(settings)

    def _setup_hotkey(self, settings: Settings) -> None:
        self._required_modifier_types = set(settings.hotkey.modifiers)

        self._trigger_key = keyboard.Key.space
        if settings.hotkey.key != "space":
            try:
                self._trigger_key = getattr(keyboard.Key, settings.hotkey.key)
            except AttributeError:
                self._trigger_key = keyboard.KeyCode.from_char(settings.hotkey.key)

    def _check_hotkey(self) -> bool:
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

        if not self._is_hotkey_active and self._check_hotkey():
            self._is_hotkey_active = True
            self.hotkey_pressed.emit()

    def _on_release(self, key) -> None:
        try:
            self._pressed_keys.remove(key)
        except KeyError:
            pass

        if self._is_hotkey_active and not self._check_hotkey():
            self._is_hotkey_active = False
            self.hotkey_released.emit()

    def start(self) -> None:
        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._listener.start()

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
            self._listener = None
