"""
Hotkey listener for global keyboard shortcuts.

Uses pynput for Windows and Linux.
"""

from typing import Optional, Set

from PySide6.QtCore import QObject, Signal

from ...utils.logger import get_logger
from ..settings.settings import Settings, get_settings

logger = get_logger(__name__)


class HotkeyListener(QObject):
    """
    Listens for hotkey combinations.

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

        self._impl = _PynputHotkeyListenerImpl(self)

    def update_settings(self, settings: Settings) -> None:
        self._setup_hotkey(settings)
        self._impl.update_config(self._trigger_key, self._required_modifier_types)

    def _setup_hotkey(self, settings: Settings) -> None:
        self._required_modifier_types: Set[str] = set(settings.hotkey.modifiers)
        self._trigger_key = settings.hotkey.key

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

    def update_config(self, trigger_key: str, modifiers: Set[str]) -> None:
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
                    "Hotkey listener is NOT TRUSTED. Accessibility permissions not granted."
                )

    def stop(self) -> None:
        if self._keyboard_listener:
            self._keyboard_listener.stop()
            self._keyboard_listener = None
