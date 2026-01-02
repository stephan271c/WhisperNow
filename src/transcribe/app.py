"""
Main application entry point.

Coordinates all components: tray, settings, recorder, and transcriber.
"""

import sys
import signal
from typing import Optional

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QThread, Signal, QObject
from pynput import keyboard
from pynput.keyboard import Controller as KeyboardController
import time

from . import __app_name__, __version__
from .core.settings import get_settings, Settings
from .core.recorder import AudioRecorder
from .core.transcriber import TranscriptionEngine, EngineState
from .ui.tray import SystemTray, TrayStatus
from .ui.main_window import SettingsWindow
from .utils.platform import check_accessibility_permissions, request_accessibility_permissions


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
        
        # Get hotkey config from settings
        settings = get_settings()
        self._setup_hotkey(settings)
    
    def _setup_hotkey(self, settings: Settings) -> None:
        """Configure the hotkey from settings."""
        # Build the set of required keys
        self._required_modifiers = set()
        
        for mod in settings.hotkey.modifiers:
            if mod == "ctrl":
                self._required_modifiers.add(keyboard.Key.ctrl_l)
                self._required_modifiers.add(keyboard.Key.ctrl_r)
            elif mod == "alt":
                self._required_modifiers.add(keyboard.Key.alt_l)
                self._required_modifiers.add(keyboard.Key.alt_r)
            elif mod == "shift":
                self._required_modifiers.add(keyboard.Key.shift_l)
                self._required_modifiers.add(keyboard.Key.shift_r)
            elif mod == "cmd" or mod == "meta":
                self._required_modifiers.add(keyboard.Key.cmd_l)
                self._required_modifiers.add(keyboard.Key.cmd_r)
        
        self._trigger_key = keyboard.Key.space
        if settings.hotkey.key != "space":
            # Handle other keys
            try:
                self._trigger_key = getattr(keyboard.Key, settings.hotkey.key)
            except AttributeError:
                # It's a character key
                self._trigger_key = keyboard.KeyCode.from_char(settings.hotkey.key)
    
    def _check_hotkey(self) -> bool:
        """Check if the hotkey combination is currently held."""
        # Check if trigger key is pressed
        if self._trigger_key not in self._pressed_keys:
            return False
        
        # Check if at least one of each required modifier type is pressed
        has_ctrl = not any("ctrl" in str(k) for k in self._required_modifiers) or \
                   keyboard.Key.ctrl_l in self._pressed_keys or \
                   keyboard.Key.ctrl_r in self._pressed_keys
        
        return has_ctrl
    
    def _on_press(self, key) -> None:
        """Handle key press events."""
        self._pressed_keys.add(key)
        
        if not self._is_hotkey_active and self._check_hotkey():
            self._is_hotkey_active = True
            self.hotkey_pressed.emit()
    
    def _on_release(self, key) -> None:
        """Handle key release events."""
        try:
            self._pressed_keys.remove(key)
        except KeyError:
            pass
        
        if self._is_hotkey_active and not self._check_hotkey():
            self._is_hotkey_active = False
            self.hotkey_released.emit()
    
    def start(self) -> None:
        """Start listening for hotkeys."""
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self._listener.start()
    
    def stop(self) -> None:
        """Stop listening for hotkeys."""
        if self._listener:
            self._listener.stop()
            self._listener = None


class TranscribeApp(QObject):
    """
    Main application class.
    
    Coordinates the system tray, settings window, audio recorder,
    transcription engine, and hotkey listener.
    """
    
    def __init__(self):
        super().__init__()
        
        self._settings = get_settings()
        self._settings_window: Optional[SettingsWindow] = None
        
        # Initialize components
        self._tray = SystemTray()
        self._recorder = AudioRecorder(
            sample_rate=self._settings.sample_rate,
            device=self._settings.input_device
        )
        self._transcriber = TranscriptionEngine(
            model_name=self._settings.model_name,
            use_gpu=self._settings.use_gpu,
            on_state_change=self._on_engine_state_change
        )
        self._hotkey_listener = HotkeyListener()
        self._keyboard_controller = KeyboardController()
        
        # Connect signals
        self._tray.settings_requested.connect(self._show_settings)
        self._tray.quit_requested.connect(self._quit)
        self._hotkey_listener.hotkey_pressed.connect(self._start_recording)
        self._hotkey_listener.hotkey_released.connect(self._stop_recording)
        
        # Check platform permissions
        self._check_permissions()
    
    def _check_permissions(self) -> None:
        """Check for required permissions (macOS accessibility)."""
        if not check_accessibility_permissions():
            # TODO: Show a dialog explaining the need for permissions
            request_accessibility_permissions()
    
    def _on_engine_state_change(self, state: EngineState, message: str) -> None:
        """Handle transcription engine state changes."""
        status_map = {
            EngineState.NOT_LOADED: TrayStatus.IDLE,
            EngineState.DOWNLOADING: TrayStatus.LOADING,
            EngineState.LOADING: TrayStatus.LOADING,
            EngineState.READY: TrayStatus.IDLE,
            EngineState.PROCESSING: TrayStatus.PROCESSING,
            EngineState.ERROR: TrayStatus.ERROR,
        }
        self._tray.set_status(status_map.get(state, TrayStatus.IDLE), message)
    
    def _start_recording(self) -> None:
        """Start recording audio."""
        self._tray.set_status(TrayStatus.RECORDING)
        self._recorder.start()
    
    def _stop_recording(self) -> None:
        """Stop recording and transcribe."""
        audio_data = self._recorder.stop()
        
        if audio_data is None or len(audio_data) == 0:
            self._tray.set_status(TrayStatus.IDLE)
            return
        
        # Transcribe the audio
        text = self._transcriber.transcribe(audio_data, self._settings.sample_rate)
        
        if text:
            # Type the result
            if self._settings.auto_type_result:
                self._type_text(text)
            
            # Show notification
            if self._settings.show_notifications:
                self._tray.show_notification("Transcription Complete", text[:100])
    
    def _type_text(self, text: str) -> None:
        """Type text with optional streaming effect."""
        time.sleep(0.1)  # Small delay for focus
        
        if self._settings.characters_per_second <= 0:
            self._keyboard_controller.type(text)
        else:
            delay = 1.0 / self._settings.characters_per_second
            for char in text:
                self._keyboard_controller.type(char)
                time.sleep(delay)
    
    def _show_settings(self) -> None:
        """Show the settings window."""
        if self._settings_window is None:
            self._settings_window = SettingsWindow()
            self._settings_window.settings_changed.connect(self._on_settings_changed)
        
        self._settings_window.show()
        self._settings_window.raise_()
        self._settings_window.activateWindow()
    
    def _on_settings_changed(self) -> None:
        """Handle settings changes."""
        # Reload settings
        self._settings = get_settings()
        
        # Update components
        self._recorder.sample_rate = self._settings.sample_rate
        self._recorder.device = self._settings.input_device
    
    def _quit(self) -> None:
        """Quit the application."""
        self._hotkey_listener.stop()
        self._transcriber.unload()
        self._tray.hide()
        QApplication.quit()
    
    def run(self) -> None:
        """Start the application."""
        # Load model in background
        self._transcriber.load_model()
        
        # Start hotkey listener
        self._hotkey_listener.start()
        
        # Show notification that app is ready
        if not self._settings.start_minimized:
            self._show_settings()


def main():
    """Application entry point."""
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Transcribe")
    app.setQuitOnLastWindowClosed(False)  # Keep running in tray
    
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, lambda *args: QApplication.quit())
    
    # Create and run the app
    transcribe_app = TranscribeApp()
    transcribe_app.run()
    
    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
