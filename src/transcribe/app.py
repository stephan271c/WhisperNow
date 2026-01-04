"""
Main application entry point.

Coordinates all components: tray, settings, recorder, and transcriber.
"""

import sys
import signal
from typing import Optional

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, QObject
from pynput.keyboard import Controller as KeyboardController

from . import __app_name__, __version__
from .core.settings import get_settings, Settings
from .core.recorder import AudioRecorder
from .core.transcriber import TranscriptionEngine, EngineState
from .core.hotkey import HotkeyListener
from .ui.tray import SystemTray, TrayStatus
from .ui.main_window import SettingsWindow
from .ui.download_dialog import DownloadDialog
from .ui.setup_wizard import SetupWizard
from .ui.permissions_dialog import PermissionsDialog
from .utils.platform import (
    get_platform,
    check_accessibility_permissions,
    request_accessibility_permissions,
    set_autostart
)
from .utils.logger import get_logger

logger = get_logger(__name__)


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
            on_state_change=self._on_engine_state_change,
            on_download_progress=self._on_download_progress
        )
        self._hotkey_listener = HotkeyListener()
        self._keyboard_controller = KeyboardController()
        self._download_dialog: Optional[DownloadDialog] = None
        
        # Typing state for non-blocking character output
        self._typing_text: str = ""
        self._typing_index: int = 0
        self._typing_timer: Optional[QTimer] = None
        
        # Connect signals
        self._tray.settings_requested.connect(self._show_settings)
        self._tray.quit_requested.connect(self._quit)
        self._hotkey_listener.hotkey_pressed.connect(self._start_recording)
        self._hotkey_listener.hotkey_released.connect(self._stop_recording)
        
        # Check platform permissions
        self._check_permissions()
    
    def _check_permissions(self) -> None:
        """Check for required permissions (macOS accessibility)."""
        # Only relevant on macOS
        if get_platform() != "macos":
            return
        
        # If already granted and recorded, verify it's still valid
        if self._settings.accessibility_permissions_granted:
            if check_accessibility_permissions():
                return  # Still valid
            else:
                # Permission was revoked, need to re-request
                logger.warning("Accessibility permission was revoked, prompting user")
        
        # Check current status
        if check_accessibility_permissions():
            self._settings.accessibility_permissions_granted = True
            self._settings.save()
            logger.info("Accessibility permissions already granted")
            return
        
        # Show dialog explaining permissions
        logger.info("Showing accessibility permissions dialog")
        dialog = PermissionsDialog()
        dialog.exec()
        
        # Update settings with result
        granted = check_accessibility_permissions()
        self._settings.accessibility_permissions_granted = granted
        self._settings.save()
        
        if granted:
            logger.info("User granted accessibility permissions")
        else:
            logger.warning("User continued without accessibility permissions")
    
    def _on_engine_state_change(self, state: EngineState, message: str) -> None:
        """Handle transcription engine state changes."""
        # Skip PROCESSING state - we stay IDLE (green) during transcription
        # The only color change should be RECORDING (red) when hotkeys are held
        if state == EngineState.PROCESSING:
            return
        
        status_map = {
            EngineState.NOT_LOADED: TrayStatus.IDLE,
            EngineState.DOWNLOADING: TrayStatus.LOADING,
            EngineState.LOADING: TrayStatus.LOADING,
            EngineState.READY: TrayStatus.IDLE,
            EngineState.ERROR: TrayStatus.ERROR,
        }
        self._tray.set_status(status_map.get(state, TrayStatus.IDLE), message)
        
        # Show/hide download dialog
        if state == EngineState.DOWNLOADING:
            self._show_download_dialog()
        elif state in (EngineState.READY, EngineState.ERROR, EngineState.NOT_LOADED):
            self._hide_download_dialog(state == EngineState.READY)
    
    def _on_download_progress(self, progress: float) -> None:
        """Handle model download progress updates."""
        if self._download_dialog is not None:
            self._download_dialog.set_progress(progress)
    
    def _show_download_dialog(self) -> None:
        """Show the model download progress dialog."""
        if self._download_dialog is None:
            self._download_dialog = DownloadDialog(
                model_name=self._settings.model_name
            )
            self._download_dialog.cancelled.connect(self._on_download_cancelled)
        self._download_dialog.show()
    
    def _hide_download_dialog(self, success: bool = True) -> None:
        """Hide the download dialog."""
        if self._download_dialog is not None:
            self._download_dialog.finish(success)
            self._download_dialog = None
    
    def _on_download_cancelled(self) -> None:
        """Handle download cancellation."""
        self._transcriber.unload()
        self._hide_download_dialog(success=False)
    
    def _start_recording(self) -> None:
        """Start recording audio."""
        logger.debug("Starting audio recording")
        if self._recorder.start():
            self._tray.set_status(TrayStatus.RECORDING)
            logger.info("Recording started")
        else:
            error_msg = self._recorder.last_error or "Failed to start recording"
            self._tray.set_status(TrayStatus.ERROR, error_msg)
            logger.error(f"Recording error: {error_msg}")
    
    def _stop_recording(self) -> None:
        """Stop recording and transcribe."""
        import time
        start_time = time.time()
        
        logger.debug("Stopping audio recording")
        audio_data = self._recorder.stop()
        
        if audio_data is None or len(audio_data) == 0:
            logger.warning("No audio data captured")
            self._tray.set_status(TrayStatus.IDLE)
            return
        
        logger.info(f"Captured {len(audio_data)} audio samples, starting transcription")
        
        # Transcribe the audio
        text = self._transcriber.transcribe(audio_data, self._settings.sample_rate)
        
        duration = time.time() - start_time
        
        if text:
            logger.info(f"Transcription completed in {duration:.2f}s: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            # Type the result
            self._type_text(text)
        else:
            logger.warning(f"Transcription returned empty result after {duration:.2f}s")
            self._tray.set_status(TrayStatus.IDLE)
    
    def _type_text(self, text: str) -> None:
        """Type text using QTimer for non-blocking output."""
        if self._settings.instant_type or self._settings.characters_per_second <= 0:
            # Instant typing via clipboard paste - avoids pynput race conditions
            self._paste_text(text)
            self._finish_typing()
        else:
            # Character-by-character typing with timer (tray stays green)
            self._typing_text = text
            self._typing_index = 0
            
            # Create timer for character output
            delay_ms = int(1000.0 / self._settings.characters_per_second)
            self._typing_timer = QTimer(self)
            self._typing_timer.timeout.connect(self._type_next_char)
            self._typing_timer.start(delay_ms)
    
    def _type_next_char(self) -> None:
        """Type the next character in the queue."""
        if self._typing_index < len(self._typing_text):
            char = self._typing_text[self._typing_index]
            self._keyboard_controller.type(char)
            self._typing_index += 1
        else:
            # Done typing
            if self._typing_timer:
                self._typing_timer.stop()
                self._typing_timer.deleteLater()
                self._typing_timer = None
            self._typing_text = ""
            self._typing_index = 0
            self._finish_typing()
    
    def _paste_text(self, text: str) -> None:
        """Paste text using clipboard - more reliable than keyboard.type() on Linux."""
        from pynput.keyboard import Key
        import subprocess
        import time
        
        logger.debug(f"Pasting text via clipboard: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Save current clipboard
        try:
            result = subprocess.run(
                ['xclip', '-selection', 'clipboard', '-o'],
                capture_output=True, text=True, timeout=1
            )
            old_clipboard = result.stdout if result.returncode == 0 else ""
        except (subprocess.TimeoutExpired, FileNotFoundError):
            old_clipboard = ""
        
        # Set new clipboard content using xclip
        try:
            subprocess.run(
                ['xclip', '-selection', 'clipboard'],
                input=text, text=True, timeout=1, check=True
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError) as e:
            logger.error(f"Failed to set clipboard: {e}")
            # Fallback to direct typing
            self._keyboard_controller.type(text)
            return
        
        # Small delay to ensure clipboard is ready
        time.sleep(0.05)
        
        # Simulate Ctrl+V
        with self._keyboard_controller.pressed(Key.ctrl):
            self._keyboard_controller.tap('v')
        
        # Small delay before restoring clipboard
        time.sleep(0.1)
        
        # Restore old clipboard
        if old_clipboard:
            try:
                subprocess.run(
                    ['xclip', '-selection', 'clipboard'],
                    input=old_clipboard, text=True, timeout=1
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
    
    def _finish_typing(self) -> None:
        """Called when typing is complete. Set idle."""
        self._tray.set_status(TrayStatus.IDLE)
    
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
        self._hotkey_listener.update_settings(self._settings)
        
        # Apply autostart setting
        set_autostart(self._settings.auto_start_on_login, "Transcribe")
    
    def _quit(self) -> None:
        """Quit the application."""
        logger.info("Shutting down application")
        self._hotkey_listener.stop()
        self._transcriber.unload()
        self._tray.hide()
        QApplication.quit()
        logger.info("Application shutdown complete")
    
    def run(self) -> None:
        """Start the application."""
        logger.info(f"Starting {__app_name__} v{__version__}")
        logger.info(f"Settings: model={self._settings.model_name}, sample_rate={self._settings.sample_rate}")
        
        # Load model in background
        logger.info("Loading transcription model...")
        self._transcriber.load_model()
        
        # Start hotkey listener
        logger.info(f"Starting hotkey listener: {self._settings.hotkey.to_display_string()}")
        self._hotkey_listener.start()
        
        # Show notification that app is ready
        if not self._settings.start_minimized:
            self._show_settings()
        
        logger.info("Application initialization complete")


def main():
    """Application entry point."""
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Transcribe")
    app.setQuitOnLastWindowClosed(False)  # Keep running in tray
    
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, lambda *args: QApplication.quit())
    
    # Check for first run
    settings = get_settings()
    if not settings.first_run_complete:
        wizard = SetupWizard()
        if wizard.exec() != SetupWizard.Accepted:
            # User cancelled the wizard
            sys.exit(0)
    
    # Create and run the app
    transcribe_app = TranscribeApp()
    transcribe_app.run()
    
    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
