"""
Application runtime.

Coordinates all components: tray, settings, recorder, and transcriber.
"""

import sys
import signal
from datetime import datetime
from typing import Optional, Tuple

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, QObject

from . import __app_name__, __version__
from .core.settings import get_settings, TranscriptionRecord, add_history_record
from .core.audio import AudioRecorder
from .core.asr import TranscriptionEngine, EngineState, ModelLoaderThread
from .core.input import HotkeyListener
from .core.transcript_processor import LLMProcessor
from .core.output import TextOutputController
from .ui.tray import SystemTray, TrayStatus
from .ui.main_window import SettingsWindow
from .ui.download_dialog import DownloadDialog
from .ui.setup_wizard import SetupWizard
from .ui.recording_toast import RecordingToast
from .utils.platform import set_autostart, check_and_request_permissions
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
        self._recording_toast = RecordingToast(__app_name__)
        self._latest_audio_level = 0.0
        self._latest_audio_spectrum: list[float] = []
        self._audio_level_timer = QTimer(self)
        self._audio_level_timer.setInterval(50)
        self._audio_level_timer.timeout.connect(self._update_audio_level)
        self._recorder = AudioRecorder(
            sample_rate=self._settings.sample_rate,
            device=self._settings.input_device,
            on_audio_level=self._capture_audio_level,
            on_audio_spectrum=self._capture_audio_spectrum,
        )
        self._transcriber = TranscriptionEngine(
            model_name=self._settings.model_name,
            use_gpu=self._settings.use_gpu,
            on_state_change=self._on_engine_state_change,
            on_download_progress=self._on_download_progress
        )
        self._hotkey_listener = HotkeyListener()
        self._text_output = TextOutputController(on_complete=self._finish_typing)
        self._download_dialog: Optional[DownloadDialog] = None
        self._model_loader_thread: Optional[ModelLoaderThread] = None
        
        # Typing state for non-blocking character output
        self._typing_text: str = ""
        self._typing_index: int = 0
        self._typing_timer: Optional[QTimer] = None
        
        # LLM processor for text enhancement
        self._llm_processor: Optional[LLMProcessor] = None
        self._init_llm_processor()
        
        # Connect signals
        self._tray.settings_requested.connect(self._show_settings)
        self._tray.quit_requested.connect(self._quit)
        self._hotkey_listener.hotkey_pressed.connect(self._start_recording)
        self._hotkey_listener.hotkey_released.connect(self._stop_recording)
        
        # Check platform permissions
        check_and_request_permissions(self._settings)
    
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
            self._latest_audio_level = 0.0
            self._latest_audio_spectrum = []
            self._recording_toast.show_recording()
            self._audio_level_timer.start()
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
        self._audio_level_timer.stop()
        self._recording_toast.hide_recording()
        self._recording_toast.set_level(0.0)
        self._recording_toast.set_spectrum([])
        
        if audio_data is None or len(audio_data) == 0:
            logger.warning("No audio data captured")
            self._tray.set_status(TrayStatus.IDLE)
            return
        
        logger.info(f"Captured {len(audio_data)} audio samples, starting transcription")
        
        # Transcribe the audio (auto-chunks if over 30 seconds)
        raw_text = self._transcriber.transcribe_chunked(audio_data, self._settings.sample_rate)
        
        duration = time.time() - start_time
        
        if raw_text:
            logger.info(f"Transcription completed in {duration:.2f}s: '{raw_text[:50]}{'...' if len(raw_text) > 50 else ''}'")
            
            # Apply vocabulary replacements
            from .core.transcript_processor import apply_vocabulary_replacements
            processed_text = apply_vocabulary_replacements(
                raw_text,
                self._settings.vocabulary_replacements
            )
            
            # Apply LLM enhancement if active
            final_text, enhanced_text, enhancement_name, cost = self._apply_enhancement(processed_text)
            
            # If vocabulary replacements were applied but no LLM enhancement,
            # record the processed text as the "enhanced" version
            if enhanced_text is None and processed_text != raw_text:
                enhanced_text = processed_text
                enhancement_name = "Vocabulary Replacement"
            
            # Record transcription to history
            self._record_transcription(raw_text, enhanced_text, enhancement_name, cost)
            
            # Type the result
            self._type_text(final_text)
        else:
            logger.warning(f"Transcription returned empty result after {duration:.2f}s")
            self._tray.set_status(TrayStatus.IDLE)
    
    def _init_llm_processor(self) -> None:
        """Initialize or reinitialize the LLM processor."""
        # Initialize if we have an API key, active enhancement, or using local Ollama
        has_config = (
            self._settings.llm_api_key or 
            self._settings.active_enhancement_id or
            self._settings.llm_provider == "ollama"
        )
        if has_config:
            model_name = LLMProcessor.format_model_name(
                self._settings.llm_model,
                self._settings.llm_provider
            )
            self._llm_processor = LLMProcessor(
                model=model_name,
                api_key=self._settings.llm_api_key,
                api_base=self._settings.llm_api_base
            )
            logger.info(f"LLM processor initialized: model={model_name}, api_base={self._settings.llm_api_base}")
        else:
            self._llm_processor = None
    
    def _apply_enhancement(self, text: str) -> Tuple[str, Optional[str], Optional[str], Optional[float]]:
        """
        Apply the active enhancement to the text, if configured.
        
        Returns:
            Tuple of (final_text, enhanced_text, enhancement_name, cost_usd).
            If no enhancement applied, enhanced_text/enhancement_name/cost will be None.
        """
        if not self._llm_processor:
            return text, None, None, None
        
        enhancement = self._settings.get_active_enhancement()
        if not enhancement:
            return text, None, None, None
        
        if not self._llm_processor.is_configured():
            logger.warning("LLM processor not configured (no API key), skipping enhancement")
            return text, None, None, None
        
        logger.info(f"Applying enhancement: {enhancement.title}")
        response = self._llm_processor.process(text, enhancement)
        
        return response.content, response.content, enhancement.title, response.cost_usd
    
    def _record_transcription(
        self,
        raw_text: str,
        enhanced_text: Optional[str],
        enhancement_name: Optional[str],
        cost_usd: Optional[float]
    ) -> None:
        """Record a transcription to history."""
        record = TranscriptionRecord(
            timestamp=datetime.now().isoformat(),
            raw_text=raw_text,
            enhanced_text=enhanced_text,
            enhancement_name=enhancement_name,
            cost_usd=cost_usd
        )
        add_history_record(record)
        logger.debug(f"Recorded transcription to history: {len(raw_text)} chars")
    
    def _type_text(self, text: str) -> None:
        """Type text using QTimer for non-blocking output."""
        if self._settings.instant_type or self._settings.characters_per_second <= 0:
            # Instant typing via clipboard paste
            self._text_output.output_text(text, instant=True)
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
            self._text_output.type_character(char)
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
        # Track previous values for comparison
        old_model = self._transcriber.model_name
        old_gpu = self._transcriber.use_gpu
        
        # Reload settings
        self._settings = get_settings()
        
        # Update components
        self._recorder.sample_rate = self._settings.sample_rate
        self._recorder.device = self._settings.input_device
        self._hotkey_listener.update_settings(self._settings)
        
        # Reinitialize LLM processor with new settings
        self._init_llm_processor()
        
        # Reload transcriber if model or GPU setting changed
        model_changed = old_model != self._settings.model_name
        gpu_changed = old_gpu != self._settings.use_gpu
        if model_changed or gpu_changed:
            change_reason = []
            if model_changed:
                change_reason.append(f"model: {old_model} -> {self._settings.model_name}")
            if gpu_changed:
                change_reason.append(f"GPU: {old_gpu} -> {self._settings.use_gpu}")
            logger.info(f"Reloading transcription engine ({', '.join(change_reason)})")
            
            # Unload current model
            self._transcriber.unload()
            
            # Show loading state in settings window
            if self._settings_window is not None:
                self._settings_window.set_loading(True)
            
            # Start background loading
            self._start_model_loading(
                self._settings.model_name,
                self._settings.use_gpu
            )
        
        # Apply autostart setting
        set_autostart(self._settings.auto_start_on_login, "WhisperNow")
    
    def _quit(self) -> None:
        """Quit the application."""
        logger.info("Shutting down application")
        self._hotkey_listener.stop()
        self._transcriber.unload()
        self._tray.hide()
        self._recording_toast.hide()
        QApplication.quit()
        logger.info("Application shutdown complete")

    def _capture_audio_level(self, level: float) -> None:
        """Capture audio level updates from the recorder callback."""
        self._latest_audio_level = level

    def _capture_audio_spectrum(self, spectrum: list[float]) -> None:
        """Capture spectrum updates from the recorder callback."""
        self._latest_audio_spectrum = spectrum

    def _update_audio_level(self) -> None:
        """Push the latest audio level into the toast animation."""
        self._recording_toast.set_level(self._latest_audio_level)
        self._recording_toast.set_spectrum(self._latest_audio_spectrum)
    
    def _start_model_loading(self, model_name: str, use_gpu: bool) -> None:
        """Start loading a model in the background."""
        # Cancel any existing loading thread
        if self._model_loader_thread is not None and self._model_loader_thread.isRunning():
            self._model_loader_thread.wait()
        
        # Create and start new loader thread
        self._model_loader_thread = ModelLoaderThread(
            model_name=model_name,
            use_gpu=use_gpu,
            parent=self
        )
        self._model_loader_thread.state_changed.connect(self._on_engine_state_change)
        self._model_loader_thread.progress.connect(self._on_download_progress)
        self._model_loader_thread.finished.connect(self._on_model_loaded)
        self._model_loader_thread.start()
    
    def _on_model_loaded(self, success: bool, message: str) -> None:
        """Handle completion of background model loading."""
        if success and self._model_loader_thread is not None:
            # Transfer the loaded engine
            self._transcriber = self._model_loader_thread.engine
            logger.info(f"Model loading complete: {message}")
        else:
            logger.error(f"Model loading failed: {message}")
        
        # Hide loading state in settings window and refresh model list
        if self._settings_window is not None:
            self._settings_window.set_loading(False)
            # Refresh model list so newly downloaded models appear in dropdown
            if success:
                self._settings_window.refresh_asr_model_list()
        
        # Update tray status
        if success:
            self._tray.set_status(TrayStatus.IDLE, message)
        else:
            self._tray.set_status(TrayStatus.ERROR, message)
    
    def run(self) -> None:
        """Start the application."""
        logger.info(f"Starting {__app_name__} v{__version__}")
        logger.info(f"Settings: model={self._settings.model_name}, sample_rate={self._settings.sample_rate}")
        
        # Start hotkey listener
        logger.info(f"Starting hotkey listener: {self._settings.hotkey.to_display_string()}")
        self._hotkey_listener.start()
        
        # Show window first (if not minimized), then start loading
        if not self._settings.start_minimized:
            self._show_settings()
        
        # Defer model loading to after the event loop starts
        # This allows the UI to appear first, then show loading state
        logger.info("Deferring model load to after UI is shown...")
        QTimer.singleShot(100, self._start_deferred_model_loading)
        
        logger.info("Application initialization complete")
    
    def _start_deferred_model_loading(self) -> None:
        """Start model loading after UI has been displayed."""
        logger.info("Starting deferred transcription model loading...")
        self._start_model_loading(
            self._settings.model_name,
            self._settings.use_gpu
        )


def main():
    """Application entry point."""
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("WhisperNow")
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
