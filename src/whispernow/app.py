"""Application runtime."""

import signal
import sys
from datetime import datetime
from typing import Optional, Tuple

from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QApplication

from whispernow import __app_name__, __version__
from whispernow.core.asr import (
    EngineState,
    ModelLoaderThread,
    TranscriptionEngine,
    TranscriptionWorkerThread,
)
from whispernow.core.audio import AudioRecorder
from whispernow.core.input import HotkeyListener
from whispernow.core.output import TextOutputController
from whispernow.core.settings import (
    TranscriptionRecord,
    add_history_record,
    get_settings,
)
from whispernow.core.transcript_processor import LLMProcessor
from whispernow.ui.download_dialog import DownloadDialog
from whispernow.ui.main_window import SettingsWindow
from whispernow.ui.recording_toast import RecordingToast
from whispernow.ui.setup_wizard import SetupWizard
from whispernow.ui.tray import SystemTray, TrayStatus
from whispernow.utils.logger import get_logger
from whispernow.utils.platform import check_and_request_permissions, set_autostart

logger = get_logger(__name__)


class TranscribeApp(QObject):

    def __init__(self):
        super().__init__()

        self._settings = get_settings()
        self._settings_window: Optional[SettingsWindow] = None

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
            on_download_progress=self._on_download_progress,
        )
        self._hotkey_listener = HotkeyListener()
        self._text_output = TextOutputController(on_complete=self._finish_typing)
        self._download_dialog: Optional[DownloadDialog] = None
        self._model_loader_thread: Optional[ModelLoaderThread] = None
        self._transcription_worker: Optional[TranscriptionWorkerThread] = None

        self._typing_text: str = ""
        self._typing_index: int = 0
        self._typing_timer: Optional[QTimer] = None

        self._llm_processor: Optional[LLMProcessor] = None
        self._init_llm_processor()

        self._tray.settings_requested.connect(self._show_settings)
        self._tray.quit_requested.connect(self._quit)
        self._hotkey_listener.hotkey_pressed.connect(self._start_recording)
        self._hotkey_listener.hotkey_released.connect(self._stop_recording)

        check_and_request_permissions(self._settings)

    def _on_engine_state_change(self, state: EngineState, message: str) -> None:
        if state in (EngineState.PROCESSING, EngineState.READY, EngineState.ERROR):
            if (
                self._transcription_worker is not None
                and self._transcription_worker.isRunning()
            ):
                return

        status_map = {
            EngineState.NOT_LOADED: TrayStatus.IDLE,
            EngineState.DOWNLOADING: TrayStatus.LOADING,
            EngineState.LOADING: TrayStatus.LOADING,
            EngineState.READY: TrayStatus.IDLE,
            EngineState.ERROR: TrayStatus.ERROR,
        }
        self._tray.set_status(status_map.get(state, TrayStatus.IDLE), message)

        if state == EngineState.DOWNLOADING:
            self._show_download_dialog()
        elif state in (EngineState.READY, EngineState.ERROR, EngineState.NOT_LOADED):
            self._hide_download_dialog(state == EngineState.READY)

    def _on_download_progress(self, progress: float) -> None:
        if self._download_dialog is not None:
            self._download_dialog.set_progress(progress)

    def _show_download_dialog(self) -> None:
        if self._download_dialog is None:
            self._download_dialog = DownloadDialog(model_name=self._settings.model_name)
            self._download_dialog.cancelled.connect(self._on_download_cancelled)
        self._download_dialog.show()

    def _hide_download_dialog(self, success: bool = True) -> None:
        if self._download_dialog is not None:
            self._download_dialog.finish(success)
            self._download_dialog = None

    def _on_download_cancelled(self) -> None:
        self._transcriber.unload()
        self._hide_download_dialog(success=False)

    def _start_recording(self) -> None:
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

        logger.info(
            f"Captured {len(audio_data)} audio samples, starting background transcription"
        )

        self._tray.set_status(TrayStatus.PROCESSING)
        QApplication.processEvents()

        enhancement = self._settings.get_active_enhancement()

        self._transcription_worker = TranscriptionWorkerThread(
            transcriber=self._transcriber,
            audio_data=audio_data,
            sample_rate=self._settings.sample_rate,
            vocabulary_replacements=self._settings.vocabulary_replacements,
            llm_processor=self._llm_processor,
            enhancement=enhancement,
            parent=self,
        )
        self._transcription_worker.finished.connect(self._on_transcription_complete)
        self._transcription_worker.error.connect(self._on_transcription_error)
        self._transcription_worker.start()

    def _on_transcription_complete(
        self,
        final_text: str,
        raw_text: str,
        enhanced_text: Optional[str],
        enhancement_name: Optional[str],
        cost: Optional[float],
    ) -> None:
        logger.info(
            f"Background transcription complete: '{final_text[:50]}{'...' if len(final_text) > 50 else ''}'"
        )
        self._record_transcription(raw_text, enhanced_text, enhancement_name, cost)
        self._type_text(final_text)

    def _on_transcription_error(self, error_message: str) -> None:
        logger.error(f"Background transcription failed: {error_message}")
        self._tray.set_status(TrayStatus.ERROR, error_message)

    def _init_llm_processor(self) -> None:
        has_config = (
            self._settings.llm_api_key
            or self._settings.active_enhancement_id
            or self._settings.llm_provider == "ollama"
        )
        if has_config:
            model_name = LLMProcessor.format_model_name(
                self._settings.llm_model, self._settings.llm_provider
            )
            self._llm_processor = LLMProcessor(
                model=model_name,
                api_key=self._settings.llm_api_key,
                api_base=self._settings.llm_api_base,
            )
            logger.info(
                f"LLM processor initialized: model={model_name}, api_base={self._settings.llm_api_base}"
            )
        else:
            self._llm_processor = None

    def _apply_enhancement(
        self, text: str
    ) -> Tuple[str, Optional[str], Optional[str], Optional[float]]:
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
            logger.warning(
                "LLM processor not configured (no API key), skipping enhancement"
            )
            return text, None, None, None

        logger.info(f"Applying enhancement: {enhancement.title}")
        response = self._llm_processor.process(text, enhancement)

        return response.content, response.content, enhancement.title, response.cost_usd

    def _record_transcription(
        self,
        raw_text: str,
        enhanced_text: Optional[str],
        enhancement_name: Optional[str],
        cost_usd: Optional[float],
    ) -> None:
        record = TranscriptionRecord(
            timestamp=datetime.now().isoformat(),
            raw_text=raw_text,
            enhanced_text=enhanced_text,
            enhancement_name=enhancement_name,
            cost_usd=cost_usd,
        )
        add_history_record(record)
        logger.debug(f"Recorded transcription to history: {len(raw_text)} chars")

    def _type_text(self, text: str) -> None:
        if self._settings.instant_type or self._settings.characters_per_second <= 0:
            self._text_output.output_text(text, instant=True)
        else:
            self._typing_text = text
            self._typing_index = 0

            delay_ms = int(1000.0 / self._settings.characters_per_second)
            self._typing_timer = QTimer(self)
            self._typing_timer.timeout.connect(self._type_next_char)
            self._typing_timer.start(delay_ms)

    def _type_next_char(self) -> None:
        if self._typing_index < len(self._typing_text):
            char = self._typing_text[self._typing_index]
            self._text_output.type_character(char)
            self._typing_index += 1
        else:
            if self._typing_timer:
                self._typing_timer.stop()
                self._typing_timer.deleteLater()
                self._typing_timer = None
            self._typing_text = ""
            self._typing_index = 0
            self._finish_typing()

    def _finish_typing(self) -> None:
        self._tray.set_status(TrayStatus.IDLE)

    def _show_settings(self) -> None:
        if self._settings_window is None:
            self._settings_window = SettingsWindow()
            self._settings_window.settings_changed.connect(self._on_settings_changed)

        self._settings_window.show()
        self._settings_window.raise_()
        self._settings_window.activateWindow()

    def _on_settings_changed(self) -> None:
        old_model = self._transcriber.model_name
        old_gpu = self._transcriber.use_gpu

        self._settings = get_settings()

        self._recorder.sample_rate = self._settings.sample_rate
        self._recorder.device = self._settings.input_device
        self._hotkey_listener.update_settings(self._settings)

        self._init_llm_processor()

        model_changed = old_model != self._settings.model_name
        gpu_changed = old_gpu != self._settings.use_gpu
        if model_changed or gpu_changed:
            change_reason = []
            if model_changed:
                change_reason.append(
                    f"model: {old_model} -> {self._settings.model_name}"
                )
            if gpu_changed:
                change_reason.append(f"GPU: {old_gpu} -> {self._settings.use_gpu}")
            logger.info(f"Reloading transcription engine ({', '.join(change_reason)})")
            self._transcriber.unload()

            if self._settings_window is not None:
                self._settings_window.set_loading(True)

            self._start_model_loading(self._settings.model_name, self._settings.use_gpu)

        set_autostart(self._settings.auto_start_on_login, "WhisperNow")

    def _quit(self) -> None:
        logger.info("Shutting down application")
        self._hotkey_listener.stop()
        self._transcriber.unload()
        self._tray.hide()
        self._recording_toast.hide()
        QApplication.quit()
        logger.info("Application shutdown complete")

    def _capture_audio_level(self, level: float) -> None:
        self._latest_audio_level = level

    def _capture_audio_spectrum(self, spectrum: list[float]) -> None:
        self._latest_audio_spectrum = spectrum

    def _update_audio_level(self) -> None:
        self._recording_toast.set_level(self._latest_audio_level)
        self._recording_toast.set_spectrum(self._latest_audio_spectrum)

    def _start_model_loading(self, model_name: str, use_gpu: bool) -> None:
        if (
            self._model_loader_thread is not None
            and self._model_loader_thread.isRunning()
        ):
            self._model_loader_thread.wait()

        self._model_loader_thread = ModelLoaderThread(
            model_name=model_name, use_gpu=use_gpu, parent=self
        )
        self._model_loader_thread.state_changed.connect(self._on_engine_state_change)
        self._model_loader_thread.progress.connect(self._on_download_progress)
        self._model_loader_thread.finished.connect(self._on_model_loaded)
        self._model_loader_thread.start()

    def _on_model_loaded(self, success: bool, message: str) -> None:
        if success and self._model_loader_thread is not None:
            self._transcriber = self._model_loader_thread.engine
            logger.info(f"Model loading complete: {message}")
        else:
            logger.error(f"Model loading failed: {message}")

        if self._settings_window is not None:
            self._settings_window.set_loading(False)
            if success:
                self._settings_window.refresh_asr_model_list()

        if success:
            self._tray.set_status(TrayStatus.IDLE, message)
        else:
            self._tray.set_status(TrayStatus.ERROR, message)

    def run(self) -> None:
        logger.info(f"Starting {__app_name__} v{__version__}")
        logger.info(
            f"Settings: model={self._settings.model_name}, sample_rate={self._settings.sample_rate}"
        )

        logger.info(
            f"Starting hotkey listener: {self._settings.hotkey.to_display_string()}"
        )
        self._hotkey_listener.start()

        if not self._settings.start_minimized:
            self._show_settings()

        logger.info("Deferring model load to after UI is shown...")
        QTimer.singleShot(100, self._start_deferred_model_loading)

        logger.info("Application initialization complete")

    def _start_deferred_model_loading(self) -> None:
        logger.info("Starting deferred transcription model loading...")
        self._start_model_loading(self._settings.model_name, self._settings.use_gpu)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("WhisperNow")
    app.setQuitOnLastWindowClosed(False)
    signal.signal(signal.SIGINT, lambda *args: QApplication.quit())

    settings = get_settings()
    if not settings.first_run_complete:
        wizard = SetupWizard()
        if wizard.exec() != SetupWizard.Accepted:
            sys.exit(0)

    transcribe_app = TranscribeApp()
    transcribe_app.run()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
