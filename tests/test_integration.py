from unittest.mock import ANY, MagicMock, patch

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from src.whispernow.app import TranscribeApp
from src.whispernow.core.asr.transcriber import EngineState
from src.whispernow.core.input.hotkey import HotkeyListener
from src.whispernow.core.settings import HotkeyConfig, Settings


@pytest.fixture
def cleanup_app():
    apps = []

    def register(app):
        apps.append(app)
        return app

    yield register

    for app in apps:
        try:
            app._hotkey_listener.stop()
            app._tray.hide()
            app._recording_toast.hide()
            if app._audio_level_timer.isActive():
                app._audio_level_timer.stop()
        except Exception:
            pass

    qapp = QApplication.instance()
    if qapp:
        qapp.processEvents()


@pytest.fixture
def mock_dependencies():
    with (
        patch("src.whispernow.app.get_settings") as mock_get_settings,
        patch("src.whispernow.app.SystemTray") as MockTray,
        patch("src.whispernow.app.AudioRecorder") as MockRecorder,
        patch("src.whispernow.app.TranscriptionEngine") as MockTranscriber,
        patch("src.whispernow.app.HotkeyListener", wraps=HotkeyListener) as MockHotkey,
        patch("src.whispernow.app.set_autostart") as mock_set_autostart,
        patch("src.whispernow.app.ModelLoaderThread") as MockLoaderThread,
    ):

        settings = Settings()
        mock_get_settings.return_value = settings

        mock_tray = MockTray.return_value
        mock_recorder = MockRecorder.return_value
        mock_recorder.stop.return_value = b"audio_data"  # Simulate audio captured

        mock_transcriber = MockTranscriber.return_value
        mock_transcriber.transcribe_chunked.return_value = "Hello World"

        mock_loader = MockLoaderThread.return_value
        mock_loader.finished = MagicMock()
        mock_loader.progress = MagicMock()
        mock_loader.state_changed = MagicMock()
        mock_loader.engine = mock_transcriber

        yield {
            "settings": settings,
            "get_settings": mock_get_settings,
            "tray": mock_tray,
            "recorder": mock_recorder,
            "transcriber": mock_transcriber,
            "set_autostart": mock_set_autostart,
            "loader_thread": MockLoaderThread,
        }


@patch("src.whispernow.app.TextOutputController")
@patch("src.whispernow.app.TranscriptionWorkerThread")
def test_full_transcription_flow(
    MockWorkerThread,
    MockTextOutput,
    mock_dependencies,
    cleanup_app,
    qtbot,
):

    mock_text_output_instance = MockTextOutput.return_value
    mock_worker_instance = MockWorkerThread.return_value
    mock_worker_instance.finished = MagicMock()
    mock_worker_instance.error = MagicMock()

    app = cleanup_app(TranscribeApp())
    app._start_recording()

    mock_dependencies["tray"].set_status.assert_called_with(ANY)
    mock_dependencies["recorder"].start.assert_called_once()

    app._stop_recording()
    mock_dependencies["recorder"].stop.assert_called_once()

    MockWorkerThread.assert_called_once()
    mock_worker_instance.start.assert_called_once()

    app._on_transcription_complete("Hello World", "Hello World", None, None, None)
    mock_text_output_instance.output_text.assert_called_once()
    call_args = mock_text_output_instance.output_text.call_args
    assert (
        call_args[0][0] == "Hello World"
    ), "Text output was not called with transcribed text"


def test_settings_hotkey_update(mock_dependencies, cleanup_app, qtbot):
    with patch("src.whispernow.app.HotkeyListener") as MockListener:
        mock_listener_instance = MockListener.return_value

        app = cleanup_app(TranscribeApp())

        new_settings = Settings(hotkey=HotkeyConfig(key="k", modifiers=["ctrl"]))
        mock_dependencies["get_settings"].return_value = new_settings

        app._on_settings_changed()
        mock_listener_instance.update_settings.assert_called_with(new_settings)


def test_autostart_update(mock_dependencies, cleanup_app, qtbot):

    with patch("src.whispernow.app.HotkeyListener") as MockListener:
        app = cleanup_app(TranscribeApp())

        new_settings = Settings(auto_start_on_login=True)
        mock_dependencies["get_settings"].return_value = new_settings

        app._on_settings_changed()
        mock_dependencies["set_autostart"].assert_called_with(True, "WhisperNow")
