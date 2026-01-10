
import pytest
from unittest.mock import MagicMock, patch, ANY
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from src.transcribe.app import TranscribeApp
from src.transcribe.core.asr.transcriber import EngineState
from src.transcribe.core.input.hotkey import HotkeyListener
from src.transcribe.core.settings import Settings, HotkeyConfig


@pytest.fixture
def cleanup_app():
    """Fixture that cleans up TranscribeApp instances after each test."""
    apps = []
    
    def register(app):
        apps.append(app)
        return app
    
    yield register
    
    # Cleanup all registered apps
    for app in apps:
        try:
            app._hotkey_listener.stop()
            app._tray.hide()
            app._recording_toast.hide()
            if app._audio_level_timer.isActive():
                app._audio_level_timer.stop()
        except Exception:
            pass
    
    # Process events to ensure cleanup
    qapp = QApplication.instance()
    if qapp:
        qapp.processEvents()


@pytest.fixture
def mock_dependencies():
    with patch('src.transcribe.app.get_settings') as mock_get_settings, \
         patch('src.transcribe.app.SystemTray') as MockTray, \
         patch('src.transcribe.app.AudioRecorder') as MockRecorder, \
         patch('src.transcribe.app.TranscriptionEngine') as MockTranscriber, \
         patch('src.transcribe.app.HotkeyListener', wraps=HotkeyListener) as MockHotkey, \
         patch('src.transcribe.app.check_and_request_permissions', return_value=True), \
         patch('src.transcribe.app.set_autostart') as mock_set_autostart, \
         patch('src.transcribe.app.ModelLoaderThread') as MockLoaderThread:
        
        # Setup mock settings
        settings = Settings()
        settings.characters_per_second = 150  # Use char-by-char typing for tests
        mock_get_settings.return_value = settings
        
        # Setup mocks
        mock_tray = MockTray.return_value
        mock_recorder = MockRecorder.return_value
        mock_recorder.stop.return_value = b"audio_data"  # Simulate audio captured
        
        mock_transcriber = MockTranscriber.return_value
        mock_transcriber.transcribe_chunked.return_value = "Hello World"
        
        # Setup loader thread mock
        mock_loader = MockLoaderThread.return_value
        # Mock signals as MagicMocks that can be connected to
        mock_loader.finished = MagicMock()
        mock_loader.progress = MagicMock()
        mock_loader.state_changed = MagicMock()
        # Mock internal engine
        mock_loader.engine = mock_transcriber
        
        yield {
            'settings': settings,
            'get_settings': mock_get_settings,
            'tray': mock_tray,
            'recorder': mock_recorder,
            'transcriber': mock_transcriber,
            'set_autostart': mock_set_autostart,
            'loader_thread': MockLoaderThread
        }

@patch('src.transcribe.core.output.text_output.subprocess.run')
@patch('src.transcribe.app.TextOutputController')
def test_full_transcription_flow(MockTextOutput, mock_subprocess, mock_dependencies, cleanup_app, qtbot):
    """Test the complete flow: Hotkey -> Record -> Transcribe -> Type"""
    
    # Set to instant typing (uses paste via subprocess)
    mock_dependencies['settings'].characters_per_second = 0
    mock_subprocess.return_value.returncode = 0
    mock_subprocess.return_value.stdout = ""
    
    # Setup mock text output
    mock_text_output_instance = MockTextOutput.return_value
    
    # Initialize App
    app = cleanup_app(TranscribeApp())
    
    # Simulate Hotkey Pressed
    app._start_recording()
    
    # Verify recording started
    mock_dependencies['tray'].set_status.assert_called_with(ANY) # Should be RECORDING
    mock_dependencies['recorder'].start.assert_called_once()
    
    # Simulate Hotkey Released
    app._stop_recording()
    
    # Verify recording stopped
    mock_dependencies['recorder'].stop.assert_called_once()
    
    # Verify transcription (using chunked method now)
    mock_dependencies['transcriber'].transcribe_chunked.assert_called_once_with(b"audio_data", 16000)

    # Verify that text was output via TextOutputController
    mock_text_output_instance.output_text.assert_called_once()
    call_args = mock_text_output_instance.output_text.call_args
    assert call_args[0][0] == "Hello World", "Text output was not called with transcribed text"

def test_settings_hotkey_update(mock_dependencies, cleanup_app, qtbot):
    """Test that changing settings updates the hotkey listener."""
    
    # Mock the internal hotkey listener of the app to check for updates
    with patch('src.transcribe.app.HotkeyListener') as MockListener:
        mock_listener_instance = MockListener.return_value
        
        app = cleanup_app(TranscribeApp())
        
        # Change settings
        new_settings = Settings(hotkey=HotkeyConfig(key="k", modifiers=["ctrl"]))
        mock_dependencies['get_settings'].return_value = new_settings
        
        # Trigger update
        app._on_settings_changed()
        
        # Verify hotkey listener was updated
        mock_listener_instance.update_settings.assert_called_with(new_settings)

def test_autostart_update(mock_dependencies, cleanup_app, qtbot):
    """Test that changing settings updates autostart."""
    
    with patch('src.transcribe.app.HotkeyListener') as MockListener:
        app = cleanup_app(TranscribeApp())
        
        # Change settings
        new_settings = Settings(auto_start_on_login=True)
        mock_dependencies['get_settings'].return_value = new_settings
        
        # Trigger update
        app._on_settings_changed()
        
        # Verify autostart called
        mock_dependencies['set_autostart'].assert_called_with(True, "Transcribe")
