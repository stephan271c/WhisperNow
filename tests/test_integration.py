
import pytest
from unittest.mock import MagicMock, patch, ANY
from PySide6.QtCore import Qt
from src.transcribe.app import TranscribeApp, HotkeyListener
from src.transcribe.core.transcriber import EngineState
from src.transcribe.core.settings import Settings, HotkeyConfig

@pytest.fixture
def mock_dependencies():
    with patch('src.transcribe.app.get_settings') as mock_get_settings, \
         patch('src.transcribe.app.SystemTray') as MockTray, \
         patch('src.transcribe.app.AudioRecorder') as MockRecorder, \
         patch('src.transcribe.app.TranscriptionEngine') as MockTranscriber, \
         patch('src.transcribe.app.HotkeyListener', wraps=HotkeyListener) as MockHotkey, \
         patch('src.transcribe.app.check_accessibility_permissions', return_value=True), \
         patch('src.transcribe.app.set_autostart') as mock_set_autostart:
        
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
        
        # We use the REAL HotkeyListener logic, but patched into the App so we control it
        # Actually, using wraps=HotkeyListener is tricky because it spawns a thread.
        # Let's mock the listener but verify the update_settings call.
        
        yield {
            'settings': settings,
            'get_settings': mock_get_settings,
            'tray': mock_tray,
            'recorder': mock_recorder,
            'transcriber': mock_transcriber,
            'set_autostart': mock_set_autostart
        }

@patch('subprocess.run')
@patch('src.transcribe.app.KeyboardController')
def test_full_transcription_flow(MockController, mock_subprocess, mock_dependencies, qtbot):
    """Test the complete flow: Hotkey -> Record -> Transcribe -> Type"""
    
    # Set to instant typing (uses paste via subprocess)
    mock_dependencies['settings'].characters_per_second = 0
    mock_subprocess.return_value.returncode = 0
    mock_subprocess.return_value.stdout = ""
    
    # Initialize App
    app = TranscribeApp()
    
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

    # Verify that text was actually "typed" (via clipboard paste in this case)
    # The app calls _paste_text, which calls subprocess.run with xclip/pbcopy etc.
    # We should verify that subprocess.run was called to put "Hello World" on clipboard
    
    # Filter calls to subprocess.run that set the clipboard (input="Hello World")
    clip_calls = [
        call for call in mock_subprocess.call_args_list 
        if call.kwargs.get('input') == "Hello World"
    ]
    assert len(clip_calls) > 0, "Clipboard was not updated with transcribed text"

def test_settings_hotkey_update(mock_dependencies, qtbot):
    """Test that changing settings updates the hotkey listener."""
    
    # Mock the internal hotkey listener of the app to check for updates
    with patch('src.transcribe.app.HotkeyListener') as MockListener:
        mock_listener_instance = MockListener.return_value
        
        app = TranscribeApp()
        
        # Change settings
        new_settings = Settings(hotkey=HotkeyConfig(key="k", modifiers=["ctrl"]))
        mock_dependencies['get_settings'].return_value = new_settings
        
        # Trigger update
        app._on_settings_changed()
        
        # Verify hotkey listener was updated
        mock_listener_instance.update_settings.assert_called_with(new_settings)

def test_autostart_update(mock_dependencies, qtbot):
    """Test that changing settings updates autostart."""
    
    with patch('src.transcribe.app.HotkeyListener') as MockListener:
        app = TranscribeApp()
        
        # Change settings
        new_settings = Settings(auto_start_on_login=True)
        mock_dependencies['get_settings'].return_value = new_settings
        
        # Trigger update
        app._on_settings_changed()
        
        # Verify autostart called
        mock_dependencies['set_autostart'].assert_called_with(True, "Transcribe")
