"""
Tests for AudioRecorder and device enumeration.

Uses mocking to avoid requiring actual audio hardware.
"""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.transcribe.core.recorder import AudioRecorder, AudioDevice


class TestAudioDevice:
    """Tests for AudioDevice dataclass."""
    
    def test_device_creation(self):
        """Test AudioDevice can be created with all fields."""
        device = AudioDevice(
            name="Test Microphone",
            index=0,
            channels=2,
            default_sample_rate=48000.0
        )
        assert device.name == "Test Microphone"
        assert device.index == 0
        assert device.channels == 2
        assert device.default_sample_rate == 48000.0


class TestAudioRecorderDeviceEnumeration:
    """Tests for device enumeration."""
    
    @patch("src.transcribe.core.recorder.sd.query_devices")
    def test_list_devices_returns_input_devices(self, mock_query):
        """Test list_devices filters to input devices only."""
        mock_query.return_value = [
            {"name": "Mic 1", "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 44100},
            {"name": "Speakers", "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48000},
            {"name": "Mic 2", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 16000},
        ]
        
        devices = AudioRecorder.list_devices()
        
        assert len(devices) == 2
        assert devices[0].name == "Mic 1"
        assert devices[0].channels == 2
        assert devices[1].name == "Mic 2"
        assert devices[1].channels == 1
    
    @patch("src.transcribe.core.recorder.sd.query_devices")
    def test_list_devices_empty(self, mock_query):
        """Test list_devices handles no devices."""
        mock_query.return_value = []
        
        devices = AudioRecorder.list_devices()
        assert devices == []


class TestAudioRecorderState:
    """Tests for recorder start/stop state."""
    
    def test_initial_state_not_recording(self):
        """Test recorder starts in non-recording state."""
        recorder = AudioRecorder()
        assert recorder.is_recording is False
    
    @patch("src.transcribe.core.recorder.sd.InputStream")
    def test_start_sets_recording(self, mock_stream_class):
        """Test start() sets is_recording to True."""
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream
        
        recorder = AudioRecorder()
        recorder.start()
        
        assert recorder.is_recording is True
        mock_stream.start.assert_called_once()
    
    @patch("src.transcribe.core.recorder.sd.InputStream")
    def test_stop_clears_recording(self, mock_stream_class):
        """Test stop() sets is_recording to False and returns audio."""
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream
        
        recorder = AudioRecorder()
        recorder.start()
        
        # Simulate some audio data in buffer
        recorder._audio_buffer = [
            np.array([[0.1], [0.2]], dtype=np.float32),
            np.array([[0.3], [0.4]], dtype=np.float32),
        ]
        
        audio = recorder.stop()
        
        assert recorder.is_recording is False
        assert audio is not None
        assert len(audio) > 0  # Sample count may vary due to resampling
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
    
    @patch("src.transcribe.core.recorder.sd.query_devices")
    @patch("src.transcribe.core.recorder.sd.InputStream")
    def test_stop_performs_resampling(self, mock_stream_class, mock_query):
        """Test stop() resamples audio when device rate differs from target."""
        # Setup device with 48kHz native rate
        mock_query.return_value = {'default_samplerate': 48000.0}
        
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream
        
        # Target 16kHz (standard for Whisper)
        recorder = AudioRecorder(sample_rate=16000)
        recorder.start()
        
        # Simulate 1 second of audio at 48kHz (48000 samples)
        # We'll simulate 2 chunks of 24000 samples each
        chunk_size = 24000
        recorder._audio_buffer = [
            np.zeros((chunk_size, 1), dtype=np.float32),
            np.zeros((chunk_size, 1), dtype=np.float32),
        ]
        
        audio = recorder.stop()
        
        # We expect 16000 samples total (1 second at 16kHz)
        # 48000 source samples / 3 = 16000 target samples
        expected_samples = 16000
        
        assert audio is not None
        # Allow small rounding difference from resampling
        assert abs(len(audio) - expected_samples) < 10
        assert recorder.sample_rate == 16000
    
    def test_stop_without_start_returns_none(self):
        """Test stop() without start() returns None."""
        recorder = AudioRecorder()
        audio = recorder.stop()
        assert audio is None
    
    @patch("src.transcribe.core.recorder.sd.InputStream")
    def test_start_twice_is_idempotent(self, mock_stream_class):
        """Test calling start() twice only starts once."""
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream
        
        recorder = AudioRecorder()
        recorder.start()
        recorder.start()  # Second call should be no-op
        
        assert mock_stream_class.call_count == 1


class TestAudioRecorderConfiguration:
    """Tests for recorder configuration."""
    
    def test_default_sample_rate(self):
        """Test default sample rate is 16000 Hz."""
        recorder = AudioRecorder()
        assert recorder.sample_rate == 16000
    
    def test_custom_sample_rate(self):
        """Test custom sample rate is accepted."""
        recorder = AudioRecorder(sample_rate=44100)
        assert recorder.sample_rate == 44100
    
    def test_default_channels_mono(self):
        """Test default is mono (1 channel)."""
        recorder = AudioRecorder()
        assert recorder.channels == 1
    
    def test_device_selection(self):
        """Test device name can be specified."""
        recorder = AudioRecorder(device="My USB Mic")
        assert recorder.device == "My USB Mic"


class TestAudioCallback:
    """Tests for audio level callback."""
    
    @patch("src.transcribe.core.recorder.sd.InputStream")
    def test_audio_level_callback(self, mock_stream_class):
        """Test on_audio_level callback is invoked."""
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream
        
        levels = []
        def level_callback(level):
            levels.append(level)
        
        recorder = AudioRecorder(on_audio_level=level_callback)
        recorder.start()
        
        # Simulate audio callback
        audio_data = np.array([[0.5], [0.5]], dtype=np.float32)
        recorder._audio_callback(audio_data, 2, None, None)
        
        assert len(levels) == 1
        assert 0.0 <= levels[0] <= 1.0
