"""
Audio recording functionality.

Handles microphone input using sounddevice library.
"""

import numpy as np
import sounddevice as sd
from typing import Callable, Optional, List
from dataclasses import dataclass


@dataclass
class AudioDevice:
    """Represents an audio input device."""
    name: str
    index: int
    channels: int
    default_sample_rate: float


class AudioRecorder:
    """
    Records audio from the microphone.
    
    Usage:
        recorder = AudioRecorder(sample_rate=16000)
        recorder.start()
        # ... user speaks ...
        audio_data = recorder.stop()
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device: Optional[str] = None,
        on_audio_level: Optional[Callable[[float], None]] = None
    ):
        """
        Initialize the audio recorder.
        
        Args:
            sample_rate: Sample rate in Hz (default: 16000 for ASR models)
            channels: Number of audio channels (default: 1 for mono)
            device: Device name or None for system default
            on_audio_level: Callback for audio level updates (0.0-1.0)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.on_audio_level = on_audio_level
        
        self._stream: Optional[sd.InputStream] = None
        self._audio_buffer: List[np.ndarray] = []
        self._is_recording = False
    
    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording
    
    def start(self) -> None:
        """Start recording audio."""
        if self._is_recording:
            return
        
        self._audio_buffer = []
        self._is_recording = True
        
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            device=self._get_device_index(),
            callback=self._audio_callback
        )
        self._stream.start()
    
    def stop(self) -> Optional[np.ndarray]:
        """
        Stop recording and return the audio data.
        
        Returns:
            Numpy array of audio samples, or None if no audio was recorded.
        """
        if not self._is_recording:
            return None
        
        self._is_recording = False
        
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        
        if not self._audio_buffer:
            return None
        
        return np.concatenate(self._audio_buffer, axis=0)
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        """Callback for audio stream."""
        if self._is_recording:
            self._audio_buffer.append(indata.copy())
            
            # Calculate audio level for visualization
            if self.on_audio_level is not None:
                level = np.abs(indata).mean()
                self.on_audio_level(min(1.0, level * 10))  # Scale for visibility
    
    def _get_device_index(self) -> Optional[int]:
        """Get the device index from device name."""
        if self.device is None:
            return None
        
        for device in self.list_devices():
            if device.name == self.device:
                return device.index
        
        return None
    
    @staticmethod
    def list_devices() -> List[AudioDevice]:
        """List available audio input devices."""
        devices = []
        
        for i, device in enumerate(sd.query_devices()):
            if device["max_input_channels"] > 0:
                devices.append(AudioDevice(
                    name=device["name"],
                    index=i,
                    channels=device["max_input_channels"],
                    default_sample_rate=device["default_samplerate"]
                ))
        
        return devices
