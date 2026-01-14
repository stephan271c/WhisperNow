from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import sounddevice as sd


@dataclass
class AudioDevice:
    name: str
    index: int
    channels: int
    default_sample_rate: float


class AudioRecorder:

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device: Optional[str] = None,
        on_audio_level: Optional[Callable[[float], None]] = None,
        on_audio_spectrum: Optional[Callable[[List[float]], None]] = None,
    ):

        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.on_audio_level = on_audio_level
        self.on_audio_spectrum = on_audio_spectrum

        self._stream: Optional[sd.InputStream] = None
        self._audio_buffer: List[np.ndarray] = []
        self._is_recording = False
        self._device_sample_rate: Optional[float] = None
        self._spectrum_frames: Optional[int] = None
        self._spectrum_rate: Optional[float] = None
        self._spectrum_bins: List[tuple[int, int]] = []
        self._spectrum_band_count = 8
        self._window: Optional[np.ndarray] = None

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def start(self) -> bool:
        if self._is_recording:
            return True

        self._audio_buffer = []
        self._last_error: Optional[str] = None

        try:

            self._device_sample_rate = float(self.sample_rate)

            self._stream = sd.InputStream(
                samplerate=self._device_sample_rate,
                channels=self.channels,
                device=self._get_device_index(),
                callback=self._audio_callback,
            )
            self._stream.start()
            self._is_recording = True
            return True

        except sd.PortAudioError as e:
            self._last_error = f"Audio device error: {e}"
            self._is_recording = False
            return False
        except Exception as e:
            self._last_error = f"Failed to start recording: {e}"
            self._is_recording = False
            return False

    @property
    def last_error(self) -> Optional[str]:
        return getattr(self, "_last_error", None)

    def stop(self) -> Optional[np.ndarray]:
        if not self._is_recording:
            return None

        self._is_recording = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not self._audio_buffer:
            return None

        audio_data = np.concatenate(self._audio_buffer, axis=0)

        return audio_data

    def _audio_callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        if self._is_recording:
            self._audio_buffer.append(indata.copy())

            if self.on_audio_level is not None:
                level = np.abs(indata).mean()
                self.on_audio_level(min(1.0, level * 10))

            if self.on_audio_spectrum is not None and self._device_sample_rate:
                bands = self._compute_spectrum_bands(indata, self._device_sample_rate)
                self.on_audio_spectrum(bands)

    def _get_device_index(self) -> Optional[int]:
        if self.device is None:
            return None

        for device in self.list_devices():
            if device.name == self.device:
                return device.index

        return None

    def _compute_spectrum_bands(
        self, indata: np.ndarray, sample_rate: float
    ) -> List[float]:
        if indata.size == 0:
            return [0.0] * self._spectrum_band_count

        mono = indata.mean(axis=1) if indata.ndim > 1 else indata.flatten()
        frames = mono.shape[0]
        if frames < 8:
            return [0.0] * self._spectrum_band_count

        if self._window is None or self._window.shape[0] != frames:
            self._window = np.hanning(frames)

        windowed = mono * self._window
        fft = np.fft.rfft(windowed)
        mag = np.abs(fft)
        if mag.size == 0:
            return [0.0] * self._spectrum_band_count

        bands = self._get_spectrum_bins(frames, sample_rate)
        log_mag = np.log1p(mag)
        max_mag = float(np.max(log_mag))
        if max_mag <= 0.0:
            return [0.0] * self._spectrum_band_count

        energies: List[float] = []
        for start, end in bands:
            if end <= start:
                energies.append(0.0)
                continue
            band_energy = float(np.mean(log_mag[start:end])) / max_mag
            energies.append(min(1.0, max(0.0, band_energy)))
        return energies

    def _get_spectrum_bins(
        self, frames: int, sample_rate: float
    ) -> List[tuple[int, int]]:
        if self._spectrum_frames == frames and self._spectrum_rate == sample_rate:
            return self._spectrum_bins

        low_freq = 80.0
        high_freq = min(8000.0, (sample_rate / 2.0) * 0.95)
        if high_freq <= low_freq:
            high_freq = low_freq * 2.0

        edges = np.logspace(
            np.log10(low_freq),
            np.log10(high_freq),
            num=self._spectrum_band_count + 1,
        )
        freqs = np.fft.rfftfreq(frames, 1.0 / sample_rate)

        bins: List[tuple[int, int]] = []
        for idx in range(self._spectrum_band_count):
            start = int(np.searchsorted(freqs, edges[idx], side="left"))
            end = int(np.searchsorted(freqs, edges[idx + 1], side="right"))
            if end <= start:
                end = min(start + 1, len(freqs))
            bins.append((start, end))

        self._spectrum_frames = frames
        self._spectrum_rate = sample_rate
        self._spectrum_bins = bins
        return bins

    @staticmethod
    def list_devices() -> List[AudioDevice]:
        devices = []

        for i, device in enumerate(sd.query_devices()):
            if device["max_input_channels"] > 0:
                devices.append(
                    AudioDevice(
                        name=device["name"],
                        index=i,
                        channels=device["max_input_channels"],
                        default_sample_rate=device["default_samplerate"],
                    )
                )

        return devices
