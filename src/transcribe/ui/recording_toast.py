"""
Recording toast with animated waveform.

Shows a small floating pill when microphone recording is active.
"""

from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import (
    Qt,
    QPoint,
    QPropertyAnimation,
    QParallelAnimationGroup,
    QTimer,
)
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QGuiApplication


class WaveformWidget(QWidget):
    """Animated waveform bars driven by an audio level (0.0-1.0)."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._bar_count = 11
        self._level = 0.0
        self._target_level = 0.0
        self._spectrum = [0.0] * self._bar_count
        self._target_spectrum = [0.0] * self._bar_count
        self._phase = 0.0
        self._bar_color = QColor("#F1E7D8")
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self.setFixedSize(120, 18)

    def set_level(self, level: float) -> None:
        """Set the target audio level."""
        self._target_level = max(0.0, min(1.0, level))

    def set_spectrum(self, spectrum: list[float]) -> None:
        """Set target spectrum levels for each bar (0.0-1.0)."""
        mapped = self._map_spectrum(spectrum)
        self._target_spectrum = [max(0.0, min(1.0, value)) for value in mapped]

    def set_bar_color(self, color: QColor) -> None:
        """Set the bar color."""
        self._bar_color = color
        self.update()

    def start(self) -> None:
        """Start the waveform animation."""
        if not self._timer.isActive():
            self._timer.start()

    def stop(self) -> None:
        """Stop the waveform animation and reset."""
        if self._timer.isActive():
            self._timer.stop()
        self._phase = 0.0
        self._level = 0.0
        self._target_level = 0.0
        self._spectrum = [0.0] * self._bar_count
        self._target_spectrum = [0.0] * self._bar_count
        self.update()

    def _tick(self) -> None:
        self._level += (self._target_level - self._level) * 0.25
        self._spectrum = [
            current + (target - current) * 0.2
            for current, target in zip(self._spectrum, self._target_spectrum)
        ]
        self._phase += 0.35 + self._level * 0.6
        self.update()

    def _map_spectrum(self, spectrum: list[float]) -> list[float]:
        if not spectrum:
            return [0.0] * self._bar_count
        if len(spectrum) == self._bar_count:
            return spectrum

        mapped: list[float] = []
        last_index = len(spectrum) - 1
        if last_index <= 0:
            return [spectrum[0]] * self._bar_count

        for idx in range(self._bar_count):
            position = idx * last_index / (self._bar_count - 1)
            low = int(math.floor(position))
            high = min(low + 1, last_index)
            blend = position - low
            value = spectrum[low] * (1.0 - blend) + spectrum[high] * blend
            mapped.append(value)
        return mapped

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._bar_color)

        width = self.width()
        height = self.height()

        spacing = 4
        bar_width = max(2, min(4, int((width - spacing * (self._bar_count - 1)) / self._bar_count)))
        total_width = self._bar_count * bar_width + (self._bar_count - 1) * spacing
        start_x = (width - total_width) / 2

        max_height = height - 2
        min_height = max(3.0, max_height * 0.25)
        volume = min(1.0, math.sqrt(self._level) * 1.2)
        energy = 0.2 + 0.8 * volume

        for index in range(self._bar_count):
            phase = self._phase + index * 0.55
            wave = (math.sin(phase) + 1.0) / 2.0
            band = self._spectrum[index]
            motion = (wave * 0.35) + (band * 0.65)
            bar_height = min_height + (max_height - min_height) * motion * energy
            x = start_x + index * (bar_width + spacing)
            y = (height - bar_height) / 2
            painter.drawRoundedRect(x, y, bar_width, bar_height, bar_width / 6, bar_width / 6)

        painter.end()


class RecordingToast(QWidget):
    """Floating toast shown while recording audio."""

    def __init__(self, app_name: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.Tool
            | Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.NoFocus)

        self._shown_pos = QPoint()
        self._hidden_pos = QPoint()
        self._hide_after_animation = False

        self._waveform = WaveformWidget()
        self._waveform.set_bar_color(QColor("#F5EAD9"))

        self._build_ui(app_name)
        self._build_animation()

    def _build_ui(self, app_name: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        header = QFrame()
        header.setObjectName("RecordingToastHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 10, 16, 10)
        header_layout.setAlignment(Qt.AlignCenter)

        title = QLabel(f"Dictating with {app_name}")
        title.setObjectName("RecordingToastTitle")
        title.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title)

        wave_frame = QFrame()
        wave_frame.setObjectName("RecordingToastWave")
        wave_layout = QHBoxLayout(wave_frame)
        wave_layout.setContentsMargins(12, 6, 12, 6)
        wave_layout.setAlignment(Qt.AlignCenter)
        wave_layout.addWidget(self._waveform)

        layout.addWidget(header, alignment=Qt.AlignCenter)
        layout.addWidget(wave_frame, alignment=Qt.AlignCenter)

        self.setStyleSheet(
            """
            QFrame#RecordingToastHeader {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0C0C0C, stop:1 #1A1713);
                border: 1px solid rgba(255, 255, 255, 20);
                border-radius: 20px;
            }
            QFrame#RecordingToastWave {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0A0A0A, stop:1 #13100D);
                border: 1px solid rgba(255, 255, 255, 18);
                border-radius: 16px;
            }
            QLabel#RecordingToastTitle {
                color: #F7EEDD;
                font-size: 13px;
                font-weight: 600;
                font-family: "Avenir Next", "Futura", "Gill Sans";
                letter-spacing: 0.3px;
            }
            """
        )

        self.adjustSize()
        self.hide()

    def _build_animation(self) -> None:
        self._anim_group = QParallelAnimationGroup(self)
        self._opacity_anim = QPropertyAnimation(self, b"windowOpacity")
        self._pos_anim = QPropertyAnimation(self, b"pos")
        self._opacity_anim.setDuration(160)
        self._pos_anim.setDuration(160)
        self._anim_group.addAnimation(self._opacity_anim)
        self._anim_group.addAnimation(self._pos_anim)
        self._anim_group.finished.connect(self._on_animation_finished)

    def _update_positions(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        geometry = screen.availableGeometry()
        self.adjustSize()
        width = self.width()
        height = self.height()
        x = int(geometry.center().x() - width / 2)
        y = int(geometry.bottom() - height - 48)
        self._shown_pos = QPoint(x, y)
        self._hidden_pos = QPoint(x, y + 12)

    def show_recording(self) -> None:
        """Show the toast with a short animation."""
        self._update_positions()
        self._hide_after_animation = False
        self._waveform.start()
        self._anim_group.stop()
        self.setWindowOpacity(0.0)
        self.move(self._hidden_pos)
        self.show()
        self.raise_()
        self._opacity_anim.setStartValue(self.windowOpacity())
        self._opacity_anim.setEndValue(1.0)
        self._pos_anim.setStartValue(self.pos())
        self._pos_anim.setEndValue(self._shown_pos)
        self._anim_group.start()

    def hide_recording(self) -> None:
        """Hide the toast with a short animation."""
        if not self.isVisible():
            return
        self._update_positions()
        self._hide_after_animation = True
        self._anim_group.stop()
        self._opacity_anim.setStartValue(self.windowOpacity())
        self._opacity_anim.setEndValue(0.0)
        self._pos_anim.setStartValue(self.pos())
        self._pos_anim.setEndValue(self._hidden_pos)
        self._anim_group.start()

    def set_level(self, level: float) -> None:
        """Update the waveform intensity."""
        self._waveform.set_level(level)

    def set_spectrum(self, spectrum: list[float]) -> None:
        """Update the waveform spectrum band energies."""
        self._waveform.set_spectrum(spectrum)

    def _on_animation_finished(self) -> None:
        if self._hide_after_animation:
            self.hide()
            self._waveform.stop()
