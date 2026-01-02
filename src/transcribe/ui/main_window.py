"""
Settings window with tabbed interface.

Provides a GUI for configuring application settings.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QLabel, QSlider, QComboBox, QCheckBox, QPushButton,
    QGroupBox, QFormLayout, QSpinBox, QKeySequenceEdit,
    QDialogButtonBox
)
from PySide6.QtCore import Qt, Signal
from typing import Optional

from ..core.settings import Settings, get_settings
from ..core.recorder import AudioRecorder


class SettingsWindow(QDialog):
    """
    Settings dialog with tabs for different configuration categories.
    
    Signals:
        settings_changed: Emitted when settings are saved
    """
    
    settings_changed = Signal()
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self.setWindowTitle("Transcribe Settings")
        self.setMinimumWidth(450)
        self.setMinimumHeight(400)
        
        self._settings = get_settings()
        self._setup_ui()
        self._load_settings()
    
    def _setup_ui(self) -> None:
        """Build the UI layout."""
        layout = QVBoxLayout(self)
        
        # Tab widget
        tabs = QTabWidget()
        tabs.addTab(self._create_general_tab(), "General")
        tabs.addTab(self._create_hotkey_tab(), "Hotkeys")
        tabs.addTab(self._create_audio_tab(), "Audio")
        tabs.addTab(self._create_advanced_tab(), "Advanced")
        layout.addWidget(tabs)
        
        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        buttons.accepted.connect(self._save_and_close)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self._save_settings)
        layout.addWidget(buttons)
    
    def _create_general_tab(self) -> QWidget:
        """Create the General settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Typing behavior group
        typing_group = QGroupBox("Typing Behavior")
        typing_layout = QFormLayout(typing_group)
        
        # Typing speed slider
        speed_layout = QHBoxLayout()
        self._speed_slider = QSlider(Qt.Horizontal)
        self._speed_slider.setRange(0, 300)
        self._speed_slider.setTickPosition(QSlider.TicksBelow)
        self._speed_slider.setTickInterval(50)
        self._speed_label = QLabel("150 chars/sec")
        self._speed_slider.valueChanged.connect(
            lambda v: self._speed_label.setText(f"{v} chars/sec" if v > 0 else "Instant")
        )
        speed_layout.addWidget(self._speed_slider)
        speed_layout.addWidget(self._speed_label)
        typing_layout.addRow("Typing Speed:", speed_layout)
        
        # Auto-type checkbox
        self._auto_type_cb = QCheckBox("Automatically type transcribed text")
        typing_layout.addRow("", self._auto_type_cb)
        
        layout.addWidget(typing_group)
        
        # Startup group
        startup_group = QGroupBox("Startup")
        startup_layout = QVBoxLayout(startup_group)
        
        self._start_minimized_cb = QCheckBox("Start minimized to tray")
        startup_layout.addWidget(self._start_minimized_cb)
        
        self._autostart_cb = QCheckBox("Start automatically on login")
        startup_layout.addWidget(self._autostart_cb)
        
        layout.addWidget(startup_group)
        
        # Notifications group
        notif_group = QGroupBox("Notifications")
        notif_layout = QVBoxLayout(notif_group)
        
        self._notifications_cb = QCheckBox("Show notifications when transcription completes")
        notif_layout.addWidget(self._notifications_cb)
        
        layout.addWidget(notif_group)
        
        layout.addStretch()
        return widget
    
    def _create_hotkey_tab(self) -> QWidget:
        """Create the Hotkeys settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        hotkey_group = QGroupBox("Push-to-Talk Hotkey")
        hotkey_layout = QFormLayout(hotkey_group)
        
        # Hotkey input
        self._hotkey_edit = QKeySequenceEdit()
        hotkey_layout.addRow("Hold to record:", self._hotkey_edit)
        
        # Instructions
        instructions = QLabel(
            "Click the field above and press your desired key combination.\n"
            "The recording will start when you hold the keys and stop when you release."
        )
        instructions.setStyleSheet("color: gray; font-size: 11px;")
        instructions.setWordWrap(True)
        hotkey_layout.addRow("", instructions)
        
        layout.addWidget(hotkey_group)
        layout.addStretch()
        return widget
    
    def _create_audio_tab(self) -> QWidget:
        """Create the Audio settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        audio_group = QGroupBox("Audio Input")
        audio_layout = QFormLayout(audio_group)
        
        # Input device dropdown
        self._device_combo = QComboBox()
        self._device_combo.addItem("System Default", None)
        
        # Populate with available devices
        for device in AudioRecorder.list_devices():
            self._device_combo.addItem(device.name, device.name)
        
        audio_layout.addRow("Input Device:", self._device_combo)
        
        # Sample rate dropdown
        self._sample_rate_combo = QComboBox()
        self._sample_rate_combo.addItem("16000 Hz (Recommended)", 16000)
        self._sample_rate_combo.addItem("22050 Hz", 22050)
        self._sample_rate_combo.addItem("44100 Hz", 44100)
        audio_layout.addRow("Sample Rate:", self._sample_rate_combo)
        
        layout.addWidget(audio_group)
        layout.addStretch()
        return widget
    
    def _create_advanced_tab(self) -> QWidget:
        """Create the Advanced settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        model_group = QGroupBox("Model")
        model_layout = QFormLayout(model_group)
        
        # Model selection (read-only for now)
        self._model_label = QLabel("nvidia/parakeet-tdt-0.6b-v3")
        self._model_label.setStyleSheet("font-family: monospace;")
        model_layout.addRow("Current Model:", self._model_label)
        
        # GPU checkbox
        self._use_gpu_cb = QCheckBox("Use GPU acceleration (if available)")
        model_layout.addRow("", self._use_gpu_cb)
        
        layout.addWidget(model_group)
        
        # Reset button
        reset_btn = QPushButton("Reset All Settings to Defaults")
        reset_btn.clicked.connect(self._reset_settings)
        layout.addWidget(reset_btn)
        
        layout.addStretch()
        return widget
    
    def _load_settings(self) -> None:
        """Load current settings into the UI."""
        self._speed_slider.setValue(self._settings.characters_per_second)
        self._auto_type_cb.setChecked(self._settings.auto_type_result)
        self._start_minimized_cb.setChecked(self._settings.start_minimized)
        self._autostart_cb.setChecked(self._settings.auto_start_on_login)
        self._notifications_cb.setChecked(self._settings.show_notifications)
        self._use_gpu_cb.setChecked(self._settings.use_gpu)
        
        # Set hotkey (simplified - full implementation would parse the HotkeyConfig)
        # self._hotkey_edit.setKeySequence(...)
        
        # Set sample rate
        idx = self._sample_rate_combo.findData(self._settings.sample_rate)
        if idx >= 0:
            self._sample_rate_combo.setCurrentIndex(idx)
        
        # Set input device
        if self._settings.input_device:
            idx = self._device_combo.findData(self._settings.input_device)
            if idx >= 0:
                self._device_combo.setCurrentIndex(idx)
    
    def _save_settings(self) -> None:
        """Save UI values to settings."""
        self._settings.characters_per_second = self._speed_slider.value()
        self._settings.auto_type_result = self._auto_type_cb.isChecked()
        self._settings.start_minimized = self._start_minimized_cb.isChecked()
        self._settings.auto_start_on_login = self._autostart_cb.isChecked()
        self._settings.show_notifications = self._notifications_cb.isChecked()
        self._settings.use_gpu = self._use_gpu_cb.isChecked()
        self._settings.sample_rate = self._sample_rate_combo.currentData()
        self._settings.input_device = self._device_combo.currentData()
        
        self._settings.save()
        self.settings_changed.emit()
    
    def _save_and_close(self) -> None:
        """Save settings and close the dialog."""
        self._save_settings()
        self.accept()
    
    def _reset_settings(self) -> None:
        """Reset all settings to defaults."""
        self._settings.reset_to_defaults()
        self._load_settings()
