
from typing import Optional, List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QComboBox, QCheckBox, QPushButton, QGroupBox, QFormLayout,
    QKeySequenceEdit, QLineEdit, QMessageBox, QScrollArea
)
from PySide6.QtGui import QKeySequence
from PySide6.QtCore import Qt, Signal

from ...core.settings import Settings, HotkeyConfig
from ...core.audio import AudioRecorder
from ...core.asr.model_utils import get_installed_asr_models, delete_asr_model

# Special value for custom model entry in combo box
_CUSTOM_MODEL_ENTRY = "__custom_model__"


class ConfigurationTab(QWidget):

    gpu_setting_changed = Signal(bool)

    reset_requested = Signal()

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._setup_ui()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)

        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        layout.addWidget(self._create_general_section())
        layout.addWidget(self._create_audio_section())
        layout.addWidget(self._create_hotkey_section())
        layout.addWidget(self._create_asr_model_section())

        layout.addStretch()

        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

    def _create_general_section(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        typing_group = QGroupBox("Typing Behavior")
        typing_layout = QFormLayout(typing_group)

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

        self._instant_type_cb = QCheckBox("Instantly output text")
        self._instant_type_cb.toggled.connect(self._on_instant_type_toggled)
        typing_layout.addRow("", self._instant_type_cb)

        layout.addWidget(typing_group)

        startup_group = QGroupBox("Startup")
        startup_layout = QVBoxLayout(startup_group)

        self._start_minimized_cb = QCheckBox("Start minimized to tray")
        startup_layout.addWidget(self._start_minimized_cb)

        self._autostart_cb = QCheckBox("Start automatically on login")
        startup_layout.addWidget(self._autostart_cb)

        layout.addWidget(startup_group)
        return widget

    def _on_instant_type_toggled(self, checked: bool) -> None:
        self._speed_slider.setEnabled(not checked)
        self._speed_label.setEnabled(not checked)

    def _create_audio_section(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        audio_group = QGroupBox("Audio")
        audio_layout = QFormLayout(audio_group)

        device_layout = QHBoxLayout()
        self._device_combo = QComboBox()
        self._refresh_devices()
        device_layout.addWidget(self._device_combo, 1)

        refresh_btn = QPushButton("⟳")
        refresh_btn.setFixedWidth(30)
        refresh_btn.setToolTip("Refresh device list")
        refresh_btn.clicked.connect(self._refresh_devices)
        device_layout.addWidget(refresh_btn)

        audio_layout.addRow("Input Device:", device_layout)

        self._sample_rate_combo = QComboBox()
        self._sample_rate_combo.addItem("16000 Hz (Recommended)", 16000)
        self._sample_rate_combo.addItem("22050 Hz", 22050)
        self._sample_rate_combo.addItem("44100 Hz", 44100)
        audio_layout.addRow("Sample Rate:", self._sample_rate_combo)

        layout.addWidget(audio_group)
        return widget

    def _refresh_devices(self) -> None:
        current = self._device_combo.currentData() if hasattr(self, '_device_combo') else None
        self._device_combo.clear()
        self._device_combo.addItem("System Default", None)

        for device in AudioRecorder.list_devices():
            self._device_combo.addItem(device.name, device.name)

        if current:
            idx = self._device_combo.findData(current)
            if idx >= 0:
                self._device_combo.setCurrentIndex(idx)

    def _create_hotkey_section(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        hotkey_group = QGroupBox("Push-to-Talk Hotkey")
        hotkey_layout = QFormLayout(hotkey_group)

        self._hotkey_edit = QKeySequenceEdit()
        hotkey_layout.addRow("Hold to record:", self._hotkey_edit)

        instructions = QLabel(
            "Click the field above and press your desired key combination.\n"
            "The recording will start when you hold the keys and stop when you release."
        )
        instructions.setStyleSheet("color: gray; font-size: 11px;")
        instructions.setWordWrap(True)
        hotkey_layout.addRow("", instructions)

        layout.addWidget(hotkey_group)
        return widget

    def _create_asr_model_section(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        model_group = QGroupBox("ASR Model")
        model_layout = QFormLayout(model_group)

        model_row_layout = QHBoxLayout()
        self._model_combo = QComboBox()
        self._model_combo.setStyleSheet("font-family: monospace;")
        model_row_layout.addWidget(self._model_combo, 1)

        self._delete_model_btn = QPushButton("🗑")
        self._delete_model_btn.setFixedWidth(30)
        self._delete_model_btn.setToolTip("Delete selected model from cache")
        self._delete_model_btn.clicked.connect(self._on_delete_model_clicked)
        model_row_layout.addWidget(self._delete_model_btn)

        self._refresh_model_list()
        self._model_combo.currentIndexChanged.connect(self._on_model_combo_changed)

        model_layout.addRow("Model:", model_row_layout)

        self._custom_model_edit = QLineEdit()
        self._custom_model_edit.setPlaceholderText("e.g. nvidia/parakeet-tdt-0.6b-v3")
        self._custom_model_edit.setStyleSheet("font-family: monospace;")
        self._custom_model_edit.hide()  # Hidden until "Custom model..." is selected
        model_layout.addRow("", self._custom_model_edit)

        self._custom_model_instructions = QLabel(
            "Enter a valid HuggingFace or NVIDIA model name.\n"
            "Model will be validated and downloaded on save."
        )
        self._custom_model_instructions.setStyleSheet("color: gray; font-size: 11px;")
        self._custom_model_instructions.setWordWrap(True)
        self._custom_model_instructions.hide()  # Hidden until "Custom model..." is selected
        model_layout.addRow("", self._custom_model_instructions)

        self._use_gpu_cb = QCheckBox("Use GPU acceleration (if available)")
        model_layout.addRow("", self._use_gpu_cb)

        layout.addWidget(model_group)

        reset_btn = QPushButton("Reset All Settings to Defaults")
        reset_btn.clicked.connect(self.reset_requested.emit)
        layout.addWidget(reset_btn)
        return widget

    def _refresh_model_list(self) -> None:
        current_data = self._model_combo.currentData() if self._model_combo.count() > 0 else None
        self._model_combo.clear()

        installed_models = get_installed_asr_models()
        for model in installed_models:
            self._model_combo.addItem(model, model)

        self._model_combo.addItem("Custom model...", _CUSTOM_MODEL_ENTRY)

        if current_data:
            idx = self._model_combo.findData(current_data)
            if idx >= 0:
                self._model_combo.setCurrentIndex(idx)

        self._update_delete_button_state()

    def refresh_model_list(self) -> None:
        """Public method to refresh the model list. Called after model loading."""
        self._refresh_model_list()

    def _update_delete_button_state(self) -> None:
        current_data = self._model_combo.currentData()
        is_custom = current_data == _CUSTOM_MODEL_ENTRY
        is_active_model = current_data == self._settings.model_name
        self._delete_model_btn.setEnabled(not is_custom and not is_active_model)
        if is_active_model and not is_custom:
            self._delete_model_btn.setToolTip("Cannot delete currently active model")
        else:
            self._delete_model_btn.setToolTip("Delete selected model from cache")

    def _on_model_combo_changed(self, index: int) -> None:
        is_custom = self._model_combo.currentData() == _CUSTOM_MODEL_ENTRY
        self._custom_model_edit.setVisible(is_custom)
        self._custom_model_instructions.setVisible(is_custom)
        self._update_delete_button_state()

    def _on_delete_model_clicked(self) -> None:
        model_name = self._model_combo.currentData()
        if not model_name or model_name == _CUSTOM_MODEL_ENTRY:
            return

        reply = QMessageBox.question(
            self,
            "Delete Model",
            f"Are you sure you want to delete '{model_name}' from the cache?\n\n"
            "This will free up disk space but the model will need to be\n"
            "re-downloaded if you want to use it again.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        success, message = delete_asr_model(model_name)

        if success:
            QMessageBox.information(self, "Model Deleted", message)
            self._refresh_model_list()
        else:
            QMessageBox.warning(self, "Deletion Failed", message)

    def _load_model_selection(self, model_name: str) -> None:
        idx = self._model_combo.findData(model_name)
        if idx >= 0:
            # Model is in the list, select it
            self._model_combo.setCurrentIndex(idx)
        else:
            custom_idx = self._model_combo.findData(_CUSTOM_MODEL_ENTRY)
            if custom_idx >= 0:
                self._model_combo.setCurrentIndex(custom_idx)
            self._custom_model_edit.setText(model_name)
            self._on_model_combo_changed(self._model_combo.currentIndex())

    def _get_selected_model_name(self) -> str:
        if self._model_combo.currentData() == _CUSTOM_MODEL_ENTRY:
            return self._custom_model_edit.text().strip()
        else:
            return self._model_combo.currentData() or ""

    def set_gpu_enabled(self, enabled: bool) -> None:
        self._use_gpu_cb.setEnabled(enabled)

    def load_settings(self) -> None:
        self._speed_slider.setValue(self._settings.characters_per_second)
        self._instant_type_cb.setChecked(self._settings.instant_type)
        self._on_instant_type_toggled(self._settings.instant_type)

        self._start_minimized_cb.setChecked(self._settings.start_minimized)
        self._autostart_cb.setChecked(self._settings.auto_start_on_login)
        self._use_gpu_cb.setChecked(self._settings.use_gpu)

        self._load_model_selection(self._settings.model_name)

        hotkey = self._settings.hotkey
        key_sequence = "+".join(hotkey.modifiers + [hotkey.key])
        self._hotkey_edit.setKeySequence(QKeySequence(key_sequence))

        idx = self._sample_rate_combo.findData(self._settings.sample_rate)
        if idx >= 0:
            self._sample_rate_combo.setCurrentIndex(idx)

        if self._settings.input_device:
            idx = self._device_combo.findData(self._settings.input_device)
            if idx >= 0:
                self._device_combo.setCurrentIndex(idx)

    def save_settings(self) -> bool:
        model_name = self._get_selected_model_name()
        if model_name and model_name != self._settings.model_name:
            if not self._validate_model_name(model_name):
                return False

        self._settings.characters_per_second = self._speed_slider.value()
        self._settings.instant_type = self._instant_type_cb.isChecked()
        self._settings.start_minimized = self._start_minimized_cb.isChecked()
        self._settings.auto_start_on_login = self._autostart_cb.isChecked()
        self._settings.use_gpu = self._use_gpu_cb.isChecked()
        self._settings.sample_rate = self._sample_rate_combo.currentData()
        self._settings.input_device = self._device_combo.currentData()

        if model_name:
            self._settings.model_name = model_name

        hotkey_config = self._parse_key_sequence()
        if hotkey_config:
            self._settings.hotkey = hotkey_config

        return True

    def _validate_model_name(self, model_name: str) -> bool:
        if "/" not in model_name or len(model_name.split("/")) != 2:
            QMessageBox.warning(
                self,
                "Invalid Model Name",
                f"Model name '{model_name}' is not in the correct format.\n\n"
                "Expected format: organization/model-name\n"
                "Example: nvidia/parakeet-tdt-0.6b-v3 or openai/whisper-large-v3"
            )
            return False

        org, model = model_name.split("/")
        if not org or not model:
            QMessageBox.warning(
                self,
                "Invalid Model Name",
                "Both organization and model name must be provided."
            )
            return False

        return True

    def _parse_key_sequence(self) -> Optional[HotkeyConfig]:
        key_sequence = self._hotkey_edit.keySequence()
        if key_sequence.isEmpty():
            return None

        seq_str = key_sequence.toString()
        if not seq_str:
            return None

        parts = seq_str.split("+")
        if not parts:
            return None

        modifiers: List[str] = []
        key = parts[-1].lower()

        for part in parts[:-1]:
            mod = part.lower()
            if mod in ("ctrl", "control"):
                modifiers.append("ctrl")
            elif mod in ("alt", "option"):
                modifiers.append("alt")
            elif mod in ("shift",):
                modifiers.append("shift")
            elif mod in ("meta", "cmd", "command", "win"):
                modifiers.append("cmd")

        return HotkeyConfig(modifiers=modifiers, key=key)
