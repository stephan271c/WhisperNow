from typing import Optional

from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWizard,
    QWizardPage,
)

from ..core.asr.models.registry import get_all_models_with_status, is_model_downloaded
from ..core.audio import AudioRecorder
from ..core.settings import HotkeyConfig, Settings, get_settings
from .download_dialog import ModelDownloadThread


class WelcomePage(QWizardPage):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Welcome to WhisperNow")
        self.setSubTitle("Let's set up your speech-to-text assistant.")

        layout = QVBoxLayout(self)

        intro = QLabel(
            "WhisperNow lets you dictate text anywhere on your computer.\n\n"
            "Just hold down a hotkey and speak — your words will be "
            "typed automatically.\n\n"
            "This wizard will help you configure:\n"
            "• Speech recognition model\n"
            "• Microphone input\n"
            "• Push-to-talk hotkey\n"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        layout.addStretch()


class MicrophonePage(QWizardPage):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Select Microphone")
        self.setSubTitle("Choose which microphone to use for recording.")

        layout = QVBoxLayout(self)

        self._device_combo = QComboBox()
        self._device_combo.addItem("System Default", None)

        for device in AudioRecorder.list_devices():
            self._device_combo.addItem(device.name, device.name)

        layout.addWidget(QLabel("Input Device:"))
        layout.addWidget(self._device_combo)

        refresh_btn = QPushButton("Refresh Devices")
        refresh_btn.clicked.connect(self._refresh_devices)
        layout.addWidget(refresh_btn)

        layout.addStretch()

        self.registerField("input_device", self._device_combo, "currentData")

    def _refresh_devices(self) -> None:
        current = self._device_combo.currentData()
        self._device_combo.clear()
        self._device_combo.addItem("System Default", None)

        for device in AudioRecorder.list_devices():
            self._device_combo.addItem(device.name, device.name)

        if current:
            idx = self._device_combo.findData(current)
            if idx >= 0:
                self._device_combo.setCurrentIndex(idx)

    def get_selected_device(self) -> Optional[str]:
        return self._device_combo.currentData()


class ModelSelectionPage(QWizardPage):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Select Speech Recognition Model")
        self.setSubTitle("Choose which model to use for transcription.")

        layout = QVBoxLayout(self)

        model_row = QHBoxLayout()
        self._model_combo = QComboBox()
        self._model_combo.setStyleSheet("font-family: monospace;")
        self._refresh_model_list()
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        model_row.addWidget(self._model_combo, 1)

        self._download_btn = QPushButton("Download")
        self._download_btn.setFixedWidth(80)
        self._download_btn.clicked.connect(self._on_download_clicked)
        model_row.addWidget(self._download_btn)

        layout.addWidget(QLabel("Model:"))
        layout.addLayout(model_row)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.hide()
        layout.addWidget(self._progress_bar)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.hide()
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        layout.addWidget(self._cancel_btn)

        self._status_label = QLabel()
        self._status_label.setStyleSheet("color: gray; font-size: 11px;")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        instructions = QLabel(
            "The recommended model (Parakeet TDT v2) offers excellent accuracy.\n"
            "Smaller models like Whisper Tiny are faster but less accurate."
        )
        instructions.setStyleSheet("color: gray; font-size: 11px;")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        layout.addStretch()

        self._download_thread: Optional[ModelDownloadThread] = None
        self._update_button_state()

    def _refresh_model_list(self) -> None:
        self._model_combo.clear()
        for model, status in get_all_models_with_status():
            indicator = "✓" if status == "downloaded" else "↓"
            display_text = f"{model.name}  {indicator}"
            self._model_combo.addItem(display_text, model.id)

    def _on_model_changed(self, index: int) -> None:
        self._update_button_state()
        self.completeChanged.emit()

    def _update_button_state(self) -> None:
        model_id = self._model_combo.currentData()
        downloaded = is_model_downloaded(model_id) if model_id else False
        self._download_btn.setEnabled(not downloaded)
        if downloaded:
            self._status_label.setText("✓ Model is ready")
        else:
            self._status_label.setText("↓ Model needs to be downloaded")

    def _on_download_clicked(self) -> None:
        model_id = self._model_combo.currentData()
        if not model_id:
            return

        self._progress_bar.setValue(0)
        self._progress_bar.show()
        self._cancel_btn.show()
        self._download_btn.setEnabled(False)
        self._model_combo.setEnabled(False)

        self._download_thread = ModelDownloadThread(model_id)
        self._download_thread.progress.connect(self._on_progress)
        self._download_thread.status_changed.connect(self._on_status_changed)
        self._download_thread.finished.connect(self._on_download_finished)
        self._download_thread.start()

    def _on_progress(self, downloaded: int, total: int) -> None:
        if total > 0:
            percent = int((downloaded / total) * 100)
            self._progress_bar.setValue(percent)

    def _on_status_changed(self, status: str) -> None:
        self._status_label.setText(status)

    def _on_download_finished(self, success: bool) -> None:
        self._progress_bar.hide()
        self._cancel_btn.hide()
        self._model_combo.setEnabled(True)
        self._download_thread = None

        current_model_id = self._model_combo.currentData()
        if success:
            self._refresh_model_list()
        if current_model_id:
            idx = self._model_combo.findData(current_model_id)
            if idx >= 0:
                self._model_combo.setCurrentIndex(idx)
        self._update_button_state()
        self.completeChanged.emit()

    def _on_cancel_clicked(self) -> None:
        if self._download_thread:
            self._download_thread.cancel()
            self._cancel_btn.setEnabled(False)
            self._cancel_btn.setText("Cancelling...")

    def isComplete(self) -> bool:
        model_id = self._model_combo.currentData()
        return is_model_downloaded(model_id) if model_id else False

    def get_selected_model(self) -> str:
        return self._model_combo.currentData() or ""


class HotkeyPage(QWizardPage):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Set Push-to-Talk Hotkey")
        self.setSubTitle("Choose the key combination to hold while speaking.")

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Click the field and press your desired hotkey:"))

        self._hotkey_edit = QKeySequenceEdit()
        self._hotkey_edit.setKeySequence(QKeySequence("Ctrl+Space"))
        layout.addWidget(self._hotkey_edit)

        instructions = QLabel(
            "Recommended: Ctrl + Space\n\n"
            "Hold this key combination to start recording. "
            "Release to stop and transcribe."
        )
        instructions.setStyleSheet("color: gray; font-size: 11px;")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        layout.addStretch()

    def get_hotkey_config(self) -> HotkeyConfig:
        key_sequence = self._hotkey_edit.keySequence()
        if key_sequence.isEmpty():
            return HotkeyConfig()  # Default

        seq_str = key_sequence.toString()
        if not seq_str:
            return HotkeyConfig()

        parts = seq_str.split("+")
        if not parts:
            return HotkeyConfig()

        modifiers = []
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


class CompletePage(QWizardPage):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Setup Complete!")
        self.setSubTitle("You're ready to start using WhisperNow.")

        layout = QVBoxLayout(self)

        summary = QLabel(
            "Your settings have been saved.\n\n"
            "Look for the WhisperNow icon in your system tray. "
            "Right-click it to access settings or quit.\n\n"
            "To transcribe:\n"
            "1. Hold your hotkey\n"
            "2. Speak clearly\n"
            "3. Release to transcribe and type\n\n"
            "Enjoy!"
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)
        layout.addStretch()


class SetupWizard(QWizard):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("WhisperNow Setup")
        self.setWizardStyle(QWizard.ModernStyle)
        self.setMinimumSize(500, 400)

        self._settings = get_settings()

        self.addPage(WelcomePage())

        self._model_page = ModelSelectionPage()
        self.addPage(self._model_page)

        self._mic_page = MicrophonePage()
        self.addPage(self._mic_page)

        self._hotkey_page = HotkeyPage()
        self.addPage(self._hotkey_page)

        self.addPage(CompletePage())

    def accept(self) -> None:
        self._settings.model_id = self._model_page.get_selected_model()
        self._settings.input_device = self._mic_page.get_selected_device()
        self._settings.hotkey = self._hotkey_page.get_hotkey_config()
        self._settings.first_run_complete = True
        self._settings.save()

        super().accept()
