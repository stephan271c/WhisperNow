from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWizard,
    QWizardPage,
)

from ..core.audio import AudioRecorder
from ..core.settings import HotkeyConfig, Settings, get_settings


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

        self._mic_page = MicrophonePage()
        self.addPage(self._mic_page)

        self._hotkey_page = HotkeyPage()
        self.addPage(self._hotkey_page)

        self.addPage(CompletePage())

    def accept(self) -> None:
        self._settings.input_device = self._mic_page.get_selected_device()

        self._settings.hotkey = self._hotkey_page.get_hotkey_config()

        self._settings.first_run_complete = True

        self._settings.save()

        super().accept()
