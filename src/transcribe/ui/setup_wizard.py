"""
First-run setup wizard for new users.

Provides a guided setup experience for configuring the application.
"""

from PySide6.QtWidgets import (
    QWizard, QWizardPage, QVBoxLayout, QLabel, QComboBox,
    QKeySequenceEdit, QPushButton, QHBoxLayout
)
from PySide6.QtGui import QKeySequence
from PySide6.QtCore import Qt
from typing import Optional

from ..core.settings import Settings, get_settings, HotkeyConfig
from ..core.recorder import AudioRecorder
from ..utils.platform import (
    get_platform, check_accessibility_permissions, 
    request_accessibility_permissions
)


class WelcomePage(QWizardPage):
    """Welcome page introducing the application."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Welcome to Transcribe")
        self.setSubTitle("Let's set up your speech-to-text assistant.")
        
        layout = QVBoxLayout(self)
        
        intro = QLabel(
            "Transcribe lets you dictate text anywhere on your computer.\n\n"
            "Just hold down a hotkey and speak — your words will be "
            "typed automatically.\n\n"
            "This wizard will help you configure:\n"
            "• Microphone input\n"
            "• Push-to-talk hotkey\n"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        layout.addStretch()


class PermissionsPage(QWizardPage):
    """macOS permissions page for accessibility access."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Accessibility Permissions")
        self.setSubTitle("Transcribe needs permission to type on your behalf.")
        
        layout = QVBoxLayout(self)
        
        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)
        
        # Button to open System Preferences
        btn_layout = QHBoxLayout()
        self._open_prefs_btn = QPushButton("Open System Preferences")
        self._open_prefs_btn.clicked.connect(request_accessibility_permissions)
        btn_layout.addWidget(self._open_prefs_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        self._check_btn = QPushButton("Check Permission")
        self._check_btn.clicked.connect(self._check_permission)
        btn_layout.addWidget(self._check_btn)
        
        layout.addStretch()
        
        self._update_status()
    
    def _update_status(self) -> None:
        """Update the permission status display."""
        if check_accessibility_permissions():
            self._status_label.setText(
                "✅ Accessibility permission granted!\n\n"
                "You're all set. Click Next to continue."
            )
            self._status_label.setStyleSheet("color: green;")
            self._open_prefs_btn.setEnabled(False)
        else:
            self._status_label.setText(
                "⚠️ Accessibility permission required.\n\n"
                "Transcribe needs accessibility access to:\n"
                "• Listen for your hotkey\n"
                "• Type the transcribed text\n\n"
                "Click the button below to open System Preferences, "
                "then add Transcribe to the allowed apps."
            )
            self._status_label.setStyleSheet("")
    
    def _check_permission(self) -> None:
        """Re-check permission status."""
        self._update_status()
    
    def isComplete(self) -> bool:
        """Allow proceeding even without permission (user knows the risk)."""
        return True


class MicrophonePage(QWizardPage):
    """Microphone selection page."""
    
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
        
        # Refresh button
        refresh_btn = QPushButton("Refresh Devices")
        refresh_btn.clicked.connect(self._refresh_devices)
        layout.addWidget(refresh_btn)
        
        layout.addStretch()
        
        # Register field for wizard data access
        self.registerField("input_device", self._device_combo, "currentData")
    
    def _refresh_devices(self) -> None:
        """Refresh the device list."""
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
        """Get the selected device name."""
        return self._device_combo.currentData()


class HotkeyPage(QWizardPage):
    """Hotkey configuration page."""
    
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
        """Parse the hotkey into a HotkeyConfig."""
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
    """Setup complete page."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Setup Complete!")
        self.setSubTitle("You're ready to start using Transcribe.")
        
        layout = QVBoxLayout(self)
        
        summary = QLabel(
            "Your settings have been saved.\n\n"
            "Look for the Transcribe icon in your system tray. "
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
    """First-run setup wizard."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Transcribe Setup")
        self.setWizardStyle(QWizard.ModernStyle)
        self.setMinimumSize(500, 400)
        
        self._settings = get_settings()
        
        # Add pages
        self.addPage(WelcomePage())
        
        # Only show permissions page on macOS
        if get_platform() == "macos":
            self.addPage(PermissionsPage())
        
        self._mic_page = MicrophonePage()
        self.addPage(self._mic_page)
        
        self._hotkey_page = HotkeyPage()
        self.addPage(self._hotkey_page)
        
        self.addPage(CompletePage())
    
    def accept(self) -> None:
        """Save settings when wizard completes."""
        # Save microphone selection
        self._settings.input_device = self._mic_page.get_selected_device()
        
        # Save hotkey configuration
        self._settings.hotkey = self._hotkey_page.get_hotkey_config()
        
        # Mark first run as complete
        self._settings.first_run_complete = True
        
        self._settings.save()
        
        super().accept()
