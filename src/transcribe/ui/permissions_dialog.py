"""
Dialog for explaining and requesting accessibility permissions on macOS.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from ..utils.platform import check_accessibility_permissions, request_accessibility_permissions
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PermissionsDialog(QDialog):
    """
    Dialog explaining why accessibility permissions are needed.
    
    Shows on macOS when permissions haven't been granted yet.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Accessibility Permission Required")
        self.setMinimumWidth(450)
        self.setModal(True)
        
        self._setup_ui()
        self._update_status()
    
    def _setup_ui(self) -> None:
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        
        # Title
        title = QLabel("Accessibility Permission Required")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Explanation
        explanation = QLabel(
            "WhisperNow needs accessibility permission to:\n\n"
            "• Listen for your push-to-talk hotkey\n"
            "• Type the transcribed text into applications\n\n"
            "Without this permission, the app cannot detect when you "
            "press your hotkey or output the transcribed text."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        
        # Status indicator
        self._status_frame = QFrame()
        self._status_frame.setFrameShape(QFrame.StyledPanel)
        status_layout = QVBoxLayout(self._status_frame)
        
        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        status_layout.addWidget(self._status_label)
        
        layout.addWidget(self._status_frame)
        
        # Instructions
        instructions = QLabel(
            "<b>How to grant permission:</b><br>"
            "1. Click 'Open System Preferences' below<br>"
            "2. Click the lock icon to make changes<br>"
            "3. Find and check 'WhisperNow' in the list<br>"
            "4. Click 'Check Again' to verify"
        )
        instructions.setTextFormat(Qt.RichText)
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self._open_prefs_btn = QPushButton("Open System Preferences")
        self._open_prefs_btn.clicked.connect(self._open_preferences)
        button_layout.addWidget(self._open_prefs_btn)
        
        self._check_btn = QPushButton("Check Again")
        self._check_btn.clicked.connect(self._check_permission)
        button_layout.addWidget(self._check_btn)
        
        button_layout.addStretch()
        
        self._continue_btn = QPushButton("Continue Anyway")
        self._continue_btn.clicked.connect(self.accept)
        button_layout.addWidget(self._continue_btn)
        
        layout.addLayout(button_layout)
    
    def _update_status(self) -> None:
        """Update the permission status display."""
        if check_accessibility_permissions():
            self._status_label.setText("✅ Permission granted!")
            self._status_frame.setStyleSheet(
                "QFrame { background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 4px; }"
            )
            self._open_prefs_btn.setEnabled(False)
            self._check_btn.setEnabled(False)
            self._continue_btn.setText("Done")
            logger.info("Accessibility permissions granted")
        else:
            self._status_label.setText("⚠️ Permission not yet granted")
            self._status_frame.setStyleSheet(
                "QFrame { background-color: #fff3cd; border: 1px solid #ffeeba; border-radius: 4px; }"
            )
            self._open_prefs_btn.setEnabled(True)
            self._check_btn.setEnabled(True)
            self._continue_btn.setText("Continue Anyway")
    
    def _open_preferences(self) -> None:
        """Open System Preferences to the Accessibility pane."""
        logger.info("Opening System Preferences for accessibility permissions")
        request_accessibility_permissions()
    
    def _check_permission(self) -> None:
        """Re-check permission status."""
        self._update_status()
    
    def permissions_granted(self) -> bool:
        """Check if permissions are currently granted."""
        return check_accessibility_permissions()
