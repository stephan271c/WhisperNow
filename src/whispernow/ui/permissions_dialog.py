from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ..utils.logger import get_logger
from ..utils.platform import (
    check_accessibility_permissions,
    check_input_monitoring_permissions,
    request_accessibility_permissions,
    request_input_monitoring_permissions,
)

logger = get_logger(__name__)


class PermissionsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Permissions Required")
        self.setMinimumWidth(450)
        self.setModal(True)

        self._setup_ui()
        self._update_status()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        title = QLabel("Permissions Required")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        explanation = QLabel(
            "WhisperNow needs permission to:\n\n"
            "• Listen for your push-to-talk hotkey (Input Monitoring)\n"
            "• Type the transcribed text into applications (Accessibility)\n\n"
            "Without these permissions, the app cannot detect when you "
            "press your hotkey or output the transcribed text."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        self._status_frame = QFrame()
        self._status_frame.setFrameShape(QFrame.StyledPanel)
        status_layout = QVBoxLayout(self._status_frame)

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        status_layout.addWidget(self._status_label)

        layout.addWidget(self._status_frame)

        instructions = QLabel(
            "<b>How to grant permissions:</b><br>"
            "1. Click 'Input Monitoring Settings' below<br>"
            "2. Click the lock icon to make changes<br>"
            "3. Click '+' and navigate to Applications<br>"
            "4. Select <b>WhisperNow.app</b> and add it<br>"
            "5. Ensure the checkbox next to 'WhisperNow' is checked<br>"
            "6. Repeat in 'Accessibility Settings'<br>"
            "7. Click 'Check Again' to verify"
        )
        instructions.setTextFormat(Qt.RichText)
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        open_layout = QHBoxLayout()

        self._open_input_monitoring_btn = QPushButton("Input Monitoring Settings")
        self._open_input_monitoring_btn.clicked.connect(
            self._open_input_monitoring_preferences
        )
        open_layout.addWidget(self._open_input_monitoring_btn)

        self._open_accessibility_btn = QPushButton("Accessibility Settings")
        self._open_accessibility_btn.clicked.connect(
            self._open_accessibility_preferences
        )
        open_layout.addWidget(self._open_accessibility_btn)

        layout.addLayout(open_layout)

        button_layout = QHBoxLayout()

        self._check_btn = QPushButton("Check Again")
        self._check_btn.clicked.connect(self._check_permission)
        button_layout.addWidget(self._check_btn)

        button_layout.addStretch()

        self._continue_btn = QPushButton("Continue Anyway")
        self._continue_btn.clicked.connect(self.accept)
        button_layout.addWidget(self._continue_btn)

        layout.addLayout(button_layout)

    def _update_status(self) -> None:
        accessibility_granted = check_accessibility_permissions()
        input_monitoring_granted = check_input_monitoring_permissions()
        status_lines = [
            f"Accessibility: {'Granted' if accessibility_granted else 'Not granted'}",
            f"Input Monitoring: {'Granted' if input_monitoring_granted else 'Not granted'}",
        ]
        self._status_label.setText("\n".join(status_lines))

        if accessibility_granted and input_monitoring_granted:
            self._status_frame.setStyleSheet(
                "QFrame { background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 4px; }"
            )
            self._open_input_monitoring_btn.setEnabled(False)
            self._open_accessibility_btn.setEnabled(False)
            self._check_btn.setEnabled(False)
            self._continue_btn.setText("Done")
            logger.info("Accessibility and Input Monitoring permissions granted")
        else:
            self._status_frame.setStyleSheet(
                "QFrame { background-color: #fff3cd; border: 1px solid #ffeeba; border-radius: 4px; }"
            )
            self._open_input_monitoring_btn.setEnabled(not input_monitoring_granted)
            self._open_accessibility_btn.setEnabled(not accessibility_granted)
            self._check_btn.setEnabled(True)
            self._continue_btn.setText("Continue Anyway")

    def _open_input_monitoring_preferences(self) -> None:
        logger.info("Opening System Preferences for Input Monitoring permissions")
        request_input_monitoring_permissions()

    def _open_accessibility_preferences(self) -> None:
        logger.info("Opening System Preferences for accessibility permissions")
        request_accessibility_permissions()

    def _check_permission(self) -> None:
        self._update_status()

    def permissions_granted(self) -> bool:
        return (
            check_accessibility_permissions() and check_input_monitoring_permissions()
        )
