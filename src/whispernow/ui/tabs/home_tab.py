from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core.settings.data_manager import (
    clear_user_data,
    get_all_data_paths,
    schedule_cleanup_and_exit,
)
from ...utils.logger import shutdown_logging


class HomeTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        title = QLabel("Get started")
        title.setStyleSheet("font-size: 14px; font-weight: 600;")
        layout.addWidget(title)

        intro = QLabel("Choose a section on the left to configure WhisperNow.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        features = QLabel(
            "<ul>"
            "<li><b>Mode</b>: Configure enhancements for different tasks.</li>"
            "<li><b>Vocabulary</b>: Word substitutions.</li>"
            "<li><b>Configuration</b>: General behavior, audio input, hotkeys, and ASR model.</li>"
            "<li><b>History</b>: Review recent transcriptions.</li>"
            "</ul>"
        )
        features.setWordWrap(True)
        features.setTextFormat(Qt.RichText)
        layout.addWidget(features)

        layout.addStretch()

        # Clear data section
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        clear_data_title = QLabel("Clear Data")
        clear_data_title.setStyleSheet("font-size: 14px; font-weight: 600;")
        layout.addWidget(clear_data_title)

        clear_data_desc = QLabel(
            "Remove all WhisperNow data including settings, logs, history, and downloaded models."
        )
        clear_data_desc.setWordWrap(True)
        clear_data_desc.setStyleSheet("color: #888;")
        layout.addWidget(clear_data_desc)

        self._clear_data_btn = QPushButton("Clear All Data")
        self._clear_data_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:pressed {
                background-color: #bd2130;
            }
            """
        )
        self._clear_data_btn.setFixedWidth(180)
        self._clear_data_btn.clicked.connect(self._on_clear_data_clicked)
        layout.addWidget(self._clear_data_btn)

    def _on_clear_data_clicked(self) -> None:
        paths = get_all_data_paths()
        if not paths:
            QMessageBox.information(
                self,
                "Nothing to Remove",
                "No WhisperNow data was found.",
            )
            return

        paths_list = "\n".join(f"  • {p}" for p in paths)
        reply = QMessageBox.warning(
            self,
            "Confirm Clear Data",
            f"This will permanently delete:\n\n{paths_list}\n\n"
            "The application will close immediately, and data will be cleared in the background.\n"
            "This action cannot be undone. Continue?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )

        if reply != QMessageBox.Yes:
            return

        schedule_cleanup_and_exit()
        QApplication.quit()
