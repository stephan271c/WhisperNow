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

from ...core.settings.uninstaller import get_all_data_dirs, uninstall_user_data


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

        # Uninstall section
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        uninstall_title = QLabel("Uninstall")
        uninstall_title.setStyleSheet("font-size: 14px; font-weight: 600;")
        layout.addWidget(uninstall_title)

        uninstall_desc = QLabel(
            "Remove all WhisperNow data including settings, history, and downloaded models."
        )
        uninstall_desc.setWordWrap(True)
        uninstall_desc.setStyleSheet("color: #888;")
        layout.addWidget(uninstall_desc)

        self._uninstall_btn = QPushButton("Uninstall WhisperNow")
        self._uninstall_btn.setStyleSheet(
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
        self._uninstall_btn.setFixedWidth(180)
        self._uninstall_btn.clicked.connect(self._on_uninstall_clicked)
        layout.addWidget(self._uninstall_btn)

    def _on_uninstall_clicked(self) -> None:
        dirs = get_all_data_dirs()
        if not dirs:
            QMessageBox.information(
                self,
                "Nothing to Remove",
                "No WhisperNow data directories were found.",
            )
            return

        dirs_list = "\n".join(f"  • {d}" for d in dirs)
        reply = QMessageBox.warning(
            self,
            "Confirm Uninstall",
            f"This will permanently delete:\n\n{dirs_list}\n\n"
            "This action cannot be undone. Continue?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )

        if reply != QMessageBox.Yes:
            return

        success, errors = uninstall_user_data()

        if success:
            QMessageBox.information(
                self,
                "Uninstall Complete",
                "All WhisperNow data has been removed.\n\n"
                "The application will now close.",
            )
            QApplication.quit()
        else:
            QMessageBox.critical(
                self,
                "Uninstall Error",
                "Some files could not be removed:\n\n" + "\n".join(errors),
            )
