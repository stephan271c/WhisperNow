from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


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
