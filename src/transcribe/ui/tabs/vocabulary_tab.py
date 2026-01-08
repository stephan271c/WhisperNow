"""Vocabulary tab placeholder."""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel


class VocabularyTab(QWidget):
    """Placeholder tab for vocabulary substitutions (coming soon)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the UI layout."""
        layout = QVBoxLayout(self)

        placeholder = QLabel("Vocabulary substitutions will appear here.")
        placeholder.setStyleSheet("color: gray;")
        placeholder.setWordWrap(True)
        layout.addWidget(placeholder)

        layout.addStretch()
