import uuid
from typing import Dict, Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class EnhancementEditDialog(QDialog):

    def __init__(
        self, parent: Optional[QWidget] = None, enhancement: Optional[Dict] = None
    ):
        super().__init__(parent)

        self._enhancement = enhancement
        self._is_new = enhancement is None

        self.setWindowTitle("Add Enhancement" if self._is_new else "Edit Enhancement")
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)

        self._setup_ui()
        self._load_enhancement()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()

        self._title_edit = QLineEdit()
        self._title_edit.setPlaceholderText("e.g., Fix Grammar")
        form.addRow("Title:", self._title_edit)

        self._prompt_edit = QTextEdit()
        self._prompt_edit.setPlaceholderText(
            "Enter the system prompt for the LLM...\n\n"
            "Example: Fix any grammar or spelling errors in the following text. "
            "Only output the corrected text, nothing else."
        )
        form.addRow("Prompt:", self._prompt_edit)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load_enhancement(self) -> None:
        if self._enhancement:
            self._title_edit.setText(self._enhancement.get("title", ""))
            self._prompt_edit.setPlainText(self._enhancement.get("prompt", ""))

    def _validate_and_accept(self) -> None:
        title = self._title_edit.text().strip()
        prompt = self._prompt_edit.toPlainText().strip()

        if not title:
            QMessageBox.warning(self, "Validation Error", "Title is required.")
            return

        if not prompt:
            QMessageBox.warning(self, "Validation Error", "Prompt is required.")
            return

        self.accept()

    def get_enhancement(self) -> Dict:
        if self._is_new:
            enh_id = str(uuid.uuid4())[:8]
        else:
            enh_id = self._enhancement.get("id", str(uuid.uuid4())[:8])

        return {
            "id": enh_id,
            "title": self._title_edit.text().strip(),
            "prompt": self._prompt_edit.toPlainText().strip(),
        }
