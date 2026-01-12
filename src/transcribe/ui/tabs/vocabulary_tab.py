from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...core.settings import Settings


class VocabularyTab(QWidget):

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._setup_ui()
        self.load_settings()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        add_group = QGroupBox("Add a text replacement")
        add_layout = QHBoxLayout(add_group)

        self._original_edit = QLineEdit()
        self._original_edit.setPlaceholderText("Original")
        add_layout.addWidget(self._original_edit)

        self._replacement_edit = QLineEdit()
        self._replacement_edit.setPlaceholderText("Replacement")
        add_layout.addWidget(self._replacement_edit)

        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._add_replacement)
        add_layout.addWidget(add_btn)

        self._original_edit.returnPressed.connect(self._add_replacement)
        self._replacement_edit.returnPressed.connect(self._add_replacement)

        layout.addWidget(add_group)

        list_group = QGroupBox("Replacement Rules")
        list_layout = QVBoxLayout(list_group)

        self._table = QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Replace", "With", ""])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self._table.setColumnWidth(2, 40)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        list_layout.addWidget(self._table)
        layout.addWidget(list_group)

    def _add_replacement(self) -> None:
        original = self._original_edit.text().strip()
        replacement = self._replacement_edit.text().strip()

        if not original:
            return  # Need at least an original word

        self._add_table_row(original, replacement)

        self._original_edit.clear()
        self._replacement_edit.clear()
        self._original_edit.setFocus()

    def _add_table_row(self, original: str, replacement: str) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        original_item = QTableWidgetItem(original)
        original_item.setFlags(original_item.flags() & ~Qt.ItemIsEditable)
        self._table.setItem(row, 0, original_item)

        replacement_item = QTableWidgetItem(replacement)
        replacement_item.setFlags(replacement_item.flags() & ~Qt.ItemIsEditable)
        self._table.setItem(row, 1, replacement_item)

        delete_btn = QPushButton("🗑️")
        delete_btn.setFixedWidth(36)
        delete_btn.setStyleSheet("QPushButton { color: #e74c3c; border: none; }")
        delete_btn.clicked.connect(
            lambda checked, btn=delete_btn: self._delete_row(btn)
        )
        self._table.setCellWidget(row, 2, delete_btn)

    def _delete_row(self, button: QPushButton) -> None:
        for r in range(self._table.rowCount()):
            if self._table.cellWidget(r, 2) is button:
                self._table.removeRow(r)
                break

    def load_settings(self) -> None:
        self._table.setRowCount(0)
        for original, replacement in self._settings.vocabulary_replacements:
            self._add_table_row(original, replacement)

    def save_settings(self) -> None:
        replacements = []
        for row in range(self._table.rowCount()):
            original_item = self._table.item(row, 0)
            replacement_item = self._table.item(row, 1)
            if original_item and replacement_item:
                replacements.append((original_item.text(), replacement_item.text()))
        self._settings.vocabulary_replacements = replacements
