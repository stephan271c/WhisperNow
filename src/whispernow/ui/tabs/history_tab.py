from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...core.settings import clear_history, load_history


class HistoryTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self.refresh_history)
        self._refresh_timer.start(2000)  # Refresh every 2 seconds

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        desc = QLabel("Recent transcriptions (most recent first):")
        layout.addWidget(desc)

        self._history_table = QTableWidget()
        self._history_table.setColumnCount(3)
        self._history_table.setHorizontalHeaderLabels(
            ["Raw Transcription", "Enhanced Text", "Cost"]
        )

        header = self._history_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)

        self._history_table.setAlternatingRowColors(True)
        self._history_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._history_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._history_table.verticalHeader().setVisible(False)
        self._history_table.setWordWrap(True)
        self._history_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self._history_table.customContextMenuRequested.connect(self._show_context_menu)

        layout.addWidget(self._history_table)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        clear_btn = QPushButton("Clear History")
        clear_btn.clicked.connect(self._clear_history)
        btn_layout.addWidget(clear_btn)

        layout.addLayout(btn_layout)

        self.refresh_history()

    def refresh_history(self) -> None:
        records = load_history()

        self._history_table.setRowCount(0)

        for record in reversed(records):
            row = self._history_table.rowCount()
            self._history_table.insertRow(row)

            raw_text = record.raw_text
            if len(raw_text) > 100:
                raw_text = raw_text[:100] + "..."
            raw_item = QTableWidgetItem(raw_text)
            # Tooltips disabled: causes AMD GPU driver crash on Ubuntu (AMDGPU page flip bug)
            # Use right-click -> Copy to get full text
            raw_item.setData(Qt.UserRole, record.raw_text)
            self._history_table.setItem(row, 0, raw_item)

            if record.enhanced_text:
                enhanced_text = record.enhanced_text
                if len(enhanced_text) > 100:
                    enhanced_text = enhanced_text[:100] + "..."
                enhanced_item = QTableWidgetItem(enhanced_text)
                # Tooltips disabled: AMD GPU driver bug (see above)
                enhanced_item.setData(Qt.UserRole, record.enhanced_text)
            else:
                enhanced_item = QTableWidgetItem("—")
            self._history_table.setItem(row, 1, enhanced_item)

            if record.cost_usd is not None:
                cost_text = f"${record.cost_usd:.6f}"
            else:
                cost_text = "—"
            cost_item = QTableWidgetItem(cost_text)
            cost_item.setData(Qt.UserRole, cost_text)
            self._history_table.setItem(row, 2, cost_item)

        self._history_table.resizeRowsToContents()

    def _show_context_menu(self, pos) -> None:
        was_active = self._refresh_timer.isActive()
        if was_active:
            self._refresh_timer.stop()
        try:
            index = self._history_table.indexAt(pos)
            if not index.isValid():
                return

            model = self._history_table.model()
            text = model.data(index, Qt.UserRole) or model.data(index, Qt.DisplayRole)
            if not text:
                return

            menu = QMenu(self)
            copy_action = menu.addAction("Copy")
            selected_action = menu.exec(self._history_table.viewport().mapToGlobal(pos))
            if selected_action == copy_action:
                QGuiApplication.clipboard().setText(str(text))
        finally:
            if was_active:
                self._refresh_timer.start(2000)

    def _clear_history(self) -> None:
        reply = QMessageBox.question(
            self,
            "Clear History",
            "Are you sure you want to clear all transcription history?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            clear_history()
            self.refresh_history()

    def stop_refresh_timer(self) -> None:
        self._refresh_timer.stop()

    def start_refresh_timer(self) -> None:
        self._refresh_timer.start(2000)
