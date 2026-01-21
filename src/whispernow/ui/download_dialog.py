from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from ..core.asr.models.downloader import ModelDownloader


class ModelDownloadThread(QThread):
    """Thread for downloading ASR models without blocking the UI."""

    progress = Signal(int, int)  # bytes_downloaded, total_bytes
    status_changed = Signal(str)  # status message
    finished = Signal(bool)  # success
    error = Signal(str)  # error message

    def __init__(self, model_id: str):
        super().__init__()
        self._model_id = model_id
        self._downloader = ModelDownloader()

    def run(self):
        try:
            success = self._downloader.download(
                self._model_id,
                on_progress=self._on_progress,
                on_status=self._on_status,
            )
            self.finished.emit(success)
        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit(False)

    def _on_progress(self, downloaded: int, total: int):
        self.progress.emit(downloaded, total)

    def _on_status(self, status: str):
        self.status_changed.emit(status)

    def cancel(self):
        self._downloader.cancel()


class DownloadDialog(QDialog):

    cancelled = Signal()

    def __init__(self, model_name: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Downloading Model")
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.CustomizeWindowHint)
        self.setModal(True)
        self.setFixedWidth(400)

        self._cancelled = False

        self._setup_ui(model_name)

    def _setup_ui(self, model_name: str) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        self.title_label = QLabel("Downloading Model")
        self.title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.title_label)

        self.model_label = QLabel(model_name)
        self.model_label.setStyleSheet("color: #666;")
        self.model_label.setWordWrap(True)
        layout.addWidget(self.model_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Preparing...")
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.status_label)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._on_cancel)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

    def set_model_name(self, model_name: str) -> None:
        self.model_label.setText(model_name)

    def set_progress(self, progress: float) -> None:
        percent = int(progress * 100)
        self.progress_bar.setValue(percent)

        if progress < 1.0:
            self.status_label.setText(f"Downloading... {percent}%")
        else:
            self.status_label.setText("Download complete, loading model...")

    def set_status(self, status: str) -> None:
        self.status_label.setText(status)

    def _on_cancel(self) -> None:
        self._cancelled = True
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Cancelling...")
        self.status_label.setText("Cancelling download...")
        self.cancelled.emit()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def finish(self, success: bool = True) -> None:
        self.accept() if success else self.reject()
