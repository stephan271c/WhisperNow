from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWizardPage,
)

from ..core.deps import DependencyManager, InstallProgress


class InstallWorkerThread(QThread):
    progress_updated = Signal(object)
    finished_signal = Signal(bool, str)

    def __init__(self, manager: DependencyManager):
        super().__init__()
        self._manager = manager

    def run(self):
        success, message = self._manager.install_dependencies(
            progress_callback=self._on_progress,
        )
        self.finished_signal.emit(success, message)

    def _on_progress(self, progress: InstallProgress):
        self.progress_updated.emit(progress)


class DependencySetupPage(QWizardPage):
    installation_complete = Signal(bool)

    def __init__(self, manager: DependencyManager, parent=None):
        super().__init__(parent)
        self._manager = manager
        self._worker: InstallWorkerThread | None = None
        self._install_started = False
        self._install_complete = False
        self._install_success = False
        self._is_cancelling = False

        self.setTitle("Install AI Components")
        self.setSubTitle(
            "WhisperNow needs to download speech recognition models. "
            "This may take a few minutes."
        )

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        gpu_available = self._manager.detect_gpu_available()

        type_group = QGroupBox("Installation Info")
        type_layout = QVBoxLayout(type_group)

        size_info = self._manager.get_estimated_download_size()
        info_label = QLabel(
            f"<b>Standard Installation</b><br/>"
            f"Downloads AI models and dependencies.<br/>"
            f"Total Size: {size_info}"
        )
        info_label.setWordWrap(True)
        type_layout.addWidget(info_label)

        if gpu_available:
            gpu_status = QLabel("✅ NVIDIA GPU detected - Optimization enabled")
            gpu_status.setStyleSheet("color: green; font-size: 11px;")
        else:
            gpu_status = QLabel(
                "ℹ️ No NVIDIA GPU detected. The application will run on CPU, "
                "but the full package size (~8.7 GB) is still required."
            )
            gpu_status.setStyleSheet("color: #666; font-size: 11px;")
            gpu_status.setWordWrap(True)

        type_layout.addWidget(gpu_status)
        layout.addWidget(type_group)

        progress_group = QGroupBox("Installation Progress")
        progress_layout = QVBoxLayout(progress_group)

        self._status_label = QLabel("Click 'Install' to begin")
        self._status_label.setWordWrap(True)
        progress_layout.addWidget(self._status_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        progress_layout.addWidget(self._progress_bar)

        self._package_label = QLabel("")
        self._package_label.setStyleSheet("color: #666; font-size: 11px;")
        self._package_label.setVisible(False)
        progress_layout.addWidget(self._package_label)

        layout.addWidget(progress_group)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self._install_button = QPushButton("Install")
        self._install_button.setMinimumWidth(120)
        self._install_button.clicked.connect(self._start_installation)
        button_layout.addWidget(self._install_button)

        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.setVisible(False)
        self._cancel_button.clicked.connect(self._cancel_installation)
        button_layout.addWidget(self._cancel_button)

        layout.addLayout(button_layout)
        layout.addStretch()

    def _start_installation(self):
        self._install_started = True
        self._is_cancelling = False
        self._install_button.setEnabled(False)
        self._install_button.setText("Installing...")
        self._cancel_button.setVisible(True)
        self._cancel_button.setText("Cancel")
        self._cancel_button.setEnabled(True)
        self._status_label.setText("Preparing installation...")
        self._progress_bar.setVisible(True)
        self._package_label.setVisible(True)

        self._worker = InstallWorkerThread(self._manager)
        self._worker.progress_updated.connect(self._on_progress)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.start()

    def _cancel_installation(self):
        if self._worker and self._worker.isRunning():
            self._is_cancelling = True
            self._manager.cancel_installation()
            self._cancel_button.setEnabled(False)
            self._cancel_button.setText("Cancelling...")
            self._status_label.setText("Cancelling installation...")

    def _on_progress(self, progress: InstallProgress):
        if self._is_cancelling:
            return
        percent = int(progress.progress * 100)
        self._progress_bar.setValue(percent)
        self._status_label.setText(progress.status)
        self._package_label.setText(
            f"Installing {progress.current_package}/{progress.total_packages} • {progress.package}"
        )

    def _on_finished(self, success: bool, message: str):
        self._install_complete = True
        self._install_success = success
        self._cancel_button.setVisible(False)

        if success:
            self._progress_bar.setValue(100)
            self._status_label.setText("✅ " + message)
            self._status_label.setStyleSheet("color: green;")
            self._install_button.setText("Complete")
            self._package_label.setText("")
        else:
            self._status_label.setText("❌ " + message)
            self._status_label.setStyleSheet("color: red;")
            self._install_button.setText("Retry")
            self._install_button.setEnabled(True)

        self.installation_complete.emit(success)
        self.completeChanged.emit()

    def isComplete(self) -> bool:

        return self._install_complete and self._install_success

    def validatePage(self) -> bool:
        return self._install_success

    def cleanupPage(self) -> None:
        if self._worker and self._worker.isRunning():
            self._manager.cancel_installation()
            self._worker.wait(5000)
