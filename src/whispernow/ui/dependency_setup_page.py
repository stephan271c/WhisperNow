"""
Dependency Setup Page for the first-run wizard.

Displayed when heavy ML dependencies (PyTorch, NeMo) are not installed.
Allows user to choose GPU/CPU variant and shows download progress.
"""

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
    progress_updated = Signal(object)  # InstallProgress
    finished_signal = Signal(bool, str)  # success, message

    def __init__(self, manager: DependencyManager, use_gpu: bool):
        super().__init__()
        self._manager = manager
        self._use_gpu = use_gpu

    def run(self):
        success, message = self._manager.install_dependencies(
            use_gpu=self._use_gpu,
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

        self.setTitle("Install AI Components")
        self.setSubTitle(
            "WhisperNow needs to download speech recognition models. "
            "This may take a few minutes."
        )

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # GPU detection and selection
        gpu_available = self._manager.detect_gpu_available()

        gpu_group = QGroupBox("Installation Type")
        gpu_layout = QVBoxLayout(gpu_group)

        self._gpu_button_group = QButtonGroup(self)

        self._cpu_radio = QRadioButton(
            f"CPU Only - {self._manager.get_estimated_download_size(False)}"
        )
        self._cuda_radio = QRadioButton(
            f"NVIDIA GPU (CUDA) - {self._manager.get_estimated_download_size(True)}"
        )

        self._gpu_button_group.addButton(self._cpu_radio, 0)
        self._gpu_button_group.addButton(self._cuda_radio, 1)

        if gpu_available:
            self._cuda_radio.setChecked(True)
            gpu_status = QLabel("✅ NVIDIA GPU detected")
            gpu_status.setStyleSheet("color: green; font-size: 11px;")
        else:
            self._cpu_radio.setChecked(True)
            self._cuda_radio.setEnabled(False)
            gpu_status = QLabel("ℹ️ No NVIDIA GPU detected - using CPU mode")
            gpu_status.setStyleSheet("color: #666; font-size: 11px;")

        gpu_layout.addWidget(self._cpu_radio)
        gpu_layout.addWidget(self._cuda_radio)
        gpu_layout.addWidget(gpu_status)

        layout.addWidget(gpu_group)

        # Progress section
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

        # Install button
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
        self._install_button.setEnabled(False)
        self._install_button.setText("Installing...")
        self._cancel_button.setVisible(True)
        self._cpu_radio.setEnabled(False)
        self._cuda_radio.setEnabled(False)

        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._package_label.setVisible(True)
        self._status_label.setText("Preparing installation...")

        use_gpu = self._cuda_radio.isChecked()

        self._worker = InstallWorkerThread(self._manager, use_gpu)
        self._worker.progress_updated.connect(self._on_progress)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.start()

    def _cancel_installation(self):
        if self._worker and self._worker.isRunning():
            self._manager.cancel_installation()
            self._cancel_button.setEnabled(False)
            self._cancel_button.setText("Cancelling...")
            self._status_label.setText("Cancelling installation...")

    def _on_progress(self, progress: InstallProgress):
        percent = int(progress.progress * 100)
        self._progress_bar.setValue(percent)
        self._status_label.setText(progress.status)
        self._package_label.setText(
            f"Package {progress.current_package} of {progress.total_packages}: {progress.package}"
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
            self._cpu_radio.setEnabled(True)
            if self._manager.detect_gpu_available():
                self._cuda_radio.setEnabled(True)

        self.installation_complete.emit(success)
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        # Allow proceeding only after successful installation
        return self._install_complete and self._install_success

    def validatePage(self) -> bool:
        return self._install_success

    def cleanupPage(self) -> None:
        if self._worker and self._worker.isRunning():
            self._manager.cancel_installation()
            self._worker.wait(5000)
