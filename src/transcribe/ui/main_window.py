"""
Settings window with sidebar navigation.

Provides a GUI for configuring application settings.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QStackedWidget, QWidget,
    QLabel, QDialogButtonBox, QListWidget, QAbstractItemView, QFrame
)
from PySide6.QtCore import Qt, Signal
from typing import Optional

from ..core.settings import get_settings
from .tabs import (
    HomeTab,
    VocabularyTab,
    ConfigurationTab,
    EnhancementsTab,
    HistoryTab,
)


class SettingsWindow(QDialog):
    """
    Settings dialog with sidebar navigation for configuration categories.

    Signals:
        settings_changed: Emitted when settings are saved
    """

    settings_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setWindowTitle("Transcribe Settings")
        self.setMinimumWidth(450)
        self.setMinimumHeight(400)

        self._settings = get_settings()
        self._loading_overlay: Optional[QFrame] = None

        self._setup_ui()
        self._load_settings()

    def set_loading(self, loading: bool) -> None:
        """
        Show or hide the loading overlay.

        Args:
            loading: True to show loading overlay, False to hide
        """
        if loading:
            if self._loading_overlay is None:
                self._loading_overlay = self._create_loading_overlay()
            self._loading_overlay.show()
            self._loading_overlay.raise_()
            # Disable GPU checkbox during loading
            self._configuration_tab.set_gpu_enabled(False)
        else:
            if self._loading_overlay is not None:
                self._loading_overlay.hide()
            # Re-enable GPU checkbox
            self._configuration_tab.set_gpu_enabled(True)

    def _create_loading_overlay(self) -> QFrame:
        """Create a semi-transparent loading overlay."""
        overlay = QFrame(self)
        overlay.setStyleSheet("""
            QFrame {
                background-color: rgba(0, 0, 0, 150);
                border-radius: 8px;
            }
            QLabel {
                color: white;
                font-size: 16px;
                font-weight: bold;
            }
        """)

        layout = QVBoxLayout(overlay)
        layout.setAlignment(Qt.AlignCenter)

        label = QLabel("Loading model...")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        # Position overlay over the entire window
        overlay.setGeometry(self.rect())

        return overlay

    def resizeEvent(self, event):
        """Resize overlay when window is resized."""
        super().resizeEvent(event)
        if self._loading_overlay is not None:
            self._loading_overlay.setGeometry(self.rect())

    def _setup_ui(self) -> None:
        """Build the UI layout."""
        layout = QVBoxLayout(self)

        # Sidebar navigation
        content_layout = QHBoxLayout()
        self._nav_list = QListWidget()
        self._nav_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._nav_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._nav_list.setSpacing(2)
        self._nav_list.setFixedWidth(170)
        self._nav_list.setFrameShape(QFrame.NoFrame)

        self._stacked = QStackedWidget()

        # Create tab instances
        self._home_tab = HomeTab(self)
        self._enhancements_tab = EnhancementsTab(self._settings, self)
        self._vocabulary_tab = VocabularyTab(self._settings, self)
        self._configuration_tab = ConfigurationTab(self._settings, self)
        self._history_tab = HistoryTab(self)

        # Connect configuration tab reset signal
        self._configuration_tab.reset_requested.connect(self._reset_settings)

        pages = [
            ("Home", self._home_tab),
            ("Mode", self._enhancements_tab),
            ("Vocabulary", self._vocabulary_tab),
            ("Configuration", self._configuration_tab),
            ("History", self._history_tab),
        ]
        for title, page in pages:
            self._nav_list.addItem(title)
            self._stacked.addWidget(page)

        self._nav_list.currentRowChanged.connect(self._stacked.setCurrentIndex)
        self._nav_list.setCurrentRow(0)

        content_layout.addWidget(self._nav_list)
        content_layout.addWidget(self._stacked, 1)
        layout.addLayout(content_layout)

        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        buttons.accepted.connect(self._save_and_close)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self._save_settings)
        layout.addWidget(buttons)

    def _load_settings(self) -> None:
        """Load current settings into all tabs."""
        self._configuration_tab.load_settings()
        self._enhancements_tab.load_settings()
        self._vocabulary_tab.load_settings()

    def _save_settings(self) -> None:
        """Save UI values from all tabs to settings."""
        # Configuration tab returns False if validation fails
        if not self._configuration_tab.save_settings():
            return

        self._enhancements_tab.save_settings()
        self._vocabulary_tab.save_settings()

        self._settings.save()
        self.settings_changed.emit()

    def _save_and_close(self) -> None:
        """Save settings and close the dialog."""
        self._save_settings()
        self.accept()

    def _reset_settings(self) -> None:
        """Reset all settings to defaults."""
        self._settings.reset_to_defaults()
        self._load_settings()
