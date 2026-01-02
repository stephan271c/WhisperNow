"""
System tray icon and menu using PySide6.

Provides a system tray icon with status indicator and context menu.
"""

from PySide6.QtWidgets import QSystemTrayIcon, QMenu, QApplication
from PySide6.QtGui import QIcon, QAction
from PySide6.QtCore import Signal, QObject
from typing import Optional, Callable
from enum import Enum, auto


class TrayStatus(Enum):
    """Status indicators for the tray icon."""
    IDLE = auto()        # Ready, waiting for input
    LOADING = auto()     # Model loading/downloading
    RECORDING = auto()   # Currently recording
    PROCESSING = auto()  # Transcribing audio
    ERROR = auto()       # Error state


class SystemTray(QObject):
    """
    System tray icon with context menu.
    
    Signals:
        settings_requested: Emitted when user clicks "Settings"
        quit_requested: Emitted when user clicks "Quit"
        toggle_requested: Emitted when user clicks "Start/Pause"
    """
    
    settings_requested = Signal()
    quit_requested = Signal()
    toggle_requested = Signal()
    
    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        
        self._status = TrayStatus.LOADING
        self._tray_icon: Optional[QSystemTrayIcon] = None
        self._menu: Optional[QMenu] = None
        
        self._setup_tray()
    
    def _setup_tray(self) -> None:
        """Initialize the system tray icon and menu."""
        self._tray_icon = QSystemTrayIcon(self)
        
        # Create context menu
        self._menu = QMenu()
        
        # Status label (non-clickable)
        self._status_action = QAction("Loading...", self._menu)
        self._status_action.setEnabled(False)
        self._menu.addAction(self._status_action)
        
        self._menu.addSeparator()
        
        # Toggle action
        self._toggle_action = QAction("Pause", self._menu)
        self._toggle_action.triggered.connect(self.toggle_requested.emit)
        self._menu.addAction(self._toggle_action)
        
        # Settings action
        settings_action = QAction("Settings...", self._menu)
        settings_action.triggered.connect(self.settings_requested.emit)
        self._menu.addAction(settings_action)
        
        self._menu.addSeparator()
        
        # Quit action
        quit_action = QAction("Quit", self._menu)
        quit_action.triggered.connect(self.quit_requested.emit)
        self._menu.addAction(quit_action)
        
        self._tray_icon.setContextMenu(self._menu)
        
        # Set initial icon
        self._update_icon()
        
        # Show the tray icon
        self._tray_icon.show()
    
    def set_status(self, status: TrayStatus, message: str = "") -> None:
        """Update the tray status and icon."""
        self._status = status
        self._update_icon()
        
        # Update status text in menu
        status_texts = {
            TrayStatus.IDLE: "Ready - Hold hotkey to speak",
            TrayStatus.LOADING: "Loading model...",
            TrayStatus.RECORDING: "🔴 Recording...",
            TrayStatus.PROCESSING: "Processing...",
            TrayStatus.ERROR: f"Error: {message}"
        }
        self._status_action.setText(status_texts.get(status, "Unknown"))
        
        # Update toggle action text
        if status == TrayStatus.IDLE:
            self._toggle_action.setText("Pause")
            self._toggle_action.setEnabled(True)
        elif status == TrayStatus.LOADING:
            self._toggle_action.setText("Start")
            self._toggle_action.setEnabled(False)
        else:
            self._toggle_action.setEnabled(True)
    
    def _update_icon(self) -> None:
        """Update the tray icon based on current status."""
        # TODO: Use actual icons for each state
        # For now, use a placeholder or the app icon
        # Icons should be in src/transcribe/ui/resources/
        
        # Placeholder: Create a simple colored icon
        # In production, load from resources
        icon = QIcon.fromTheme("audio-input-microphone")
        if icon.isNull():
            # Fallback - create a simple icon
            pass
        
        if self._tray_icon:
            self._tray_icon.setIcon(icon)
            self._tray_icon.setToolTip("Transcribe")
    
    def show_notification(self, title: str, message: str, duration_ms: int = 3000) -> None:
        """Show a balloon notification from the tray icon."""
        if self._tray_icon and QSystemTrayIcon.supportsMessages():
            self._tray_icon.showMessage(title, message, QSystemTrayIcon.Information, duration_ms)
    
    def hide(self) -> None:
        """Hide the tray icon."""
        if self._tray_icon:
            self._tray_icon.hide()
