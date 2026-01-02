"""
System tray icon and menu using PySide6.

Provides a system tray icon with status indicator and context menu.
"""

from PySide6.QtWidgets import QSystemTrayIcon, QMenu, QApplication
from PySide6.QtGui import QIcon, QAction, QPixmap, QPainter, QColor, QBrush, QPen
from PySide6.QtCore import Signal, QObject, Qt
from typing import Optional, Callable, Dict
from enum import Enum, auto


class TrayStatus(Enum):
    """Status indicators for the tray icon."""
    IDLE = auto()        # Ready, waiting for input
    LOADING = auto()     # Model loading/downloading
    RECORDING = auto()   # Currently recording
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
        # Color mapping for each status
        status_colors: Dict[TrayStatus, QColor] = {
            TrayStatus.IDLE: QColor("#4CAF50"),       # Green
            TrayStatus.LOADING: QColor("#FF9800"),    # Orange
            TrayStatus.RECORDING: QColor("#F44336"),  # Red
            TrayStatus.ERROR: QColor("#F44336"),      # Red
        }
        
        # Status tooltips
        status_tooltips: Dict[TrayStatus, str] = {
            TrayStatus.IDLE: "Transcribe - Ready",
            TrayStatus.LOADING: "Transcribe - Loading...",
            TrayStatus.RECORDING: "Transcribe - Recording",
            TrayStatus.ERROR: "Transcribe - Error",
        }
        
        # Create a colored circle icon
        size = 22
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        color = status_colors.get(self._status, QColor("#808080"))
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(color.darker(120), 1))
        
        # Draw filled circle
        margin = 2
        painter.drawEllipse(margin, margin, size - 2 * margin, size - 2 * margin)
        
        # For error state, add an X overlay
        if self._status == TrayStatus.ERROR:
            painter.setPen(QPen(QColor("#FFFFFF"), 2))
            inner_margin = 6
            painter.drawLine(inner_margin, inner_margin, size - inner_margin, size - inner_margin)
            painter.drawLine(size - inner_margin, inner_margin, inner_margin, size - inner_margin)
        
        painter.end()
        
        icon = QIcon(pixmap)
        
        if self._tray_icon:
            self._tray_icon.setIcon(icon)
            self._tray_icon.setToolTip(status_tooltips.get(self._status, "Transcribe"))
    
    
    def hide(self) -> None:
        """Hide the tray icon."""
        if self._tray_icon:
            self._tray_icon.hide()
