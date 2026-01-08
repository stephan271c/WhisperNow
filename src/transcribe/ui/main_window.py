"""
Settings window with sidebar navigation.

Provides a GUI for configuring application settings.
"""

import uuid
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QStackedWidget, QWidget,
    QLabel, QSlider, QComboBox, QCheckBox, QPushButton,
    QGroupBox, QFormLayout, QSpinBox, QKeySequenceEdit,
    QDialogButtonBox, QLineEdit, QMessageBox, QListWidget,
    QListWidgetItem, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QFrame, QMenu
)
from PySide6.QtGui import QKeySequence, QGuiApplication
from PySide6.QtCore import Qt, Signal, QTimer
from typing import Optional, List, Dict

from ..core.settings import Settings, get_settings, HotkeyConfig, load_history, clear_history
from ..core.recorder import AudioRecorder
from ..core.llm_processor import Enhancement, PROVIDERS, get_models_for_provider


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
        
        # Auto-refresh timer for history tab
        self._history_refresh_timer = QTimer(self)
        self._history_refresh_timer.timeout.connect(self._refresh_history)
        self._history_refresh_timer.start(2000)  # Refresh every 2 seconds
    
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
            if hasattr(self, '_use_gpu_cb'):
                self._use_gpu_cb.setEnabled(False)
        else:
            if self._loading_overlay is not None:
                self._loading_overlay.hide()
            # Re-enable GPU checkbox
            if hasattr(self, '_use_gpu_cb'):
                self._use_gpu_cb.setEnabled(True)
    
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
        
        pages = [
            ("Home", self._create_home_tab()),
            ("Mode", self._create_enhancements_tab()),
            ("Vocabulary", self._create_vocabulary_tab()),
            ("Configuration", self._create_configuration_tab()),
            ("History", self._create_history_tab()),
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
    
    def _create_home_tab(self) -> QWidget:
        """Create the Home tab with a brief overview."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        title = QLabel("Get started")
        title.setStyleSheet("font-size: 14px; font-weight: 600;")
        layout.addWidget(title)
        
        intro = QLabel("Choose a section on the left to configure Transcribe.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        
        features = QLabel(
            "<ul>"
            "<li><b>Mode</b>: Configure enhancements for different tasks.</li>"
            "<li><b>Vocabulary</b>: Word substitutions (coming soon).</li>"
            "<li><b>Configuration</b>: General behavior, audio input, hotkeys, and ASR model.</li>"
            "<li><b>History</b>: Review recent transcriptions.</li>"
            "</ul>"
        )
        features.setWordWrap(True)
        features.setTextFormat(Qt.RichText)
        layout.addWidget(features)
        
        layout.addStretch()
        return widget
    
    def _create_vocabulary_tab(self) -> QWidget:
        """Create the Vocabulary placeholder tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        placeholder = QLabel("Vocabulary substitutions will appear here.")
        placeholder.setStyleSheet("color: gray;")
        placeholder.setWordWrap(True)
        layout.addWidget(placeholder)
        
        layout.addStretch()
        return widget
    
    def _create_configuration_tab(self) -> QWidget:
        """Create the Configuration tab with all settings grouped."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        general_section = self._create_general_section()
        layout.addWidget(general_section)
        layout.addWidget(self._create_audio_section())
        layout.addWidget(self._create_hotkey_section())
        layout.addWidget(self._create_asr_model_section())
        
        layout.addStretch()
        return widget
    
    def _create_general_section(self) -> QWidget:
        """Create the General settings section."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        # Typing behavior group
        typing_group = QGroupBox("Typing Behavior")
        typing_layout = QFormLayout(typing_group)
        
        # Typing speed slider
        speed_layout = QHBoxLayout()
        self._speed_slider = QSlider(Qt.Horizontal)
        self._speed_slider.setRange(0, 300)
        self._speed_slider.setTickPosition(QSlider.TicksBelow)
        self._speed_slider.setTickInterval(50)
        self._speed_label = QLabel("150 chars/sec")
        self._speed_slider.valueChanged.connect(
            lambda v: self._speed_label.setText(f"{v} chars/sec" if v > 0 else "Instant")
        )
        speed_layout.addWidget(self._speed_slider)
        speed_layout.addWidget(self._speed_label)
        typing_layout.addRow("Typing Speed:", speed_layout)
        
        # Auto-type checkbox
        self._instant_type_cb = QCheckBox("Instantly output text")
        self._instant_type_cb.toggled.connect(self._on_instant_type_toggled)
        typing_layout.addRow("", self._instant_type_cb)
        
        layout.addWidget(typing_group)
        
        # Startup group
        startup_group = QGroupBox("Startup")
        startup_layout = QVBoxLayout(startup_group)
        
        self._start_minimized_cb = QCheckBox("Start minimized to tray")
        startup_layout.addWidget(self._start_minimized_cb)
        
        self._autostart_cb = QCheckBox("Start automatically on login")
        startup_layout.addWidget(self._autostart_cb)
        
        layout.addWidget(startup_group)
        return widget
    
    def _on_instant_type_toggled(self, checked: bool) -> None:
        """Enable/disable speed slider based on instant type setting."""
        self._speed_slider.setEnabled(not checked)
        self._speed_label.setEnabled(not checked)
    
    def _create_enhancements_tab(self) -> QWidget:
        """Create the Mode settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # LLM Settings group
        llm_group = QGroupBox("LLM Settings")
        llm_layout = QFormLayout(llm_group)
        
        # Provider selector
        self._provider_combo = QComboBox()
        for provider_id, (display_name, _, _) in PROVIDERS.items():
            self._provider_combo.addItem(display_name, provider_id)
        self._provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        llm_layout.addRow("Provider:", self._provider_combo)
        
        # Model selector with refresh button
        model_layout = QHBoxLayout()
        self._llm_model_combo = QComboBox()
        self._llm_model_combo.setEditable(True)  # Allow custom model names
        self._llm_model_combo.setMinimumWidth(250)
        model_layout.addWidget(self._llm_model_combo, 1)
        
        refresh_btn = QPushButton("⟳")
        refresh_btn.setFixedWidth(30)
        refresh_btn.setToolTip("Refresh model list from provider")
        refresh_btn.clicked.connect(self._refresh_model_list)
        model_layout.addWidget(refresh_btn)
        llm_layout.addRow("Model:", model_layout)
        
        # API Base URL (shown for certain providers)
        self._api_base_edit = QLineEdit()
        self._api_base_edit.setPlaceholderText("e.g., http://localhost:11434")
        self._api_base_label = QLabel("API Base URL:")
        llm_layout.addRow(self._api_base_label, self._api_base_edit)
        
        # API Key input
        self._api_key_edit = QLineEdit()
        self._api_key_edit.setEchoMode(QLineEdit.Password)
        self._api_key_edit.setPlaceholderText("Enter your API key")
        llm_layout.addRow("API Key:", self._api_key_edit)
        
        layout.addWidget(llm_group)
        
        # Enhancements list group
        enhance_group = QGroupBox("Enhancements")
        enhance_layout = QVBoxLayout(enhance_group)
        
        # Active enhancement selector
        active_layout = QHBoxLayout()
        active_layout.addWidget(QLabel("Active Enhancement:"))
        self._active_enhancement_combo = QComboBox()
        active_layout.addWidget(self._active_enhancement_combo, 1)
        enhance_layout.addLayout(active_layout)
        
        # Enhancement list
        self._enhancement_list = QListWidget()
        self._enhancement_list.itemDoubleClicked.connect(self._edit_enhancement)
        enhance_layout.addWidget(self._enhancement_list)
        
        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._add_enhancement)
        edit_btn = QPushButton("Edit")
        edit_btn.clicked.connect(self._edit_selected_enhancement)
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self._delete_enhancement)
        
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(edit_btn)
        btn_layout.addWidget(delete_btn)
        btn_layout.addStretch()
        enhance_layout.addLayout(btn_layout)
        
        layout.addWidget(enhance_group)
        
        return widget
    
    def _on_provider_changed(self) -> None:
        """Handle provider selection change."""
        provider_id = self._provider_combo.currentData()
        if not provider_id:
            return
        
        # Get provider config
        _, default_api_base, _ = PROVIDERS.get(provider_id, (None, None, None))
        
        # Show/hide API base field based on provider
        # Only Ollama (local) and Other need custom API base - LiteLLM handles others automatically
        needs_api_base = provider_id in ("ollama", "other")
        self._api_base_label.setVisible(needs_api_base)
        self._api_base_edit.setVisible(needs_api_base)
        
        # Set default API base if switching to a provider with one
        if default_api_base and not self._api_base_edit.text():
            self._api_base_edit.setText(default_api_base)
        
        # Refresh model list and reset to first model for new provider
        self._refresh_model_list(reset_selection=True)
    
    def _refresh_model_list(self, reset_selection: bool = False) -> None:
        """Refresh the model dropdown with models from the selected provider."""
        provider_id = self._provider_combo.currentData()
        if not provider_id:
            return
        
        # Save current selection (only if not resetting)
        current_model = "" if reset_selection else self._llm_model_combo.currentText()
        
        # Clear and populate
        self._llm_model_combo.clear()
        
        # Get models for provider
        models = get_models_for_provider(provider_id)
        if models:
            self._llm_model_combo.addItems(models)
        
        # Restore selection if still valid and not resetting, otherwise use first item
        if not reset_selection and current_model:
            idx = self._llm_model_combo.findText(current_model)
            if idx >= 0:
                self._llm_model_combo.setCurrentIndex(idx)
            else:
                # Keep custom model even if not in list (editable combo)
                self._llm_model_combo.setCurrentText(current_model)
    
    def _refresh_enhancement_list(self) -> None:
        """Refresh the enhancement list and active combo."""
        self._enhancement_list.clear()
        self._active_enhancement_combo.clear()
        self._active_enhancement_combo.addItem("None (disabled)", None)
        
        for enh_dict in self._settings.enhancements:
            item = QListWidgetItem(enh_dict.get("title", "Untitled"))
            item.setData(Qt.UserRole, enh_dict)
            self._enhancement_list.addItem(item)
            
            self._active_enhancement_combo.addItem(
                enh_dict.get("title", "Untitled"),
                enh_dict.get("id")
            )
        
        # Select active enhancement
        if self._settings.active_enhancement_id:
            idx = self._active_enhancement_combo.findData(self._settings.active_enhancement_id)
            if idx >= 0:
                self._active_enhancement_combo.setCurrentIndex(idx)
    
    def _add_enhancement(self) -> None:
        """Add a new enhancement."""
        dialog = EnhancementEditDialog(self)
        if dialog.exec() == QDialog.Accepted:
            enh = dialog.get_enhancement()
            self._settings.enhancements.append(enh)
            self._refresh_enhancement_list()
    
    def _edit_enhancement(self, item: QListWidgetItem) -> None:
        """Edit an enhancement from the list."""
        enh_dict = item.data(Qt.UserRole)
        dialog = EnhancementEditDialog(self, enh_dict)
        if dialog.exec() == QDialog.Accepted:
            updated = dialog.get_enhancement()
            # Update in list
            for i, e in enumerate(self._settings.enhancements):
                if e.get("id") == updated.get("id"):
                    self._settings.enhancements[i] = updated
                    break
            self._refresh_enhancement_list()
    
    def _edit_selected_enhancement(self) -> None:
        """Edit the currently selected enhancement."""
        item = self._enhancement_list.currentItem()
        if item:
            self._edit_enhancement(item)
    
    def _delete_enhancement(self) -> None:
        """Delete the selected enhancement."""
        item = self._enhancement_list.currentItem()
        if not item:
            return
        
        enh_dict = item.data(Qt.UserRole)
        reply = QMessageBox.question(
            self,
            "Delete Enhancement",
            f"Delete enhancement '{enh_dict.get('title')}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self._settings.enhancements = [
                e for e in self._settings.enhancements
                if e.get("id") != enh_dict.get("id")
            ]
            # Clear active if deleted
            if self._settings.active_enhancement_id == enh_dict.get("id"):
                self._settings.active_enhancement_id = None
            self._refresh_enhancement_list()
    
    def _create_hotkey_section(self) -> QWidget:
        """Create the Hotkey settings section."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        hotkey_group = QGroupBox("Push-to-Talk Hotkey")
        hotkey_layout = QFormLayout(hotkey_group)
        
        # Hotkey input
        self._hotkey_edit = QKeySequenceEdit()
        hotkey_layout.addRow("Hold to record:", self._hotkey_edit)
        
        # Instructions
        instructions = QLabel(
            "Click the field above and press your desired key combination.\n"
            "The recording will start when you hold the keys and stop when you release."
        )
        instructions.setStyleSheet("color: gray; font-size: 11px;")
        instructions.setWordWrap(True)
        hotkey_layout.addRow("", instructions)
        
        layout.addWidget(hotkey_group)
        return widget
    
    def _create_audio_section(self) -> QWidget:
        """Create the Audio settings section."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        audio_group = QGroupBox("Audio")
        audio_layout = QFormLayout(audio_group)
        
        # Input device dropdown with refresh button
        device_layout = QHBoxLayout()
        self._device_combo = QComboBox()
        self._refresh_devices()
        device_layout.addWidget(self._device_combo, 1)
        
        refresh_btn = QPushButton("⟳")
        refresh_btn.setFixedWidth(30)
        refresh_btn.setToolTip("Refresh device list")
        refresh_btn.clicked.connect(self._refresh_devices)
        device_layout.addWidget(refresh_btn)
        
        audio_layout.addRow("Input Device:", device_layout)
        
        # Sample rate dropdown
        self._sample_rate_combo = QComboBox()
        self._sample_rate_combo.addItem("16000 Hz (Recommended)", 16000)
        self._sample_rate_combo.addItem("22050 Hz", 22050)
        self._sample_rate_combo.addItem("44100 Hz", 44100)
        audio_layout.addRow("Sample Rate:", self._sample_rate_combo)
        
        layout.addWidget(audio_group)
        return widget
    
    def _refresh_devices(self) -> None:
        """Refresh the audio device dropdown."""
        current = self._device_combo.currentData() if hasattr(self, '_device_combo') else None
        self._device_combo.clear()
        self._device_combo.addItem("System Default", None)
        
        for device in AudioRecorder.list_devices():
            self._device_combo.addItem(device.name, device.name)
        
        # Restore selection if still available
        if current:
            idx = self._device_combo.findData(current)
            if idx >= 0:
                self._device_combo.setCurrentIndex(idx)
    
    def _create_history_tab(self) -> QWidget:
        """Create the History tab showing recent transcriptions."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Description
        desc = QLabel("Recent transcriptions (most recent first):")
        layout.addWidget(desc)
        
        # Table with 3 columns: Raw, Enhanced, Cost
        self._history_table = QTableWidget()
        self._history_table.setColumnCount(3)
        self._history_table.setHorizontalHeaderLabels(["Raw Transcription", "Enhanced Text", "Cost"])
        
        # Configure table appearance
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
        self._history_table.customContextMenuRequested.connect(self._show_history_context_menu)
        
        layout.addWidget(self._history_table)
        
        # Button row
        btn_layout = QHBoxLayout()
        
        btn_layout.addStretch()
        
        clear_btn = QPushButton("Clear History")
        clear_btn.clicked.connect(self._clear_history)
        btn_layout.addWidget(clear_btn)
        
        layout.addLayout(btn_layout)
        
        # Load initial data
        self._refresh_history()
        
        return widget
    
    def _refresh_history(self) -> None:
        """Refresh the history table with data from history.json."""
        records = load_history()
        
        # Clear existing rows
        self._history_table.setRowCount(0)
        
        # Add rows in reverse order (newest first)
        for record in reversed(records):
            row = self._history_table.rowCount()
            self._history_table.insertRow(row)
            
            # Raw text (truncate for display)
            raw_text = record.raw_text
            if len(raw_text) > 100:
                raw_text = raw_text[:100] + "..."
            raw_item = QTableWidgetItem(raw_text)
            raw_item.setToolTip(record.raw_text)  # Full text on hover
            raw_item.setData(Qt.UserRole, record.raw_text)
            self._history_table.setItem(row, 0, raw_item)
            
            # Enhanced text
            if record.enhanced_text:
                enhanced_text = record.enhanced_text
                if len(enhanced_text) > 100:
                    enhanced_text = enhanced_text[:100] + "..."
                enhanced_item = QTableWidgetItem(enhanced_text)
                enhanced_item.setToolTip(record.enhanced_text)
                enhanced_item.setData(Qt.UserRole, record.enhanced_text)
            else:
                enhanced_item = QTableWidgetItem("—")
            self._history_table.setItem(row, 1, enhanced_item)
            
            # Cost
            if record.cost_usd is not None:
                cost_text = f"${record.cost_usd:.6f}"
            else:
                cost_text = "—"
            cost_item = QTableWidgetItem(cost_text)
            cost_item.setData(Qt.UserRole, cost_text)
            self._history_table.setItem(row, 2, cost_item)
        
        # Resize rows to content
        self._history_table.resizeRowsToContents()

    def _show_history_context_menu(self, pos) -> None:
        """Show a context menu for copying cell text."""
        was_active = self._history_refresh_timer.isActive()
        if was_active:
            self._history_refresh_timer.stop()
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
                self._history_refresh_timer.start(2000)
    
    def _clear_history(self) -> None:
        """Clear all transcription history."""
        reply = QMessageBox.question(
            self,
            "Clear History",
            "Are you sure you want to clear all transcription history?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            clear_history()
            self._refresh_history()

    def _create_asr_model_section(self) -> QWidget:
        """Create the ASR model settings section."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        model_group = QGroupBox("ASR Model")
        model_layout = QFormLayout(model_group)
        
        # Editable model name
        self._model_edit = QLineEdit()
        self._model_edit.setPlaceholderText("e.g. nvidia/parakeet-tdt-0.6b-v3")
        self._model_edit.setStyleSheet("font-family: monospace;")
        model_layout.addRow("Model Name:", self._model_edit)
        
        # Instructions
        model_instructions = QLabel(
            "Enter a valid HuggingFace or NVIDIA model name.\n"
            "Model will be validated and downloaded on save."
        )
        model_instructions.setStyleSheet("color: gray; font-size: 11px;")
        model_instructions.setWordWrap(True)
        model_layout.addRow("", model_instructions)
        
        # GPU checkbox
        self._use_gpu_cb = QCheckBox("Use GPU acceleration (if available)")
        model_layout.addRow("", self._use_gpu_cb)
        
        layout.addWidget(model_group)
        
        # Reset button
        reset_btn = QPushButton("Reset All Settings to Defaults")
        reset_btn.clicked.connect(self._reset_settings)
        layout.addWidget(reset_btn)
        return widget
    
    def _load_settings(self) -> None:
        """Load current settings into the UI."""
        self._speed_slider.setValue(self._settings.characters_per_second)
        self._instant_type_cb.setChecked(self._settings.instant_type)
        self._on_instant_type_toggled(self._settings.instant_type)
        
        self._start_minimized_cb.setChecked(self._settings.start_minimized)
        self._autostart_cb.setChecked(self._settings.auto_start_on_login)
        self._use_gpu_cb.setChecked(self._settings.use_gpu)
        
        # Set model name
        self._model_edit.setText(self._settings.model_name)
        
        # Set hotkey from HotkeyConfig
        hotkey = self._settings.hotkey
        key_sequence = "+".join(hotkey.modifiers + [hotkey.key])
        self._hotkey_edit.setKeySequence(QKeySequence(key_sequence))
        
        # Set sample rate
        idx = self._sample_rate_combo.findData(self._settings.sample_rate)
        if idx >= 0:
            self._sample_rate_combo.setCurrentIndex(idx)
        
        # Set input device
        if self._settings.input_device:
            idx = self._device_combo.findData(self._settings.input_device)
            if idx >= 0:
                self._device_combo.setCurrentIndex(idx)
        
        # Load LLM settings
        # Set provider first (this triggers _on_provider_changed which populates models)
        provider_idx = self._provider_combo.findData(self._settings.llm_provider)
        if provider_idx >= 0:
            self._provider_combo.setCurrentIndex(provider_idx)
        
        # Set API base (before triggering provider change clears it)
        if self._settings.llm_api_base:
            self._api_base_edit.setText(self._settings.llm_api_base)
        
        # Trigger provider change to show/hide API base and populate models
        self._on_provider_changed()
        
        # Set model after models are populated
        self._llm_model_combo.setCurrentText(self._settings.llm_model)
        
        if self._settings.llm_api_key:
            self._api_key_edit.setText(self._settings.llm_api_key)
        
        # Load enhancements
        self._refresh_enhancement_list()
    
    def _save_settings(self) -> None:
        """Save UI values to settings."""
        # Validate model name first
        model_name = self._model_edit.text().strip()
        if model_name and model_name != self._settings.model_name:
            if not self._validate_model_name(model_name):
                return  # Validation failed, don't save
        
        self._settings.characters_per_second = self._speed_slider.value()
        self._settings.instant_type = self._instant_type_cb.isChecked()
        self._settings.start_minimized = self._start_minimized_cb.isChecked()
        self._settings.auto_start_on_login = self._autostart_cb.isChecked()
        self._settings.use_gpu = self._use_gpu_cb.isChecked()
        self._settings.sample_rate = self._sample_rate_combo.currentData()
        self._settings.input_device = self._device_combo.currentData()
        
        # Save model name
        if model_name:
            self._settings.model_name = model_name
        
        # Parse and save hotkey
        hotkey_config = self._parse_key_sequence()
        if hotkey_config:
            self._settings.hotkey = hotkey_config
        
        # Save LLM settings
        self._settings.llm_provider = self._provider_combo.currentData()
        self._settings.llm_model = self._llm_model_combo.currentText()
        api_key = self._api_key_edit.text().strip()
        if api_key:
            self._settings.llm_api_key = api_key
        if self._api_base_edit.isVisible():
            api_base = self._api_base_edit.text().strip()
            self._settings.llm_api_base = api_base if api_base else None
        else:
            self._settings.llm_api_base = None
        
        # Save active enhancement
        self._settings.active_enhancement_id = self._active_enhancement_combo.currentData()
        
        self._settings.save()
        self.settings_changed.emit()
    
    def _validate_model_name(self, model_name: str) -> bool:
        """
        Validate that the model name is a valid HuggingFace/NVIDIA model.
        
        Checks format (org/model) and attempts to verify existence.
        Returns True if valid, False otherwise.
        """
        # Basic format check: should be org/model format
        if "/" not in model_name or len(model_name.split("/")) != 2:
            QMessageBox.warning(
                self,
                "Invalid Model Name",
                f"Model name '{model_name}' is not in the correct format.\n\n"
                "Expected format: organization/model-name\n"
                "Example: nvidia/parakeet-tdt-0.6b-v3 or openai/whisper-large-v3"
            )
            return False
        
        org, model = model_name.split("/")
        if not org or not model:
            QMessageBox.warning(
                self,
                "Invalid Model Name",
                "Both organization and model name must be provided."
            )
            return False
        
        # Note: Full validation (checking if model exists on HuggingFace)
        # requires a network request and is done when loading the model.
        # Here we just validate the format.
        return True
    
    def _parse_key_sequence(self) -> Optional[HotkeyConfig]:
        """Parse QKeySequence from the hotkey editor into HotkeyConfig."""
        key_sequence = self._hotkey_edit.keySequence()
        if key_sequence.isEmpty():
            return None
        
        # Get string representation like "Ctrl+Space"
        seq_str = key_sequence.toString()
        if not seq_str:
            return None
        
        parts = seq_str.split("+")
        if not parts:
            return None
        
        # Last part is the key, rest are modifiers
        modifiers: List[str] = []
        key = parts[-1].lower()
        
        for part in parts[:-1]:
            mod = part.lower()
            # Normalize modifier names
            if mod in ("ctrl", "control"):
                modifiers.append("ctrl")
            elif mod in ("alt", "option"):
                modifiers.append("alt")
            elif mod in ("shift",):
                modifiers.append("shift")
            elif mod in ("meta", "cmd", "command", "win"):
                modifiers.append("cmd")
        
        return HotkeyConfig(modifiers=modifiers, key=key)
    
    def _save_and_close(self) -> None:
        """Save settings and close the dialog."""
        self._save_settings()
        self.accept()
    
    def _reset_settings(self) -> None:
        """Reset all settings to defaults."""
        self._settings.reset_to_defaults()
        self._load_settings()


class EnhancementEditDialog(QDialog):
    """Dialog for creating or editing an enhancement."""
    
    def __init__(self, parent: Optional[QWidget] = None, enhancement: Optional[Dict] = None):
        super().__init__(parent)
        
        self._enhancement = enhancement
        self._is_new = enhancement is None
        
        self.setWindowTitle("Add Enhancement" if self._is_new else "Edit Enhancement")
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)
        
        self._setup_ui()
        self._load_enhancement()
    
    def _setup_ui(self) -> None:
        """Build the dialog UI."""
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
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _load_enhancement(self) -> None:
        """Load enhancement data into the form."""
        if self._enhancement:
            self._title_edit.setText(self._enhancement.get("title", ""))
            self._prompt_edit.setPlainText(self._enhancement.get("prompt", ""))
    
    def _validate_and_accept(self) -> None:
        """Validate input and accept if valid."""
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
        """Get the enhancement data from the form."""
        if self._is_new:
            enh_id = str(uuid.uuid4())[:8]
        else:
            enh_id = self._enhancement.get("id", str(uuid.uuid4())[:8])
        
        return {
            "id": enh_id,
            "title": self._title_edit.text().strip(),
            "prompt": self._prompt_edit.toPlainText().strip(),
        }
