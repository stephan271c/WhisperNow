"""Enhancements/Mode tab for LLM settings and enhancement management."""

from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QGroupBox, QFormLayout, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QDialog, QStackedWidget
)
from PySide6.QtCore import Qt, Signal

from ...core.settings import Settings
from ...core.transcript_processor import PROVIDERS, get_models_for_provider
from .enhancement_edit_dialog import EnhancementEditDialog


class LLMSettingsWidget(QWidget):
    """Widget for configuring LLM provider, model, and API credentials."""

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the LLM settings UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        llm_group = QGroupBox("LLM Settings")
        llm_layout = QFormLayout(llm_group)

        # Provider selector
        self._provider_combo = QComboBox()
        for provider_id, (display_name, _, _) in PROVIDERS.items():
            self._provider_combo.addItem(display_name, provider_id)
        self._provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        llm_layout.addRow("Provider:", self._provider_combo)

        # Model selector - uses stacked widget for combo vs text input
        model_layout = QHBoxLayout()
        
        self._model_stack = QStackedWidget()
        
        # Option 1: ComboBox for providers with curated lists
        self._llm_model_combo = QComboBox()
        self._llm_model_combo.setEditable(True)
        self._llm_model_combo.setMinimumWidth(250)
        self._model_stack.addWidget(self._llm_model_combo)  # Index 0
        
        # Option 2: LineEdit for providers with many models
        self._llm_model_edit = QLineEdit()
        self._llm_model_edit.setPlaceholderText("e.g., openrouter/anthropic/claude-3-sonnet")
        self._llm_model_edit.setMinimumWidth(250)
        self._model_stack.addWidget(self._llm_model_edit)  # Index 1
        
        model_layout.addWidget(self._model_stack, 1)

        # Refresh button (only visible for combo mode)
        self._refresh_btn = QPushButton("⟳")
        self._refresh_btn.setFixedWidth(30)
        self._refresh_btn.setToolTip("Refresh model list from provider")
        self._refresh_btn.clicked.connect(lambda: self._refresh_model_list())
        model_layout.addWidget(self._refresh_btn)
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

    def _uses_text_input(self, provider_id: str) -> bool:
        """Return True if provider should use text input instead of dropdown."""
        return provider_id in ("openrouter", "ollama", "other")

    def _on_provider_changed(self) -> None:
        """Handle provider selection change."""
        provider_id = self._provider_combo.currentData()
        if not provider_id:
            return

        # Get provider config
        _, default_api_base, _ = PROVIDERS.get(provider_id, (None, None, None))

        # Show/hide API base field based on provider
        needs_api_base = provider_id in ("ollama", "other")
        self._api_base_label.setVisible(needs_api_base)
        self._api_base_edit.setVisible(needs_api_base)

        # Set default API base if switching to a provider with one
        if default_api_base and not self._api_base_edit.text():
            self._api_base_edit.setText(default_api_base)

        # Handle API key based on whether we're switching back to saved provider
        if provider_id == self._settings.llm_provider:
            if self._settings.llm_api_key:
                self._api_key_edit.setText(self._settings.llm_api_key)
        else:
            self._api_key_edit.clear()

        # Switch model input mode
        use_text = self._uses_text_input(provider_id)
        self._model_stack.setCurrentIndex(1 if use_text else 0)
        self._refresh_btn.setVisible(not use_text)
        
        if use_text:
            placeholders = {
                "openrouter": "e.g., openrouter/anthropic/claude-3-sonnet",
                "ollama": "e.g., ollama/llama3.2",
                "other": "e.g., your-provider/model-name",
            }
            self._llm_model_edit.setPlaceholderText(placeholders.get(provider_id, ""))
            
            if provider_id == self._settings.llm_provider:
                if self._settings.llm_model:
                    self._llm_model_edit.setText(self._settings.llm_model)
            else:
                self._llm_model_edit.clear()
        else:
            self._refresh_model_list(reset_selection=True)
            if provider_id == self._settings.llm_provider and self._settings.llm_model:
                self._llm_model_combo.setCurrentText(self._settings.llm_model)

    def _refresh_model_list(self, reset_selection: bool = False) -> None:
        """Refresh the model dropdown with models from the selected provider."""
        provider_id = self._provider_combo.currentData()
        if not provider_id:
            return

        current_model = "" if reset_selection else self._llm_model_combo.currentText()
        self._llm_model_combo.clear()

        models = get_models_for_provider(provider_id)
        if models:
            self._llm_model_combo.addItems(models)

        if not reset_selection and current_model:
            idx = self._llm_model_combo.findText(current_model)
            if idx >= 0:
                self._llm_model_combo.setCurrentIndex(idx)
            else:
                self._llm_model_combo.setCurrentText(current_model)

    def load_settings(self) -> None:
        """Load current settings into the UI."""
        provider_idx = self._provider_combo.findData(self._settings.llm_provider)
        if provider_idx >= 0:
            self._provider_combo.setCurrentIndex(provider_idx)

        if self._settings.llm_api_base:
            self._api_base_edit.setText(self._settings.llm_api_base)

        self._on_provider_changed()

        if self._uses_text_input(self._settings.llm_provider):
            self._llm_model_edit.setText(self._settings.llm_model or "")
        else:
            self._llm_model_combo.setCurrentText(self._settings.llm_model or "")

        if self._settings.llm_api_key:
            self._api_key_edit.setText(self._settings.llm_api_key)

    def save_settings(self) -> None:
        """Save UI values to settings."""
        provider = self._provider_combo.currentData()
        self._settings.llm_provider = provider
        
        if self._uses_text_input(provider):
            self._settings.llm_model = self._llm_model_edit.text().strip()
        else:
            self._settings.llm_model = self._llm_model_combo.currentText()
        
        api_key = self._api_key_edit.text().strip()
        if api_key:
            self._settings.llm_api_key = api_key
            
        if self._api_base_edit.isVisible():
            api_base = self._api_base_edit.text().strip()
            self._settings.llm_api_base = api_base if api_base else None
        else:
            self._settings.llm_api_base = None


class EnhancementListWidget(QWidget):
    """Widget for managing enhancement prompts."""

    enhancements_changed = Signal()

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the enhancement list UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

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

    def refresh_list(self) -> None:
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
            self.refresh_list()
            self.enhancements_changed.emit()

    def _edit_enhancement(self, item: QListWidgetItem) -> None:
        """Edit an enhancement from the list."""
        enh_dict = item.data(Qt.UserRole)
        dialog = EnhancementEditDialog(self, enh_dict)
        if dialog.exec() == QDialog.Accepted:
            updated = dialog.get_enhancement()
            for i, e in enumerate(self._settings.enhancements):
                if e.get("id") == updated.get("id"):
                    self._settings.enhancements[i] = updated
                    break
            self.refresh_list()
            self.enhancements_changed.emit()

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
            if self._settings.active_enhancement_id == enh_dict.get("id"):
                self._settings.active_enhancement_id = None
            self.refresh_list()
            self.enhancements_changed.emit()

    def load_settings(self) -> None:
        """Load current settings into the UI."""
        self.refresh_list()

    def save_settings(self) -> None:
        """Save UI values to settings."""
        self._settings.active_enhancement_id = self._active_enhancement_combo.currentData()


class EnhancementsTab(QWidget):
    """Tab for configuring LLM settings and managing enhancements."""

    enhancements_changed = Signal()

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the UI layout by composing child widgets."""
        layout = QVBoxLayout(self)

        # LLM Settings section
        self._llm_settings = LLMSettingsWidget(self._settings, self)
        layout.addWidget(self._llm_settings)

        # Enhancement list section
        self._enhancement_list = EnhancementListWidget(self._settings, self)
        self._enhancement_list.enhancements_changed.connect(self.enhancements_changed.emit)
        layout.addWidget(self._enhancement_list)

    def refresh_enhancement_list(self) -> None:
        """Refresh the enhancement list (for external callers)."""
        self._enhancement_list.refresh_list()

    def load_settings(self) -> None:
        """Load current settings into the UI."""
        self._llm_settings.load_settings()
        self._enhancement_list.load_settings()

    def save_settings(self) -> None:
        """Save UI values to settings."""
        self._llm_settings.save_settings()
        self._enhancement_list.save_settings()
