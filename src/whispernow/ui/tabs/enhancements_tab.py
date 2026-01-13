from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...core.settings import Settings
from ...core.transcript_processor import PROVIDERS, get_models_for_provider
from .enhancement_edit_dialog import EnhancementEditDialog

# Special value for custom model entry in combo box
_CUSTOM_MODEL_ENTRY = "__custom_model__"


class LLMSettingsWidget(QWidget):

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        llm_group = QGroupBox("LLM Settings")
        llm_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        llm_layout = QFormLayout(llm_group)

        self._provider_combo = QComboBox()
        for provider_id, (display_name, _, _) in PROVIDERS.items():
            self._provider_combo.addItem(display_name, provider_id)
        self._provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        llm_layout.addRow("Provider:", self._provider_combo)

        model_layout = QHBoxLayout()
        self._model_stack = QStackedWidget()

        self._llm_model_combo = QComboBox()
        self._llm_model_combo.setEditable(True)
        self._llm_model_combo.setMinimumWidth(250)
        self._model_stack.addWidget(self._llm_model_combo)  # Index 0

        self._llm_model_edit = QLineEdit()
        self._llm_model_edit.setPlaceholderText(
            "e.g., openrouter/anthropic/claude-3-sonnet"
        )
        self._llm_model_edit.setMinimumWidth(250)
        self._model_stack.addWidget(self._llm_model_edit)  # Index 1

        model_layout.addWidget(self._model_stack, 1)

        self._refresh_btn = QPushButton("⟳")
        self._refresh_btn.setFixedWidth(30)
        self._refresh_btn.setToolTip("Refresh model list from provider")
        self._refresh_btn.clicked.connect(lambda: self._refresh_model_list())
        model_layout.addWidget(self._refresh_btn)
        llm_layout.addRow("Model:", model_layout)

        self._custom_model_edit = QLineEdit()
        self._custom_model_edit.setPlaceholderText("e.g., ollama/gemma3:1b")
        self._custom_model_edit.hide()
        self._custom_model_label = QLabel("")
        self._custom_model_label.hide()
        llm_layout.addRow(self._custom_model_label, self._custom_model_edit)

        self._llm_model_combo.currentIndexChanged.connect(self._on_model_combo_changed)

        self._api_base_edit = QLineEdit()
        self._api_base_edit.setPlaceholderText("e.g., http://localhost:11434")
        self._api_base_label = QLabel("API Base URL:")
        llm_layout.addRow(self._api_base_label, self._api_base_edit)

        self._api_key_edit = QLineEdit()
        self._api_key_edit.setEchoMode(QLineEdit.Password)
        self._api_key_edit.setPlaceholderText("Enter your API key")
        self._api_key_label = QLabel("API Key:")
        llm_layout.addRow(self._api_key_label, self._api_key_edit)

        layout.addWidget(llm_group)

    def _uses_text_input(self, provider_id: str) -> bool:
        return provider_id in ("openrouter", "other")

    def _on_model_combo_changed(self) -> None:
        is_custom = self._llm_model_combo.currentData() == _CUSTOM_MODEL_ENTRY
        self._custom_model_edit.setVisible(is_custom)
        self._custom_model_label.setVisible(is_custom)

    def _on_provider_changed(self) -> None:
        provider_id = self._provider_combo.currentData()
        if not provider_id:
            return

        _, default_api_base, _ = PROVIDERS.get(provider_id, (None, None, None))
        provider_settings = self._settings.get_provider_settings(provider_id)

        needs_api_base = provider_id in ("ollama", "other")
        self._api_base_label.setVisible(needs_api_base)
        self._api_base_edit.setVisible(needs_api_base)

        if provider_settings.api_base:
            self._api_base_edit.setText(provider_settings.api_base)
        elif default_api_base:
            self._api_base_edit.setText(default_api_base)
        else:
            self._api_base_edit.clear()

        _, _, env_var_name = PROVIDERS.get(provider_id, (None, None, None))
        needs_api_key = env_var_name is not None or provider_id == "other"
        self._api_key_label.setVisible(needs_api_key)
        self._api_key_edit.setVisible(needs_api_key)

        if needs_api_key and provider_settings.api_key:
            self._api_key_edit.setText(provider_settings.api_key)
        else:
            self._api_key_edit.clear()

        use_text = self._uses_text_input(provider_id)
        self._model_stack.setCurrentIndex(1 if use_text else 0)
        self._refresh_btn.setVisible(not use_text and provider_id != "ollama")

        self._custom_model_edit.hide()
        self._custom_model_label.hide()

        if use_text:
            placeholders = {
                "openrouter": "e.g., openrouter/anthropic/claude-3-sonnet",
                "other": "e.g., your-provider/model-name",
            }
            self._llm_model_edit.setPlaceholderText(placeholders.get(provider_id, ""))
            if provider_settings.model:
                self._llm_model_edit.setText(provider_settings.model)
            else:
                self._llm_model_edit.clear()
        elif provider_id == "ollama":
            self._populate_ollama_models(provider_settings)
        else:
            self._refresh_model_list(reset_selection=True)
            if provider_settings.model:
                self._llm_model_combo.setCurrentText(provider_settings.model)

    def _populate_ollama_models(self, provider_settings) -> None:
        self._llm_model_combo.blockSignals(True)
        self._llm_model_combo.clear()

        models_to_show = list(provider_settings.saved_models)
        current_model = provider_settings.model
        if current_model and current_model not in models_to_show:
            models_to_show.insert(0, current_model)

        for model in models_to_show:
            self._llm_model_combo.addItem(model, model)

        self._llm_model_combo.addItem("Custom Model...", _CUSTOM_MODEL_ENTRY)

        if current_model:
            idx = self._llm_model_combo.findData(current_model)
            if idx >= 0:
                self._llm_model_combo.setCurrentIndex(idx)
                self._llm_model_combo.setEditText(current_model)

        self._llm_model_combo.blockSignals(False)
        self._on_model_combo_changed()

    def _refresh_model_list(self, reset_selection: bool = False) -> None:
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
        provider_idx = self._provider_combo.findData(self._settings.llm_provider)
        if provider_idx >= 0:
            self._provider_combo.setCurrentIndex(provider_idx)
        self._on_provider_changed()

    def save_settings(self) -> None:
        provider = self._provider_combo.currentData()
        self._settings.llm_provider = provider

        if self._uses_text_input(provider):
            self._settings.llm_model = self._llm_model_edit.text().strip()
        elif provider == "ollama":
            self._save_ollama_model()
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

    def _save_ollama_model(self) -> None:
        is_custom = self._llm_model_combo.currentData() == _CUSTOM_MODEL_ENTRY

        if is_custom:
            model = self._custom_model_edit.text().strip()
        else:
            model = self._llm_model_combo.currentData() or ""

        if not model:
            return

        self._settings.llm_model = model

        provider_settings = self._settings.get_provider_settings("ollama")
        if model and model not in provider_settings.saved_models:
            provider_settings.saved_models.append(model)
            self._settings.set_provider_settings("ollama", provider_settings)


class EnhancementListWidget(QWidget):

    enhancements_changed = Signal()

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        enhance_group = QGroupBox("Enhancements")
        enhance_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        enhance_layout = QVBoxLayout(enhance_group)

        active_layout = QHBoxLayout()
        active_layout.addWidget(QLabel("Active Enhancement:"))
        self._active_enhancement_combo = QComboBox()
        active_layout.addWidget(self._active_enhancement_combo, 1)
        enhance_layout.addLayout(active_layout)

        self._enhancement_list = QListWidget()
        self._enhancement_list.setMaximumHeight(150)
        self._enhancement_list.itemDoubleClicked.connect(self._edit_enhancement)
        enhance_layout.addWidget(self._enhancement_list)

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
        self._enhancement_list.clear()
        self._active_enhancement_combo.clear()
        self._active_enhancement_combo.addItem("None (disabled)", None)

        for enh_dict in self._settings.enhancements:
            item = QListWidgetItem(enh_dict.get("title", "Untitled"))
            item.setData(Qt.UserRole, enh_dict)
            self._enhancement_list.addItem(item)

            self._active_enhancement_combo.addItem(
                enh_dict.get("title", "Untitled"), enh_dict.get("id")
            )

        if self._settings.active_enhancement_id:
            idx = self._active_enhancement_combo.findData(
                self._settings.active_enhancement_id
            )
            if idx >= 0:
                self._active_enhancement_combo.setCurrentIndex(idx)

    def _add_enhancement(self) -> None:
        dialog = EnhancementEditDialog(self)
        if dialog.exec() == QDialog.Accepted:
            enh = dialog.get_enhancement()
            self._settings.enhancements.append(enh)
            self.refresh_list()
            self.enhancements_changed.emit()

    def _edit_enhancement(self, item: QListWidgetItem) -> None:
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
        item = self._enhancement_list.currentItem()
        if item:
            self._edit_enhancement(item)

    def _delete_enhancement(self) -> None:
        item = self._enhancement_list.currentItem()
        if not item:
            return

        enh_dict = item.data(Qt.UserRole)
        reply = QMessageBox.question(
            self,
            "Delete Enhancement",
            f"Delete enhancement '{enh_dict.get('title')}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._settings.enhancements = [
                e
                for e in self._settings.enhancements
                if e.get("id") != enh_dict.get("id")
            ]
            if self._settings.active_enhancement_id == enh_dict.get("id"):
                self._settings.active_enhancement_id = None
            self.refresh_list()
            self.enhancements_changed.emit()

    def load_settings(self) -> None:
        self.refresh_list()

    def save_settings(self) -> None:
        self._settings.active_enhancement_id = (
            self._active_enhancement_combo.currentData()
        )


class EnhancementsTab(QWidget):

    enhancements_changed = Signal()

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._setup_ui()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.load_settings()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._llm_settings = LLMSettingsWidget(self._settings, self)
        layout.addWidget(self._llm_settings)

        self._enhancement_list = EnhancementListWidget(self._settings, self)
        self._enhancement_list.enhancements_changed.connect(
            self.enhancements_changed.emit
        )
        layout.addWidget(self._enhancement_list)

        layout.addStretch(1)

    def refresh_enhancement_list(self) -> None:
        self._enhancement_list.refresh_list()

    def load_settings(self) -> None:
        self._llm_settings.load_settings()
        self._enhancement_list.load_settings()

    def save_settings(self) -> None:
        self._llm_settings.save_settings()
        self._enhancement_list.save_settings()
