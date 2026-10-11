//! One control saves at a time; its edit is shown until the desktop confirms or refuses it.

use whispernow_domain::ipc::{SaveSettingsResponse, SettingsView};
use whispernow_domain::{Settings, ThemePreference};

use super::ManagementState;
use crate::NoticeScope;
use crate::components::ui::Tone;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum SettingKey {
    InputDevice,
    /// The recording shortcut's keys and mode.
    RecordingShortcut,
    CancelShortcut,
    StartMinimized,
    StartOnLogin,
    Theme,
    HistoryLimit,
    KeepAudio,
    ActiveProvider,
    ProviderModel,
    ProviderApiBase,
    CleanupEnabled,
    Vocabulary,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum SaveStatus {
    #[default]
    Idle,
    Saving,
    Saved,
    /// The change was not saved; the control shows the saved value again.
    Failed(String),
}

impl ManagementState {
    pub fn install_settings(&mut self, settings: SettingsView) {
        if let Some(warning) = settings.warnings.first() {
            self.notify(Tone::Warning, NoticeScope::Settings, warning.clone());
            self.cleanup_warning_notice = Some(self.next_notice);
        } else {
            self.clear_notice_scope(NoticeScope::Settings);
            self.cleanup_warning_notice = None;
        }
        self.settings = Some(settings);
    }

    #[must_use]
    pub fn shown_settings(&self) -> Option<&Settings> {
        self.edited
            .as_ref()
            .or_else(|| self.settings.as_ref().map(|view| &view.settings))
    }

    /// The document `data-theme` for the chosen theme, following a theme change at once;
    /// `None` follows the system colour scheme.
    #[must_use]
    #[cfg_attr(
        not(any(test, target_arch = "wasm32")),
        expect(dead_code, reason = "called by the browser's window root and by tests")
    )]
    pub fn theme_attribute(&self) -> Option<&'static str> {
        let theme = self
            .shown_settings()
            .map_or(ThemePreference::System, |settings| settings.theme);
        match theme {
            ThemePreference::System => None,
            ThemePreference::Light => Some("light"),
            ThemePreference::Dark => Some("dark"),
        }
    }

    #[must_use]
    pub fn settings_pending(&self) -> bool {
        self.saving.is_some()
            || self
                .busy
                .iter()
                .any(|operation| operation.changes_settings())
    }

    pub(super) fn begin_settings_change(&self) -> bool {
        !self.settings_pending()
    }

    pub fn begin_settings_update(
        &mut self,
        key: SettingKey,
        edit: impl FnOnce(&mut Settings),
    ) -> Option<Settings> {
        if !self.begin_settings_change() {
            return None;
        }
        let mut settings = self
            .shown_settings()
            .cloned()
            .expect("settings controls render after loading");
        edit(&mut settings);
        self.edited = Some(settings.clone());
        self.saving = Some(key);
        self.setting_errors.remove(&key);
        self.saved_setting = None;
        Some(settings)
    }

    /// Shows why a value typed into a control cannot be saved; nothing is sent.
    pub fn reject_setting(&mut self, key: SettingKey, message: impl Into<String>) {
        if self.saved_setting == Some(key) {
            self.saved_setting = None;
        }
        self.setting_errors.insert(key, message.into());
    }

    #[must_use]
    pub fn setting_status(&self, key: SettingKey) -> SaveStatus {
        if let Some(message) = self.setting_errors.get(&key) {
            SaveStatus::Failed(message.clone())
        } else if self.saving == Some(key) {
            SaveStatus::Saving
        } else if self.saved_setting == Some(key) {
            SaveStatus::Saved
        } else {
            SaveStatus::Idle
        }
    }

    pub fn finish_settings_update(&mut self, answer: Result<SaveSettingsResponse, String>) {
        let key = self.saving.take().expect("a settings save is pending");
        match answer {
            Ok(SaveSettingsResponse { settings, warnings }) => {
                let cleanup_warning_only = !warnings.is_empty() && warnings == settings.warnings;
                self.settings = Some(settings);
                if warnings.is_empty() {
                    if let Some(id) = self.cleanup_warning_notice.take() {
                        self.dismiss_notice(id);
                    }
                    self.saved_setting = Some(key);
                } else {
                    let id = self.notify_saved(NoticeScope::Settings, "Saved", &warnings);
                    self.cleanup_warning_notice = cleanup_warning_only.then_some(id);
                }
            }
            Err(message) => {
                self.reject_setting(key, message);
            }
        }
        self.edited = None;
    }
}
