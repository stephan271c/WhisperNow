//! One control saves at a time; its edit is shown until the desktop confirms or refuses it.

use whispernow_domain::ipc::{SaveSettingsResponse, SettingsView};
use whispernow_domain::{AppError, SettingChange, Settings, ThemePreference};

use super::ManagementState;
use crate::NoticeScope;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum SettingKey {
    InputDevice,
    RecordingShortcut,
    CancelShortcut,
    StartMinimized,
    StartOnLogin,
    Theme,
    HistoryLimit,
    HistoryRetention,
    KeepAudio,
    ActiveProvider,
    ProviderModel,
    ProviderEndpoint,
    CleanupEnabled,
    CleanupUsesGpu,
    CleanupModel,
    Vocabulary,
}

impl SettingKey {
    #[must_use]
    pub const fn of(change: &SettingChange) -> Self {
        match change {
            SettingChange::InputDevice(_) => Self::InputDevice,
            SettingChange::RecordingShortcut { .. } => Self::RecordingShortcut,
            SettingChange::CancelShortcut(_) => Self::CancelShortcut,
            SettingChange::StartMinimized(_) => Self::StartMinimized,
            SettingChange::Theme(_) => Self::Theme,
            SettingChange::HistoryLimit(_) => Self::HistoryLimit,
            SettingChange::HistoryRetention(_) => Self::HistoryRetention,
            SettingChange::KeepAudio(_) => Self::KeepAudio,
            SettingChange::ActiveProvider(_) => Self::ActiveProvider,
            SettingChange::ChooseProviderModel { .. }
            | SettingChange::RemoveProviderModel { .. } => Self::ProviderModel,
            SettingChange::ProviderEndpoint { .. } => Self::ProviderEndpoint,
            SettingChange::CleanupEnabled(_) => Self::CleanupEnabled,
            SettingChange::CleanupUsesGpu(_) => Self::CleanupUsesGpu,
            SettingChange::CleanupModel(_) => Self::CleanupModel,
            SettingChange::Vocabulary(_) => Self::Vocabulary,
        }
    }
}

/// A control's change, shown until the desktop confirms or refuses it.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct PendingSave {
    key: SettingKey,
    settings: Settings,
}

impl ManagementState {
    pub fn install_settings(&mut self, settings: SettingsView) {
        self.settings = Some(settings);
    }

    #[must_use]
    pub fn shown_settings(&self) -> Option<&Settings> {
        self.pending_save
            .as_ref()
            .map(|pending| &pending.settings)
            .or_else(|| self.settings.as_ref().map(|view| &view.settings))
    }

    /// The document `data-theme` for the chosen theme, following a theme change at once;
    /// `None` follows the system colour scheme.
    #[must_use]
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
        self.pending_save.is_some()
            || self
                .busy
                .iter()
                .any(|operation| operation.changes_settings())
    }

    /// Shows `edit` applied to the shown settings until the desktop answers; `false`, with nothing
    /// changed, while another save is pending.
    pub fn begin_settings_update(
        &mut self,
        key: SettingKey,
        edit: impl FnOnce(&mut Settings),
    ) -> bool {
        if self.settings_pending() {
            return false;
        }
        let mut settings = self
            .shown_settings()
            .cloned()
            .expect("settings controls render after loading");
        edit(&mut settings);
        self.pending_save = Some(PendingSave { key, settings });
        self.setting_errors.remove(&key);
        true
    }

    pub fn reject_setting(&mut self, key: SettingKey, message: impl Into<String>) {
        self.setting_errors.insert(key, message.into());
    }

    /// Why the last change to `key` was not saved; the control shows the saved value again.
    #[must_use]
    pub fn setting_error(&self, key: SettingKey) -> Option<String> {
        self.setting_errors.get(&key).cloned()
    }

    pub fn finish_settings_update(&mut self, answer: Result<SaveSettingsResponse, AppError>) {
        let PendingSave { key, .. } = self
            .pending_save
            .take()
            .expect("a settings save is pending");
        match answer {
            Ok(SaveSettingsResponse { settings, warnings }) => {
                self.install_settings(settings);
                if !warnings.is_empty() {
                    self.notify_saved(NoticeScope::Settings, "Saved", &warnings);
                }
            }
            Err(error) => self.reject_setting(key, error.summary),
        }
    }
}
