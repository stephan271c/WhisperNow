//! One control saves at a time; its edit is shown until the desktop confirms or refuses it.

use whispernow_domain::ipc::{SaveSettingsResponse, SettingsView};
use whispernow_domain::{Settings, ThemePreference};

use super::ManagementState;
use crate::components::ui::Tone;
use crate::{NoticeScope, UiOperation};

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
        match settings.warnings.first() {
            Some(warning) => self.notify(Tone::Warning, NoticeScope::Settings, warning.clone()),
            None => self.clear_notice_scope(NoticeScope::Settings),
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
            || self.busy.iter().any(|operation| {
                matches!(
                    operation,
                    UiOperation::LoadSettings
                        | UiOperation::CredentialMutation
                        | UiOperation::ModelSwitch
                        | UiOperation::SaveSetup
                        | UiOperation::CompleteSetup
                        | UiOperation::RestartSetup
                        | UiOperation::ClearData
                )
            })
    }

    pub fn begin_settings_update(
        &mut self,
        key: SettingKey,
        edit: impl FnOnce(&mut Settings),
    ) -> Option<Settings> {
        if self.settings_pending() {
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
    pub fn setting_status(&self, key: &SettingKey) -> SaveStatus {
        if let Some(message) = self.setting_errors.get(key) {
            SaveStatus::Failed(message.clone())
        } else if self.saving.as_ref() == Some(key) {
            SaveStatus::Saving
        } else if self.saved_setting.as_ref() == Some(key) {
            SaveStatus::Saved
        } else {
            SaveStatus::Idle
        }
    }

    pub fn finish_settings_update(&mut self, answer: Result<SaveSettingsResponse, String>) {
        let key = self.saving.take().expect("a settings save is pending");
        match answer {
            Ok(SaveSettingsResponse { settings, warnings }) => {
                let resolved_cleanup_warning = settings
                    .settings
                    .cleanup_warning()
                    .is_none()
                    .then(|| {
                        self.settings
                            .as_ref()
                            .and_then(|view| view.settings.cleanup_warning())
                    })
                    .flatten();
                self.settings = Some(settings);
                if warnings.is_empty() {
                    if let Some(warning) = resolved_cleanup_warning
                        && let Some(notice) = &self.notice
                        && notice.scope == NoticeScope::Settings
                        && notice.tone == Tone::Warning
                        && (notice.message == warning
                            || notice.message
                                == format!("Saved, but not everything took effect. {warning}"))
                    {
                        self.dismiss_notice(notice.id);
                    }
                    self.saved_setting = Some(key);
                } else {
                    self.notify_saved(NoticeScope::Settings, "Saved", &warnings);
                }
            }
            Err(message) => {
                self.reject_setting(key, message);
            }
        }
        self.edited = None;
    }
}
