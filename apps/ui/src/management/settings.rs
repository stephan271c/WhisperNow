//! Every control saves its own edit. The edit shows at once in the window's copy of the
//! settings, which goes to the desktop whole; edits made while it is there go together once
//! it answers. A refused edit drops out again while its control shows why.

use whispernow_api::{
    ApiErrorDto, SaveSettingsResponse, Settings, SettingsView, TestPromptRequest,
    TestPromptResponse, ThemePreference, UpdateSettingsRequest,
};

use super::ManagementState;
use crate::NoticeScope;
use crate::components::ui::Tone;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum SettingKey {
    InputDevice,
    HoldShortcut,
    ToggleShortcut,
    CancelShortcut,
    StartMinimized,
    StartOnLogin,
    Theme,
    OverlayStyle,
    OverlayPosition,
    HistoryLimit,
    KeepAudio,
    ActiveProvider,
    ProviderModel,
    ProviderApiBase,
    LlmTimeout,
    /// Whether AI cleanup is on, and with which prompt.
    ActivePrompt,
    /// Adding and deleting prompts.
    Prompts,
    PromptTitle(String),
    /// A prompt's instructions, including choosing between them and a newer default.
    PromptText(String),
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PromptTest {
    pub prompt_id: String,
    /// `None` while the provider is working.
    pub outcome: Option<Result<String, String>>,
}

impl ManagementState {
    pub fn install_settings(&mut self, settings: SettingsView) {
        match settings.warnings.first() {
            Some(warning) => self.notify(Tone::Warning, NoticeScope::Settings, warning.clone()),
            None => self.clear_notice_scope(NoticeScope::Settings),
        }
        self.settings = Some(settings);
    }

    /// The settings as the controls show them: the saved settings with every edit the desktop
    /// has not saved yet.
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

    /// Applies the edit of control `key` to the shown settings; `false` before the settings
    /// have loaded. The desktop checks it, and a refusal shows beside the control.
    pub fn change_setting(&mut self, key: SettingKey, edit: impl FnOnce(&mut Settings)) -> bool {
        let Some(mut settings) = self.shown_settings().cloned() else {
            return false;
        };
        edit(&mut settings);
        self.edited = Some(settings);
        self.setting_errors.remove(&key);
        self.saved_settings.clear();
        self.unsent.push(key);
        true
    }

    /// Shows why a value typed into a control cannot be saved; nothing is sent.
    pub fn reject_setting(&mut self, key: SettingKey, message: impl Into<String>) {
        self.saved_settings.retain(|saved| *saved != key);
        self.setting_errors.insert(key, message.into());
    }

    /// The shown settings, when they hold edits and no other save is with the desktop.
    pub fn next_settings_update(&mut self) -> Option<UpdateSettingsRequest> {
        if !self.sending.is_empty() || self.unsent.is_empty() {
            return None;
        }
        self.sending = std::mem::take(&mut self.unsent);
        Some(UpdateSettingsRequest {
            settings: self.edited.clone()?,
        })
    }

    #[must_use]
    pub fn settings_saving(&self) -> bool {
        !self.sending.is_empty() || !self.unsent.is_empty()
    }

    #[must_use]
    pub fn setting_status(&self, key: &SettingKey) -> SaveStatus {
        if let Some(message) = self.setting_errors.get(key) {
            SaveStatus::Failed(message.clone())
        } else if self.sending.contains(key) || self.unsent.contains(key) {
            SaveStatus::Saving
        } else if self.saved_settings.contains(key) {
            SaveStatus::Saved
        } else {
            SaveStatus::Idle
        }
    }

    /// Installs the desktop's answer to the save it has: the settings it saved, with the
    /// native side effects that could not follow them as the notice, or why it refused them.
    /// Edits made meanwhile stay shown and are sent next.
    pub fn finish_settings_update(&mut self, answer: Result<SaveSettingsResponse, String>) {
        let sent = std::mem::take(&mut self.sending);
        match answer {
            Ok(SaveSettingsResponse { settings, warnings }) => {
                self.settings = Some(settings);
                if warnings.is_empty() {
                    self.saved_settings = sent;
                } else {
                    self.notify_saved(NoticeScope::Settings, "Saved", &warnings);
                }
            }
            Err(message) => {
                for key in sent {
                    self.reject_setting(key, message.clone());
                }
            }
        }
        if self.unsent.is_empty() {
            self.edited = None;
        }
    }

    pub fn finish_credential_mutation(&mut self, response: SettingsView) {
        self.settings = Some(response);
    }

    pub fn begin_prompt_test(
        &mut self,
        prompt_id: &str,
        sample_text: &str,
    ) -> Option<TestPromptRequest> {
        let request = TestPromptRequest {
            prompt_id: prompt_id.to_owned(),
            sample_text: sample_text.trim().to_owned(),
        };
        if request.sample_text.is_empty() {
            self.prompt_test = Some(PromptTest {
                prompt_id: prompt_id.to_owned(),
                outcome: Some(Err("Enter some sample text to test the prompt".to_owned())),
            });
            return None;
        }
        self.prompt_test = Some(PromptTest {
            prompt_id: prompt_id.to_owned(),
            outcome: None,
        });
        Some(request)
    }

    pub fn finish_prompt_test(
        &mut self,
        request: &TestPromptRequest,
        outcome: Result<TestPromptResponse, String>,
    ) {
        if let Some(test) = &mut self.prompt_test
            && test.prompt_id == request.prompt_id
        {
            test.outcome = Some(outcome.map(|response| response.text));
        }
    }
}

/// A save confirmation that names every side effect the desktop could not apply.
pub(super) fn saved_notice(saved: &str, warnings: &[ApiErrorDto]) -> String {
    if warnings.is_empty() {
        return saved.to_owned();
    }
    let summaries = warnings
        .iter()
        .map(|warning| warning.summary.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    format!("{saved}, but not everything took effect. {summaries}")
}
