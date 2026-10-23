use serde::{Deserialize, Serialize};

use super::known_cleanup_model;
use crate::{
    AudioDeviceId, HistoryLimit, HistoryRetention, LlmProvider, ModelId, ProviderEndpoint,
    RecordingMode, Settings, SettingsError, Shortcut, ThemePreference, VocabularyReplacement,
};

/// One edit the settings window saves. The window shows it with [`Self::apply`] until the
/// desktop, which saves it with the same rule, answers.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SettingChange {
    InputDevice(Option<AudioDeviceId>),
    RecordingShortcut {
        shortcut: Shortcut,
        mode: RecordingMode,
    },
    CancelShortcut(Option<Shortcut>),
    StartMinimized(bool),
    Theme(ThemePreference),
    HistoryLimit(HistoryLimit),
    HistoryRetention(HistoryRetention),
    KeepAudio(bool),
    ActiveProvider(LlmProvider),
    ChooseProviderModel {
        provider: LlmProvider,
        model: String,
    },
    RemoveProviderModel {
        provider: LlmProvider,
        model: String,
    },
    ProviderEndpoint {
        provider: LlmProvider,
        api_base: Option<ProviderEndpoint>,
    },
    CleanupEnabled(bool),
    CleanupUsesGpu(bool),
    CleanupModel(ModelId),
    Vocabulary(Vec<VocabularyReplacement>),
}

impl SettingChange {
    pub fn apply(self, settings: &mut Settings) {
        match self {
            Self::InputDevice(device) => settings.input_device = device,
            Self::RecordingShortcut { shortcut, mode } => {
                settings.recording_shortcut = shortcut;
                settings.recording_mode = mode;
            }
            Self::CancelShortcut(shortcut) => settings.cancel_shortcut = shortcut,
            Self::StartMinimized(start_minimized) => settings.start_minimized = start_minimized,
            Self::Theme(theme) => settings.theme = theme,
            Self::HistoryLimit(limit) => settings.history_limit = limit,
            Self::HistoryRetention(retention) => settings.history_retention = retention,
            Self::KeepAudio(keep_audio) => settings.keep_audio = keep_audio,
            Self::ActiveProvider(provider) => settings.llm_provider = provider,
            Self::ChooseProviderModel { provider, model } => {
                settings.provider_settings_mut(provider).choose_model(model);
            }
            Self::RemoveProviderModel { provider, model } => {
                settings
                    .provider_settings_mut(provider)
                    .remove_model(&model);
            }
            Self::ProviderEndpoint { provider, api_base } => {
                settings.provider_settings_mut(provider).api_base = api_base;
            }
            Self::CleanupEnabled(enabled) => settings.cleanup_enabled = enabled,
            Self::CleanupUsesGpu(uses_gpu) => settings.cleanup_uses_gpu = uses_gpu,
            Self::CleanupModel(model) => {
                settings.cleanup_model = known_cleanup_model(model.as_str());
            }
            Self::Vocabulary(rules) => settings.vocabulary_replacements = rules,
        }
    }

    /// The first rule `edited`, the settings this change made, breaks in what the change
    /// edits. A rule broken elsewhere, such as by a hand edit, does not refuse it.
    pub fn check(&self, edited: &Settings) -> Result<(), SettingsError> {
        edited
            .problems()
            .into_iter()
            .find(|problem| self.can_break(problem))
            .map_or(Ok(()), Err)
    }

    fn can_break(&self, problem: &SettingsError) -> bool {
        match (self, problem) {
            (
                Self::RecordingShortcut { .. },
                SettingsError::RecordingShortcut | SettingsError::CancelIsRecordingShortcut,
            )
            | (
                Self::CancelShortcut(_),
                SettingsError::CancelShortcut | SettingsError::CancelIsRecordingShortcut,
            )
            | (Self::Vocabulary(_), SettingsError::VocabularySource) => true,
            (
                Self::ChooseProviderModel { provider, .. }
                | Self::RemoveProviderModel { provider, .. },
                SettingsError::ModelName { provider: broken },
            ) => provider == broken,
            _ => false,
        }
    }
}
