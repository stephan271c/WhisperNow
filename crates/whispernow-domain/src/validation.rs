use thiserror::Error;

use crate::{LlmProvider, Settings, SetupCheckpoint};

/// Its message is shown beside the control that broke it.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum SettingsError {
    #[error("Choose a modifier and key for the recording shortcut")]
    RecordingShortcut,
    #[error("Use Escape or a function key alone, or a modifier and a key")]
    CancelShortcut,
    #[error("The cancel shortcut and the recording shortcut need different keys")]
    CancelIsRecordingShortcut,
    #[error("Model names cannot be empty or contain line breaks")]
    ModelName { provider: LlmProvider },
    #[error("Enter the text to replace")]
    VocabularySource,
}

impl Settings {
    /// Every rule these settings break, in the order the settings window shows its sections.
    #[must_use]
    pub fn problems(&self) -> Vec<SettingsError> {
        let mut problems = Vec::new();
        problems.extend(self.shortcut_problems());
        for (&provider, settings) in &self.llm_provider_settings {
            let mut unique_models = settings.saved_models.clone();
            unique_models.sort_unstable();
            unique_models.dedup();
            if settings.model.contains(['\n', '\r'])
                || unique_models.len() != settings.saved_models.len()
                || settings
                    .saved_models
                    .iter()
                    .any(|model| model.trim().is_empty() || model.contains(['\n', '\r']))
            {
                problems.push(SettingsError::ModelName { provider });
            }
        }
        if self
            .vocabulary_replacements
            .iter()
            .any(|rule| rule.source.trim().is_empty())
        {
            problems.push(SettingsError::VocabularySource);
        }
        problems
    }

    pub fn validate(&self) -> Result<(), SettingsError> {
        self.problems().into_iter().next().map_or(Ok(()), Err)
    }

    /// Checks settings an edit turned `before` into. A rule `before` already broke does not
    /// block the edit: settings saved by an older version, or before a provider changed its
    /// rules, still accept other changes and the change that repairs them.
    pub fn validate_update(&self, before: &Self) -> Result<(), SettingsError> {
        let existing = before.problems();
        self.problems()
            .into_iter()
            .find(|problem| !existing.contains(problem))
            .map_or(Ok(()), Err)
    }

    /// Every rule the recording and cancel shortcuts break; none may be registered while
    /// they break one.
    #[must_use]
    pub fn shortcut_problems(&self) -> Vec<SettingsError> {
        let mut problems = Vec::new();
        if !self.recording_shortcut.is_recording_shortcut() {
            problems.push(SettingsError::RecordingShortcut);
        }
        if let Some(cancel) = &self.cancel_shortcut {
            if !cancel.is_cancel_shortcut() {
                problems.push(SettingsError::CancelShortcut);
            } else if cancel == &self.recording_shortcut {
                problems.push(SettingsError::CancelIsRecordingShortcut);
            }
        }
        problems
    }
}

impl SetupCheckpoint {
    #[must_use]
    pub fn shortcuts_ready(&self) -> bool {
        self.recording_shortcut.is_recording_shortcut()
    }
}
