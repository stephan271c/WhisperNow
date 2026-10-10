use thiserror::Error;

use crate::{LlmProvider, Settings, SetupCheckpoint, Shortcut};

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
            let mut saved = settings.saved_models.clone();
            saved.sort_unstable();
            saved.dedup();
            if settings.model.contains(['\n', '\r'])
                || saved.len() != settings.saved_models.len()
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
        shortcut_problems(&self.recording_shortcut, self.cancel_shortcut.as_ref())
    }

    #[must_use]
    pub fn cleanup_warning(&self) -> Option<String> {
        if !self.cleanup_enabled || self.llm_provider == LlmProvider::LocalCleanup {
            return None;
        }
        let provider = self.provider_settings(self.llm_provider);
        let missing = if self.llm_provider.request_model(&provider.model).is_empty() {
            "a model"
        } else if self.llm_provider == LlmProvider::Custom && provider.api_base.is_none() {
            "an API endpoint"
        } else {
            return None;
        };
        Some(format!(
            "AI cleanup needs {missing} for {}; the plain transcript will be pasted until you enter one.",
            self.llm_provider.label()
        ))
    }
}

impl SetupCheckpoint {
    #[must_use]
    pub fn shortcuts_ready(&self) -> bool {
        shortcut_problems(&self.recording_shortcut, None).is_empty()
    }
}

fn shortcut_problems(recording: &Shortcut, cancel: Option<&Shortcut>) -> Vec<SettingsError> {
    let mut problems = Vec::new();
    if !recording.is_recording_shortcut() {
        problems.push(SettingsError::RecordingShortcut);
    }
    if let Some(cancel) = cancel {
        if !cancel.is_cancel_shortcut() {
            problems.push(SettingsError::CancelShortcut);
        } else if cancel == recording {
            problems.push(SettingsError::CancelIsRecordingShortcut);
        }
    }
    problems
}
