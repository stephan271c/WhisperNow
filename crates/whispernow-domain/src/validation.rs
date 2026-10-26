use thiserror::Error;

use crate::{Settings, SetupCheckpoint, VocabularyReplacement};

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
    ModelName,
    #[error("Enter the text to replace")]
    VocabularySource,
}

impl Settings {
    /// Every rule these settings break, in the order the settings window shows its sections.
    #[must_use]
    pub fn problems(&self) -> Vec<SettingsError> {
        let mut problems = Vec::new();
        problems.extend(self.shortcut_problems());
        if self
            .vocabulary_replacements
            .iter()
            .any(VocabularyReplacement::has_blank_source)
        {
            problems.push(SettingsError::VocabularySource);
        }
        problems
    }

    pub fn validate(&self) -> Result<(), SettingsError> {
        self.problems().into_iter().next().map_or(Ok(()), Err)
    }

    /// Whether the recording and cancel shortcuts break no rule; none may be registered while
    /// they break one.
    #[must_use]
    pub fn shortcuts_valid(&self) -> bool {
        self.shortcut_problems().is_empty()
    }

    fn shortcut_problems(&self) -> Vec<SettingsError> {
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

/// A typed provider model name without its surrounding spaces, unless it is blank or spans
/// lines.
pub(crate) fn model_name(typed: &str) -> Option<&str> {
    let name = typed.trim();
    (!name.is_empty() && !name.contains(['\n', '\r'])).then_some(name)
}

impl SetupCheckpoint {
    #[must_use]
    pub fn shortcuts_ready(&self) -> bool {
        self.recording_shortcut.is_recording_shortcut()
    }
}
