use thiserror::Error;

use crate::{
    CLEAN_UP_PROMPT_ID, LLM_TIMEOUT_SECONDS_RANGE, LlmProvider, Settings, SetupCheckpoint, Shortcut,
};

/// Its message is shown beside the control that broke it.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum SettingsError {
    #[error("Keep at least one recording shortcut on")]
    NoRecordingShortcut,
    #[error("Choose a modifier and key for hold-to-record")]
    HoldShortcut,
    #[error("Choose a modifier and key for toggle-to-record")]
    ToggleShortcut,
    #[error("Hold-to-record and toggle-to-record need different shortcuts")]
    SameRecordingShortcuts,
    #[error("Use Escape or a function key alone, or a modifier and a key")]
    CancelShortcut,
    #[error("The cancel shortcut and the recording shortcuts need different keys")]
    CancelIsRecordingShortcut,
    #[error("Enter a time limit between 5 and 120 seconds")]
    LlmTimeout,
    #[error("Model names cannot be empty or contain line breaks")]
    ModelName { provider: LlmProvider },
    #[error("Give the prompt a name")]
    PromptTitle { id: String },
    #[error("Write the instructions for this prompt")]
    PromptText { id: String },
    #[error("Each prompt needs its own identifier")]
    PromptId,
    #[error("The chosen prompt was deleted")]
    UnknownActivePrompt,
    #[error("Enter the text to replace")]
    VocabularySource,
    #[error("AI cleanup needs {missing} for {provider}; enter one, or turn AI cleanup off", provider = .provider.label())]
    CleanupProvider {
        provider: LlmProvider,
        missing: &'static str,
    },
    #[error(
        "The local cleanup model only does Clean Up & Format; choose that prompt, or another provider"
    )]
    LocalCleanupPrompt,
}

impl Settings {
    /// Every rule these settings break, in the order the settings window shows its sections.
    #[must_use]
    pub fn problems(&self) -> Vec<SettingsError> {
        let mut problems = Vec::new();
        problems.extend(self.shortcut_problems());
        if !LLM_TIMEOUT_SECONDS_RANGE.contains(&self.llm_timeout_seconds) {
            problems.push(SettingsError::LlmTimeout);
        }
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
        self.prompt_problems(&mut problems);
        if self
            .vocabulary_replacements
            .iter()
            .any(|rule| rule.source.trim().is_empty())
        {
            problems.push(SettingsError::VocabularySource);
        }
        if let Some(missing) = self.cleanup_provider_missing() {
            problems.push(SettingsError::CleanupProvider {
                provider: self.llm_provider,
                missing,
            });
        }
        if self.llm_provider == LlmProvider::LocalCleanup
            && self
                .active_prompt_id
                .as_deref()
                .is_some_and(|id| id != CLEAN_UP_PROMPT_ID)
        {
            problems.push(SettingsError::LocalCleanupPrompt);
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
        shortcut_problems(
            &self.hold_shortcut,
            self.hold_shortcut_enabled,
            self.toggle_shortcut.as_ref(),
            self.cancel_shortcut.as_ref(),
        )
    }

    #[must_use]
    fn cleanup_provider_missing(&self) -> Option<&'static str> {
        self.active_prompt_id.as_ref()?;
        // The local cleanup model runs in process: it needs no model name or endpoint.
        if self.llm_provider == LlmProvider::LocalCleanup {
            return None;
        }
        let provider = self.provider_settings(self.llm_provider);
        if provider.model.trim().is_empty() {
            Some("a model")
        } else if self.llm_provider == LlmProvider::Custom && provider.api_base.is_none() {
            Some("an API endpoint")
        } else {
            None
        }
    }

    fn prompt_problems(&self, problems: &mut Vec<SettingsError>) {
        for prompt in &self.prompts {
            if prompt.title.trim().is_empty() {
                problems.push(SettingsError::PromptTitle {
                    id: prompt.id.clone(),
                });
            }
            if prompt.prompt.trim().is_empty() {
                problems.push(SettingsError::PromptText {
                    id: prompt.id.clone(),
                });
            }
        }
        let mut ids = self
            .prompts
            .iter()
            .map(|prompt| prompt.id.as_str())
            .collect::<Vec<_>>();
        ids.sort_unstable();
        ids.dedup();
        if ids.len() != self.prompts.len() || ids.iter().any(|id| id.trim().is_empty()) {
            problems.push(SettingsError::PromptId);
        }
        if self
            .active_prompt_id
            .as_ref()
            .is_some_and(|active| !self.prompts.iter().any(|prompt| &prompt.id == active))
        {
            problems.push(SettingsError::UnknownActivePrompt);
        }
    }

    #[must_use]
    pub fn next_prompt_id(&self) -> String {
        // One more candidate than there are prompts, so at least one is free.
        (1..=self.prompts.len() + 1)
            .map(|number| format!("prompt-{number}"))
            .find(|candidate| self.prompts.iter().all(|item| item.id != *candidate))
            .unwrap_or_default()
    }

    pub fn delete_prompt(&mut self, id: &str) {
        self.prompts.retain(|prompt| prompt.id != id);
        if self.active_prompt_id.as_deref() == Some(id) {
            self.active_prompt_id = None;
        }
    }
}

impl SetupCheckpoint {
    #[must_use]
    pub fn shortcuts_ready(&self) -> bool {
        shortcut_problems(
            &self.hold_shortcut,
            self.hold_shortcut_enabled,
            self.toggle_shortcut.as_ref(),
            None,
        )
        .is_empty()
    }
}

fn shortcut_problems(
    hold: &Shortcut,
    hold_enabled: bool,
    toggle: Option<&Shortcut>,
    cancel: Option<&Shortcut>,
) -> Vec<SettingsError> {
    let mut problems = Vec::new();
    if !hold_enabled && toggle.is_none() {
        problems.push(SettingsError::NoRecordingShortcut);
    }
    // A turned-off hold shortcut is kept for turning it back on, so it stays valid too.
    if !hold.is_recording_shortcut() {
        problems.push(SettingsError::HoldShortcut);
    }
    let hold = hold_enabled.then_some(hold);
    if let Some(toggle) = toggle {
        if !toggle.is_recording_shortcut() {
            problems.push(SettingsError::ToggleShortcut);
        } else if hold == Some(toggle) {
            problems.push(SettingsError::SameRecordingShortcuts);
        }
    }
    if let Some(cancel) = cancel {
        if !cancel.is_cancel_shortcut() {
            problems.push(SettingsError::CancelShortcut);
        } else if hold
            .into_iter()
            .chain(toggle)
            .any(|recording| recording == cancel)
        {
            problems.push(SettingsError::CancelIsRecordingShortcut);
        }
    }
    problems
}
