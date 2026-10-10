mod data;
mod history;
pub mod ipc;
mod models;
mod provider;
mod settings;
mod shortcut;
mod validation;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use data::ClearDataCategory;
pub use history::{
    EntryId, EntryStatus, HistoryCursor, HistoryEntry, HistoryFilter, HistoryLimit, HistoryPage,
    HistoryPageRequest, NewHistoryEntry, UtcMillis,
};
pub use models::{
    ArchiveFormat, CLEANUP_MODEL_FILE, CLEANUP_MODEL_IDS, CUDA_PACK_ID, DEFAULT_CLEANUP_MODEL_ID,
    DownloadPhase, MINICPM_CLEANUP_MODEL_ID, ModelAvailability, ModelDefinition, ModelFamily,
    ModelId, ModelLayout, ModelProfile, ModelRegistry, ModelSource, Rating, SpeechBenchmark,
    WerSource,
};
pub use provider::{EndpointError, LlmProvider, ProviderEndpoint};
pub use settings::{
    ApiKey, InvalidApiKey, ProviderSettings, PythonSettings, Settings, SetupCheckpoint, SetupStep,
    ThemePreference, VocabularyReplacement, WindowGeometry,
};
pub use shortcut::{RecordingMode, Shortcut, ShortcutKey, ShortcutModifier};
pub use validation::SettingsError;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCategory {
    General,
    NotFound,
    Audio,
    Permission,
    Model,
    Recognition,
    Cleanup,
    Output,
    HistoryUnavailable,
    Settings,
}

#[derive(Clone, Debug, Deserialize, Error, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
#[error("{code}: {summary}")]
pub struct AppError {
    pub category: ErrorCategory,
    pub code: String,
    /// Must not contain captured or secret data.
    pub summary: String,
}

impl AppError {
    #[must_use]
    pub fn new(
        category: ErrorCategory,
        code: impl Into<String>,
        summary: impl Into<String>,
    ) -> Self {
        Self {
            category,
            code: code.into(),
            summary: summary.into(),
        }
    }
}
