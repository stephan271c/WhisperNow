mod audio_device;
mod cleanup;
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

pub use audio_device::AudioDeviceId;
pub use cleanup::{CleanupTarget, MissingCleanupField, RemoteCleanupTarget, RemoteProvider};
pub use data::ClearDataCategory;
pub use history::{
    EntryId, EntryStatus, HistoryEntry, HistoryFilter, HistoryLimit, HistoryPage,
    HistoryPageRequest, HistoryRetention, NewHistoryEntry, UtcMillis,
};
pub use models::{
    CLEANUP_MODEL_IDS, CleanupBenchmark, DEFAULT_CLEANUP_MODEL_ID, DownloadPhase,
    ModelAvailability, ModelBenchmark, ModelDefinition, ModelFamily, ModelId, ModelLayout,
    ModelRegistry, ModelSource, Rating, SpeechBenchmark, WerSource,
};
pub use provider::{EndpointError, LlmProvider, ProviderEndpoint};
pub use settings::{
    ApiKey, InvalidApiKey, ProviderSettings, PythonSettings, SettingChange, Settings,
    SetupCheckpoint, SetupProgress, SetupStep, ThemePreference, VocabularyReplacement,
    WindowGeometry,
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

pub const TEXT_OUTPUT_PERMISSION_DENIED: &str = "text_output_permission_denied";

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
