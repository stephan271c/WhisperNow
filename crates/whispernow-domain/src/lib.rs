//! Framework-independent domain primitives shared by every composition root.

mod data;
mod history;
mod models;
mod prompts;
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
    ArchiveFormat, AsrCapabilities, CLEANUP_MODEL_FILE, CLEANUP_MODEL_IDS, CUDA_PACK_ID,
    DEFAULT_CLEANUP_MODEL_ID, DownloadPhase, MINICPM_CLEANUP_MODEL_ID, ModelAvailability,
    ModelDefinition, ModelFamily, ModelId, ModelLayout, ModelProfile, ModelRegistry, ModelSource,
    RelativeRating,
};
pub use prompts::{CLEAN_UP_PROMPT_ID, CleanupPrompt, default_prompts};
pub use provider::{EndpointError, LlmProvider, ProviderEndpoint};
pub use settings::{
    ApiKey, DEFAULT_LLM_TIMEOUT_SECONDS, InvalidApiKey, LLM_TIMEOUT_SECONDS_RANGE, OverlayPosition,
    OverlayStyle, ProviderSettings, PythonSettings, Settings, SetupCheckpoint, SetupStep,
    ThemePreference, VocabularyReplacement, WindowGeometry,
};
pub use shortcut::{RecordingMode, Shortcut, ShortcutKey, ShortcutModifier};
pub use validation::SettingsError;

/// A stable, content-free error suitable for crossing an application boundary.
#[derive(Clone, Debug, Deserialize, Error, Eq, PartialEq, Serialize)]
#[error("{code}: {summary}")]
pub struct AppError {
    /// Stable machine-readable error code.
    pub code: String,
    /// Safe summary that must not contain captured or secret data.
    pub summary: String,
}

impl AppError {
    /// Creates a redacted application error.
    #[must_use]
    pub fn redacted(code: impl Into<String>, summary: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            summary: summary.into(),
        }
    }
}
