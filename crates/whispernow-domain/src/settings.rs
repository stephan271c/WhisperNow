use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};

use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use thiserror::Error;

use crate::{
    CLEANUP_MODEL_IDS, DEFAULT_CLEANUP_MODEL_ID, HistoryLimit, HistoryRetention, LlmProvider,
    ModelId, ProviderEndpoint, RecordingMode, Shortcut,
};

mod legacy;

pub use legacy::PythonSettings;

/// An unfinished setup keeps its step and choices in [`Settings::setup`] so an interrupted wizard
/// resumes without claiming setup is complete.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SetupStep {
    #[default]
    Welcome,
    Permissions,
    Model,
    Shortcuts,
    TryIt,
}

impl SetupStep {
    /// Every step in wizard order.
    pub const ALL: [Self; 5] = [
        Self::Welcome,
        Self::Permissions,
        Self::Model,
        Self::Shortcuts,
        Self::TryIt,
    ];
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ThemePreference {
    #[default]
    System,
    Light,
    Dark,
}

/// Download operation identifiers are intentionally excluded because they are process-scoped. The
/// saved settings stay in effect until setup finishes and commits these choices.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct SetupCheckpoint {
    pub step: SetupStep,
    pub model_id: ModelId,
    pub input_device: Option<String>,
    pub recording_shortcut: Shortcut,
    pub recording_mode: RecordingMode,
}

/// A secret API key, kept in the OS credential store. It crosses to the desktop only when the
/// user enters it, and is never saved with the settings.
///
/// Its `Debug` implementation is always redacted. Callers must deliberately expose
/// the value when passing it to a provider client.
#[derive(Clone, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct ApiKey(String);

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
#[error("Enter the API key on one line")]
pub struct InvalidApiKey;

impl ApiKey {
    pub fn new(value: impl Into<String>) -> Result<Self, InvalidApiKey> {
        let value = value.into();
        let key = value.trim();
        if key.is_empty() || key.chars().any(char::is_control) {
            return Err(InvalidApiKey);
        }
        Ok(Self(key.to_owned()))
    }

    #[must_use]
    pub fn expose_secret(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for ApiKey {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::new(String::deserialize(deserializer)?).map_err(D::Error::custom)
    }
}

impl fmt::Debug for ApiKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ApiKey([REDACTED])")
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct ProviderSettings {
    pub model: String,
    pub api_base: Option<ProviderEndpoint>,
    pub saved_models: Vec<String>,
}

static DEFAULT_PROVIDER_SETTINGS: ProviderSettings = ProviderSettings {
    model: String::new(),
    api_base: None,
    saved_models: Vec::new(),
};

/// A literal vocabulary replacement, stored as a `[source, replacement]` pair.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(from = "(String, String)", into = "(String, String)")]
pub struct VocabularyReplacement {
    pub source: String,
    pub replacement: String,
}

impl VocabularyReplacement {
    #[must_use]
    pub fn new(source: impl Into<String>, replacement: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            replacement: replacement.into(),
        }
    }
}

impl From<(String, String)> for VocabularyReplacement {
    fn from((source, replacement): (String, String)) -> Self {
        Self {
            source,
            replacement,
        }
    }
}

impl From<VocabularyReplacement> for (String, String) {
    fn from(rule: VocabularyReplacement) -> Self {
        (rule.source, rule.replacement)
    }
}

/// Settings-window bounds in physical pixels, stored as an `[x, y, width, height]` array.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(from = "(i32, i32, u32, u32)", into = "(i32, i32, u32, u32)")]
pub struct WindowGeometry {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl From<(i32, i32, u32, u32)> for WindowGeometry {
    fn from((x, y, width, height): (i32, i32, u32, u32)) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

impl From<WindowGeometry> for (i32, i32, u32, u32) {
    fn from(geometry: WindowGeometry) -> Self {
        (geometry.x, geometry.y, geometry.width, geometry.height)
    }
}

#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct Settings {
    /// Selected input device name, or the host default.
    pub input_device: Option<String>,
    pub recording_shortcut: Shortcut,
    pub recording_mode: RecordingMode,
    /// Shortcut that cancels the active dictation; `None` turns it off.
    pub cancel_shortcut: Option<Shortcut>,
    pub model_id: ModelId,
    pub start_minimized: bool,
    pub theme: ThemePreference,
    pub first_run_complete: bool,
    pub setup: Option<SetupCheckpoint>,
    pub cleanup_enabled: bool,
    /// Whether the local cleanup model may load on a GPU; off keeps it on the CPU.
    pub cleanup_uses_gpu: bool,
    pub llm_provider: LlmProvider,
    pub llm_provider_settings: BTreeMap<LlmProvider, ProviderSettings>,
    /// Providers whose API key is saved in the OS credential store.
    pub api_keys: BTreeSet<LlmProvider>,
    /// Ordered literal replacements applied to transcripts.
    pub vocabulary_replacements: Vec<VocabularyReplacement>,
    pub window_geometry: Option<WindowGeometry>,
    pub history_limit: HistoryLimit,
    /// Entries older than this are deleted with the history, or nothing when `Forever`.
    pub history_retention: HistoryRetention,
    /// Whether dictations keep their audio in history.
    pub keep_audio: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            input_device: None,
            recording_shortcut: Shortcut::default(),
            recording_mode: RecordingMode::default(),
            cancel_shortcut: Some(Shortcut::default_cancel()),
            model_id: ModelId::default(),
            start_minimized: false,
            theme: ThemePreference::System,
            first_run_complete: false,
            setup: None,
            cleanup_enabled: false,
            cleanup_uses_gpu: true,
            llm_provider: LlmProvider::OpenAi,
            llm_provider_settings: BTreeMap::new(),
            api_keys: BTreeSet::new(),
            vocabulary_replacements: Vec::new(),
            window_geometry: None,
            history_limit: HistoryLimit::default(),
            history_retention: HistoryRetention::default(),
            keep_audio: false,
        }
    }
}

impl Settings {
    /// The local cleanup model AI cleanup uses with the Local cleanup model provider: the one
    /// saved as that provider's model, or the default when none, or an unknown one, is saved.
    #[must_use]
    pub fn cleanup_model_id(&self) -> ModelId {
        let id = self
            .llm_provider_settings
            .get(&LlmProvider::LocalCleanup)
            .and_then(|saved| CLEANUP_MODEL_IDS.into_iter().find(|id| *id == saved.model))
            .unwrap_or(DEFAULT_CLEANUP_MODEL_ID);
        ModelId::new(id).expect("cleanup registry IDs are valid")
    }

    #[must_use]
    pub fn provider_settings(&self, provider: LlmProvider) -> &ProviderSettings {
        self.llm_provider_settings
            .get(&provider)
            .unwrap_or(&DEFAULT_PROVIDER_SETTINGS)
    }

    pub fn provider_settings_mut(&mut self, provider: LlmProvider) -> &mut ProviderSettings {
        self.llm_provider_settings.entry(provider).or_default()
    }
}
