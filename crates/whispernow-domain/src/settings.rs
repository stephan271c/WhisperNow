use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};

use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use thiserror::Error;

use crate::{
    AudioDeviceId, CLEANUP_MODEL_IDS, DEFAULT_CLEANUP_MODEL_ID, HistoryLimit, HistoryRetention,
    LlmProvider, ModelId, ModelRegistry, ProviderEndpoint, RecordingMode, Shortcut,
};

mod change;
mod legacy;

pub use change::SettingChange;
pub use legacy::PythonSettings;

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

/// An unfinished setup keeps its step and choices in [`Settings::setup`] so an interrupted wizard
/// resumes without claiming setup is complete. The saved settings stay in effect until setup
/// finishes and commits these choices.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct SetupCheckpoint {
    pub step: SetupStep,
    #[serde(deserialize_with = "speech_model")]
    pub model_id: ModelId,
    #[serde(deserialize_with = "input_device")]
    pub input_device: Option<AudioDeviceId>,
    pub recording_shortcut: Shortcut,
    pub recording_mode: RecordingMode,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SetupProgress {
    Unfinished(SetupCheckpoint),
    Complete,
}

impl Default for SetupProgress {
    fn default() -> Self {
        Self::Unfinished(SetupCheckpoint::default())
    }
}

impl SetupProgress {
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        matches!(self, Self::Complete)
    }

    #[must_use]
    pub const fn checkpoint(&self) -> Option<&SetupCheckpoint> {
        match self {
            Self::Unfinished(checkpoint) => Some(checkpoint),
            Self::Complete => None,
        }
    }
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
    pub fn new(value: impl AsRef<str>) -> Result<Self, InvalidApiKey> {
        let key = value.as_ref().trim();
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

/// A remote provider's model choices. The active model is always one of `saved_models`.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct ProviderSettings {
    pub model: Option<String>,
    pub api_base: Option<ProviderEndpoint>,
    pub saved_models: Vec<String>,
}

static DEFAULT_PROVIDER_SETTINGS: ProviderSettings = ProviderSettings {
    model: None,
    api_base: None,
    saved_models: Vec::new(),
};

impl ProviderSettings {
    pub fn choose_model(&mut self, model: String) {
        if !self.saved_models.contains(&model) {
            self.saved_models.push(model.clone());
        }
        self.model = Some(model);
    }

    /// Forgets the saved model `removed`; if it was active, the first saved model takes over.
    pub fn remove_model(&mut self, removed: &str) {
        self.saved_models.retain(|saved| saved != removed);
        if self.model.as_deref() == Some(removed) {
            self.model = self.saved_models.first().cloned();
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
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

    #[must_use]
    pub fn has_blank_source(&self) -> bool {
        self.source.trim().is_empty()
    }
}

/// Settings-window bounds in physical pixels.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WindowGeometry {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

/// Missing fields load as their defaults and unknown ones are ignored, so a file written by
/// another version still loads.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct Settings {
    /// The chosen microphone; `None` is the system default, and a blank id reads as `None`.
    #[serde(deserialize_with = "input_device")]
    pub input_device: Option<AudioDeviceId>,
    pub recording_shortcut: Shortcut,
    pub recording_mode: RecordingMode,
    /// Shortcut that cancels the active dictation; `None` turns it off.
    pub cancel_shortcut: Option<Shortcut>,
    /// The speech model dictation records with; an unknown or cleanup model id loads as the
    /// default.
    #[serde(deserialize_with = "speech_model")]
    pub model_id: ModelId,
    pub start_minimized: bool,
    pub theme: ThemePreference,
    pub setup: SetupProgress,
    pub cleanup_enabled: bool,
    /// Whether the local cleanup model may load on a GPU; off keeps it on the CPU.
    pub cleanup_uses_gpu: bool,
    /// The model the Local cleanup model provider runs; an unknown id loads as the default.
    #[serde(deserialize_with = "cleanup_model")]
    pub cleanup_model: ModelId,
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
            setup: SetupProgress::default(),
            cleanup_enabled: false,
            cleanup_uses_gpu: true,
            cleanup_model: default_cleanup_model(),
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

fn default_cleanup_model() -> ModelId {
    ModelId::new(DEFAULT_CLEANUP_MODEL_ID).expect("cleanup registry IDs are valid")
}

pub(crate) fn known_cleanup_model(id: &str) -> ModelId {
    CLEANUP_MODEL_IDS
        .into_iter()
        .find(|known| *known == id)
        .map_or_else(default_cleanup_model, |known| {
            ModelId::new(known).expect("cleanup registry IDs are valid")
        })
}

fn cleanup_model<'de, D: Deserializer<'de>>(deserializer: D) -> Result<ModelId, D::Error> {
    Ok(known_cleanup_model(&String::deserialize(deserializer)?))
}

pub(crate) fn known_speech_model(id: &str) -> ModelId {
    ModelRegistry::builtin()
        .speech_models()
        .find(|model| model.id.as_str() == id)
        .map_or_else(ModelId::default, |model| model.id.clone())
}

fn speech_model<'de, D: Deserializer<'de>>(deserializer: D) -> Result<ModelId, D::Error> {
    Ok(known_speech_model(&String::deserialize(deserializer)?))
}

fn input_device<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<AudioDeviceId>, D::Error> {
    Ok(Option::<String>::deserialize(deserializer)?.and_then(AudioDeviceId::new))
}

impl Settings {
    #[must_use]
    pub fn provider_settings(&self, provider: LlmProvider) -> &ProviderSettings {
        self.llm_provider_settings
            .get(&provider)
            .unwrap_or(&DEFAULT_PROVIDER_SETTINGS)
    }

    pub fn provider_settings_mut(&mut self, provider: LlmProvider) -> &mut ProviderSettings {
        self.llm_provider_settings.entry(provider).or_default()
    }

    /// Setup's model while setup is unfinished, since its test dictation records with it;
    /// otherwise the saved model.
    #[must_use]
    pub fn selected_model_id(&self) -> &ModelId {
        self.setup
            .checkpoint()
            .map_or(&self.model_id, |checkpoint| &checkpoint.model_id)
    }
}
