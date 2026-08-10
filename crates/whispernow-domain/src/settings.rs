//! User settings.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};

use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use thiserror::Error;

use crate::{
    CLEANUP_MODEL_IDS, CleanupPrompt, DEFAULT_CLEANUP_MODEL_ID, HistoryLimit, LlmProvider, ModelId,
    ProviderEndpoint, Shortcut, default_prompts, prompts::upgrade_unedited_defaults,
};

mod legacy;

pub use legacy::PythonSettings;

/// Default total time allowed for AI cleanup before the deterministic
/// transcript is used instead.
pub const DEFAULT_LLM_TIMEOUT_SECONDS: u32 = 20;

/// Accepted range for the configurable AI cleanup deadline.
pub const LLM_TIMEOUT_SECONDS_RANGE: std::ops::RangeInclusive<u32> = 5..=120;

/// First-run step. An unfinished setup keeps its step and choices in
/// [`Settings::setup`] so an interrupted wizard resumes without claiming setup is complete.
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

/// Colour theme applied to the desktop windows.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ThemePreference {
    /// Follow the operating system's light or dark appearance.
    #[default]
    System,
    Light,
    Dark,
}

/// How much the recording overlay shows while dictating.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OverlayStyle {
    /// Never show the overlay.
    Off,
    /// Show phases, level bars and outcomes without transcript text.
    Minimal,
    /// Also show the committed partial transcript while listening.
    #[default]
    WithLiveText,
}

/// Screen edge the recording overlay is placed against.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OverlayPosition {
    Top,
    #[default]
    Bottom,
}

/// Non-secret, resumable first-run draft. Download operation identifiers are
/// intentionally excluded because they are process-scoped. The saved settings stay in
/// effect until setup finishes and commits these choices.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct SetupCheckpoint {
    pub step: SetupStep,
    pub model_id: ModelId,
    pub input_device: Option<String>,
    #[serde(rename = "hotkey")]
    pub hold_shortcut: Shortcut,
    /// Whether the hold-to-record shortcut is enabled.
    #[serde(rename = "hotkey_enabled")]
    pub hold_shortcut_enabled: bool,
    /// Optional press-once/press-again recording shortcut.
    #[serde(rename = "toggle_hotkey")]
    pub toggle_shortcut: Option<Shortcut>,
}

impl Default for SetupCheckpoint {
    fn default() -> Self {
        Self {
            step: SetupStep::Welcome,
            model_id: ModelId::default(),
            input_device: None,
            hold_shortcut: Shortcut::default(),
            hold_shortcut_enabled: true,
            toggle_shortcut: None,
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

/// A typed API key that is empty or spans more than one line.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
#[error("Enter the API key on one line")]
pub struct InvalidApiKey;

impl ApiKey {
    /// The key typed as `value`, trimmed.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidApiKey`] when the trimmed key is empty or holds a control character.
    pub fn new(value: impl Into<String>) -> Result<Self, InvalidApiKey> {
        let value = value.into();
        let key = value.trim();
        if key.is_empty() || key.chars().any(char::is_control) {
            return Err(InvalidApiKey);
        }
        Ok(Self(key.to_owned()))
    }

    /// Deliberately exposes the key for authentication at the provider boundary.
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

/// Provider-specific model and endpoint settings. The API key is kept in the OS
/// credential store, never here.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct ProviderSettings {
    /// Provider model name.
    pub model: String,
    /// Optional custom provider endpoint.
    pub api_base: Option<ProviderEndpoint>,
    /// Provider model names saved by the user.
    pub saved_models: Vec<String>,
}

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

/// Persisted application settings. Fields missing from a saved document take their defaults.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct Settings {
    /// Selected input device name, or the host default.
    pub input_device: Option<String>,
    /// The hold-to-record shortcut.
    #[serde(rename = "hotkey")]
    pub hold_shortcut: Shortcut,
    /// Whether the hold-to-record shortcut is enabled.
    #[serde(rename = "hotkey_enabled")]
    pub hold_shortcut_enabled: bool,
    /// Optional press-once/press-again recording shortcut.
    #[serde(rename = "toggle_hotkey")]
    pub toggle_shortcut: Option<Shortcut>,
    /// Shortcut that cancels the active dictation; `None` turns it off. Documents from
    /// before this setting get the Escape default.
    #[serde(rename = "cancel_hotkey")]
    pub cancel_shortcut: Option<Shortcut>,
    /// Selected speech model.
    pub model_id: ModelId,
    /// Whether the desktop window starts hidden.
    pub start_minimized: bool,
    /// Whether the OS should start `WhisperNow` on login.
    pub auto_start_on_login: bool,
    /// Colour theme. Absent from documents before this setting.
    pub theme: ThemePreference,
    /// Recording overlay detail. Absent from documents before this setting.
    pub overlay_style: OverlayStyle,
    /// Recording overlay screen edge. Absent from documents before this setting.
    pub overlay_position: OverlayPosition,
    /// Whether first-run setup has completed.
    pub first_run_complete: bool,
    /// The unfinished first-run setup's step and choices.
    pub setup: Option<SetupCheckpoint>,
    /// The AI cleanup prompts. Unedited built-in prompts load as their current text, and an
    /// empty list loads as the built-in prompts.
    #[serde(rename = "enhancements", deserialize_with = "upgraded_prompts")]
    pub prompts: Vec<CleanupPrompt>,
    /// The prompt AI cleanup uses; `None` turns AI cleanup off.
    #[serde(rename = "active_enhancement_id")]
    pub active_prompt_id: Option<String>,
    /// Selected LLM provider.
    pub llm_provider: LlmProvider,
    /// Settings isolated by provider.
    pub llm_provider_settings: BTreeMap<LlmProvider, ProviderSettings>,
    /// Providers whose API key is saved in the OS credential store.
    pub api_keys: BTreeSet<LlmProvider>,
    /// Total seconds AI cleanup may take before the deterministic
    /// transcript is used instead. Absent from documents before this setting.
    pub llm_timeout_seconds: u32,
    /// Ordered literal replacements applied to transcripts.
    pub vocabulary_replacements: Vec<VocabularyReplacement>,
    /// The settings window's last bounds.
    pub window_geometry: Option<WindowGeometry>,
    /// How many history entries to keep.
    pub history_limit: HistoryLimit,
    /// Whether successful dictations keep their audio. Failed and cancelled ones always do.
    pub keep_audio: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            input_device: None,
            hold_shortcut: Shortcut::default(),
            hold_shortcut_enabled: true,
            toggle_shortcut: None,
            cancel_shortcut: Some(Shortcut::default_cancel()),
            model_id: ModelId::default(),
            start_minimized: false,
            auto_start_on_login: false,
            theme: ThemePreference::System,
            overlay_style: OverlayStyle::default(),
            overlay_position: OverlayPosition::default(),
            first_run_complete: false,
            setup: None,
            prompts: default_prompts(),
            active_prompt_id: None,
            llm_provider: LlmProvider::OpenAi,
            llm_provider_settings: BTreeMap::new(),
            api_keys: BTreeSet::new(),
            llm_timeout_seconds: DEFAULT_LLM_TIMEOUT_SECONDS,
            vocabulary_replacements: Vec::new(),
            window_geometry: None,
            history_limit: HistoryLimit::default(),
            keep_audio: false,
        }
    }
}

impl Settings {
    /// The local cleanup model AI cleanup uses with the Local cleanup model provider: the one
    /// saved as that provider's model, or the default when none, or an unknown one, is saved.
    #[must_use]
    pub fn cleanup_model_id(&self) -> &'static str {
        self.llm_provider_settings
            .get(&LlmProvider::LocalCleanup)
            .and_then(|saved| CLEANUP_MODEL_IDS.into_iter().find(|id| *id == saved.model))
            .unwrap_or(DEFAULT_CLEANUP_MODEL_ID)
    }

    /// Returns settings for a provider, or provider defaults when absent.
    #[must_use]
    pub fn provider_settings(&self, provider: LlmProvider) -> ProviderSettings {
        self.llm_provider_settings
            .get(&provider)
            .cloned()
            .unwrap_or_default()
    }

    /// The settings of `provider` to edit, created with defaults when absent.
    pub fn provider_settings_mut(&mut self, provider: LlmProvider) -> &mut ProviderSettings {
        self.llm_provider_settings.entry(provider).or_default()
    }
}

fn upgraded_prompts<'de, D>(deserializer: D) -> Result<Vec<CleanupPrompt>, D::Error>
where
    D: Deserializer<'de>,
{
    let mut prompts = Vec::<CleanupPrompt>::deserialize(deserializer)?;
    if prompts.is_empty() {
        return Ok(default_prompts());
    }
    upgrade_unedited_defaults(&mut prompts);
    Ok(prompts)
}
