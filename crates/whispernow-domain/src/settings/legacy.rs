//! The Python release's `settings.json`, imported once into [`Settings`].

use std::collections::{BTreeMap, BTreeSet, btree_map::Entry};

use serde::{Deserialize, de::DeserializeOwned};
use serde_json::Value;

use super::{known_cleanup_model, known_speech_model};
use crate::validation::model_name;
use crate::{
    ApiKey, HistoryLimit, HistoryRetention, LlmProvider, ProviderEndpoint, ProviderSettings,
    RecordingMode, Settings, SetupProgress, Shortcut, ShortcutKey, ShortcutModifier,
    VocabularyReplacement, WindowGeometry,
};

/// Settings read from the Python release's `settings.json`, and the API keys it held in
/// plain text.
#[derive(Debug)]
pub struct PythonSettings {
    pub settings: Settings,
    pub api_keys: BTreeMap<LlmProvider, ApiKey>,
    pub enable_autostart: bool,
}

impl PythonSettings {
    /// A missing or unreadable value takes its default, and everything the Python release did
    /// not have is default.
    #[must_use]
    pub fn import(document: &Value) -> Self {
        let mut settings = Settings::default();
        settings.input_device = read(document, "input_device").unwrap_or(settings.input_device);
        if let Some(model_id) = read::<String>(document, "model_id") {
            settings.model_id = known_speech_model(&model_id);
        }
        settings.start_minimized =
            read(document, "start_minimized").unwrap_or(settings.start_minimized);
        if read(document, "first_run_complete").unwrap_or(false) {
            settings.setup = SetupProgress::Complete;
        }
        settings.cleanup_enabled = document
            .get("active_enhancement_id")
            .is_some_and(Value::is_string);
        settings.llm_provider = read(document, "llm_provider").unwrap_or(settings.llm_provider);
        settings.vocabulary_replacements =
            read::<Vec<(String, String)>>(document, "vocabulary_replacements")
                .unwrap_or_default()
                .into_iter()
                .map(|(source, replacement)| VocabularyReplacement::new(source, replacement))
                .collect();
        settings.window_geometry = read::<(i32, i32, u32, u32)>(document, "window_geometry").map(
            |(x, y, width, height)| WindowGeometry {
                x,
                y,
                width,
                height,
            },
        );
        if let Some(shortcut) = document
            .get("hotkey")
            .filter(|hotkey| hotkey.is_object())
            .and_then(|hotkey| StoredShortcut::deserialize(hotkey).ok())
            .map(Shortcut::from)
            .filter(Shortcut::is_recording_shortcut)
        {
            settings.recording_shortcut = shortcut;
        }
        // The Python release could run hold and toggle shortcuts side by side; hold wins
        // unless it was turned off.
        let hold_enabled = read(document, "hotkey_enabled").unwrap_or(true);
        if !hold_enabled
            && let Some(shortcut) =
                read::<Shortcut>(document, "toggle_hotkey").filter(Shortcut::is_recording_shortcut)
        {
            settings.recording_shortcut = shortcut;
            settings.recording_mode = RecordingMode::Toggle;
        }
        if let Some(max_entries) = document
            .get("text_retention")
            .and_then(|retention| retention.get("max_entries"))
        {
            settings.history_limit = match max_entries {
                Value::Null => HistoryLimit::Unlimited,
                value => value
                    .as_u64()
                    .map_or_else(HistoryLimit::default, HistoryLimit::at_least),
            };
        }
        if let Some(max_age_days) = document
            .get("text_retention")
            .and_then(|retention| retention.get("max_age_days"))
        {
            settings.history_retention = match max_age_days {
                Value::Null => HistoryRetention::Forever,
                value => value
                    .as_u64()
                    .map_or_else(HistoryRetention::default, HistoryRetention::at_least_days),
            };
        }
        if let Some(keep_audio) = document
            .get("audio_retention")
            .and_then(|retention| read(retention, "enabled"))
        {
            settings.keep_audio = keep_audio;
        }

        let api_keys = import_providers(document, &mut settings);
        Self {
            settings,
            api_keys,
            enable_autostart: document
                .get("auto_start_on_login")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        }
    }

    /// Removes every API key field except an imported key the credential store refused.
    #[must_use]
    pub fn copy_with_refused_keys(
        &self,
        document: &Value,
        refused: &BTreeSet<LlmProvider>,
    ) -> Value {
        let keep_key = |provider: Option<LlmProvider>, value: &Value| {
            provider
                .filter(|provider| refused.contains(provider))
                .and_then(|provider| self.api_keys.get(&provider))
                .is_some_and(|key| {
                    value
                        .as_str()
                        .is_some_and(|value| value.trim() == key.expose_secret())
                })
        };
        let mut copy = document.clone();
        if let Some(providers) = copy
            .get_mut("llm_provider_settings")
            .and_then(Value::as_object_mut)
        {
            for (name, entry) in providers {
                if let Some(fields) = entry.as_object_mut()
                    && !fields
                        .get("api_key")
                        .is_some_and(|value| keep_key(name.parse().ok(), value))
                {
                    fields.remove("api_key");
                }
            }
        }
        if let Some(fields) = copy.as_object_mut()
            && !fields
                .get("llm_api_key")
                .is_some_and(|value| keep_key(Some(self.settings.llm_provider), value))
        {
            fields.remove("llm_api_key");
        }
        copy
    }
}

fn import_providers(document: &Value, settings: &mut Settings) -> BTreeMap<LlmProvider, ApiKey> {
    let mut api_keys = BTreeMap::new();
    let providers = document
        .get("llm_provider_settings")
        .and_then(Value::as_object)
        .into_iter()
        .flatten();
    for (name, entry) in providers {
        let Ok(provider) = name.parse::<LlmProvider>() else {
            continue;
        };
        let model = read::<String>(entry, "model");
        if provider == LlmProvider::LocalCleanup {
            if let Some(model) = model {
                settings.cleanup_model = known_cleanup_model(&model);
            }
            continue;
        }
        settings.llm_provider_settings.insert(
            provider,
            provider_settings(
                provider,
                model,
                read(entry, "api_base"),
                &read::<Vec<String>>(entry, "saved_models").unwrap_or_default(),
            ),
        );
        if let Some(key) = text(entry, "api_key").and_then(|key| ApiKey::new(key).ok()) {
            api_keys.insert(provider, key);
        }
    }
    // The earliest release kept one provider's settings at the top level.
    let provider = settings.llm_provider;
    if provider != LlmProvider::LocalCleanup
        && let Entry::Vacant(entry) = settings.llm_provider_settings.entry(provider)
    {
        let model = text(document, "llm_model");
        let api_base = text(document, "llm_api_base");
        if model.is_some() || api_base.is_some() {
            let api_base = api_base.and_then(|api_base| ProviderEndpoint::parse(api_base).ok());
            entry.insert(provider_settings(provider, model, api_base, &[]));
        }
        if let Some(key) = text(document, "llm_api_key").and_then(|key| ApiKey::new(key).ok()) {
            api_keys.insert(provider, key);
        }
    }
    api_keys
}

/// Saves the active model with the others, every name once and without its routing prefix.
/// A name that is blank or spans lines is dropped.
fn provider_settings(
    provider: LlmProvider,
    model: Option<String>,
    api_base: Option<ProviderEndpoint>,
    saved: &[String],
) -> ProviderSettings {
    let mut saved_models = Vec::new();
    for name in saved
        .iter()
        .filter_map(|saved| provider_model(provider, saved))
    {
        if !saved_models.contains(&name) {
            saved_models.push(name);
        }
    }
    let model = model.and_then(|model| provider_model(provider, &model));
    if let Some(model) = &model
        && !saved_models.contains(model)
    {
        saved_models.insert(0, model.clone());
    }
    ProviderSettings {
        model,
        api_base,
        saved_models,
    }
}

/// Drops the `LiteLLM` routing prefix the Python release saved with a model name
/// (`LLMProcessor.format_model_name`), so `openrouter/auto` imports as `auto`.
fn provider_model(provider: LlmProvider, model: &str) -> Option<String> {
    let model = model.trim();
    let prefix = match provider {
        LlmProvider::OpenAi => "openai/",
        LlmProvider::Anthropic => "anthropic/",
        LlmProvider::Gemini => "gemini/",
        LlmProvider::Ollama => "ollama/",
        LlmProvider::OpenRouter => "openrouter/",
        LlmProvider::LocalCleanup | LlmProvider::Custom => "",
    };
    model_name(model.strip_prefix(prefix).unwrap_or(model)).map(str::to_owned)
}

fn read<T: DeserializeOwned>(document: &Value, name: &str) -> Option<T> {
    T::deserialize(document.get(name)?).ok()
}

/// A stored shortcut missing its modifiers or key takes the default's.
#[derive(Deserialize)]
#[serde(default)]
struct StoredShortcut {
    modifiers: BTreeSet<ShortcutModifier>,
    key: ShortcutKey,
}

impl Default for StoredShortcut {
    fn default() -> Self {
        let Shortcut { modifiers, key } = Shortcut::default();
        Self { modifiers, key }
    }
}

impl From<StoredShortcut> for Shortcut {
    fn from(stored: StoredShortcut) -> Self {
        Self {
            modifiers: stored.modifiers,
            key: stored.key,
        }
    }
}

fn text(document: &Value, name: &str) -> Option<String> {
    document
        .get(name)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}
