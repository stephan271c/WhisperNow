//! The Python release's `settings.json`, imported once into [`Settings`].

use std::collections::{BTreeMap, BTreeSet, btree_map::Entry};

use serde::de::DeserializeOwned;
use serde_json::Value;

use super::known_cleanup_model;
use crate::{
    ApiKey, HistoryLimit, HistoryRetention, LlmProvider, ModelId, ModelRegistry, ProviderEndpoint,
    ProviderSettings, RecordingMode, Settings, SetupProgress, Shortcut, VocabularyReplacement,
    WindowGeometry,
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
        pick(document, "input_device", &mut settings.input_device);
        pick(document, "model_id", &mut settings.model_id);
        if !ModelRegistry::builtin()
            .speech_models()
            .any(|model| model.id == settings.model_id)
        {
            settings.model_id = ModelId::default();
        }
        pick(document, "start_minimized", &mut settings.start_minimized);
        let mut first_run_complete = false;
        pick(document, "first_run_complete", &mut first_run_complete);
        if first_run_complete {
            settings.setup = SetupProgress::Complete;
        }
        settings.cleanup_enabled = document
            .get("active_enhancement_id")
            .is_some_and(Value::is_string);
        pick(document, "llm_provider", &mut settings.llm_provider);
        let mut vocabulary = Vec::<(String, String)>::new();
        pick(document, "vocabulary_replacements", &mut vocabulary);
        settings.vocabulary_replacements = vocabulary
            .into_iter()
            .map(|(source, replacement)| VocabularyReplacement::new(source, replacement))
            .collect();
        let mut geometry = None::<(i32, i32, u32, u32)>;
        pick(document, "window_geometry", &mut geometry);
        settings.window_geometry = geometry.map(|(x, y, width, height)| WindowGeometry {
            x,
            y,
            width,
            height,
        });
        // A stored shortcut missing its modifiers or key takes the default's.
        if let Some(Value::Object(stored)) = document.get("hotkey") {
            let mut shortcut_fields = serde_json::to_value(&settings.recording_shortcut)
                .ok()
                .and_then(|value| value.as_object().cloned())
                .unwrap_or_default();
            shortcut_fields.extend(stored.clone());
            if let Ok(shortcut) = serde_json::from_value::<Shortcut>(Value::Object(shortcut_fields))
                && shortcut.is_recording_shortcut()
            {
                settings.recording_shortcut = shortcut;
            }
        }
        // The Python release could run hold and toggle shortcuts side by side; hold wins
        // unless it was turned off.
        let mut hold_enabled = true;
        pick(document, "hotkey_enabled", &mut hold_enabled);
        let mut toggle_shortcut = None::<Shortcut>;
        pick(document, "toggle_hotkey", &mut toggle_shortcut);
        if !hold_enabled
            && let Some(shortcut) = toggle_shortcut.filter(Shortcut::is_recording_shortcut)
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
        pick(
            document.get("audio_retention").unwrap_or(&Value::Null),
            "enabled",
            &mut settings.keep_audio,
        );

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
                        .is_some_and(|value| keep_key(LlmProvider::from_name(name), value))
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
        let Some(provider) = LlmProvider::from_name(name) else {
            continue;
        };
        let mut model = None::<String>;
        pick(entry, "model", &mut model);
        if provider == LlmProvider::LocalCleanup {
            if let Some(model) = model {
                settings.cleanup_model = known_cleanup_model(&model);
            }
            continue;
        }
        let mut api_base = None;
        let mut saved_models = Vec::new();
        pick(entry, "api_base", &mut api_base);
        pick(entry, "saved_models", &mut saved_models);
        settings.llm_provider_settings.insert(
            provider,
            provider_settings(provider, model, api_base, saved_models),
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
            entry.insert(provider_settings(provider, model, api_base, Vec::new()));
        }
        if let Some(key) = text(document, "llm_api_key").and_then(|key| ApiKey::new(key).ok()) {
            api_keys.insert(provider, key);
        }
    }
    api_keys
}

/// Saves the active model with the others, every name without its routing prefix.
fn provider_settings(
    provider: LlmProvider,
    model: Option<String>,
    api_base: Option<ProviderEndpoint>,
    saved_models: Vec<String>,
) -> ProviderSettings {
    let mut saved_models: Vec<String> = saved_models
        .into_iter()
        .map(|saved| provider_model(provider, &saved))
        .collect();
    let model = model
        .map(|model| provider_model(provider, &model))
        .filter(|model| !model.is_empty());
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
fn provider_model(provider: LlmProvider, model: &str) -> String {
    let model = model.trim();
    let prefix = match provider {
        LlmProvider::OpenAi => "openai/",
        LlmProvider::Anthropic => "anthropic/",
        LlmProvider::Gemini => "gemini/",
        LlmProvider::Ollama => "ollama/",
        LlmProvider::OpenRouter => "openrouter/",
        LlmProvider::LocalCleanup | LlmProvider::Custom => return model.to_owned(),
    };
    model.strip_prefix(prefix).unwrap_or(model).to_owned()
}

fn pick<T: DeserializeOwned>(document: &Value, name: &str, target: &mut T) {
    if let Some(value) = document
        .get(name)
        .and_then(|value| serde_json::from_value(value.clone()).ok())
    {
        *target = value;
    }
}

fn text(document: &Value, name: &str) -> Option<String> {
    document
        .get(name)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}
