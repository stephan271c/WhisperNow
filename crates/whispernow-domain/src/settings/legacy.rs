//! The Python release's `settings.json`, imported once into [`Settings`].

use std::collections::{BTreeMap, btree_map::Entry};

use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::{
    ApiKey, HistoryLimit, LlmProvider, ModelId, ModelRegistry, ProviderEndpoint, ProviderSettings,
    RecordingMode, Settings, Shortcut,
};

/// Settings read from the Python release's `settings.json`, and the API keys it held in
/// plain text.
#[derive(Debug)]
pub struct PythonSettings {
    pub settings: Settings,
    pub api_keys: BTreeMap<LlmProvider, ApiKey>,
}

impl PythonSettings {
    /// A missing or unreadable value takes its default, and everything the Python release did
    /// not have is default.
    #[must_use]
    pub fn import(document: &Value) -> Self {
        let mut settings = Settings::default();
        pick(document, "input_device", &mut settings.input_device);
        if settings
            .input_device
            .as_ref()
            .is_some_and(|device| device.trim().is_empty())
        {
            settings.input_device = None;
        }
        pick(document, "model_id", &mut settings.model_id);
        if !ModelRegistry::builtin()
            .speech_models()
            .any(|model| model.id == settings.model_id)
        {
            settings.model_id = ModelId::default();
        }
        pick(document, "start_minimized", &mut settings.start_minimized);
        pick(
            document,
            "auto_start_on_login",
            &mut settings.auto_start_on_login,
        );
        pick(
            document,
            "first_run_complete",
            &mut settings.first_run_complete,
        );
        settings.cleanup_enabled = document
            .get("active_enhancement_id")
            .is_some_and(Value::is_string);
        pick(document, "llm_provider", &mut settings.llm_provider);
        pick(
            document,
            "vocabulary_replacements",
            &mut settings.vocabulary_replacements,
        );
        pick(document, "window_geometry", &mut settings.window_geometry);
        // A stored shortcut missing its modifiers or key takes the default's.
        if let Some(Value::Object(stored)) = document.get("hotkey") {
            let mut shortcut = serde_json::to_value(&settings.recording_shortcut)
                .ok()
                .and_then(|value| value.as_object().cloned())
                .unwrap_or_default();
            shortcut.extend(stored.clone());
            if let Ok(shortcut) = serde_json::from_value::<Shortcut>(Value::Object(shortcut))
                && shortcut.is_recording_shortcut()
            {
                settings.recording_shortcut = shortcut;
            }
        }
        // The Python release could run hold and toggle shortcuts side by side; hold wins
        // unless it was turned off.
        let mut hold_enabled = true;
        pick(document, "hotkey_enabled", &mut hold_enabled);
        let mut toggle = None::<Shortcut>;
        pick(document, "toggle_hotkey", &mut toggle);
        if !hold_enabled && let Some(toggle) = toggle.filter(Shortcut::is_recording_shortcut) {
            settings.recording_shortcut = toggle;
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
                    .map_or_else(HistoryLimit::default, |entries| {
                        HistoryLimit::at_least(Some(u32::try_from(entries).unwrap_or(u32::MAX)))
                    }),
            };
        }
        pick(
            document.get("audio_retention").unwrap_or(&Value::Null),
            "enabled",
            &mut settings.keep_audio,
        );

        let api_keys = import_providers(document, &mut settings);
        Self { settings, api_keys }
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
        let provider_settings = settings.provider_settings_mut(provider);
        pick(entry, "model", &mut provider_settings.model);
        pick(entry, "api_base", &mut provider_settings.api_base);
        pick(entry, "saved_models", &mut provider_settings.saved_models);
        if let Some(key) = text(entry, "api_key").and_then(|key| ApiKey::new(key).ok()) {
            api_keys.insert(provider, key);
        }
    }
    // The earliest release kept one provider's settings at the top level.
    let provider = settings.llm_provider;
    if let Entry::Vacant(entry) = settings.llm_provider_settings.entry(provider) {
        let model = text(document, "llm_model");
        let api_base = text(document, "llm_api_base");
        if model.is_some() || api_base.is_some() {
            entry.insert(ProviderSettings {
                model: model.unwrap_or_default(),
                api_base: api_base.and_then(|api_base| ProviderEndpoint::parse(api_base).ok()),
                saved_models: Vec::new(),
            });
        }
        if let Some(key) = text(document, "llm_api_key").and_then(|key| ApiKey::new(key).ok()) {
            api_keys.insert(provider, key);
        }
    }
    api_keys
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
