use std::collections::BTreeSet;

use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use whispernow_domain::{
    ApiKey, HistoryLimit, HistoryRetention, LlmProvider, ModelId, ProviderEndpoint,
    ProviderSettings, PythonSettings, RecordingMode, Settings, SetupCheckpoint, SetupStep,
    Shortcut, ShortcutKey, ShortcutModifier, VocabularyReplacement,
};

fn parse_json<T: DeserializeOwned>(document: &str) -> T {
    match serde_json::from_str(document) {
        Ok(value) => value,
        Err(error) => panic!("test fixture must be valid JSON: {error}"),
    }
}

#[test]
fn fresh_settings_take_the_defaults() {
    let settings = Settings::default();

    assert_eq!(
        settings.recording_shortcut.modifiers,
        [ShortcutModifier::Control].into()
    );
    assert_eq!(settings.recording_shortcut.key, ShortcutKey::Space);
    assert_eq!(settings.recording_mode, RecordingMode::Hold);
    assert_eq!(
        settings.model_id.as_str(),
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
    );
}

#[test]
fn settings_round_trip_and_missing_fields_take_defaults() {
    let mut settings = Settings {
        recording_mode: RecordingMode::Toggle,
        recording_shortcut: Shortcut {
            modifiers: [ShortcutModifier::Shift, ShortcutModifier::Meta].into(),
            key: ShortcutKey::F8,
        },
        cleanup_enabled: true,
        cleanup_uses_gpu: false,
        history_limit: HistoryLimit::Last1000,
        history_retention: HistoryRetention::Days30,
        keep_audio: true,
        setup: Some(SetupCheckpoint {
            step: SetupStep::TryIt,
            ..SetupCheckpoint::default()
        }),
        api_keys: [LlmProvider::Gemini].into(),
        ..Settings::default()
    };
    *settings.provider_settings_mut(LlmProvider::Gemini) = ProviderSettings {
        model: "gemini-test".to_owned(),
        api_base: None,
        saved_models: vec!["gemini-test".to_owned()],
    };
    let document = serde_json::to_string(&settings).expect("serialize settings");
    assert_eq!(parse_json::<Settings>(&document), settings);
    let saved: Value = parse_json(&document);
    assert_eq!(saved["history_retention"], "30");
    assert!(saved.get("recording_shortcut").is_some());
    assert!(saved.get("cancel_shortcut").is_some());
    assert!(saved["setup"].get("recording_shortcut").is_some());
    for key in ["hotkey", "cancel_hotkey", "auto_start_on_login"] {
        assert!(saved.get(key).is_none());
        assert!(saved["setup"].get(key).is_none());
    }

    let partial: Settings = parse_json(r#"{"keep_audio":true,"unknown":{"nested":true}}"#);
    assert_eq!(
        partial,
        Settings {
            keep_audio: true,
            ..Settings::default()
        }
    );
    assert!(serde_json::from_str::<Settings>(r#"{"keep_audio":"fast"}"#).is_err());
}

#[test]
fn python_settings_import_picks_known_fields() {
    let document: Value = parse_json(include_str!("fixtures/current_settings.json"));
    let imported = PythonSettings::import(&document);
    let settings = imported.settings;
    assert!(imported.api_keys.is_empty());
    assert_eq!(settings.input_device.as_deref(), Some("Studio Microphone"));
    assert_eq!(
        settings.recording_shortcut,
        Shortcut {
            modifiers: [ShortcutModifier::Control, ShortcutModifier::Shift].into(),
            key: ShortcutKey::R,
        }
    );
    assert_eq!(settings.recording_mode, RecordingMode::Hold);
    // The fixture's Whisper Small is no longer offered.
    assert_eq!(settings.model_id, ModelId::default());
    assert!(settings.start_minimized && imported.enable_autostart);
    assert!(settings.first_run_complete);
    assert!(
        settings.cleanup_enabled,
        "any chosen prompt turns cleanup on"
    );
    assert_eq!(settings.llm_provider, LlmProvider::Ollama);
    assert_eq!(
        settings.provider_settings(LlmProvider::Ollama),
        &ProviderSettings {
            model: "llama3.2".to_owned(),
            api_base: ProviderEndpoint::parse("http://localhost:11434").ok(),
            saved_models: vec!["llama3.2".to_owned(), "qwen2.5".to_owned()],
        }
    );
    assert_eq!(
        settings.vocabulary_replacements,
        [
            VocabularyReplacement::new("whisper now", "WhisperNow"),
            VocabularyReplacement::new("rust lang", "Rust")
        ]
    );
    assert_eq!(
        settings.window_geometry,
        Some(whispernow_domain::WindowGeometry {
            x: 120,
            y: 80,
            width: 1024,
            height: 720,
        })
    );
    assert_eq!(
        (settings.cancel_shortcut, settings.history_limit),
        (Some(Shortcut::default_cancel()), HistoryLimit::Last20),
        "what the Python release did not have is default"
    );
}

#[test]
fn python_model_names_lose_their_litellm_routing_prefix() {
    let imported = PythonSettings::import(&json!({
        "llm_provider": "openai",
        "llm_provider_settings": {
            "openai": { "model": "openai/gpt-4o", "saved_models": ["openai/gpt-4o", "gpt-5"] },
            "openrouter": { "model": "openrouter/auto", "saved_models": ["openai/gpt-5.2"] }
        }
    }));
    let settings = imported.settings;
    assert_eq!(
        settings.provider_settings(LlmProvider::OpenAi),
        &ProviderSettings {
            model: "gpt-4o".to_owned(),
            api_base: None,
            saved_models: vec!["gpt-4o".to_owned(), "gpt-5".to_owned()],
        }
    );
    assert_eq!(
        settings.provider_settings(LlmProvider::OpenRouter),
        &ProviderSettings {
            model: "auto".to_owned(),
            api_base: None,
            saved_models: vec!["openai/gpt-5.2".to_owned()],
        },
        "only the provider's own routing prefix goes"
    );
}

#[test]
fn python_blank_microphones_normalize_to_the_system_default() {
    for input_device in ["", " ", "\t\n"] {
        let imported = PythonSettings::import(&json!({ "input_device": input_device }));
        assert_eq!(imported.settings.input_device, None);
    }
}

#[test]
fn python_settings_import_moves_keys_out_and_reads_the_older_layouts() {
    let secret = ["test", "-only", "-key"].concat();
    let imported = PythonSettings::import(&json!({
        "hotkey_enabled": "unreadable",
        "hotkey": { "modifiers": ["alt"] },
        "toggle_hotkey": { "modifiers": [], "key": "r" },
        "llm_provider": "openrouter",
        "llm_provider_settings": {
            "openai": { "model": "gpt-test", "api_key": secret },
            "future-provider": { "model": "unknown" }
        },
        "llm_model": "anthropic/claude-3-sonnet",
        "llm_api_base": "https://openrouter.ai/api/v1",
        "llm_api_key": "flat-key",
        "text_retention": { "max_entries": 50, "max_age_days": 7 },
        "audio_retention": { "enabled": true, "max_recordings": 5 }
    }));
    let settings = &imported.settings;
    assert_eq!(
        settings.recording_mode,
        RecordingMode::Hold,
        "an unreadable value is default"
    );
    assert_eq!(
        settings.recording_shortcut.modifiers,
        [ShortcutModifier::Alt].into()
    );
    assert_eq!(
        settings.recording_shortcut.key,
        ShortcutKey::Space,
        "a missing key takes the default's"
    );
    assert!(!settings.cleanup_enabled);
    assert_eq!(
        settings.provider_settings(LlmProvider::OpenAi).model,
        "gpt-test"
    );
    assert_eq!(
        settings.provider_settings(LlmProvider::OpenRouter),
        &ProviderSettings {
            model: "anthropic/claude-3-sonnet".to_owned(),
            api_base: ProviderEndpoint::parse("https://openrouter.ai/api/v1").ok(),
            saved_models: Vec::new(),
        }
    );
    assert_eq!(settings.llm_provider_settings.len(), 2);
    assert_eq!(settings.history_limit, HistoryLimit::Last100);
    assert!(settings.keep_audio);
    assert!(settings.api_keys.is_empty(), "the store records the keys");
    assert_eq!(
        imported.api_keys,
        [
            (
                LlmProvider::OpenAi,
                ApiKey::new(secret.clone()).expect("a key")
            ),
            (
                LlmProvider::OpenRouter,
                ApiKey::new("flat-key").expect("a key")
            ),
        ]
        .into()
    );
    assert!(!format!("{imported:?}").contains(&secret));

    let toggle_only = PythonSettings::import(&json!({
        "hotkey_enabled": false,
        "toggle_hotkey": { "modifiers": ["alt"], "key": "r" },
    }));
    assert_eq!(
        (
            toggle_only.settings.recording_mode,
            toggle_only.settings.recording_shortcut
        ),
        (
            RecordingMode::Toggle,
            Shortcut {
                modifiers: [ShortcutModifier::Alt].into(),
                key: ShortcutKey::R,
            }
        )
    );
    let invalid_toggle = PythonSettings::import(&json!({
        "hotkey_enabled": false,
        "toggle_hotkey": { "modifiers": [], "key": "r" },
    }));
    assert_eq!(
        invalid_toggle.settings.recording_mode,
        RecordingMode::Hold,
        "a bare typing key is refused, so hold stays"
    );

    let unlimited = PythonSettings::import(&json!({ "text_retention": { "max_entries": null } }));
    assert_eq!(unlimited.settings.history_limit, HistoryLimit::Unlimited);
}

#[test]
fn python_max_age_days_imports_as_the_choice_that_keeps_it() {
    let week = PythonSettings::import(&json!({
        "text_retention": { "max_age_days": 7 },
    }));
    assert_eq!(week.settings.history_retention, HistoryRetention::Days7);
    let no_age_limit = PythonSettings::import(&json!({
        "text_retention": { "max_age_days": null },
    }));
    assert_eq!(
        no_age_limit.settings.history_retention,
        HistoryRetention::Forever
    );
    let ancient = PythonSettings::import(&json!({
        "text_retention": { "max_age_days": 200 },
    }));
    assert_eq!(
        ancient.settings.history_retention,
        HistoryRetention::Forever
    );
}

#[test]
fn python_settings_copy_keeps_only_refused_keys_in_both_layouts() {
    let document = json!({
        "llm_provider": "openrouter",
        "llm_api_key": " flat-key ",
        "llm_model": "legacy-model",
        "llm_provider_settings": {
            "openai": { "api_key": "accepted-key", "model": "gpt-test" },
            "gemini": { "api_key": "refused-key", "model": "gemini-test" },
            "anthropic": { "api_key": "invalid\nkey", "model": "claude-test" },
            "future-provider": { "api_key": "unknown-key", "model": "future-model" }
        },
        "unknown": { "values": [true, null, 42] }
    });
    let imported = PythonSettings::import(&document);
    let mut expected = json!({
        "llm_provider": "openrouter",
        "llm_model": "legacy-model",
        "llm_provider_settings": {
            "openai": { "model": "gpt-test" },
            "gemini": { "model": "gemini-test" },
            "anthropic": { "model": "claude-test" },
            "future-provider": { "model": "future-model" }
        },
        "unknown": { "values": [true, null, 42] }
    });
    assert_eq!(
        imported.copy_with_refused_keys(&document, &BTreeSet::new()),
        expected
    );

    expected["llm_api_key"] = document["llm_api_key"].clone();
    expected["llm_provider_settings"]["gemini"]["api_key"] =
        document["llm_provider_settings"]["gemini"]["api_key"].clone();
    assert_eq!(
        imported.copy_with_refused_keys(
            &document,
            &[LlmProvider::Gemini, LlmProvider::OpenRouter].into()
        ),
        expected
    );
}

#[test]
fn saved_shortcuts_keep_their_key_names() {
    let shortcut =
        parse_json::<Shortcut>(r#"{"modifiers":["shift","ctrl","ctrl"],"key":" Space "}"#);
    assert_eq!(
        shortcut,
        Shortcut {
            modifiers: [ShortcutModifier::Control, ShortcutModifier::Shift].into(),
            key: ShortcutKey::Space,
        }
    );
    assert_eq!(
        serde_json::to_value(&shortcut).expect("serialize"),
        serde_json::json!({"modifiers": ["ctrl", "shift"], "key": "space"})
    );
    assert!(serde_json::from_str::<Shortcut>(r#"{"key":"f25"}"#).is_err());
}
