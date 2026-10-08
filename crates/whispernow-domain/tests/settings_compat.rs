use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use whispernow_domain::{
    ApiKey, HistoryLimit, LlmProvider, ModelId, ProviderEndpoint, ProviderSettings, PythonSettings,
    RecordingMode, Settings, SetupCheckpoint, SetupStep, Shortcut, ShortcutKey, ShortcutModifier,
    VocabularyReplacement,
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

    assert_eq!(settings.input_device, None);
    assert_eq!(
        settings.recording_shortcut.modifiers,
        [ShortcutModifier::Control].into()
    );
    assert_eq!(settings.recording_shortcut.key, ShortcutKey::Space);
    assert_eq!(settings.recording_mode, RecordingMode::Hold);
    assert_eq!(settings.cancel_shortcut, Some(Shortcut::default_cancel()));
    assert_eq!(
        settings.model_id.as_str(),
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
    );
    assert!(!settings.first_run_complete);
    assert_eq!(settings.setup, None);
    assert!(!settings.cleanup_enabled);
    assert_eq!(settings.llm_provider, LlmProvider::OpenAi);
    assert!(settings.llm_provider_settings.is_empty());
    assert!(settings.api_keys.is_empty());
    assert_eq!(settings.history_limit, HistoryLimit::Last20);
    assert!(!settings.keep_audio);
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
        history_limit: HistoryLimit::Last1000,
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

    let partial: Settings = parse_json(r#"{"keep_audio":true,"unknown":{"nested":true}}"#);
    assert_eq!(
        partial,
        Settings {
            keep_audio: true,
            ..Settings::default()
        }
    );
    assert_eq!(
        parse_json::<Settings>(r#"{"cancel_hotkey":null}"#).cancel_shortcut,
        None,
        "the cancel shortcut can be off"
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
    assert!(settings.start_minimized && settings.auto_start_on_login);
    assert!(settings.first_run_complete);
    assert!(
        settings.cleanup_enabled,
        "any chosen prompt turns cleanup on"
    );
    assert_eq!(settings.llm_provider, LlmProvider::Ollama);
    assert_eq!(
        settings.provider_settings(LlmProvider::Ollama),
        ProviderSettings {
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
        ProviderSettings {
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
    let hold_off_without_toggle = PythonSettings::import(&json!({
        "hotkey_enabled": false,
        "toggle_hotkey": { "modifiers": [], "key": "r" },
    }));
    assert_eq!(
        hold_off_without_toggle.settings.recording_mode,
        RecordingMode::Hold,
        "a bare typing key is refused, so hold stays"
    );

    let unlimited = PythonSettings::import(&json!({ "text_retention": { "max_entries": null } }));
    assert_eq!(unlimited.settings.history_limit, HistoryLimit::Unlimited);
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

#[test]
fn the_recording_shortcut_loads_and_saves_under_the_old_key_name() {
    let document = json!({
        "hotkey": { "modifiers": ["ctrl"], "key": "r" },
        "recording_mode": "toggle",
        "cancel_hotkey": null,
        "setup": { "step": "shortcuts", "hotkey": { "modifiers": ["alt"], "key": "t" } },
    });
    let settings: Settings = parse_json(&document.to_string());
    assert_eq!(
        settings.recording_shortcut,
        Shortcut {
            modifiers: [ShortcutModifier::Control].into(),
            key: ShortcutKey::R,
        }
    );
    assert_eq!(settings.recording_mode, RecordingMode::Toggle);
    assert_eq!(settings.cancel_shortcut, None);
    let setup = settings.setup.clone().expect("setup checkpoint");
    assert_eq!(setup.step, SetupStep::Shortcuts);
    assert_eq!(setup.recording_shortcut.key, ShortcutKey::T);

    let saved = serde_json::to_value(&settings).expect("serialize settings");
    assert!(saved.get("hotkey").is_some() && saved.get("cancel_hotkey").is_some());
    assert!(saved["setup"].get("hotkey").is_some());
}
