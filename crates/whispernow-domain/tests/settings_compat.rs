use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use whispernow_domain::{
    ApiKey, CleanupPrompt, DEFAULT_LLM_TIMEOUT_SECONDS, HistoryLimit, LlmProvider,
    ProviderEndpoint, ProviderSettings, PythonSettings, Settings, SetupCheckpoint, SetupStep,
    Shortcut, ShortcutKey, ShortcutModifier, VocabularyReplacement, default_prompts,
};

fn parse_json<T: DeserializeOwned>(document: &str) -> T {
    match serde_json::from_str(document) {
        Ok(value) => value,
        Err(error) => panic!("test fixture must be valid JSON: {error}"),
    }
}

#[test]
fn rust_defaults_prefer_native_48khz_capture() {
    let settings = Settings::default();

    assert_eq!(settings.input_device, None);
    assert_eq!(
        settings.hold_shortcut.modifiers,
        [ShortcutModifier::Control].into()
    );
    assert_eq!(settings.hold_shortcut.key, ShortcutKey::Space);
    assert!(settings.hold_shortcut_enabled);
    assert_eq!(settings.toggle_shortcut, None);
    assert_eq!(settings.cancel_shortcut, Some(Shortcut::default_cancel()));
    assert_eq!(
        settings.model_id.as_str(),
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16"
    );
    assert!(!settings.first_run_complete);
    assert_eq!(settings.setup, None);
    assert_eq!(settings.prompts, default_prompts());
    assert_eq!(settings.llm_timeout_seconds, DEFAULT_LLM_TIMEOUT_SECONDS);
    assert_eq!(settings.llm_provider, LlmProvider::OpenAi);
    assert!(settings.llm_provider_settings.is_empty());
    assert!(settings.api_keys.is_empty());
    assert_eq!(settings.history_limit, HistoryLimit::Last20);
    assert!(!settings.keep_audio);
}

#[test]
fn settings_round_trip_and_missing_fields_take_defaults() {
    let mut settings = Settings {
        hold_shortcut_enabled: false,
        toggle_shortcut: Some(Shortcut {
            modifiers: [ShortcutModifier::Shift, ShortcutModifier::Meta].into(),
            key: ShortcutKey::F8,
        }),
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
fn unedited_shipped_prompts_upgrade_and_edited_ones_report_the_newer_default() {
    let shipped: Vec<CleanupPrompt> = parse_json(include_str!("fixtures/default_prompts.json"));
    let loaded: Settings = parse_json(&json!({ "enhancements": shipped }).to_string());
    // The retired "concise" prompt is no longer built in, so it stays as the user's own.
    assert_eq!(loaded.prompts[..2], default_prompts()[..]);
    assert_eq!(loaded.prompts[2], shipped[2]);
    assert!(
        loaded
            .prompts
            .iter()
            .all(|prompt| prompt.newer_default().is_none())
    );

    let mut edited = shipped;
    edited[0].prompt.push_str(" Keep British spelling.");
    let expected_text = edited[0].prompt.clone();
    let loaded: Settings = parse_json(&json!({ "enhancements": edited }).to_string());
    let fix_grammar = &loaded.prompts[0];
    assert_eq!(fix_grammar.prompt, expected_text);
    assert_eq!(
        fix_grammar.newer_default(),
        Some(default_prompts()[0].prompt.as_str())
    );

    let emptied: Settings = parse_json(r#"{"enhancements":[]}"#);
    assert_eq!(emptied.prompts, default_prompts());
}

#[test]
fn python_settings_import_picks_known_fields() {
    let document: Value = parse_json(include_str!("fixtures/current_settings.json"));
    let imported = PythonSettings::import(&document);
    let settings = imported.settings;
    assert!(imported.api_keys.is_empty());
    assert_eq!(settings.input_device.as_deref(), Some("Studio Microphone"));
    assert_eq!(
        settings.hold_shortcut,
        Shortcut {
            modifiers: [ShortcutModifier::Control, ShortcutModifier::Shift].into(),
            key: ShortcutKey::R,
        }
    );
    assert_eq!(settings.model_id.as_str(), "sherpa-onnx-whisper-small");
    assert!(settings.start_minimized && settings.auto_start_on_login);
    assert!(settings.first_run_complete);
    assert_eq!(settings.prompts.len(), 1);
    assert_eq!(settings.active_prompt_id.as_deref(), Some("notes"));
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
    assert!(
        settings.hold_shortcut_enabled,
        "an unreadable value is default"
    );
    assert_eq!(
        settings.hold_shortcut.modifiers,
        [ShortcutModifier::Alt].into()
    );
    assert_eq!(
        settings.hold_shortcut.key,
        ShortcutKey::Space,
        "a missing key takes the default's"
    );
    assert_eq!(
        settings.toggle_shortcut, None,
        "a bare typing key is refused"
    );
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
fn settings_saved_under_the_old_key_names_load_and_save_under_them() {
    let ctrl_r = json!({ "modifiers": ["ctrl"], "key": "r" });
    let alt_t = json!({ "modifiers": ["alt"], "key": "t" });
    let document = json!({
        "hotkey": ctrl_r,
        "hotkey_enabled": false,
        "toggle_hotkey": alt_t,
        "cancel_hotkey": null,
        "enhancements": [{ "id": "notes", "title": "Notes", "prompt": "Make notes." }],
        "active_enhancement_id": "notes",
        "setup": {
            "step": "shortcuts",
            "hotkey": alt_t,
            "hotkey_enabled": false,
            "toggle_hotkey": ctrl_r,
        },
    });
    let settings: Settings = parse_json(&document.to_string());
    let ctrl_r = Shortcut {
        modifiers: [ShortcutModifier::Control].into(),
        key: ShortcutKey::R,
    };
    let alt_t = Shortcut {
        modifiers: [ShortcutModifier::Alt].into(),
        key: ShortcutKey::T,
    };
    assert_eq!(settings.hold_shortcut, ctrl_r);
    assert!(!settings.hold_shortcut_enabled);
    assert_eq!(settings.toggle_shortcut, Some(alt_t.clone()));
    assert_eq!(settings.cancel_shortcut, None);
    assert_eq!(settings.prompts.len(), 1);
    assert_eq!(settings.prompts[0].prompt, "Make notes.");
    assert_eq!(settings.active_prompt_id.as_deref(), Some("notes"));
    let setup = settings.setup.clone().expect("setup checkpoint");
    assert_eq!(setup.step, SetupStep::Shortcuts);
    assert_eq!(setup.hold_shortcut, alt_t);
    assert!(!setup.hold_shortcut_enabled);
    assert_eq!(setup.toggle_shortcut, Some(ctrl_r));

    let saved = serde_json::to_value(&settings).expect("serialize settings");
    for key in [
        "hotkey",
        "hotkey_enabled",
        "toggle_hotkey",
        "cancel_hotkey",
        "enhancements",
        "active_enhancement_id",
    ] {
        assert!(saved.get(key).is_some(), "saved under {key}");
    }
    for key in ["hotkey", "hotkey_enabled", "toggle_hotkey"] {
        assert!(saved["setup"].get(key).is_some(), "setup saved under {key}");
    }
}
