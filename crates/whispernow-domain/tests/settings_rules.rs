use whispernow_domain::{
    ApiKey, CLEANUP_MODEL_IDS, CleanupTarget, DEFAULT_CLEANUP_MODEL_ID, InvalidApiKey, LlmProvider,
    MissingCleanupField, ProviderEndpoint, RemoteProvider, Settings, SettingsError,
    SetupCheckpoint, Shortcut, ShortcutKey, ShortcutModifier, VocabularyReplacement,
};

fn shortcut(modifiers: &[ShortcutModifier], key: &str) -> Shortcut {
    Shortcut {
        modifiers: modifiers.iter().copied().collect(),
        key: ShortcutKey::from_name(key).expect("a shortcut key"),
    }
}

#[test]
fn default_settings_break_no_rule() {
    assert_eq!(Settings::default().problems(), []);
}

#[test]
fn the_recording_shortcut_needs_a_modifier() {
    let settings = Settings {
        recording_shortcut: shortcut(&[], "f9"),
        ..Settings::default()
    };
    assert_eq!(settings.validate(), Err(SettingsError::RecordingShortcut));
}

#[test]
fn the_cancel_shortcut_is_a_lone_non_typing_key_or_a_combination_and_never_a_recording_one() {
    let mut settings = Settings::default();
    for cancel in [
        Shortcut::default_cancel(),
        shortcut(&[], "f8"),
        shortcut(&[ShortcutModifier::Alt], "q"),
    ] {
        settings.cancel_shortcut = Some(cancel);
        assert_eq!(settings.validate(), Ok(()));
    }
    settings.cancel_shortcut = Some(shortcut(&[], "q"));
    assert_eq!(settings.validate(), Err(SettingsError::CancelShortcut));
    settings.cancel_shortcut = Some(settings.recording_shortcut.clone());
    assert_eq!(
        settings.validate(),
        Err(SettingsError::CancelIsRecordingShortcut)
    );
    settings.cancel_shortcut = None;
    assert_eq!(settings.validate(), Ok(()));
}

#[test]
fn model_names_are_checked_for_every_provider() {
    let mut settings = Settings::default();
    for saved in [
        vec![" ".to_owned()],
        vec!["same".to_owned(), "same".to_owned()],
    ] {
        settings
            .provider_settings_mut(LlmProvider::OpenRouter)
            .saved_models = saved;
        assert_eq!(
            settings.validate(),
            Err(SettingsError::ModelName {
                provider: LlmProvider::OpenRouter
            })
        );
    }
}

#[test]
fn requests_drop_the_providers_own_prefix_from_a_model_name() {
    assert_eq!(
        LlmProvider::OpenAi.request_model(" openai/gpt-4o "),
        "gpt-4o"
    );
    assert_eq!(
        LlmProvider::OpenRouter.request_model("anthropic/claude-3"),
        "anthropic/claude-3"
    );
    assert_eq!(LlmProvider::Custom.request_model("openai/x"), "openai/x");
}

#[test]
fn an_api_key_is_one_trimmed_line() {
    assert_eq!(
        ApiKey::new(" key ").map(|key| key.expose_secret().to_owned()),
        Ok("key".to_owned())
    );
    assert_eq!(ApiKey::new(" ").err(), Some(InvalidApiKey));
    assert_eq!(ApiKey::new("a\nb").err(), Some(InvalidApiKey));
    assert!(serde_json::from_str::<ApiKey>(r#""a\nb""#).is_err());
}

#[test]
fn a_vocabulary_rule_needs_text_to_replace() {
    let settings = Settings {
        vocabulary_replacements: vec![VocabularyReplacement {
            source: " ".to_owned(),
            replacement: "x".to_owned(),
        }],
        ..Settings::default()
    };
    assert_eq!(settings.validate(), Err(SettingsError::VocabularySource));
}

#[test]
fn incomplete_cleanup_settings_save_with_a_warning_until_repaired() {
    let mut settings = Settings {
        llm_provider: LlmProvider::Custom,
        ..Settings::default()
    };
    settings.provider_settings_mut(LlmProvider::Custom).model = String::new();
    assert_eq!(settings.validate(), Ok(()));
    assert_eq!(settings.cleanup_warning(), None);

    settings.cleanup_enabled = true;
    assert_eq!(settings.validate(), Ok(()));
    assert_eq!(settings.cleanup_target(), Err(MissingCleanupField::Model));
    let warning = settings
        .cleanup_warning()
        .expect("warning for the missing model");
    assert!(warning.contains(LlmProvider::Custom.label()));
    settings.provider_settings_mut(LlmProvider::Custom).model = "local".to_owned();
    assert_eq!(
        settings.cleanup_target(),
        Err(MissingCleanupField::Endpoint)
    );
    let warning = settings
        .cleanup_warning()
        .expect("warning for the missing endpoint");
    assert!(warning.contains(LlmProvider::Custom.label()));
    settings.provider_settings_mut(LlmProvider::Custom).api_base =
        Some(ProviderEndpoint::parse("http://localhost:8080/v1").expect("endpoint"));
    assert_eq!(settings.cleanup_warning(), None);
    let CleanupTarget::Remote(target) = settings.cleanup_target().expect("repaired target") else {
        panic!("remote cleanup target");
    };
    assert_eq!(target.model, "local");
    let RemoteProvider::Custom(endpoint) = target.provider else {
        panic!("custom provider endpoint");
    };
    assert_eq!(endpoint.expose_url(), "http://localhost:8080/v1");
}

#[test]
fn the_cleanup_model_is_the_one_saved_for_its_provider_else_the_default() {
    let mut settings = Settings::default();
    assert_eq!(
        settings.cleanup_model_id().as_str(),
        DEFAULT_CLEANUP_MODEL_ID
    );
    settings
        .provider_settings_mut(LlmProvider::LocalCleanup)
        .model = CLEANUP_MODEL_IDS[1].to_owned();
    assert_eq!(settings.cleanup_model_id().as_str(), CLEANUP_MODEL_IDS[1]);
    // A model saved before there was a choice, or one no longer offered, is the default.
    for saved in ["", "cleanup", "whispernow-cleanup-gone"] {
        settings
            .provider_settings_mut(LlmProvider::LocalCleanup)
            .model = saved.to_owned();
        assert_eq!(
            settings.cleanup_model_id().as_str(),
            DEFAULT_CLEANUP_MODEL_ID
        );
    }
    // Another provider's model never chooses it.
    settings.provider_settings_mut(LlmProvider::Custom).model = CLEANUP_MODEL_IDS[1].to_owned();
    assert_eq!(
        settings.cleanup_model_id().as_str(),
        DEFAULT_CLEANUP_MODEL_ID
    );
}

#[test]
fn an_update_is_refused_only_for_a_rule_it_breaks() {
    let before = Settings {
        vocabulary_replacements: vec![VocabularyReplacement::new(" ", "x")],
        ..Settings::default()
    };
    let mut edited = Settings {
        start_minimized: !before.start_minimized,
        ..before.clone()
    };
    assert_eq!(
        edited.validate_update(&before),
        Ok(()),
        "a rule broken before the edit does not block other changes"
    );
    edited.cancel_shortcut = Some(edited.recording_shortcut.clone());
    assert_eq!(
        edited.validate_update(&before),
        Err(SettingsError::CancelIsRecordingShortcut)
    );
}

#[test]
fn setup_shortcuts_are_ready_when_the_recording_shortcut_is_valid() {
    let mut checkpoint = SetupCheckpoint::default();
    assert!(checkpoint.shortcuts_ready());
    checkpoint.recording_shortcut = shortcut(&[], "space");
    assert!(!checkpoint.shortcuts_ready());
}

#[test]
fn credentials_serialize_the_value_and_redact_debug_output() {
    let key = ApiKey::new("write-only-secret").expect("a key");
    assert_eq!(
        serde_json::to_value(&key).expect("serialize credential mutation"),
        serde_json::json!("write-only-secret")
    );
    assert!(!format!("{key:?}").contains("write-only-secret"));
}
