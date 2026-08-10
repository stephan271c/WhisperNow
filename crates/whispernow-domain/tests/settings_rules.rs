use whispernow_domain::{
    ApiKey, CLEAN_UP_PROMPT_ID, CleanupPrompt, DEFAULT_CLEANUP_MODEL_ID, InvalidApiKey,
    LlmProvider, MINICPM_CLEANUP_MODEL_ID, Settings, SettingsError, SetupCheckpoint, Shortcut,
    ShortcutKey, ShortcutModifier, VocabularyReplacement,
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
fn recording_shortcuts_need_one_on_each_valid_and_distinct() {
    let mut settings = Settings {
        hold_shortcut_enabled: false,
        toggle_shortcut: None,
        ..Settings::default()
    };
    assert_eq!(settings.validate(), Err(SettingsError::NoRecordingShortcut));

    settings.toggle_shortcut = Some(shortcut(&[], "f9"));
    assert_eq!(settings.validate(), Err(SettingsError::ToggleShortcut));

    settings.hold_shortcut_enabled = true;
    settings.toggle_shortcut = Some(settings.hold_shortcut.clone());
    assert_eq!(
        settings.validate(),
        Err(SettingsError::SameRecordingShortcuts)
    );
    settings.hold_shortcut_enabled = false;
    assert_eq!(
        settings.validate(),
        Ok(()),
        "a hold shortcut that is off may share the toggle's combination"
    );

    settings.hold_shortcut = shortcut(&[], "space");
    assert_eq!(
        settings.validate(),
        Err(SettingsError::HoldShortcut),
        "a hold shortcut that is off is kept valid for turning it back on"
    );
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
    settings.cancel_shortcut = Some(settings.hold_shortcut.clone());
    assert_eq!(
        settings.validate(),
        Err(SettingsError::CancelIsRecordingShortcut)
    );
    settings.cancel_shortcut = None;
    assert_eq!(settings.validate(), Ok(()));
}

#[test]
fn numbers_stay_in_their_ranges() {
    let mut settings = Settings {
        llm_timeout_seconds: 4,
        ..Settings::default()
    };
    assert_eq!(settings.validate(), Err(SettingsError::LlmTimeout));
    settings.llm_timeout_seconds = 121;
    assert_eq!(settings.validate(), Err(SettingsError::LlmTimeout));
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
fn prompts_need_a_title_text_and_their_own_id_and_the_chosen_one_must_exist() {
    let prompt = |id: &str, title: &str, text: &str| CleanupPrompt {
        id: id.to_owned(),
        title: title.to_owned(),
        prompt: text.to_owned(),
        default_hash: None,
    };
    let mut settings = Settings {
        prompts: vec![prompt("a", "", "text")],
        ..Settings::default()
    };
    assert_eq!(
        settings.validate(),
        Err(SettingsError::PromptTitle { id: "a".to_owned() })
    );
    settings.prompts = vec![prompt("a", "Title", " ")];
    assert_eq!(
        settings.validate(),
        Err(SettingsError::PromptText { id: "a".to_owned() })
    );
    settings.prompts = vec![prompt("a", "One", "text"), prompt("a", "Two", "text")];
    assert_eq!(settings.validate(), Err(SettingsError::PromptId));

    settings.prompts = vec![prompt("a", "One", "text")];
    settings.active_prompt_id = Some("missing".to_owned());
    assert_eq!(settings.validate(), Err(SettingsError::UnknownActivePrompt));

    settings.active_prompt_id = Some("a".to_owned());
    settings.delete_prompt("a");
    assert_eq!(settings.active_prompt_id, None);
    assert_eq!(settings.next_prompt_id(), "prompt-1");
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
fn ai_cleanup_needs_a_usable_provider_only_while_it_is_on() {
    let mut settings = Settings {
        llm_provider: LlmProvider::Custom,
        active_prompt_id: None,
        ..Settings::default()
    };
    settings.provider_settings_mut(LlmProvider::Custom).model = String::new();
    assert_eq!(settings.validate(), Ok(()));

    settings.active_prompt_id = Some(settings.prompts[0].id.clone());
    assert_eq!(
        settings.validate().expect_err("refused").to_string(),
        "AI cleanup needs a model for Custom provider; enter one, or turn AI cleanup off"
    );
    settings.provider_settings_mut(LlmProvider::Custom).model = "local".to_owned();
    assert_eq!(
        settings.validate(),
        Err(SettingsError::CleanupProvider {
            provider: LlmProvider::Custom,
            missing: "an API endpoint"
        })
    );
}

#[test]
fn the_local_cleanup_model_takes_only_clean_up() {
    let mut settings = Settings {
        llm_provider: LlmProvider::LocalCleanup,
        active_prompt_id: Some(CLEAN_UP_PROMPT_ID.to_owned()),
        ..Settings::default()
    };
    settings
        .provider_settings_mut(LlmProvider::LocalCleanup)
        .model = "cleanup".to_owned();
    assert_eq!(settings.validate(), Ok(()));

    settings.active_prompt_id = Some("professional".to_owned());
    assert_eq!(settings.validate(), Err(SettingsError::LocalCleanupPrompt));
    assert!(
        settings
            .validate()
            .expect_err("refused")
            .to_string()
            .starts_with("The local cleanup model only does Clean Up & Format")
    );

    settings.llm_provider = LlmProvider::Ollama;
    settings.provider_settings_mut(LlmProvider::Ollama).model = "local".to_owned();
    settings.active_prompt_id = Some("professional".to_owned());
    assert_eq!(settings.validate(), Ok(()));
}

#[test]
fn the_cleanup_model_is_the_one_saved_for_its_provider_else_the_default() {
    let mut settings = Settings::default();
    assert_eq!(settings.cleanup_model_id(), DEFAULT_CLEANUP_MODEL_ID);
    settings
        .provider_settings_mut(LlmProvider::LocalCleanup)
        .model = MINICPM_CLEANUP_MODEL_ID.to_owned();
    assert_eq!(settings.cleanup_model_id(), MINICPM_CLEANUP_MODEL_ID);
    // A model saved before there was a choice, or one no longer offered, is the default.
    for saved in ["", "cleanup", "whispernow-cleanup-gone"] {
        settings
            .provider_settings_mut(LlmProvider::LocalCleanup)
            .model = saved.to_owned();
        assert_eq!(settings.cleanup_model_id(), DEFAULT_CLEANUP_MODEL_ID);
    }
    // Another provider's model never chooses it.
    settings.provider_settings_mut(LlmProvider::Custom).model = MINICPM_CLEANUP_MODEL_ID.to_owned();
    assert_eq!(settings.cleanup_model_id(), DEFAULT_CLEANUP_MODEL_ID);
}

#[test]
fn an_update_is_refused_only_for_a_rule_it_breaks() {
    let before = Settings {
        llm_timeout_seconds: 1,
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
    edited.toggle_shortcut = Some(edited.hold_shortcut.clone());
    assert_eq!(
        edited.validate_update(&before),
        Err(SettingsError::SameRecordingShortcuts)
    );
}

#[test]
fn setup_shortcuts_are_ready_when_one_is_on_each_is_valid_and_they_differ() {
    let mut checkpoint = SetupCheckpoint::default();
    assert!(checkpoint.shortcuts_ready());
    checkpoint.toggle_shortcut = Some(checkpoint.hold_shortcut.clone());
    assert!(!checkpoint.shortcuts_ready());
    checkpoint.hold_shortcut_enabled = false;
    assert!(checkpoint.shortcuts_ready());
    checkpoint.toggle_shortcut = None;
    assert!(!checkpoint.shortcuts_ready());
    checkpoint.hold_shortcut_enabled = true;
    checkpoint.hold_shortcut = shortcut(&[], "space");
    assert!(!checkpoint.shortcuts_ready());
}
