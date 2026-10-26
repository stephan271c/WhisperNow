use whispernow_domain::{
    ApiKey, CLEANUP_MODEL_IDS, CleanupTarget, DEFAULT_CLEANUP_MODEL_ID, InvalidApiKey, LlmProvider,
    MissingCleanupField, ModelId, ModelRegistry, ProviderEndpoint, ProviderSettings, RecordingMode,
    RemoteProvider, SettingChange, Settings, SettingsError, SetupCheckpoint, Shortcut,
    ShortcutModifier, VocabularyReplacement,
};

fn shortcut(modifiers: &[ShortcutModifier], key: &str) -> Shortcut {
    Shortcut {
        modifiers: modifiers.iter().copied().collect(),
        key: key.parse().expect("a shortcut key"),
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
fn shortcuts_are_valid_unless_a_shortcut_rule_is_broken() {
    let mut settings = Settings::default();
    assert!(settings.shortcuts_valid());
    settings.cancel_shortcut = Some(Shortcut::default_cancel());
    assert!(settings.shortcuts_valid());
    settings.cancel_shortcut = Some(shortcut(&[], "q"));
    assert!(!settings.shortcuts_valid());
    settings.cancel_shortcut = Some(settings.recording_shortcut.clone());
    assert!(!settings.shortcuts_valid());
    settings.cancel_shortcut = None;
    settings.recording_shortcut = shortcut(&[], "f9");
    assert!(!settings.shortcuts_valid());

    settings.recording_shortcut = Settings::default().recording_shortcut;
    settings.vocabulary_replacements = vec![VocabularyReplacement::new(" ", "x")];
    assert!(settings.shortcuts_valid());
}

#[test]
fn a_chosen_model_name_is_trimmed_and_refused_when_blank_or_multi_line() {
    let mut settings = Settings::default();
    let choose = |model: &str| SettingChange::ChooseProviderModel {
        provider: LlmProvider::OpenRouter,
        model: model.to_owned(),
    };
    for refused in [" ", "a\nb", "a\rb"] {
        let change = choose(refused);
        let mut edited = settings.clone();
        change.clone().apply(&mut edited);
        assert_eq!(edited, settings, "{refused:?} is not shown as chosen");
        assert_eq!(change.check(&edited), Err(SettingsError::ModelName));
    }

    let change = choose("  auto ");
    change.clone().apply(&mut settings);
    assert_eq!(change.check(&settings), Ok(()));
    let openrouter = settings.provider_settings(LlmProvider::OpenRouter);
    assert_eq!(openrouter.model.as_deref(), Some("auto"));
    assert_eq!(openrouter.saved_models, ["auto"]);
}

#[test]
fn bad_saved_model_names_block_no_choice_or_removal() {
    let mut saved = Settings::default();
    saved
        .provider_settings_mut(LlmProvider::OpenRouter)
        .saved_models = vec![" ".to_owned(), "same".to_owned(), "same".to_owned()];
    assert_eq!(saved.validate(), Ok(()));
    for change in [
        SettingChange::ChooseProviderModel {
            provider: LlmProvider::OpenRouter,
            model: "auto".to_owned(),
        },
        SettingChange::RemoveProviderModel {
            provider: LlmProvider::OpenRouter,
            model: "same".to_owned(),
        },
    ] {
        let mut edited = saved.clone();
        change.clone().apply(&mut edited);
        assert_eq!(change.check(&edited), Ok(()), "{change:?}");
    }
}

#[test]
fn the_model_list_keeps_the_active_model_and_follows_choices_and_removals() {
    let mut openai = ProviderSettings::default();
    openai.choose_model("gpt-5".to_owned());
    assert_eq!(openai.saved_models, ["gpt-5"]);

    openai.choose_model("gpt-5-mini".to_owned());
    assert_eq!(openai.saved_models, ["gpt-5", "gpt-5-mini"]);
    assert_eq!(openai.model.as_deref(), Some("gpt-5-mini"));

    openai.choose_model("gpt-5".to_owned());
    assert_eq!(openai.saved_models, ["gpt-5", "gpt-5-mini"]);

    openai.remove_model("gpt-5");
    assert_eq!(openai.saved_models, ["gpt-5-mini"]);
    assert_eq!(
        openai.model.as_deref(),
        Some("gpt-5-mini"),
        "the first saved model takes over"
    );

    openai.remove_model("gpt-5-mini");
    assert!(openai.saved_models.is_empty());
    assert_eq!(openai.model, None);
}

#[test]
fn the_cleanup_target_keeps_the_model_name_as_saved() {
    let mut settings = Settings {
        llm_provider: LlmProvider::OpenRouter,
        ..Settings::default()
    };
    settings
        .provider_settings_mut(LlmProvider::OpenRouter)
        .choose_model("openrouter/auto".to_owned());
    let CleanupTarget::Remote(target) = settings.cleanup_target().expect("a remote target") else {
        panic!("remote cleanup target");
    };
    assert_eq!(target.model, "openrouter/auto");
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
    assert_eq!(settings.validate(), Ok(()));
    assert_eq!(settings.cleanup_warning(), None);

    settings.cleanup_enabled = true;
    assert_eq!(settings.validate(), Ok(()));
    assert_eq!(settings.cleanup_target(), Err(MissingCleanupField::Model));
    let warning = settings
        .cleanup_warning()
        .expect("warning for the missing model");
    assert!(warning.contains(LlmProvider::Custom.label()));
    settings
        .provider_settings_mut(LlmProvider::Custom)
        .choose_model("local".to_owned());
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
fn cleanup_has_an_active_target_only_while_enabled_and_resolved() {
    let mut settings = Settings {
        llm_provider: LlmProvider::LocalCleanup,
        ..Settings::default()
    };
    assert_eq!(settings.active_cleanup_target(), None);

    settings.cleanup_enabled = true;
    assert_eq!(
        settings.active_cleanup_target(),
        Some(CleanupTarget::Local {
            model: settings.cleanup_model.clone()
        })
    );

    settings.llm_provider = LlmProvider::Custom;
    settings
        .provider_settings_mut(LlmProvider::Custom)
        .choose_model("local".to_owned());
    assert_eq!(settings.active_cleanup_target(), None);

    settings.provider_settings_mut(LlmProvider::Custom).api_base =
        Some(ProviderEndpoint::parse("http://localhost:8080/v1").expect("endpoint"));
    assert_eq!(
        settings.active_cleanup_target(),
        settings.cleanup_target().ok()
    );
    assert!(settings.active_cleanup_target().is_some());

    settings.cleanup_enabled = false;
    assert_eq!(settings.active_cleanup_target(), None);
}

#[test]
fn an_unknown_cleanup_model_loads_as_the_default() {
    assert_eq!(
        Settings::default().cleanup_model.as_str(),
        DEFAULT_CLEANUP_MODEL_ID
    );
    let chosen: Settings =
        serde_json::from_value(serde_json::json!({ "cleanup_model": CLEANUP_MODEL_IDS[1] }))
            .expect("settings");
    assert_eq!(chosen.cleanup_model.as_str(), CLEANUP_MODEL_IDS[1]);
    // A model no longer offered is the default.
    for saved in ["", "cleanup", "whispernow-cleanup-gone"] {
        let settings: Settings =
            serde_json::from_value(serde_json::json!({ "cleanup_model": saved }))
                .expect("settings");
        assert_eq!(settings.cleanup_model.as_str(), DEFAULT_CLEANUP_MODEL_ID);
    }
}

#[test]
fn an_unknown_or_cleanup_speech_model_loads_as_the_default() {
    let speech = ModelRegistry::builtin()
        .speech_models()
        .map(|model| model.id.clone())
        .find(|id| *id != ModelId::default())
        .expect("another speech model");
    let read = |id: &str| {
        let settings: Settings = serde_json::from_value(serde_json::json!({
            "model_id": id,
            "setup": { "unfinished": { "model_id": id } },
        }))
        .expect("settings");
        let checkpoint = settings.setup.checkpoint().expect("unfinished setup");
        (settings.model_id.clone(), checkpoint.model_id.clone())
    };
    assert_eq!(read(speech.as_str()), (speech.clone(), speech));
    for saved in ["", "speech-model-gone", DEFAULT_CLEANUP_MODEL_ID] {
        assert_eq!(
            read(saved),
            (ModelId::default(), ModelId::default()),
            "{saved:?}"
        );
    }
}

#[test]
fn a_change_is_refused_only_for_a_rule_it_breaks() {
    let saved = Settings {
        vocabulary_replacements: vec![VocabularyReplacement::new(" ", "x")],
        ..Settings::default()
    };
    let save = |change: SettingChange| {
        let mut edited = saved.clone();
        change.clone().apply(&mut edited);
        change.check(&edited)
    };
    assert_eq!(
        save(SettingChange::StartMinimized(true)),
        Ok(()),
        "a rule broken elsewhere does not block other changes"
    );
    assert_eq!(
        save(SettingChange::CancelShortcut(Some(
            saved.recording_shortcut.clone()
        ))),
        Err(SettingsError::CancelIsRecordingShortcut)
    );
    assert_eq!(
        save(SettingChange::Vocabulary(vec![VocabularyReplacement::new(
            "", "y"
        )])),
        Err(SettingsError::VocabularySource)
    );
    assert_eq!(
        save(SettingChange::Vocabulary(Vec::new())),
        Ok(()),
        "the change that repairs a broken rule is accepted"
    );
    assert_eq!(
        save(SettingChange::ChooseProviderModel {
            provider: LlmProvider::Gemini,
            model: "a\nb".to_owned(),
        }),
        Err(SettingsError::ModelName)
    );
}

#[test]
fn a_change_edits_only_its_own_setting() {
    let mut settings = Settings::default();
    SettingChange::ChooseProviderModel {
        provider: LlmProvider::Gemini,
        model: "gemini-test".to_owned(),
    }
    .apply(&mut settings);
    SettingChange::RecordingShortcut {
        shortcut: shortcut(&[ShortcutModifier::Alt], "f8"),
        mode: RecordingMode::Toggle,
    }
    .apply(&mut settings);
    SettingChange::CleanupModel(ModelId::new("whispernow-cleanup-gone").expect("model id"))
        .apply(&mut settings);

    let mut expected = Settings {
        recording_shortcut: shortcut(&[ShortcutModifier::Alt], "f8"),
        recording_mode: RecordingMode::Toggle,
        ..Settings::default()
    };
    expected
        .provider_settings_mut(LlmProvider::Gemini)
        .choose_model("gemini-test".to_owned());
    assert_eq!(
        settings, expected,
        "an unknown cleanup model is the default"
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
