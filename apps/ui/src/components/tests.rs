use dioxus::prelude::*;
use whispernow_domain::AppError;
use whispernow_domain::ipc::{
    AppState, ClearDataItem, ClearDataResult, DataLocation, ErrorCategory, MicrophoneTestResponse,
    ModelState, ModelSummary, Phase, SaveSettingsResponse, SettingsView, SetupStateView,
};
use whispernow_domain::{
    ClearDataCategory, HistoryLimit, LlmProvider, ModelAvailability, ModelId, ProviderSettings,
    Rating, RecordingMode, Settings, SetupCheckpoint, SetupStep, Shortcut, ShortcutKey,
    ShortcutModifier, SpeechBenchmark, VocabularyReplacement, WerSource,
};

use crate::client::fake;
use crate::components::ui::Tone;
use crate::test_support::{Page, Probe, focused, label_in_name_violations, snapshot, tag, text};
use crate::{AppShell, ManagementState, NoticeScope, UiState};

#[component]
fn Shell(snapshot: AppState, management: ManagementState, probe: Probe) -> Element {
    let state = use_signal(move || {
        let mut state = UiState::default();
        state.hydrate(snapshot);
        state
    });
    let management = use_signal(move || management);
    probe.provide(state, management);
    rsx! { AppShell {} }
}

pub(super) fn shell(snapshot: AppState, management: ManagementState) -> Page {
    Page::new(|probe| {
        VirtualDom::new_with_props(
            Shell,
            ShellProps {
                snapshot,
                management,
                probe,
            },
        )
    })
}

pub(super) fn shortcut() -> Shortcut {
    Shortcut {
        modifiers: [ShortcutModifier::Control].into(),
        key: ShortcutKey::Space,
    }
}

fn settings() -> Settings {
    Settings {
        model_id: ModelId::new("sherpa-onnx-whisper-tiny").expect("fixture model id"),
        recording_shortcut: shortcut(),
        first_run_complete: true,
        history_limit: HistoryLimit::Last20,
        llm_provider_settings: [(
            LlmProvider::OpenAi,
            ProviderSettings {
                model: "gpt-5".to_owned(),
                ..ProviderSettings::default()
            },
        )]
        .into(),
        ..Settings::default()
    }
}

pub(super) fn loaded() -> ManagementState {
    let mut management = ManagementState::default();
    management.install_settings(SettingsView {
        settings: settings(),
        warnings: Vec::new(),
    });
    management
}

fn unfinished_setup() -> ManagementState {
    let mut management = loaded();
    management.install_setup(SetupStateView {
        first_run_complete: false,
        checkpoint: SetupCheckpoint {
            step: SetupStep::Model,
            model_id: ModelId::new("sherpa-onnx-whisper-tiny").expect("fixture model id"),
            input_device: None,
            recording_shortcut: shortcut(),
            recording_mode: RecordingMode::Hold,
        },
    });
    management
}

fn current_links(html: &str) -> usize {
    html.matches("aria-current=\"page\"").count()
}

#[test]
fn home_is_the_first_page_and_shows_the_setup_checklist_without_redirecting() {
    let mut page = shell(snapshot(Phase::Idle), unfinished_setup());
    let html = page.html();

    assert_eq!(text(&html, "page-heading"), "Home", "no redirect to setup");
    assert_eq!(html.matches("<h1").count(), 1, "one h1 per view");
    assert!(
        tag(&html, "nav-home")
            .expect("home link")
            .contains("aria-current=\"page\"")
    );
    assert_eq!(current_links(&html), 1);
    let checklist = text(&html, "setup-checklist");
    assert!(
        checklist.contains("Allow microphone access (done)"),
        "{checklist}"
    );
    assert!(
        checklist.contains("Choose a speech model (to do)"),
        "{checklist}"
    );
    assert!(tag(&html, "continue-setup").is_some());
    let skip = tag(&html, "main-content").expect("main landmark");
    assert!(skip.starts_with("<main"), "{skip}");
    assert!(html.contains("href=\"#main-content\""), "skip link");
    page.mount("page-heading");
    assert_eq!(focused(), None, "the first page leaves focus alone");

    page.click("continue-setup");
    page.settle();
    let html = page.html();
    assert_eq!(text(&html, "page-heading"), "Set up WhisperNow");
    assert_eq!(
        focused().as_deref(),
        Some("page-heading"),
        "continuing setup moves focus to its heading"
    );

    let page = shell(snapshot(Phase::Idle), loaded());
    let html = page.html();
    assert!(tag(&html, "setup-checklist").is_none(), "setup is complete");
}

#[test]
fn sidebar_links_mark_one_current_page_and_move_focus_to_its_heading() {
    let mut page = shell(snapshot(Phase::Idle), loaded());
    let html = page.html();
    assert!(
        tag(&html, "nav-settings-general").is_none(),
        "sub-pages are collapsed"
    );
    page.mount("page-heading");

    page.click("nav-settings");
    page.settle();
    assert_eq!(focused().as_deref(), Some("page-heading"));
    let html = page.html();
    assert_eq!(text(&html, "page-heading"), "General &#38; Shortcuts");
    assert!(
        tag(&html, "nav-settings-general")
            .expect("general link")
            .contains("aria-current=\"page\"")
    );
    assert!(
        tag(&html, "nav-settings")
            .expect("settings link")
            .contains("aria-current=\"false\"")
    );
    assert_eq!(current_links(&html), 1);
    for slug in [
        "general",
        "audio",
        "models",
        "ai-cleanup",
        "vocabulary",
        "privacy",
        "about",
    ] {
        assert!(
            tag(&html, &format!("nav-settings-{slug}")).is_some(),
            "{slug}"
        );
    }

    page.click("nav-settings-audio");
    page.settle();
    let html = page.html();
    assert_eq!(text(&html, "page-heading"), "Audio");
    assert_eq!(current_links(&html), 1);
    assert_eq!(focused().as_deref(), Some("page-heading"));

    page.click("nav-history");
    let html = page.html();
    assert_eq!(text(&html, "page-heading"), "History");
    assert!(tag(&html, "nav-settings-audio").is_none());
    assert!(
        tag(&html, "nav-history")
            .expect("history")
            .contains("aria-current=\"page\"")
    );
    assert_eq!(html.matches("<h1").count(), 1);
}

#[test]
fn notices_announce_by_severity_and_appear_beside_the_page_that_hosts_them() {
    let mut page = shell(snapshot(Phase::Idle), loaded());
    let mut management = page.context().management;
    for (severity, role) in [
        (Tone::Info, "status"),
        (Tone::Success, "status"),
        (Tone::Warning, "alert"),
        (Tone::Error, "alert"),
    ] {
        management.write().notify(
            severity,
            NoticeScope::Global,
            format!("{severity:?} message"),
        );
        page.settle();
        let html = page.html();
        let notice = tag(&html, "notice").expect("notice");
        assert!(
            notice.contains(&format!("role=\"{role}\"")),
            "{severity:?}: {notice}"
        );
        assert!(notice.contains("notice-toast"), "global notices are toasts");
        assert_eq!(text(&html, "notice"), format!("{severity:?} message"));
        assert!(tag(&html, "notice-dismiss").is_some());
    }

    // A Home notice sits in the microphone card while Home is open.
    management.write().notify(
        Tone::Warning,
        NoticeScope::Home,
        "No sound reached the microphone",
    );
    page.settle();
    let html = page.html();
    assert!(
        tag(&html, "notice")
            .expect("notice")
            .contains("notice-inline")
    );

    // A settings notice raised on Home is a toast until its page is open.
    management.write().notify(
        Tone::Error,
        NoticeScope::Settings,
        "Fix the highlighted fields",
    );
    page.settle();
    let html = page.html();
    assert!(
        tag(&html, "notice")
            .expect("notice")
            .contains("notice-toast")
    );
    page.click("nav-settings");
    let html = page.html();
    assert!(
        tag(&html, "notice")
            .expect("notice")
            .contains("notice-inline")
    );
}

#[test]
fn test_microphone_runs_once_and_shows_the_measured_level() {
    let mut page = shell(snapshot(Phase::Idle), loaded());
    page.click("test-microphone");
    let mut management = page.context().management;
    assert_eq!(
        fake::count("test_microphone"),
        1,
        "the click invokes the desktop test"
    );
    let html = page.html();
    let button = tag(&html, "test-microphone").expect("button");
    assert!(
        button.contains(" disabled="),
        "one test at a time: {button}"
    );
    assert!(text(&html, "test-microphone").contains("Listening…"));
    assert!(tag(&html, "microphone-level").is_none());

    management.write().microphone_test = Some(MicrophoneTestResponse {
        levels: vec![0.1, 0.3, 0.2],
        peak: 0.46,
    });
    page.settle();
    let html = page.html();
    let meter = tag(&html, "microphone-level").expect("meter");
    assert!(meter.contains("role=\"meter\""), "{meter}");
    assert!(meter.contains("aria-label=\"Microphone level\""), "{meter}");
    assert!(meter.contains("aria-valuenow=\"46\""), "{meter}");
}

#[test]
fn disabled_controls_explain_why_in_visible_text() {
    let mut dictating = snapshot(Phase::Idle);
    dictating.phase = crate::test_support::RECORDING;
    let page = shell(dictating, loaded());
    let html = page.html();
    let button = tag(&html, "test-microphone").expect("button");
    assert!(button.contains(" disabled="), "{button}");
    assert!(
        button.contains("aria-describedby=\"test-microphone-disabled-reason\""),
        "{button}"
    );
    assert_eq!(
        text(&html, "test-microphone-disabled-reason"),
        "Wait for the current dictation to finish"
    );
}

#[test]
fn a_problem_on_home_links_to_the_page_that_fixes_it() {
    let mut failed = snapshot(Phase::Idle);
    failed.model_state = ModelState::Failed;
    let mut page = shell(failed, loaded());
    let html = page.html();
    let status = tag(&html, "home-status").expect("status");
    assert!(status.contains("role=\"alert\""), "{status}");
    assert_eq!(text(&html, "home-status-fix"), "Open speech models");

    page.click("home-status-fix");
    let html = page.html();
    assert_eq!(text(&html, "page-heading"), "Speech models");
    assert!(
        tag(&html, "nav-settings-models")
            .expect("models link")
            .contains("aria-current=\"page\"")
    );
}

#[test]
fn a_cleanup_failure_on_home_opens_ai_cleanup_settings() {
    let mut failed = snapshot(Phase::Idle);
    failed.model_state = ModelState::Failed;
    failed.error = Some(AppError::new(
        ErrorCategory::Cleanup,
        "cleanup_failed",
        "AI cleanup was unavailable, so the transcript was kept as recognized",
    ));
    let mut page = shell(failed, loaded());
    assert_eq!(text(&page.html(), "home-status-fix"), "Open AI cleanup");

    page.click("home-status-fix");
    assert_eq!(text(&page.html(), "page-heading"), "AI cleanup");
    assert!(
        tag(&page.html(), "nav-settings-ai-cleanup")
            .expect("AI cleanup link")
            .contains("aria-current=\"page\"")
    );
}

#[test]
fn the_latest_transcript_can_be_copied_from_home() {
    let mut transcribed = snapshot(Phase::Idle);
    transcribed.last_transcript = Some("Meet on Thursday".to_owned());
    let mut page = shell(transcribed, loaded());
    let html = page.html();
    assert_eq!(text(&html, "latest-transcript"), "Meet on Thursday");
    assert!(text(&html, "home-shortcuts-title").contains("Shortcuts"));

    page.click("copy-latest");
    assert_eq!(
        fake::requests::<String>("write_clipboard"),
        ["Meet on Thursday"]
    );
}

fn open(page: &mut Page, ids: &[&str]) {
    for id in ids {
        page.click(id);
    }
}

/// Every settings change sent to the desktop so far.
fn sent_settings() -> Vec<Settings> {
    fake::requests::<Settings>("update_settings")
}

pub(super) fn model(id: &str, name: &str, availability: ModelAvailability) -> ModelSummary {
    ModelSummary {
        id: ModelId::new(id).expect("fixture model id"),
        display_name: name.to_owned(),
        family: whispernow_domain::ModelFamily::Whisper,
        availability,
        languages: vec!["en".to_owned()],
        size_bytes: 75_000_000,
        speed: Rating::new(5),
        accuracy: Rating::new(3),
        benchmark: None,
        recommended: name == "Base",
    }
}

/// Settings with every section populated: AI cleanup on, a saved provider model, a vocabulary
/// rule, kept audio, and three models.
fn configured() -> ManagementState {
    let mut settings = settings();
    settings.cleanup_enabled = true;
    settings
        .provider_settings_mut(LlmProvider::OpenAi)
        .saved_models = vec!["gpt-5-mini".to_owned()];
    settings.vocabulary_replacements = vec![
        VocabularyReplacement::new("teh", "the"),
        VocabularyReplacement::new("whisper now", "WhisperNow"),
    ];
    settings.keep_audio = true;
    settings.api_keys = [LlmProvider::OpenAi].into();
    let mut management = ManagementState::default();
    management.install_settings(SettingsView {
        settings,
        warnings: Vec::new(),
    });
    management.install_models(vec![
        model(
            "sherpa-onnx-whisper-tiny",
            "Tiny",
            ModelAvailability::Downloaded,
        ),
        model(
            "sherpa-onnx-whisper-base",
            "Base",
            ModelAvailability::Downloaded,
        ),
        model(
            "sherpa-onnx-whisper-small",
            "Small",
            ModelAvailability::NotInstalled,
        ),
    ]);
    management
}

const SETTINGS_PAGES: [&str; 7] = [
    "nav-settings-general",
    "nav-settings-audio",
    "nav-settings-models",
    "nav-settings-ai-cleanup",
    "nav-settings-vocabulary",
    "nav-settings-privacy",
    "nav-settings-about",
];

#[test]
fn the_ai_cleanup_page_saves_api_keys_during_recording() {
    let mut page = shell(snapshot(crate::test_support::RECORDING), loaded());
    open(&mut page, &["nav-settings", "nav-settings-ai-cleanup"]);
    page.input("provider-credential", "sk-test");
    let html = page.html();
    let save = tag(&html, "save-credential").expect("save key");
    assert!(!save.contains(" disabled="), "{save}");

    page.click("save-credential");
    let requests = fake::requests::<whispernow_domain::ipc::SetApiKeyRequest>("set_api_key");
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].provider, LlmProvider::OpenAi);
    assert_eq!(requests[0].key.expose_secret(), "sk-test");
}

#[test]
fn a_settings_toggle_is_disabled_during_recording_and_sends_nothing() {
    let mut page = shell(snapshot(crate::test_support::RECORDING), loaded());
    open(&mut page, &["nav-settings"]);
    let html = page.html();
    assert!(
        tag(&html, "start-minimized")
            .expect("toggle")
            .contains(" disabled=")
    );
    assert_eq!(
        text(&html, "settings-disabled-reason"),
        "Wait for the current dictation to finish"
    );

    page.click("start-minimized");
    assert!(sent_settings().is_empty());
    assert!(
        tag(&page.html(), "start-minimized")
            .expect("toggle")
            .contains("aria-checked=\"false\"")
    );

    page.publish(whispernow_domain::ipc::EventPayload::StateChanged {
        state: snapshot(Phase::Idle),
    });
    assert!(
        !tag(&page.html(), "start-minimized")
            .expect("toggle")
            .contains(" disabled=")
    );
    page.click("start-minimized");
    assert_eq!(sent_settings().len(), 1);
    assert!(sent_settings()[0].start_minimized);
}

#[test]
fn settings_controls_on_each_blocked_page_are_disabled() {
    let cases: [(&str, &[&str]); 5] = [
        (
            "nav-settings-general",
            &[
                "recording-mode-toggle",
                "recording-shortcut",
                "cancel-dictation",
                "theme",
                "start-minimized",
                "start-on-login",
            ],
        ),
        ("nav-settings-audio", &["input-device"]),
        (
            "nav-settings-ai-cleanup",
            &[
                "ai-cleanup-enabled",
                "cleanup-source-local",
                "provider-name",
                "provider-model",
                "provider-endpoint",
            ],
        ),
        (
            "nav-settings-vocabulary",
            &[
                "vocabulary-source",
                "vocabulary-replacement",
                "vocabulary-add",
                "vocabulary-rule-1-source",
                "vocabulary-rule-1-replacement",
                "vocabulary-rule-1-delete",
                "vocabulary-import",
            ],
        ),
        (
            "nav-settings-privacy",
            &["history-limit", "history-keep-audio"],
        ),
    ];
    for (route, controls) in cases {
        let mut page = shell(snapshot(crate::test_support::RECORDING), configured());
        open(&mut page, &["nav-settings", route]);
        let html = page.html();
        assert!(tag(&html, "settings-disabled-reason").is_some(), "{route}");
        for control in controls {
            let element = tag(&html, control).unwrap_or_else(|| panic!("{route} {control}"));
            assert!(element.contains(" disabled="), "{route} {element}");
        }
    }
}

#[test]
fn a_settings_toggle_saves_once_and_a_refusal_restores_it_with_the_error_beside_it() {
    let mut page = shell(snapshot(Phase::Idle), loaded());
    open(&mut page, &["nav-settings"]);
    let html = page.html();
    assert!(
        tag(&html, "start-minimized")
            .expect("toggle")
            .contains("aria-checked=\"false\"")
    );

    open(&mut page, &["start-minimized"]);
    let sent = sent_settings();
    assert_eq!(sent.len(), 1, "one click saves once");
    assert!(sent[0].start_minimized);
    let html = page.html();
    assert!(
        tag(&html, "start-minimized")
            .expect("toggle")
            .contains("aria-checked=\"true\""),
        "the switch moves before the desktop answers"
    );
    let status = tag(&html, "start-minimized-status").expect("save status");
    assert!(status.contains("role=\"status\""), "{status}");
    assert!(!text(&html, "start-minimized-status").is_empty());
    assert!(tag(&html, "start-minimized-error").is_none());

    fake::fail("update_settings");
    page.settle();
    let html = page.html();
    let toggle = tag(&html, "start-minimized").expect("toggle");
    assert!(toggle.contains("aria-checked=\"false\""), "{toggle}");
    assert!(
        toggle.contains("start-minimized-error"),
        "the error describes the control: {toggle}"
    );
    let error = tag(&html, "start-minimized-error").expect("error beside the control");
    assert!(error.contains("role=\"alert\""), "{error}");
    assert!(!text(&html, "start-minimized-error").is_empty());
    assert_eq!(sent_settings().len(), 1, "a refusal is not retried");
}

#[test]
fn text_fields_save_when_committed_not_on_each_keystroke() {
    let mut page = shell(snapshot(Phase::Idle), configured());
    open(&mut page, &["nav-settings", "nav-settings-ai-cleanup"]);
    for typed in ["g", "gp", "gpt-4", "gpt-4o"] {
        page.fire("provider-model", "input", typed);
    }
    assert!(sent_settings().is_empty(), "typing saves nothing");
    let html = page.html();
    assert!(
        tag(&html, "provider-model")
            .expect("model field")
            .contains("value=\"gpt-4o\""),
        "the field keeps what was typed"
    );

    page.fire("provider-model", "change", "gpt-4o");
    let sent = sent_settings();
    assert_eq!(sent.len(), 1, "leaving the field saves once");
    assert_eq!(
        sent[0].provider_settings(LlmProvider::OpenAi).model,
        "gpt-4o"
    );
}

#[test]
fn deleting_a_downloaded_model_needs_a_confirmation() {
    let mut page = shell(snapshot(Phase::Idle), configured());
    open(&mut page, &["nav-settings", "nav-settings-models"]);
    let deletes = || fake::count("delete_model");
    let html = page.html();
    assert!(
        tag(&html, "model-sherpa-onnx-whisper-tiny-delete").is_none(),
        "the model in use cannot be deleted"
    );

    open(&mut page, &["model-sherpa-onnx-whisper-base-delete"]);
    assert_eq!(deletes(), 0, "the first press only asks");
    let html = page.html();
    let confirm = tag(&html, "model-sherpa-onnx-whisper-base-delete-confirm").expect("confirm");
    assert!(
        confirm.contains("Base"),
        "the confirmation names the model: {confirm}"
    );

    open(
        &mut page,
        &["model-sherpa-onnx-whisper-base-delete-confirm"],
    );
    assert_eq!(deletes(), 1);
}

#[test]
fn a_cancelled_model_delete_sends_nothing() {
    let mut page = shell(snapshot(Phase::Idle), configured());
    open(
        &mut page,
        &[
            "nav-settings",
            "nav-settings-models",
            "model-sherpa-onnx-whisper-base-delete",
            "model-sherpa-onnx-whisper-base-delete-cancel",
        ],
    );
    assert_eq!(fake::count("delete_model"), 0);
    let html = page.html();
    assert!(tag(&html, "model-sherpa-onnx-whisper-base-delete").is_some());
}

#[test]
fn every_settings_control_names_itself_with_its_visible_label() {
    let mut page = shell(snapshot(Phase::Idle), configured());
    open(&mut page, &["nav-settings"]);
    for section in SETTINGS_PAGES {
        open(&mut page, &[section]);
        let html = page.html();
        let violations = label_in_name_violations(&html);
        assert!(violations.is_empty(), "{section}: {violations:#?}");
    }
}

#[test]
fn repeated_controls_name_the_row_they_act_on() {
    let mut page = shell(snapshot(Phase::Idle), configured());
    open(&mut page, &["nav-settings", "nav-settings-vocabulary"]);
    let html = page.html();
    for (source, replacement) in [("teh", "the"), ("whisper now", "WhisperNow")] {
        let rule_controls = html
            .split("aria-label=\"")
            .skip(1)
            .filter_map(|rest| rest.split('"').next())
            .filter(|name| name.contains(source) && name.contains(replacement))
            .count();
        assert!(
            rule_controls >= 3,
            "move up, move down and delete name {source} → {replacement}"
        );
    }
}

#[test]
fn a_new_vocabulary_rule_is_added_with_enter() {
    let mut page = shell(snapshot(Phase::Idle), configured());
    open(&mut page, &["nav-settings", "nav-settings-vocabulary"]);
    page.fire("vocabulary-source", "input", "recieve");
    page.fire("vocabulary-replacement", "input", "receive");
    assert!(sent_settings().is_empty());
    let html = page.html();
    let form = tag(&html, "vocabulary-add-form").expect("add form");
    assert!(form.starts_with("<form"), "{form}");
    page.fire("vocabulary-add-form", "submit", "");
    let sent = sent_settings();
    assert_eq!(sent.len(), 1);
    assert_eq!(
        sent[0].vocabulary_replacements.last(),
        Some(&VocabularyReplacement::new("recieve", "receive"))
    );
}

#[test]
fn typed_text_survives_a_refused_or_failed_save_until_the_user_reverts() {
    let mut page = shell(snapshot(Phase::Idle), configured());
    open(&mut page, &["nav-settings", "nav-settings-ai-cleanup"]);
    let management = page.context().management;

    // Refused before it is sent.
    page.fire("provider-endpoint", "input", "http://api.example.com");
    page.fire("provider-endpoint", "change", "http://api.example.com");
    assert!(sent_settings().is_empty());
    let html = page.html();
    let field = tag(&html, "provider-endpoint").expect("endpoint field");
    assert!(
        field.contains("value=\"http://api.example.com\""),
        "{field}"
    );
    assert!(field.contains("aria-invalid=\"true\""), "{field}");

    // Taken, then refused by the desktop.
    page.fire("provider-model", "input", "gpt-4o");
    page.fire("provider-model", "change", "gpt-4o");
    let sent = sent_settings();
    assert_eq!(sent.len(), 1);
    fake::fail("update_settings");
    page.settle();
    let html = page.html();
    let field = tag(&html, "provider-model").expect("model field");
    assert!(
        field.contains("value=\"gpt-4o\""),
        "the typed model stays: {field}"
    );
    assert!(field.contains("aria-invalid=\"true\""), "{field}");
    assert_eq!(
        management
            .read()
            .shown_settings()
            .cloned()
            .expect("loaded")
            .provider_settings(LlmProvider::OpenAi)
            .model,
        "gpt-5",
        "the saved model is unchanged"
    );

    page.key("provider-model", Key::Escape);
    let html = page.html();
    assert!(
        tag(&html, "provider-model")
            .expect("model field")
            .contains("value=\"gpt-5\""),
        "Escape goes back to the saved model"
    );
}

#[test]
fn a_refused_provider_keeps_its_fields_shown_and_is_asked_for_again_once_edited() {
    let mut management = configured();
    let mut settings = management.shown_settings().cloned().expect("loaded");
    *settings.provider_settings_mut(LlmProvider::Anthropic) = ProviderSettings::default();
    management.install_settings(SettingsView {
        settings,
        warnings: Vec::new(),
    });
    let mut page = shell(snapshot(Phase::Idle), management);
    open(&mut page, &["nav-settings", "nav-settings-ai-cleanup"]);
    let management = page.context().management;
    let saved = |settings: Settings| {
        Ok::<_, AppError>(SaveSettingsResponse {
            settings: SettingsView {
                settings,
                warnings: Vec::new(),
            },
            warnings: Vec::new(),
        })
    };

    page.fire("provider-name", "change", "anthropic");
    assert_eq!(sent_settings()[0].llm_provider, LlmProvider::Anthropic);
    fake::answer(
        "update_settings",
        Err::<(), _>(AppError::new(
            ErrorCategory::General,
            "invalid_settings",
            "AI cleanup needs a model for Anthropic",
        )),
    );
    page.settle();
    let html = page.html();
    let error = tag(&html, "provider-name-error").expect("the desktop's refusal");
    assert!(error.contains("role=\"alert\""), "{error}");
    assert!(
        tag(&html, "provider-model")
            .expect("the chosen provider's model field")
            .contains("value=\"\""),
        "the fields shown are the chosen provider's"
    );

    page.fire("provider-model", "change", "claude-test");
    let effective = management.read().shown_settings().cloned().expect("loaded");
    assert_eq!(effective.llm_provider, LlmProvider::Anthropic);
    let sent = sent_settings();
    assert_eq!(sent.len(), 2);
    assert_eq!(
        sent[1].provider_settings(LlmProvider::Anthropic).model,
        "claude-test"
    );
    assert_eq!(
        sent[1].llm_provider,
        LlmProvider::OpenAi,
        "the model goes first; the switch is queued behind it"
    );
    fake::answer("update_settings", saved(sent[1].clone()));
    page.settle();
    let sent = sent_settings();
    assert_eq!(sent.len(), 3);
    assert_eq!(sent[2].llm_provider, LlmProvider::Anthropic);
    fake::answer("update_settings", saved(sent[2].clone()));
    page.settle();
    let html = page.html();
    assert!(tag(&html, "provider-name-error").is_none());
}

fn cleanup_model(id: &str, name: &str, availability: ModelAvailability) -> ModelSummary {
    ModelSummary {
        family: whispernow_domain::ModelFamily::Cleanup,
        languages: Vec::new(),
        recommended: false,
        ..model(id, name, availability)
    }
}

/// Configured settings that clean up with the app's own models, two of them listed with the
/// CUDA pack.
fn local_cleanup(small: &str, large: &str, pack: &str) -> ManagementState {
    let mut management = configured();
    let mut settings = management.shown_settings().cloned().expect("loaded");
    settings.llm_provider = LlmProvider::LocalCleanup;
    small.clone_into(
        &mut settings
            .provider_settings_mut(LlmProvider::LocalCleanup)
            .model,
    );
    management.install_settings(SettingsView {
        settings,
        warnings: Vec::new(),
    });
    management.install_models(vec![
        model(
            "sherpa-onnx-whisper-base",
            "Base",
            ModelAvailability::Downloaded,
        ),
        cleanup_model(
            small,
            "Local cleanup model (LFM2.5 350M)",
            ModelAvailability::Downloaded,
        ),
        cleanup_model(
            large,
            "Local cleanup model (MiniCPM5 1B)",
            ModelAvailability::Downloaded,
        ),
        ModelSummary {
            family: whispernow_domain::ModelFamily::CudaPack,
            ..cleanup_model(
                pack,
                "NVIDIA CUDA for cleanup models",
                ModelAvailability::NotInstalled,
            )
        },
    ]);
    management
}

#[test]
fn speech_models_show_speed_and_accuracy_as_bars() {
    let mut page = shell(snapshot(Phase::Idle), configured());
    open(&mut page, &["nav-settings", "nav-settings-models"]);
    let html = page.html();
    assert!(html.contains("aria-label=\"Speed: 5 of 5\""), "{html}");
    assert!(html.contains("aria-label=\"Accuracy: 3 of 5\""));
}

#[test]
fn speed_and_accuracy_bars_explain_themselves_on_hover() {
    let mut management = configured();
    management.models[0].speed = Rating::new(3);
    management.models[0].benchmark = Some(SpeechBenchmark {
        english_wer: 5.39,
        wer_source: WerSource::Leaderboard,
        speed: 20.9,
    });
    management.models[1].benchmark = Some(SpeechBenchmark {
        english_wer: 11.0,
        wer_source: WerSource::Estimated,
        speed: 55.6,
    });
    let mut page = shell(snapshot(Phase::Idle), management);
    open(&mut page, &["nav-settings", "nav-settings-models"]);
    let html = page.html();
    assert!(
        html.contains("aria-label=\"Speed: 3 of 5. 21× real time\""),
        "{html}"
    );
    assert!(html.contains("title=\"21× real time\""), "{html}");
    assert!(html.contains("title=\"English word error rate: 5.39% (Open ASR Leaderboard)\""));
    assert!(html.contains(
        "title=\"English word error rate: about 11.0% (estimated; not on the Open ASR Leaderboard)\""
    ));
}

#[test]
fn one_language_is_named_and_several_open_as_a_list_by_name() {
    let mut management = configured();
    management.models[0].languages = vec!["en".to_owned()];
    management.models[1].languages = ["uk", "de", "en", "xx"].map(str::to_owned).to_vec();
    let mut page = shell(snapshot(Phase::Idle), management);
    open(&mut page, &["nav-settings", "nav-settings-models"]);
    let html = page.html();
    assert!(html.contains("<span>English</span>"), "{html}");
    assert!(html.contains("<summary>4 languages</summary>"), "{html}");
    // Sorted by name; a tag without a name shows as itself.
    assert!(html.contains("English, German, Ukrainian, xx"), "{html}");
}

#[test]
fn ai_cleanup_lists_the_cleanup_models_and_the_cuda_pack_and_saves_the_choice() {
    const SMALL: &str = "whispernow-cleanup-lfm2.5-350m-q8_0";
    const LARGE: &str = "whispernow-cleanup-minicpm5-1b-q8_0";
    const PACK: &str = "llama-cuda-13.4";
    let management = local_cleanup(SMALL, LARGE, PACK);
    assert!(
        management
            .speech_models()
            .all(|model| model.id.as_str() != PACK)
    );
    let mut page = shell(snapshot(Phase::Idle), management);
    open(&mut page, &["nav-settings", "nav-settings-models"]);
    let html = page.html();
    assert!(
        tag(&html, "model-whispernow-cleanup-lfm2-5-350m-q8_0-name").is_none(),
        "cleanup models are not speech models"
    );
    assert!(tag(&html, "cuda-pack-title").is_none());

    open(&mut page, &["nav-settings-ai-cleanup"]);
    let html = page.html();
    assert!(
        tag(&html, "cleanup-source-whispernow")
            .expect("the source choice")
            .contains("aria-pressed=\"true\"")
    );
    assert!(
        tag(&html, "model-whispernow-cleanup-lfm2-5-350m-q8_0-use").is_none(),
        "the chosen model is in use"
    );
    let section = html
        .find("id=\"cuda-pack-title\"")
        .expect("the CUDA section");
    let row = html
        .find("id=\"model-llama-cuda-13-4-name\"")
        .expect("the CUDA row");
    assert!(row > section, "listed under its own section");
    assert!(tag(&html, "model-llama-cuda-13-4-download").is_some());
    assert!(tag(&html, "model-llama-cuda-13-4-use").is_none());

    open(
        &mut page,
        &["model-whispernow-cleanup-minicpm5-1b-q8_0-use"],
    );
    let sent = sent_settings();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].cleanup_model_id(), LARGE);
}

#[test]
fn choosing_a_cleanup_source_picks_its_first_provider_and_shows_its_fields() {
    let mut page = shell(snapshot(Phase::Idle), configured());
    open(&mut page, &["nav-settings", "nav-settings-ai-cleanup"]);
    let html = page.html();
    assert!(
        tag(&html, "cleanup-source-api")
            .expect("the source choice")
            .contains("aria-pressed=\"true\"")
    );
    assert!(tag(&html, "provider-credential").is_some());

    open(&mut page, &["cleanup-source-local"]);
    let sent = sent_settings();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].llm_provider, LlmProvider::Ollama);
    let html = page.html();
    assert!(tag(&html, "provider-model").is_some());
    assert!(
        tag(&html, "provider-credential").is_none(),
        "Ollama needs no key"
    );
}

#[test]
fn the_recording_shortcut_switches_between_hold_and_toggle() {
    let mut page = shell(snapshot(Phase::Idle), configured());
    open(&mut page, &["nav-settings", "nav-settings-general"]);
    let html = page.html();
    assert!(
        tag(&html, "recording-mode-hold")
            .expect("hold")
            .contains("aria-pressed=\"true\"")
    );
    assert!(tag(&html, "hold-to-record").is_none());
    assert!(tag(&html, "overlay-style").is_none());

    open(&mut page, &["recording-mode-toggle"]);
    let sent = sent_settings();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].recording_mode, RecordingMode::Toggle);
    assert_eq!(sent[0].recording_shortcut, shortcut(), "the keys stay");
}

#[test]
fn deleting_a_rule_names_it_and_moves_focus_to_its_neighbour() {
    let mut page = shell(snapshot(Phase::Idle), configured());
    open(&mut page, &["nav-settings", "nav-settings-vocabulary"]);
    for id in ["vocabulary-rule-1", "vocabulary-add-form"] {
        page.mount(id);
    }

    open(&mut page, &["vocabulary-rule-1-delete"]);
    page.settle();
    let sent = sent_settings();
    assert_eq!(sent.len(), 1);
    assert_eq!(
        sent[0].vocabulary_replacements,
        [VocabularyReplacement::new("whisper now", "WhisperNow")]
    );
    let html = page.html();
    assert!(
        tag(&html, "vocabulary-rule-1")
            .expect("remaining rule")
            .contains("WhisperNow")
    );
    assert_eq!(
        focused().as_deref(),
        Some("vocabulary-rule-1"),
        "the remaining rule, now first, takes focus"
    );

    open(&mut page, &["vocabulary-rule-1-delete"]);
    page.settle();
    assert_eq!(
        focused().as_deref(),
        Some("vocabulary-add-form"),
        "with no rule left, focus goes to the form that adds one"
    );
}

#[test]
fn clearing_data_lists_where_it_lives_and_removes_it_after_a_confirmation() {
    let mut page = shell(snapshot(Phase::Idle), loaded());
    open(
        &mut page,
        &["nav-settings", "nav-settings-privacy", "preview-clear-data"],
    );
    // General, where Settings opens, checks how paste works.
    assert_eq!(fake::commands(), ["get_permissions", "data_locations"]);
    fake::answer(
        "data_locations",
        Ok(vec![DataLocation {
            category: ClearDataCategory::History,
            path: "/data/history.sqlite3".into(),
        }]),
    );
    page.settle();
    assert!(page.html().contains("/data/history.sqlite3"));

    open(&mut page, &["execute-clear-data"]);
    assert_eq!(fake::count("clear_data"), 0, "the first press only asks");
    open(&mut page, &["execute-clear-data-confirm"]);
    assert_eq!(
        fake::requests::<Vec<ClearDataCategory>>("clear_data"),
        [ClearDataCategory::ALL.to_vec()]
    );

    fake::answer(
        "clear_data",
        Ok(ClearDataResult {
            items: vec![ClearDataItem {
                category: ClearDataCategory::History,
                path: "/data/history.sqlite3".to_owned(),
                deleted: true,
            }],
            restart_required: false,
        }),
    );
    page.settle();
    let html = page.html();
    assert!(html.contains("Removed"), "{html}");
    for read in [
        "get_state",
        "get_settings",
        "get_setup_state",
        "list_models",
        "get_history",
    ] {
        assert_eq!(
            fake::count(read),
            1,
            "{read}: the window reads everything again"
        );
    }
}

#[test]
fn a_decorated_window_keeps_the_native_frame_and_no_drag_region() {
    let page = shell(snapshot(Phase::Idle), ManagementState::default());
    let html = page.html();

    assert!(!html.contains("custom-chrome"), "{html}");
    assert!(!html.contains("window-close"), "{html}");
    assert!(!html.contains("data-tauri-drag-region"), "{html}");
}
