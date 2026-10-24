use dioxus::prelude::*;
use whispernow_domain::ipc::{
    ActiveDownload, AppState, CleanupDevice, ClearDataItem, ClearDataResult, DataLocation,
    EventPayload, MainWindowSection, MicrophoneTestResponse, ModelState, ModelSummary, Phase,
    SaveSettingsResponse, SettingsView,
};
use whispernow_domain::{AppError, ErrorCategory};
use whispernow_domain::{
    ClearDataCategory, DownloadPhase, LlmProvider, ModelAvailability, ModelId, ProviderEndpoint,
    ProviderSettings, RecordingMode, SettingChange, Settings, SetupCheckpoint, SetupProgress,
    SetupStep, Shortcut, ShortcutKey, SpeechBenchmark, VocabularyReplacement, WerSource,
};

use crate::client::fake;
use crate::components::ui::Tone;
use crate::test_support::{
    Page, editable_settings, focused, label_in_name_violations, management_with, settings_view,
    snapshot, tag, text,
};
use crate::{ManagementState, NoticeScope, UiState};

pub(super) fn shell(snapshot: AppState, management: ManagementState) -> Page {
    let mut state = UiState::default();
    state.hydrate(snapshot);
    Page::host(state, management, crate::AppShell)
}

pub(super) fn shortcut() -> Shortcut {
    Shortcut::default()
}

pub(super) fn loaded() -> ManagementState {
    let mut management = management_with(editable_settings());
    management.setup = Some(SetupCheckpoint::default());
    management
}

pub(super) fn unfinished_setup(checkpoint: SetupCheckpoint) -> ManagementState {
    let mut management = loaded();
    let mut settings = editable_settings();
    settings.setup = SetupProgress::Unfinished(checkpoint.clone());
    management.install_settings(SettingsView::new(&settings));
    management.setup = Some(checkpoint);
    management
}

fn current_links(html: &str) -> usize {
    html.matches("aria-current=\"page\"").count()
}

#[test]
fn home_is_the_first_page_and_shows_the_setup_checklist_without_redirecting() {
    let checkpoint = SetupCheckpoint {
        step: SetupStep::Model,
        model_id: ModelId::new("sherpa-onnx-whisper-tiny").expect("fixture model id"),
        input_device: None,
        recording_shortcut: shortcut(),
        recording_mode: RecordingMode::Hold,
    };
    let mut page = shell(snapshot(Phase::Idle), unfinished_setup(checkpoint));
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
fn an_overlay_navigation_request_opens_its_route_and_moves_focus() {
    let mut page = shell(snapshot(Phase::Idle), loaded());
    page.mount("page-heading");
    assert_eq!(focused(), None);
    for (section, id, title) in [
        (MainWindowSection::History, "nav-history", "History"),
        (
            MainWindowSection::AiCleanup,
            "nav-settings-ai-cleanup",
            "AI cleanup",
        ),
    ] {
        page.publish(EventPayload::NavigationRequested { section });
        let html = page.html();
        assert!(
            tag(&html, id)
                .expect("navigation item")
                .contains("aria-current=\"page\""),
            "{section:?}"
        );
        assert_eq!(text(&html, "page-heading"), title, "{section:?}");
        assert_eq!(
            focused().as_deref(),
            Some("page-heading"),
            "{section:?}: navigation moves focus to the heading"
        );
    }
}

#[test]
fn global_notices_show_the_message_and_allow_dismissal() {
    let mut page = shell(snapshot(Phase::Idle), loaded());
    page.context().management.write().notify(
        Tone::Warning,
        NoticeScope::Global,
        "Check the microphone",
    );
    page.settle();
    let html = page.html();
    assert!(
        tag(&html, "notice")
            .expect("notice")
            .contains("role=\"alert\"")
    );
    assert_eq!(text(&html, "notice"), "Check the microphone");
    page.click("notice-dismiss");
    assert!(tag(&page.html(), "notice").is_none());
}

#[test]
fn error_toasts_and_notices_beside_controls_leave_on_their_own() {
    for (scope, placement) in [
        (NoticeScope::Global, "notice-toast"),
        (NoticeScope::Home, "notice-inline"),
    ] {
        let mut page = shell(snapshot(Phase::Idle), loaded());
        page.context().management.write().notify(
            Tone::Error,
            scope,
            "The desktop connection failed",
        );
        page.settle();
        assert!(
            tag(&page.html(), "notice")
                .expect("notice")
                .contains(placement),
            "{scope:?}"
        );
        page.wait();
        assert!(tag(&page.html(), "notice").is_none(), "{scope:?}");
    }
}

#[test]
fn test_microphone_runs_once_and_shows_the_measured_level() {
    let mut page = shell(snapshot(Phase::Idle), loaded());
    page.click("test-microphone");
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

    page.answer(
        "test_microphone",
        Ok(MicrophoneTestResponse {
            levels: vec![0.1, 0.3, 0.2],
            peak: 0.46,
        }),
    );
    let html = page.html();
    assert!(
        !tag(&html, "test-microphone")
            .expect("button")
            .contains(" disabled=")
    );
    let meter = tag(&html, "microphone-level").expect("meter");
    assert!(meter.contains("role=\"meter\""), "{meter}");
    assert!(meter.contains("aria-label=\"Microphone level\""), "{meter}");
    assert!(meter.contains("aria-valuenow=\"46\""), "{meter}");
}

#[test]
fn the_level_meter_hides_after_a_moment_and_a_new_test_replaces_it() {
    let mut page = shell(snapshot(Phase::Idle), loaded());
    page.click("test-microphone");
    page.answer(
        "test_microphone",
        Ok(MicrophoneTestResponse {
            levels: vec![0.1, 0.3, 0.2],
            peak: 0.46,
        }),
    );
    assert!(tag(&page.html(), "microphone-level").is_some());

    page.click("test-microphone");
    assert!(
        tag(&page.html(), "microphone-level").is_none(),
        "a new test replaces the shown meter"
    );
    page.answer(
        "test_microphone",
        Ok(MicrophoneTestResponse {
            levels: vec![0.2, 0.4],
            peak: 0.51,
        }),
    );
    assert!(tag(&page.html(), "microphone-level").is_some());
    page.wait();
    assert!(
        tag(&page.html(), "microphone-level").is_none(),
        "the meter hides itself after a moment"
    );
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
        "Stop the current recording first"
    );
}

#[test]
fn home_says_where_cleanup_runs_once_its_model_is_loaded() {
    let home_line = |device: Option<CleanupDevice>| {
        let mut state = snapshot(Phase::Idle);
        state.cleanup_device = device;
        shell(state, loaded()).html()
    };
    assert_eq!(
        text(
            &home_line(Some(CleanupDevice::Gpu {
                name: "NVIDIA GeForce RTX 5070".to_owned(),
            })),
            "home-cleanup-device"
        ),
        "Cleanup runs on the GPU (NVIDIA GeForce RTX 5070)."
    );
    assert_eq!(
        text(&home_line(Some(CleanupDevice::Cpu)), "home-cleanup-device"),
        "Cleanup runs on the CPU."
    );
    assert!(tag(&home_line(None), "home-cleanup-device").is_none());
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
fn copying_the_latest_transcript_confirms_on_its_button() {
    let mut transcribed = snapshot(Phase::Idle);
    transcribed.last_transcript = Some("Meet on Thursday".to_owned());
    let mut page = shell(transcribed, loaded());
    let html = page.html();
    assert_eq!(text(&html, "latest-transcript"), "Meet on Thursday");

    page.click("copy-latest");
    assert_eq!(fake::requests::<String>("copy_text"), ["Meet on Thursday"]);
    page.answer("copy_text", Ok(()));
    assert_eq!(text(&page.html(), "copy-latest").trim(), "Copied");
    assert!(
        page.context().management.read().notice.is_none(),
        "no notice card"
    );
    page.wait();
    assert_eq!(text(&page.html(), "copy-latest").trim(), "Copy");
}

fn open(page: &mut Page, ids: &[&str]) {
    for id in ids {
        page.click(id);
    }
}

/// Every settings change sent to the desktop so far.
fn sent_changes() -> Vec<SettingChange> {
    fake::requests::<SettingChange>("update_settings")
}

/// What the page shows while a save is pending, which is what the desktop saves for its change.
fn shown(page: &Page) -> Settings {
    page.context()
        .management
        .read()
        .shown_settings()
        .cloned()
        .expect("loaded")
}

fn confirm_save(page: &mut Page) -> Settings {
    let saved = shown(page);
    page.answer(
        "update_settings",
        Ok(SaveSettingsResponse {
            settings: SettingsView::new(&saved),
            warnings: Vec::new(),
        }),
    );
    saved
}

pub(super) fn model(id: &str, name: &str, availability: ModelAvailability) -> ModelSummary {
    ModelSummary {
        id: ModelId::new(id).expect("fixture model id"),
        display_name: name.to_owned(),
        family: whispernow_domain::ModelFamily::Whisper,
        availability,
        languages: vec!["en".to_owned()],
        size_bytes: 75_000_000,
        benchmark: None,
        recommended: name == "Base",
    }
}

/// Settings with every section populated: AI cleanup on, a saved provider model, a vocabulary
/// rule, kept audio, and three models.
fn configured() -> ManagementState {
    let mut settings = editable_settings();
    settings.cleanup_enabled = true;
    settings
        .provider_settings_mut(LlmProvider::OpenAi)
        .saved_models = vec!["gpt-5".to_owned(), "gpt-5-mini".to_owned()];
    settings.vocabulary_replacements = vec![
        VocabularyReplacement::new("teh", "the"),
        VocabularyReplacement::new("whisper now", "WhisperNow"),
    ];
    settings.keep_audio = true;
    settings.api_keys = [LlmProvider::OpenAi].into();
    let mut management = management_with(settings);
    management.models = vec![
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
    ];
    management.models_loaded = true;
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
fn a_settings_toggle_becomes_available_when_recording_finishes_then_saves() {
    let mut page = shell(snapshot(crate::test_support::RECORDING), loaded());
    open(&mut page, &["nav-settings"]);
    page.publish(EventPayload::StateChanged {
        state: snapshot(Phase::Idle),
    });
    assert!(
        !tag(&page.html(), "start-minimized")
            .expect("toggle")
            .contains(" disabled=")
    );
    page.click("start-minimized");
    assert_eq!(sent_changes(), [SettingChange::StartMinimized(true)]);
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
                "run-setup-again",
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
                "provider-model-remove",
                "provider-model-name",
                "provider-model-add",
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
            &["history-limit", "history-retention", "history-keep-audio"],
        ),
    ];
    for saving in [false, true] {
        for (route, controls) in cases {
            let phase = if saving {
                Phase::Idle
            } else {
                crate::test_support::RECORDING
            };
            let mut page = shell(snapshot(phase), configured());
            if saving {
                page.act(|actions| {
                    actions.change_setting(SettingChange::StartMinimized(true));
                });
            }
            open(&mut page, &["nav-settings", route]);
            let html = page.html();
            let reason = tag(&html, "settings-disabled-reason").expect(route);
            assert_eq!(
                reason.contains("visually-hidden"),
                saving,
                "only a save in flight stays out of the layout: {route}"
            );
            for control in controls {
                let element = tag(&html, control).unwrap_or_else(|| panic!("{route} {control}"));
                assert!(element.contains(" disabled="), "{route} {element}");
            }
            if saving && route == "nav-settings-ai-cleanup" {
                for control in [
                    "provider-credential",
                    "save-credential",
                    "remove-credential",
                ] {
                    assert!(
                        tag(&html, control)
                            .expect("credential control")
                            .contains(" disabled=")
                    );
                }
            }
        }
    }
}

#[test]
fn a_refused_shortcut_save_blocks_theme_until_it_finishes_then_theme_saves_confirmed_keys() {
    let mut page = shell(snapshot(Phase::Idle), loaded());
    open(&mut page, &["nav-settings"]);
    page.act(|actions| {
        actions.change_setting(SettingChange::RecordingShortcut {
            shortcut: Shortcut {
                key: ShortcutKey::R,
                ..shortcut()
            },
            mode: RecordingMode::Hold,
        });
    });
    assert_eq!(sent_changes().len(), 1);
    assert!(text(&page.html(), "recording-shortcut-keys").contains('R'));
    try_theme(&mut page);
    assert_eq!(
        sent_changes().len(),
        1,
        "a second edit is unavailable while saving"
    );
    page.answer(
        "update_settings",
        Err::<SaveSettingsResponse, _>(AppError::new(
            ErrorCategory::Settings,
            "shortcut_refused",
            "The recording shortcut is taken",
        )),
    );
    assert!(text(&page.html(), "recording-shortcut-keys").contains("Space"));
    assert!(text(&page.html(), "recording-mode-error").contains("shortcut is taken"));
    assert!(
        !tag(&page.html(), "theme")
            .expect("theme")
            .contains(" disabled=")
    );
    page.choose("theme", "Dark");
    let requests = sent_changes();
    assert_eq!(requests.len(), 2);
    assert_eq!(
        requests[1],
        SettingChange::Theme(whispernow_domain::ThemePreference::Dark)
    );
    assert_eq!(shown(&page).recording_shortcut, shortcut());
    confirm_save(&mut page);
    assert!(text(&page.html(), "theme").contains("Dark"));
    assert!(tag(&page.html(), "theme-error").is_none());
    assert!(text(&page.html(), "recording-mode-error").contains("shortcut is taken"));
}

#[test]
fn a_credential_mutation_blocks_document_saves_until_its_confirmed_settings_arrive() {
    let mut page = shell(snapshot(Phase::Idle), configured());
    open(&mut page, &["nav-settings", "nav-settings-ai-cleanup"]);
    page.input("provider-credential", "sk-test")
        .click("save-credential");
    for control in [
        "provider-name",
        "provider-model",
        "provider-endpoint",
        "ai-cleanup-enabled",
    ] {
        assert!(
            tag(&page.html(), control)
                .expect("provider control")
                .contains(" disabled=")
        );
    }
    open(&mut page, &["nav-settings-general"]);
    assert!(
        tag(&page.html(), "theme")
            .expect("theme")
            .contains(" disabled=")
    );
    try_theme(&mut page);
    assert!(sent_changes().is_empty());
    let mut confirmed = editable_settings();
    confirmed.api_keys.insert(LlmProvider::OpenAi);
    page.answer("set_api_key", Ok(SettingsView::new(&confirmed)));
    assert!(
        !tag(&page.html(), "theme")
            .expect("theme")
            .contains(" disabled=")
    );
    page.choose("theme", "Dark");
    assert_eq!(
        sent_changes(),
        [SettingChange::Theme(
            whispernow_domain::ThemePreference::Dark
        )]
    );
    assert!(shown(&page).api_keys.contains(&LlmProvider::OpenAi));
}

#[test]
fn a_settings_reload_blocks_edits_until_it_installs_the_desktops_confirmed_values() {
    let mut page = shell(snapshot(Phase::Idle), loaded());
    open(&mut page, &["nav-settings"]);
    page.act(crate::UiContext::refresh_all);
    assert!(
        tag(&page.html(), "theme")
            .expect("theme")
            .contains(" disabled=")
    );
    try_theme(&mut page);
    assert!(sent_changes().is_empty());
    let mut confirmed = editable_settings();
    confirmed.recording_shortcut.key = ShortcutKey::T;
    confirmed.theme = whispernow_domain::ThemePreference::Light;
    page.answer("get_settings", Ok(SettingsView::new(&confirmed)));
    assert!(
        !tag(&page.html(), "theme")
            .expect("theme")
            .contains(" disabled=")
    );
    page.choose("theme", "Dark");
    assert_eq!(
        sent_changes(),
        [SettingChange::Theme(
            whispernow_domain::ThemePreference::Dark
        )]
    );
    assert_eq!(shown(&page).recording_shortcut.key, ShortcutKey::T);
}

#[test]
fn a_pending_settings_save_disables_speech_model_selection() {
    let mut page = shell(snapshot(Phase::Idle), configured());
    open(
        &mut page,
        &["nav-settings", "start-minimized", "nav-settings-models"],
    );
    let choose = "model-sherpa-onnx-whisper-base-use";
    assert!(
        tag(&page.html(), choose)
            .expect("model choice")
            .contains(" disabled=")
    );
    page.act(|context| {
        context.switch_model(ModelId::new("sherpa-onnx-whisper-base").expect("fixture model"));
    });
    assert_eq!(fake::count("select_model"), 0);
    assert!(
        !tag(&page.html(), "model-sherpa-onnx-whisper-small-download")
            .expect("download")
            .contains(" disabled=")
    );
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
    assert_eq!(
        sent_changes(),
        [SettingChange::StartMinimized(true)],
        "one click saves once"
    );
    let html = page.html();
    assert!(
        tag(&html, "start-minimized")
            .expect("toggle")
            .contains("aria-checked=\"true\""),
        "the switch moves before the desktop answers"
    );
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
    assert_eq!(sent_changes().len(), 1, "a refusal is not retried");

    page.click("start-minimized");
    assert_eq!(
        sent_changes(),
        vec![SettingChange::StartMinimized(true); 2],
        "the next attempt saves once"
    );
    assert!(
        tag(&page.html(), "start-minimized-error").is_none(),
        "a new attempt clears the old error"
    );
}

#[test]
fn text_fields_save_when_committed_not_on_each_keystroke() {
    let mut page = shell(snapshot(Phase::Idle), configured());
    open(&mut page, &["nav-settings", "nav-settings-ai-cleanup"]);
    let endpoint = "https://api.example.com/v1";
    for typed in ["h", "https://", "https://api.example.com", endpoint] {
        page.fire("provider-endpoint", "input", typed);
    }
    assert!(sent_changes().is_empty(), "typing saves nothing");
    let html = page.html();
    assert!(
        tag(&html, "provider-endpoint")
            .expect("endpoint field")
            .contains(&format!("value=\"{endpoint}\"")),
        "the field keeps what was typed"
    );

    page.fire("provider-endpoint", "change", endpoint);
    let sent = sent_changes();
    assert_eq!(sent.len(), 1, "leaving the field saves once");
    let SettingChange::ProviderEndpoint { provider, api_base } = &sent[0] else {
        panic!("{sent:?}");
    };
    assert_eq!(*provider, LlmProvider::OpenAi);
    assert_eq!(
        api_base.as_ref().map(ProviderEndpoint::expose_url),
        Some(endpoint)
    );
}

#[test]
fn the_model_picker_chooses_adds_and_removes_saved_models() {
    let mut page = shell(snapshot(Phase::Idle), configured());
    open(&mut page, &["nav-settings", "nav-settings-ai-cleanup"]);
    let saved = |page: &mut Page| {
        confirm_save(page)
            .provider_settings(LlmProvider::OpenAi)
            .clone()
    };
    assert_eq!(text(&page.html(), "provider-model"), "gpt-5");

    page.choose("provider-model", "gpt-5-mini");
    let openai = saved(&mut page);
    assert_eq!(openai.model.as_deref(), Some("gpt-5-mini"));
    assert_eq!(openai.saved_models, ["gpt-5", "gpt-5-mini"]);

    page.input("provider-model-name", " gpt-5.1 ")
        .fire("provider-model-form", "submit", "");
    let openai = saved(&mut page);
    assert_eq!(
        openai.model.as_deref(),
        Some("gpt-5.1"),
        "an added model is chosen"
    );
    assert_eq!(openai.saved_models, ["gpt-5", "gpt-5-mini", "gpt-5.1"]);
    assert!(
        tag(&page.html(), "provider-model-name")
            .expect("name field")
            .contains("value=\"\"")
    );

    let html = page.html();
    assert!(
        tag(&html, "provider-model-remove")
            .expect("remove")
            .contains("aria-label=\"Remove gpt-5.1\"")
    );
    page.click("provider-model-remove");
    let openai = saved(&mut page);
    assert_eq!(openai.saved_models, ["gpt-5", "gpt-5-mini"]);
    assert_eq!(
        openai.model.as_deref(),
        Some("gpt-5"),
        "the first remaining model is chosen"
    );
    let provider = LlmProvider::OpenAi;
    assert_eq!(
        sent_changes(),
        [
            SettingChange::ChooseProviderModel {
                provider,
                model: "gpt-5-mini".to_owned()
            },
            SettingChange::ChooseProviderModel {
                provider,
                model: "gpt-5.1".to_owned()
            },
            SettingChange::RemoveProviderModel {
                provider,
                model: "gpt-5.1".to_owned()
            },
        ]
    );
}

#[test]
fn saved_models_without_an_active_one_ask_for_a_choice() {
    let mut management = configured();
    let mut settings = management.shown_settings().cloned().expect("loaded");
    let openai = settings.provider_settings_mut(LlmProvider::OpenAi);
    openai.model = None;
    openai.saved_models = vec!["gpt-5-mini".to_owned()];
    management.install_settings(SettingsView::new(&settings));
    let mut page = shell(snapshot(Phase::Idle), management);
    open(&mut page, &["nav-settings", "nav-settings-ai-cleanup"]);
    let html = page.html();
    assert_eq!(text(&html, "provider-model"), "Choose a model");
    assert!(
        tag(&html, "provider-model-remove")
            .expect("remove")
            .contains(" disabled=")
    );
}

fn suggested(html: &str) -> Vec<String> {
    html.match_indices("id=\"provider-model-name-option-")
        .filter_map(|(at, _)| html[at + "id=\"".len()..].split('"').next())
        .map(|option| text(html, option))
        .collect()
}

#[test]
fn the_model_name_field_suggests_the_provider_models_starting_with_the_typed_text() {
    let mut page = shell(snapshot(Phase::Idle), configured());
    open(&mut page, &["nav-settings", "nav-settings-ai-cleanup"]);
    let asked = || fake::requests::<LlmProvider>("list_provider_models");
    assert_eq!(asked(), [LlmProvider::OpenAi]);
    page.answer(
        "list_provider_models",
        Ok(["gpt-6-mini", "gpt-6-nano", "o5-gpt-6"]),
    );
    page.input("provider-model-name", "gpt-6-n");
    assert_eq!(suggested(&page.html()), ["gpt-6-nano"]);

    page.click("provider-model-name-option-0")
        .fire("provider-model-form", "submit", "");
    assert_eq!(
        sent_changes(),
        [SettingChange::ChooseProviderModel {
            provider: LlmProvider::OpenAi,
            model: "gpt-6-nano".to_owned(),
        }]
    );
    confirm_save(&mut page);
    assert_eq!(
        asked(),
        [LlmProvider::OpenAi],
        "a model change asks nothing"
    );

    page.fire("provider-endpoint", "change", "https://proxy.example/v1");
    confirm_save(&mut page);
    assert_eq!(
        asked(),
        [LlmProvider::OpenAi; 2],
        "a saved endpoint is asked again"
    );

    open(&mut page, &["cleanup-source-local"]);
    assert_eq!(
        asked(),
        [
            LlmProvider::OpenAi,
            LlmProvider::OpenAi,
            LlmProvider::Ollama
        ]
    );
}

#[test]
fn a_model_list_that_cannot_load_leaves_the_name_to_be_typed() {
    let mut page = shell(snapshot(Phase::Idle), configured());
    open(&mut page, &["nav-settings", "nav-settings-ai-cleanup"]);
    page.answer(
        "list_provider_models",
        Err::<Vec<String>, _>(AppError::new(
            ErrorCategory::Cleanup,
            "provider_models_unavailable",
            "The provider's models could not be listed",
        )),
    );
    page.input("provider-model-name", "gpt-typed");
    let html = page.html();
    assert!(tag(&html, "provider-model-name-listbox").is_none());
    assert!(
        tag(&html, "notice").is_none(),
        "the failure is not announced"
    );

    page.fire("provider-model-form", "submit", "");
    assert_eq!(
        sent_changes(),
        [SettingChange::ChooseProviderModel {
            provider: LlmProvider::OpenAi,
            model: "gpt-typed".to_owned(),
        }]
    );
}

#[test]
fn deleting_a_downloaded_model_names_it_and_sends_its_id() {
    let mut page = shell(snapshot(Phase::Idle), configured());
    open(&mut page, &["nav-settings", "nav-settings-models"]);
    let html = page.html();
    assert!(
        tag(&html, "model-sherpa-onnx-whisper-tiny-delete").is_none(),
        "the model in use cannot be deleted"
    );

    open(&mut page, &["model-sherpa-onnx-whisper-base-delete"]);
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
    assert_eq!(
        fake::requests::<ModelId>("delete_model"),
        [ModelId::new("sherpa-onnx-whisper-base").expect("fixture model id")]
    );
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
    assert!(sent_changes().is_empty());
    let html = page.html();
    let form = tag(&html, "vocabulary-add-form").expect("add form");
    assert!(form.starts_with("<form"), "{form}");
    page.fire("vocabulary-add-form", "submit", "");
    let sent = sent_changes();
    assert_eq!(sent.len(), 1);
    let SettingChange::Vocabulary(rules) = &sent[0] else {
        panic!("{sent:?}");
    };
    assert_eq!(
        rules.last(),
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
    assert!(sent_changes().is_empty());
    let html = page.html();
    let field = tag(&html, "provider-endpoint").expect("endpoint field");
    assert!(
        field.contains("value=\"http://api.example.com\""),
        "{field}"
    );
    assert!(field.contains("aria-invalid=\"true\""), "{field}");

    // Taken, then refused by the desktop.
    page.fire("provider-endpoint", "input", "https://api.example.com");
    page.fire("provider-endpoint", "change", "https://api.example.com");
    assert_eq!(sent_changes().len(), 1);
    fake::fail("update_settings");
    page.settle();
    let html = page.html();
    let field = tag(&html, "provider-endpoint").expect("endpoint field");
    assert!(
        field.contains("value=\"https://api.example.com\""),
        "the typed endpoint stays: {field}"
    );
    assert!(field.contains("aria-invalid=\"true\""), "{field}");
    assert_eq!(
        management
            .read()
            .shown_settings()
            .cloned()
            .expect("loaded")
            .provider_settings(LlmProvider::OpenAi)
            .api_base,
        None,
        "the saved endpoint is unchanged"
    );

    page.key("provider-endpoint", Key::Escape);
    let html = page.html();
    assert!(
        tag(&html, "provider-endpoint")
            .expect("endpoint field")
            .contains("value=\"\""),
        "Escape goes back to the saved endpoint"
    );
}

#[test]
fn an_incomplete_provider_saves_with_a_warning_and_adding_a_model_repairs_it() {
    let mut management = configured();
    let mut settings = management.shown_settings().cloned().expect("loaded");
    settings.cleanup_enabled = true;
    *settings.provider_settings_mut(LlmProvider::Anthropic) = ProviderSettings::default();
    management.install_settings(SettingsView::new(&settings));
    let mut page = shell(snapshot(Phase::Idle), management);
    open(&mut page, &["nav-settings", "nav-settings-ai-cleanup"]);
    page.choose("provider-name", "Anthropic");
    assert_eq!(
        sent_changes(),
        [SettingChange::ActiveProvider(LlmProvider::Anthropic)]
    );
    confirm_save(&mut page);
    assert!(
        page.html()
            .contains("AI cleanup needs a model for Anthropic")
    );
    assert!(
        tag(&page.html(), "provider-model").is_none(),
        "there is no model to choose"
    );
    page.input("provider-model-name", "claude-test")
        .fire("provider-model-form", "submit", "");
    assert_eq!(
        sent_changes()[1],
        SettingChange::ChooseProviderModel {
            provider: LlmProvider::Anthropic,
            model: "claude-test".to_owned(),
        }
    );
    let saved = confirm_save(&mut page);
    let anthropic = saved.provider_settings(LlmProvider::Anthropic);
    assert_eq!(anthropic.model.as_deref(), Some("claude-test"));
    assert_eq!(anthropic.saved_models, ["claude-test"]);
    assert!(!page.html().contains("AI cleanup needs a model"));
}

#[test]
fn standing_settings_warnings_and_save_warnings_are_shown_together() {
    let mut management = configured();
    let mut settings = management.shown_settings().cloned().expect("loaded");
    settings.cleanup_enabled = true;
    settings.llm_provider = LlmProvider::Anthropic;
    settings
        .provider_settings_mut(LlmProvider::Anthropic)
        .choose_model("claude-initial".to_owned());
    management.install_settings(SettingsView::new(&settings));
    let mut page = shell(snapshot(Phase::Idle), management);
    open(&mut page, &["nav-settings", "nav-settings-ai-cleanup"]);

    page.click("provider-model-remove");
    assert_eq!(
        sent_changes(),
        [SettingChange::RemoveProviderModel {
            provider: LlmProvider::Anthropic,
            model: "claude-initial".to_owned(),
        }]
    );
    let removed = shown(&page);
    assert_eq!(
        removed.provider_settings(LlmProvider::Anthropic).model,
        None
    );
    page.answer(
        "update_settings",
        Ok(SaveSettingsResponse {
            settings: SettingsView::new(&removed),
            warnings: vec!["A native setting could not take effect".to_owned()],
        }),
    );
    let html = page.html();
    assert!(html.contains("AI cleanup needs a model for Anthropic"));
    assert!(text(&html, "notice").contains("Saved, but not everything took effect"));
    assert!(text(&html, "notice").contains("A native setting could not take effect"));

    page.input("provider-model-name", "claude-test")
        .fire("provider-model-form", "submit", "");
    confirm_save(&mut page);
    let html = page.html();
    assert!(!html.contains("AI cleanup needs a model"));
    assert!(text(&html, "notice").contains("A native setting could not take effect"));
}

fn cleanup_model(id: &str, name: &str, availability: ModelAvailability) -> ModelSummary {
    ModelSummary {
        family: whispernow_domain::ModelFamily::Cleanup,
        languages: Vec::new(),
        recommended: false,
        ..model(id, name, availability)
    }
}

/// Configured settings that clean up with the app's own `small` model, `large` also downloaded.
fn local_cleanup(small: &str, large: &str) -> ManagementState {
    let mut management = configured();
    let mut settings = management.shown_settings().cloned().expect("loaded");
    settings.llm_provider = LlmProvider::LocalCleanup;
    settings.cleanup_model = ModelId::new(small).expect("cleanup model id");
    management.install_settings(settings_view(settings));
    management.models = vec![
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
    ];
    management.models_loaded = true;
    management
}

#[test]
fn speech_model_controls_use_download_cancel_and_refresh_the_finished_download() {
    let mut page = shell(snapshot(Phase::Idle), configured());
    open(&mut page, &["nav-settings", "nav-settings-models"]);
    let base = ModelId::new("sherpa-onnx-whisper-base").expect("base model");
    let small = ModelId::new("sherpa-onnx-whisper-small").expect("small model");
    page.click("model-sherpa-onnx-whisper-base-use");
    assert_eq!(
        fake::requests::<ModelId>("select_model").as_slice(),
        std::slice::from_ref(&base)
    );
    page.answer("select_model", Ok(()));
    let mut active = snapshot(Phase::Idle);
    active.selected_model_id = base;
    page.publish(EventPayload::StateChanged {
        state: active.clone(),
    });
    assert!(page.html().contains("In use · Ready"));

    page.click("model-sherpa-onnx-whisper-small-download");
    assert_eq!(
        fake::requests::<ModelId>("download_model").as_slice(),
        std::slice::from_ref(&small)
    );
    page.answer("download_model", Ok(()));
    active.active_downloads.push(ActiveDownload {
        model_id: small.clone(),
        phase: DownloadPhase::Downloading,
        downloaded_bytes: 512,
        total_bytes: 1_024,
        bytes_per_second: None,
    });
    page.publish(EventPayload::StateChanged {
        state: active.clone(),
    });
    assert!(page.html().contains("0.5 KiB of 1.0 KiB"));
    page.click_named("Cancel download Small");
    assert_eq!(
        fake::requests::<ModelId>("cancel_download").as_slice(),
        std::slice::from_ref(&small)
    );
    page.answer("cancel_download", Ok(()));
    active.active_downloads.clear();
    page.publish(EventPayload::StateChanged {
        state: active.clone(),
    });
    assert_eq!(fake::count("list_models"), 0);
    page.publish(EventPayload::ModelsChanged);
    assert_eq!(fake::count("list_models"), 1);
    page.answer(
        "list_models",
        Ok(vec![
            model(
                "sherpa-onnx-whisper-base",
                "Base",
                ModelAvailability::Downloaded,
            ),
            model(
                "sherpa-onnx-whisper-small",
                "Small",
                ModelAvailability::Downloaded,
            ),
        ]),
    );
    assert!(tag(&page.html(), "model-sherpa-onnx-whisper-small-use").is_some());
    assert!(tag(&page.html(), "model-sherpa-onnx-whisper-small-download").is_none());
}

#[test]
fn speed_and_accuracy_bars_explain_themselves_on_hover() {
    let mut management = configured();
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
    assert!(html.contains("aria-label=\"Speed: 3 of 5"));
    assert!(html.contains("aria-label=\"Speed: 5 of 5"));
    assert!(html.contains("aria-label=\"Accuracy: 5 of 5"));
    assert!(html.contains("aria-label=\"Accuracy: 1 of 5"));
    let titles = html
        .split("title=\"")
        .skip(1)
        .filter_map(|title| title.split('"').next())
        .filter(|title| title.contains("English word error rate"))
        .collect::<Vec<_>>();
    assert!(
        titles
            .iter()
            .any(|title| title.contains("Leaderboard") && !title.contains("estimated"))
    );
    assert!(titles.iter().any(|title| title.contains("estimated")));
}

#[test]
fn one_language_is_named_and_several_open_as_a_list_by_name() {
    let mut management = configured();
    management.models.truncate(2);
    management.models[0].languages = vec!["en".to_owned()];
    management.models[1].languages = ["uk", "de", "en", "xx"].map(str::to_owned).to_vec();
    let mut page = shell(snapshot(Phase::Idle), management);
    open(&mut page, &["nav-settings", "nav-settings-models"]);
    let html = page.html();
    let names = text(&html, "main-content");
    assert_eq!(
        names.matches("English").count(),
        2,
        "both models name the language"
    );
    assert_eq!(
        html.matches("<details").count(),
        1,
        "only the multilingual model expands"
    );
    assert!(names.contains("4 languages"));
    let positions = ["English", "German", "Ukrainian", "xx"]
        .map(|name| names.rfind(name).expect("language name"));
    assert!(
        positions.windows(2).all(|pair| pair[0] < pair[1]),
        "languages sort by name: {names}"
    );
}

#[test]
fn ai_cleanup_lists_the_cleanup_models_and_saves_the_choice() {
    const SMALL: &str = "whispernow-cleanup-lfm2.5-350m-r112-q8_0";
    const LARGE: &str = "whispernow-cleanup-minicpm5-1b-r112-q8_0";
    let management = local_cleanup(SMALL, LARGE);
    let mut page = shell(snapshot(Phase::Idle), management);
    open(&mut page, &["nav-settings", "nav-settings-models"]);
    let html = page.html();
    assert!(
        tag(&html, "model-whispernow-cleanup-lfm2-5-350m-r112-q8_0-name").is_none(),
        "cleanup models are not speech models"
    );

    open(&mut page, &["nav-settings-ai-cleanup"]);
    let html = page.html();
    assert!(
        tag(&html, "cleanup-source-whispernow")
            .expect("the source choice")
            .contains("aria-pressed=\"true\"")
    );
    assert!(
        tag(&html, "model-whispernow-cleanup-lfm2-5-350m-r112-q8_0-use").is_none(),
        "the chosen model is in use"
    );
    assert!(
        !html.contains("aria-label=\"Speed:") && !html.contains("aria-label=\"Accuracy:"),
        "cleanup models are not rated"
    );

    open(
        &mut page,
        &["model-whispernow-cleanup-minicpm5-1b-r112-q8_0-use"],
    );
    assert_eq!(
        sent_changes(),
        [SettingChange::CleanupModel(
            ModelId::new(LARGE).expect("fixture model id")
        )]
    );
    let other = "model-whispernow-cleanup-lfm2-5-350m-r112-q8_0-use";
    assert!(
        tag(&page.html(), other)
            .expect("other cleanup model")
            .contains(" disabled=")
    );

    assert!(tag(&page.html(), "cleanup-model-error").is_none());
    fake::fail("update_settings");
    page.settle();
    assert!(
        tag(&page.html(), "cleanup-model-error").is_some(),
        "a refused choice says why"
    );
}

#[test]
fn a_cleanup_model_is_marked_selected_only_once_downloaded() {
    const SMALL: &str = "whispernow-cleanup-lfm2.5-350m-r112-q8_0";
    const LARGE: &str = "whispernow-cleanup-minicpm5-1b-r112-q8_0";
    for availability in [
        ModelAvailability::NotInstalled,
        ModelAvailability::Downloaded,
    ] {
        let mut management = local_cleanup(SMALL, LARGE);
        management.models[1].availability = availability;
        let mut page = shell(snapshot(Phase::Idle), management);
        open(&mut page, &["nav-settings", "nav-settings-ai-cleanup"]);
        assert_eq!(
            page.html().contains(">Selected</span>"),
            availability == ModelAvailability::Downloaded,
            "{availability:?}"
        );
    }
}

#[test]
fn the_gpu_switch_shows_only_for_the_apps_own_models_and_saves() {
    let mut page = shell(snapshot(Phase::Idle), configured());
    open(&mut page, &["nav-settings", "nav-settings-ai-cleanup"]);
    assert!(tag(&page.html(), "cleanup-uses-gpu").is_none());

    let management = local_cleanup(
        "whispernow-cleanup-lfm2.5-350m-r112-q8_0",
        "whispernow-cleanup-minicpm5-1b-r112-q8_0",
    );
    let mut page = shell(snapshot(Phase::Idle), management);
    open(&mut page, &["nav-settings", "nav-settings-ai-cleanup"]);
    assert!(
        tag(&page.html(), "cleanup-uses-gpu")
            .expect("the GPU switch")
            .contains("aria-checked=\"true\"")
    );
    page.click("cleanup-uses-gpu");
    assert_eq!(sent_changes(), [SettingChange::CleanupUsesGpu(false)]);
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
    assert_eq!(
        sent_changes(),
        [SettingChange::ActiveProvider(LlmProvider::Ollama)]
    );
    let html = page.html();
    assert!(tag(&html, "provider-model-name").is_some());
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

    open(&mut page, &["recording-mode-toggle"]);
    assert_eq!(
        sent_changes(),
        [SettingChange::RecordingShortcut {
            shortcut: shortcut(),
            mode: RecordingMode::Toggle,
        }],
        "the keys stay"
    );
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
    assert_eq!(
        sent_changes(),
        [SettingChange::Vocabulary(vec![VocabularyReplacement::new(
            "whisper now",
            "WhisperNow"
        )])]
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

    confirm_save(&mut page);

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
    assert_eq!(fake::count("data_locations"), 1);
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
    open(&mut page, &["execute-clear-data-confirm"]);
    assert_eq!(fake::count("clear_data"), 1);
    let pending = page.html();
    let button = tag(&pending, "execute-clear-data").expect("the remove button");
    assert!(button.contains("disabled"));
    assert!(button.contains(r#"aria-describedby="execute-disabled-reason""#));
    assert!(tag(&pending, "execute-disabled-reason").is_some());

    fake::answer(
        "clear_data",
        Ok(ClearDataResult {
            items: vec![ClearDataItem {
                location: DataLocation {
                    category: ClearDataCategory::History,
                    path: "/data/history.sqlite3".into(),
                },
                deleted: true,
            }],
            restart_required: false,
        }),
    );
    page.settle();
    let html = page.html();
    assert!(html.contains("Removed"), "{html}");
    for read in ["get_state", "get_settings", "list_models", "get_history"] {
        assert_eq!(
            fake::count(read),
            1,
            "{read}: the window reads everything again"
        );
    }
}

/// Asks for the dark theme as the theme select would, for a select that is disabled.
fn try_theme(page: &mut Page) {
    page.act(|actions| {
        actions.change_setting(SettingChange::Theme(
            whispernow_domain::ThemePreference::Dark,
        ));
    });
}
