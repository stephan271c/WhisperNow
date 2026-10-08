use dioxus::prelude::*;
use whispernow_api::{
    ApiErrorDto, AppStateDto, CleanupPrompt, ClearDataCategory, ClearDataRequest, ClearDataResult,
    ClearDataResultItemDto, DataLocationDto, ErrorCategoryDto, HistoryLimit, ListModelsResponse,
    LlmProvider, MicrophoneTestResponse, ModelAvailability, ModelDto, ModelId, ModelStateDto,
    OverlayPosition, OverlayStyle, PhaseDto, ProviderSettings, RelativeRating,
    SaveSettingsResponse, Settings, SettingsView, SetupCheckpoint, SetupStateView, SetupStep,
    Shortcut, ShortcutKey, ShortcutModifier, TranscriptResponse, UpdateSettingsRequest,
    VocabularyReplacement,
};

use crate::client::{ClientError, fake};
use crate::components::ui::Tone;
use crate::test_support::{Page, Probe, focused, label_in_name_violations, snapshot, tag, text};
use crate::{AppShell, ManagementState, NoticeScope, UiState};

#[component]
fn Shell(snapshot: AppStateDto, management: ManagementState, probe: Probe) -> Element {
    let state = use_signal(move || {
        let mut state = UiState::default();
        state.hydrate(snapshot);
        state
    });
    let management = use_signal(move || management);
    probe.provide(state, management);
    rsx! { AppShell {} }
}

pub(super) fn shell(snapshot: AppStateDto, management: ManagementState) -> Page {
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
        hold_shortcut: shortcut(),
        overlay_style: OverlayStyle::WithLiveText,
        overlay_position: OverlayPosition::Bottom,
        first_run_complete: true,
        history_limit: HistoryLimit::Last20,
        prompts: Vec::new(),
        llm_provider_settings: [(
            LlmProvider::OpenAi,
            ProviderSettings {
                model: "gpt-5".to_owned(),
                ..ProviderSettings::default()
            },
        )]
        .into(),
        llm_timeout_seconds: 20,
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
            hold_shortcut: shortcut(),
            hold_shortcut_enabled: true,
            toggle_shortcut: None,
        },
    });
    management
}

fn current_links(html: &str) -> usize {
    html.matches("aria-current=\"page\"").count()
}

#[test]
fn home_is_the_first_page_and_shows_the_setup_checklist_without_redirecting() {
    let mut page = shell(snapshot(PhaseDto::Idle), unfinished_setup());
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

    let page = shell(snapshot(PhaseDto::Idle), loaded());
    let html = page.html();
    assert!(tag(&html, "setup-checklist").is_none(), "setup is complete");
}

#[test]
fn sidebar_links_mark_one_current_page_and_move_focus_to_its_heading() {
    let mut page = shell(snapshot(PhaseDto::Idle), loaded());
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
    let mut page = shell(snapshot(PhaseDto::Idle), loaded());
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
    let mut page = shell(snapshot(PhaseDto::Idle), loaded());
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
    let mut dictating = snapshot(PhaseDto::Idle);
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
    let mut failed = snapshot(PhaseDto::Idle);
    failed.model_state = ModelStateDto::Failed;
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
fn the_latest_transcript_can_be_copied_from_home() {
    let mut transcribed = snapshot(PhaseDto::Idle);
    transcribed.last_transcript = Some(TranscriptResponse {
        raw_text: "raw".to_owned(),
        final_text: "Meet on Thursday".to_owned(),
        text_output_status: None,
        cleanup_failed: false,
        history_save_failed: false,
    });
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
    fake::requests::<UpdateSettingsRequest>("update_settings")
        .into_iter()
        .map(|request| request.settings)
        .collect()
}

pub(super) fn model(id: &str, name: &str, availability: ModelAvailability) -> ModelDto {
    ModelDto {
        id: ModelId::new(id).expect("fixture model id"),
        display_name: name.to_owned(),
        family: whispernow_api::ModelFamily::Whisper,
        availability,
        languages: vec!["en".to_owned()],
        automatic_language_detection: false,
        size_bytes: 75_000_000,
        speed: RelativeRating::High,
        accuracy: RelativeRating::Medium,
        recommended: name == "Base",
    }
}

/// Settings with every section populated: both shortcuts, AI cleanup with a prompt, a saved
/// provider model, a vocabulary rule, kept audio, and three models.
fn configured() -> ManagementState {
    let mut settings = settings();
    settings.toggle_shortcut = Some(Shortcut {
        modifiers: [ShortcutModifier::Alt].into(),
        key: ShortcutKey::T,
    });
    // An edited built-in prompt, so a newer default is offered.
    settings.active_prompt_id = Some("fix_grammar".to_owned());
    settings.prompts = vec![CleanupPrompt {
        id: "fix_grammar".to_owned(),
        title: "Clean up".to_owned(),
        prompt: "Fix punctuation".to_owned(),
        default_hash: None,
    }];
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
    management.install_models(ListModelsResponse {
        models: vec![
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
        ],
    });
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
fn a_settings_toggle_saves_once_and_a_refusal_restores_it_with_the_error_beside_it() {
    let mut page = shell(snapshot(PhaseDto::Idle), loaded());
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
    let mut page = shell(snapshot(PhaseDto::Idle), configured());
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

    // Instructions are multi-line and save the same way, queued behind the model.
    let management = page.context().management;
    for typed in ["F", "Fi", "Fix"] {
        page.fire("prompt-0-instructions", "input", typed);
    }
    let prompt = || {
        management
            .read()
            .shown_settings()
            .cloned()
            .expect("loaded")
            .prompts[0]
            .prompt
            .clone()
    };
    assert_eq!(prompt(), "Fix punctuation", "typing changes nothing yet");
    page.fire("prompt-0-instructions", "change", "Fix");
    assert_eq!(prompt(), "Fix");
    assert_eq!(
        sent_settings().len(),
        1,
        "one change at a time goes to the desktop"
    );
}

#[test]
fn deleting_a_downloaded_model_needs_a_confirmation() {
    let mut page = shell(snapshot(PhaseDto::Idle), configured());
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
    let mut page = shell(snapshot(PhaseDto::Idle), configured());
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
    let mut page = shell(snapshot(PhaseDto::Idle), configured());
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
    let mut page = shell(snapshot(PhaseDto::Idle), configured());
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
    let mut page = shell(snapshot(PhaseDto::Idle), configured());
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
    let mut page = shell(snapshot(PhaseDto::Idle), configured());
    open(&mut page, &["nav-settings", "nav-settings-ai-cleanup"]);
    let management = page.context().management;

    // Refused before it is sent.
    page.fire("llm-timeout", "input", "soon");
    page.fire("llm-timeout", "change", "soon");
    assert!(sent_settings().is_empty());
    let html = page.html();
    let field = tag(&html, "llm-timeout").expect("time limit field");
    assert!(field.contains("value=\"soon\""), "{field}");
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
    let mut page = shell(snapshot(PhaseDto::Idle), management);
    open(&mut page, &["nav-settings", "nav-settings-ai-cleanup"]);
    let management = page.context().management;
    let saved = |settings: Settings| {
        Ok::<_, ClientError>(SaveSettingsResponse {
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
        Err::<(), _>(ClientError::Api(ApiErrorDto::new(
            ErrorCategoryDto::General,
            "invalid_settings",
            "AI cleanup needs a model for Anthropic",
        ))),
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

#[test]
fn the_local_cleanup_provider_offers_each_cleanup_model_and_saves_the_choice() {
    const SMALL: &str = "whispernow-cleanup-lfm2.5-350m-q8_0";
    const LARGE: &str = "whispernow-cleanup-minicpm5-1b-q8_0";
    let cleanup = |id: &str, name: &str, availability| ModelDto {
        family: whispernow_api::ModelFamily::Cleanup,
        languages: Vec::new(),
        recommended: false,
        ..model(id, name, availability)
    };
    let mut management = configured();
    let mut settings = management.shown_settings().cloned().expect("loaded");
    settings.llm_provider = LlmProvider::LocalCleanup;
    settings.active_prompt_id = None;
    management.install_settings(SettingsView {
        settings,
        warnings: Vec::new(),
    });
    management.install_models(ListModelsResponse {
        models: vec![
            model(
                "sherpa-onnx-whisper-base",
                "Base",
                ModelAvailability::Downloaded,
            ),
            cleanup(
                SMALL,
                "Local cleanup model (LFM2.5 350M)",
                ModelAvailability::Downloaded,
            ),
            cleanup(
                LARGE,
                "Local cleanup model (MiniCPM5 1B)",
                ModelAvailability::NotInstalled,
            ),
        ],
    });
    let mut page = shell(snapshot(PhaseDto::Idle), management);
    open(&mut page, &["nav-settings", "nav-settings-ai-cleanup"]);
    let html = page.html();
    let choice = tag(&html, "provider-cleanup-model-choice").expect("the cleanup model picker");
    assert!(choice.contains(&format!("value=\"{SMALL}\"")), "{choice}");
    assert!(
        html.contains("Local cleanup model (MiniCPM5 1B)"),
        "both are offered"
    );
    assert!(
        tag(&html, "provider-cleanup-model-download").is_none(),
        "the default is installed"
    );

    page.fire("provider-cleanup-model-choice", "change", LARGE);
    let sent = sent_settings();
    assert_eq!(sent.len(), 1);
    assert_eq!(
        sent[0].provider_settings(LlmProvider::LocalCleanup).model,
        LARGE
    );
    assert_eq!(sent[0].cleanup_model_id(), LARGE);
    page.settle();
    let html = page.html();
    assert!(
        tag(&html, "provider-cleanup-model-download").is_some(),
        "the larger model offers its download"
    );
}

#[test]
fn the_cuda_pack_is_offered_in_its_own_section_and_never_as_a_speech_model() {
    const PACK: &str = "llama-cuda-13.4";
    let listed = |id: &str, name: &str, family, availability| ModelDto {
        family,
        languages: Vec::new(),
        recommended: false,
        ..model(id, name, availability)
    };
    let mut management = configured();
    management.install_models(ListModelsResponse {
        models: vec![
            model(
                "sherpa-onnx-whisper-base",
                "Base",
                ModelAvailability::Downloaded,
            ),
            listed(
                "whispernow-cleanup-lfm2.5-350m-q8_0",
                "Local cleanup model (LFM2.5 350M)",
                whispernow_api::ModelFamily::Cleanup,
                ModelAvailability::Downloaded,
            ),
            listed(
                PACK,
                "NVIDIA CUDA for cleanup models",
                whispernow_api::ModelFamily::CudaPack,
                ModelAvailability::NotInstalled,
            ),
        ],
    });
    assert!(
        management
            .speech_models()
            .all(|model| model.id.as_str() != PACK)
    );
    let mut page = shell(snapshot(PhaseDto::Idle), management);
    open(&mut page, &["nav-settings", "nav-settings-models"]);
    let html = page.html();
    let section = html
        .find("id=\"cuda-pack-title\"")
        .expect("the CUDA section");
    let row = html
        .find("id=\"model-llama-cuda-13-4-name\"")
        .expect("the CUDA row");
    assert!(row > section, "listed under its own section");
    assert!(tag(&html, "model-llama-cuda-13-4-download").is_some());
    assert!(tag(&html, "model-llama-cuda-13-4-use").is_none());
}

#[test]
fn deleting_a_rule_names_it_and_moves_focus_to_its_neighbour() {
    let mut page = shell(snapshot(PhaseDto::Idle), configured());
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
fn prompt_fields_name_the_prompt_they_edit() {
    let mut page = shell(snapshot(PhaseDto::Idle), configured());
    open(&mut page, &["nav-settings", "nav-settings-ai-cleanup"]);
    let html = page.html();
    for id in ["prompt-0-name", "prompt-0-instructions"] {
        let field = tag(&html, id).expect("prompt field");
        let name = field
            .split("aria-label=\"")
            .nth(1)
            .and_then(|rest| rest.split('"').next())
            .unwrap_or_else(|| panic!("{id} has an accessible name: {field}"));
        assert!(name.contains("Clean up"), "{id} names its prompt: {name}");
    }
    assert!(label_in_name_violations(&html).is_empty());
}

#[test]
fn clearing_data_lists_where_it_lives_and_removes_it_after_a_confirmation() {
    let mut page = shell(snapshot(PhaseDto::Idle), loaded());
    open(
        &mut page,
        &["nav-settings", "nav-settings-privacy", "preview-clear-data"],
    );
    // General, where Settings opens, checks how paste works.
    assert_eq!(fake::commands(), ["get_permissions", "data_locations"]);
    fake::answer(
        "data_locations",
        Ok(vec![DataLocationDto {
            category: ClearDataCategory::History,
            path: "/data/history.sqlite3".to_owned(),
        }]),
    );
    page.settle();
    assert!(page.html().contains("/data/history.sqlite3"));

    open(&mut page, &["execute-clear-data"]);
    assert_eq!(fake::count("clear_data"), 0, "the first press only asks");
    open(&mut page, &["execute-clear-data-confirm"]);
    assert_eq!(
        fake::requests::<ClearDataRequest>("clear_data"),
        [ClearDataRequest {
            categories: ClearDataCategory::ALL.to_vec(),
        }]
    );

    fake::answer(
        "clear_data",
        Ok(ClearDataResult {
            items: vec![ClearDataResultItemDto {
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
    let page = shell(snapshot(PhaseDto::Idle), ManagementState::default());
    let html = page.html();

    assert!(!html.contains("custom-chrome"), "{html}");
    assert!(!html.contains("window-close"), "{html}");
    assert!(!html.contains("data-tauri-drag-region"), "{html}");
}
