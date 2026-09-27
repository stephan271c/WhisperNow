//! The main window shell: routes, focus, the Home checklist, notices and disabled reasons.

use std::{cell::RefCell, rc::Rc};

use dioxus::dioxus_core::Mutations;
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
use crate::test_support::{
    click_rendered, focused, form_event_rendered, mount_rendered, use_test_context,
};
use crate::{AppShell, ManagementState, NoticeScope, UiContext, UiState};

pub(super) type Signals = (Signal<UiState>, Signal<ManagementState>);

/// Hands the shell's signals to the test after the first render.
#[derive(Clone, Default)]
pub(super) struct Probe(Rc<RefCell<Option<UiContext>>>);

impl PartialEq for Probe {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Probe {
    pub(super) fn context(&self) -> UiContext {
        (*self.0.borrow()).expect("shell rendered")
    }

    pub(super) fn signals(&self) -> Signals {
        let context = self.context();
        (context.state, context.management)
    }
}

#[component]
fn Shell(snapshot: AppStateDto, management: ManagementState, probe: Probe) -> Element {
    let state = use_signal(move || {
        let mut state = UiState::default();
        state.hydrate(snapshot);
        state
    });
    let management = use_signal(move || management);
    *probe.0.borrow_mut() = Some(use_test_context(state, management));
    rsx! { AppShell {} }
}

pub(super) fn snapshot() -> AppStateDto {
    AppStateDto {
        phase: PhaseDto::Idle,
        selected_model_id: ModelId::new("sherpa-onnx-whisper-tiny").expect("fixture model id"),
        model_state: ModelStateDto::Ready,
        last_transcript: None,
        error: None,
        active_downloads: Vec::new(),
        overlay_style: whispernow_api::OverlayStyle::default(),
        overlay_position: whispernow_api::OverlayPosition::default(),
        setup_test: None,
    }
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

pub(super) fn mount(
    snapshot: AppStateDto,
    management: ManagementState,
) -> (VirtualDom, Mutations, Probe) {
    fake::reset();
    let probe = Probe::default();
    let mut dom = VirtualDom::new_with_props(
        Shell,
        ShellProps {
            snapshot,
            management,
            probe: probe.clone(),
        },
    );
    let mutations = dom.rebuild_to_vec();
    (dom, mutations, probe)
}

/// Clicks `id`, then runs the calls the click started and renders their result.
pub(super) fn click(dom: &mut VirtualDom, rendered: &Mutations, id: &str) -> Mutations {
    let mut mutations = click_rendered(dom, rendered, id);
    dom.process_events();
    mutations.edits.extend(dom.render_immediate_to_vec().edits);
    mutations
}

/// Re-renders after a signal changed or the desktop answered outside an event handler.
pub(super) fn settle(dom: &mut VirtualDom) -> Mutations {
    let mut mutations = Mutations::default();
    for _ in 0..4 {
        dom.process_events();
        mutations.edits.extend(dom.render_immediate_to_vec().edits);
    }
    mutations
}

/// The opening tag of the element rendered with `id`.
pub(super) fn tag<'a>(html: &'a str, id: &str) -> Option<&'a str> {
    let start = html.find(&format!("id=\"{id}\""))?;
    let open = html[..start].rfind('<')?;
    let end = start + html[start..].find('>')?;
    Some(&html[open..=end])
}

/// The text of the element rendered with `id`, without nested markup.
pub(super) fn text(html: &str, id: &str) -> String {
    let opening = tag(html, id).unwrap_or_else(|| panic!("element {id}"));
    let start = html.find(opening).expect("opening tag") + opening.len();
    let name = opening[1..]
        .split(|character: char| character.is_whitespace() || character == '>')
        .next()
        .expect("tag name");
    let end = start
        + html[start..]
            .find(&format!("</{name}>"))
            .expect("closing tag");
    let mut plain = String::new();
    let mut in_tag = false;
    for character in html[start..end].chars() {
        match character {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => plain.push(character),
            _ => {}
        }
    }
    plain
}

fn current_links(html: &str) -> usize {
    html.matches("aria-current=\"page\"").count()
}

#[test]
fn home_is_the_first_page_and_shows_the_setup_checklist_without_redirecting() {
    let (mut dom, rendered, _) = mount(snapshot(), unfinished_setup());
    let html = dioxus_ssr::render(&dom);

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
    mount_rendered(&mut dom, &rendered, "page-heading");
    assert_eq!(focused(), None, "the first page leaves focus alone");

    click(&mut dom, &rendered, "continue-setup");
    settle(&mut dom);
    let html = dioxus_ssr::render(&dom);
    assert_eq!(text(&html, "page-heading"), "Set up WhisperNow");
    assert_eq!(
        focused().as_deref(),
        Some("page-heading"),
        "continuing setup moves focus to its heading"
    );

    let (dom, _, _) = mount(snapshot(), loaded());
    let html = dioxus_ssr::render(&dom);
    assert!(tag(&html, "setup-checklist").is_none(), "setup is complete");
}

#[test]
fn sidebar_links_mark_one_current_page_and_move_focus_to_its_heading() {
    let (mut dom, rendered, _) = mount(snapshot(), loaded());
    let html = dioxus_ssr::render(&dom);
    assert!(
        tag(&html, "nav-settings-general").is_none(),
        "sub-pages are collapsed"
    );
    mount_rendered(&mut dom, &rendered, "page-heading");

    let settings = click(&mut dom, &rendered, "nav-settings");
    settle(&mut dom);
    assert_eq!(focused().as_deref(), Some("page-heading"));
    let html = dioxus_ssr::render(&dom);
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

    click(&mut dom, &settings, "nav-settings-audio");
    settle(&mut dom);
    let html = dioxus_ssr::render(&dom);
    assert_eq!(text(&html, "page-heading"), "Audio");
    assert_eq!(current_links(&html), 1);
    assert_eq!(focused().as_deref(), Some("page-heading"));

    click(&mut dom, &rendered, "nav-history");
    let html = dioxus_ssr::render(&dom);
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
    let (mut dom, rendered, probe) = mount(snapshot(), loaded());
    let (_, mut management) = probe.signals();
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
        settle(&mut dom);
        let html = dioxus_ssr::render(&dom);
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
    settle(&mut dom);
    let html = dioxus_ssr::render(&dom);
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
    let rendered = settle_with(&mut dom, rendered);
    let html = dioxus_ssr::render(&dom);
    assert!(
        tag(&html, "notice")
            .expect("notice")
            .contains("notice-toast")
    );
    click(&mut dom, &rendered, "nav-settings");
    let html = dioxus_ssr::render(&dom);
    assert!(
        tag(&html, "notice")
            .expect("notice")
            .contains("notice-inline")
    );
}

/// Re-renders and keeps the earlier mutations so their elements can still be clicked.
fn settle_with(dom: &mut VirtualDom, mut rendered: Mutations) -> Mutations {
    rendered.edits.extend(settle(dom).edits);
    rendered
}

#[test]
fn test_microphone_runs_once_and_shows_the_measured_level() {
    let (mut dom, rendered, probe) = mount(snapshot(), loaded());
    click(&mut dom, &rendered, "test-microphone");
    let (_, mut management) = probe.signals();
    assert_eq!(
        fake::count("test_microphone"),
        1,
        "the click invokes the desktop test"
    );
    let html = dioxus_ssr::render(&dom);
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
    settle(&mut dom);
    let html = dioxus_ssr::render(&dom);
    let meter = tag(&html, "microphone-level").expect("meter");
    assert!(meter.contains("role=\"meter\""), "{meter}");
    assert!(meter.contains("aria-label=\"Microphone level\""), "{meter}");
    assert!(meter.contains("aria-valuenow=\"46\""), "{meter}");
}

#[test]
fn disabled_controls_explain_why_in_visible_text() {
    let mut dictating = snapshot();
    dictating.phase = crate::test_support::RECORDING;
    let (dom, _, _) = mount(dictating, loaded());
    let html = dioxus_ssr::render(&dom);
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
    let mut failed = snapshot();
    failed.model_state = ModelStateDto::Failed;
    let (mut dom, rendered, _) = mount(failed, loaded());
    let html = dioxus_ssr::render(&dom);
    let status = tag(&html, "home-status").expect("status");
    assert!(status.contains("role=\"alert\""), "{status}");
    assert_eq!(text(&html, "home-status-fix"), "Open speech models");

    click(&mut dom, &rendered, "home-status-fix");
    let html = dioxus_ssr::render(&dom);
    assert_eq!(text(&html, "page-heading"), "Speech models");
    assert!(
        tag(&html, "nav-settings-models")
            .expect("models link")
            .contains("aria-current=\"page\"")
    );
}

#[test]
fn the_latest_transcript_can_be_copied_from_home() {
    let mut transcribed = snapshot();
    transcribed.last_transcript = Some(TranscriptResponse {
        raw_text: "raw".to_owned(),
        final_text: "Meet on Thursday".to_owned(),
        text_output_status: None,
        cleanup_failed: false,
        history_save_failed: false,
    });
    let (mut dom, rendered, _) = mount(transcribed, loaded());
    let html = dioxus_ssr::render(&dom);
    assert_eq!(text(&html, "latest-transcript"), "Meet on Thursday");
    assert!(text(&html, "home-shortcuts-title").contains("Shortcuts"));

    click(&mut dom, &rendered, "copy-latest");
    assert_eq!(
        fake::requests::<String>("write_clipboard"),
        ["Meet on Thursday"]
    );
}

/// Clicks through `ids` in order, keeping every render so later ids can be found.
fn open(dom: &mut VirtualDom, mut rendered: Mutations, ids: &[&str]) -> Mutations {
    for id in ids {
        let next = click(dom, &rendered, id);
        rendered.edits.extend(next.edits);
    }
    rendered
}

/// Fires a form event at `id`, runs the calls it started, and keeps every render.
fn fire(
    dom: &mut VirtualDom,
    mut rendered: Mutations,
    id: &str,
    name: &'static str,
    value: &str,
) -> Mutations {
    let next = form_event_rendered(dom, &rendered, id, name, value);
    rendered.edits.extend(next.edits);
    rendered.edits.extend(settle(dom).edits);
    rendered
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
    let (mut dom, rendered, _) = mount(snapshot(), loaded());
    let rendered = open(&mut dom, rendered, &["nav-settings"]);
    let html = dioxus_ssr::render(&dom);
    assert!(
        tag(&html, "start-minimized")
            .expect("toggle")
            .contains("aria-checked=\"false\"")
    );

    let rendered = open(&mut dom, rendered, &["start-minimized"]);
    let sent = sent_settings();
    assert_eq!(sent.len(), 1, "one click saves once");
    assert!(sent[0].start_minimized);
    let html = dioxus_ssr::render(&dom);
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
    settle_with(&mut dom, rendered);
    let html = dioxus_ssr::render(&dom);
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
    let (mut dom, rendered, probe) = mount(snapshot(), configured());
    let mut rendered = open(
        &mut dom,
        rendered,
        &["nav-settings", "nav-settings-ai-cleanup"],
    );
    for typed in ["g", "gp", "gpt-4", "gpt-4o"] {
        rendered = fire(&mut dom, rendered, "provider-model", "input", typed);
    }
    assert!(sent_settings().is_empty(), "typing saves nothing");
    let html = dioxus_ssr::render(&dom);
    assert!(
        tag(&html, "provider-model")
            .expect("model field")
            .contains("value=\"gpt-4o\""),
        "the field keeps what was typed"
    );

    rendered = fire(&mut dom, rendered, "provider-model", "change", "gpt-4o");
    let sent = sent_settings();
    assert_eq!(sent.len(), 1, "leaving the field saves once");
    assert_eq!(
        sent[0].provider_settings(LlmProvider::OpenAi).model,
        "gpt-4o"
    );

    // Instructions are multi-line and save the same way, queued behind the model.
    let (_, management) = probe.signals();
    for typed in ["F", "Fi", "Fix"] {
        rendered = fire(&mut dom, rendered, "prompt-0-instructions", "input", typed);
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
    fire(&mut dom, rendered, "prompt-0-instructions", "change", "Fix");
    assert_eq!(prompt(), "Fix");
    assert_eq!(
        sent_settings().len(),
        1,
        "one change at a time goes to the desktop"
    );
}

#[test]
fn deleting_a_downloaded_model_needs_a_confirmation() {
    let (mut dom, rendered, _) = mount(snapshot(), configured());
    let rendered = open(&mut dom, rendered, &["nav-settings", "nav-settings-models"]);
    let deletes = || fake::count("delete_model");
    let html = dioxus_ssr::render(&dom);
    assert!(
        tag(&html, "model-sherpa-onnx-whisper-tiny-delete").is_none(),
        "the model in use cannot be deleted"
    );

    let rendered = open(
        &mut dom,
        rendered,
        &["model-sherpa-onnx-whisper-base-delete"],
    );
    assert_eq!(deletes(), 0, "the first press only asks");
    let html = dioxus_ssr::render(&dom);
    let confirm = tag(&html, "model-sherpa-onnx-whisper-base-delete-confirm").expect("confirm");
    assert!(
        confirm.contains("Base"),
        "the confirmation names the model: {confirm}"
    );

    open(
        &mut dom,
        rendered,
        &["model-sherpa-onnx-whisper-base-delete-confirm"],
    );
    assert_eq!(deletes(), 1);
}

#[test]
fn a_cancelled_model_delete_sends_nothing() {
    let (mut dom, rendered, _) = mount(snapshot(), configured());
    open(
        &mut dom,
        rendered,
        &[
            "nav-settings",
            "nav-settings-models",
            "model-sherpa-onnx-whisper-base-delete",
            "model-sherpa-onnx-whisper-base-delete-cancel",
        ],
    );
    assert_eq!(fake::count("delete_model"), 0);
    let html = dioxus_ssr::render(&dom);
    assert!(tag(&html, "model-sherpa-onnx-whisper-base-delete").is_some());
}

/// The accessible names of interactive elements that leave out the words they show.
fn label_in_name_violations(html: &str) -> Vec<String> {
    const INTERACTIVE: [&str; 5] = ["button", "a", "input", "textarea", "select"];
    let normalize = |value: &str| {
        value
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    };
    let mut violations = Vec::new();
    let mut offset = 0;
    while let Some(found) = html[offset..].find("aria-label=\"") {
        let at = offset + found;
        offset = at + 1;
        let open = html[..at].rfind('<').expect("tag start");
        let end = at + html[at..].find('>').expect("tag end");
        let opening = &html[open..=end];
        let name_start = at + "aria-label=\"".len();
        let name = &html[name_start..name_start + html[name_start..].find('"').expect("quote")];
        let tag_name = opening[1..]
            .split(|character: char| character.is_whitespace() || character == '>')
            .next()
            .expect("tag name");
        if !INTERACTIVE.contains(&tag_name) {
            continue;
        }
        let visible = if matches!(tag_name, "input" | "textarea" | "select") {
            // A form control shows the text of the label that points at it.
            let Some(id) = opening
                .split("id=\"")
                .nth(1)
                .and_then(|rest| rest.split('"').next())
            else {
                continue;
            };
            let Some(label) = html.find(&format!("for=\"{id}\"")) else {
                continue;
            };
            let start = label + html[label..].find('>').expect("label end") + 1;
            let stop = start + html[start..].find("</label>").expect("label close");
            html[start..stop].to_owned()
        } else {
            let start = end + 1;
            let stop = start
                + html[start..]
                    .find(&format!("</{tag_name}>"))
                    .expect("closing tag");
            let mut plain = String::new();
            let mut in_tag = false;
            for character in html[start..stop].chars() {
                match character {
                    '<' => in_tag = true,
                    '>' => in_tag = false,
                    _ if !in_tag => plain.push(character),
                    _ => {}
                }
            }
            plain
        };
        let visible = normalize(&visible);
        if !visible.is_empty() && !normalize(name).contains(&visible) {
            violations.push(format!("{name:?} hides {visible:?} in {opening}"));
        }
    }
    violations
}

#[test]
fn every_settings_control_names_itself_with_its_visible_label() {
    let (mut dom, rendered, _) = mount(snapshot(), configured());
    let mut rendered = open(&mut dom, rendered, &["nav-settings"]);
    for page in SETTINGS_PAGES {
        rendered = open(&mut dom, rendered, &[page]);
        let html = dioxus_ssr::render(&dom);
        let violations = label_in_name_violations(&html);
        assert!(violations.is_empty(), "{page}: {violations:#?}");
    }
}

#[test]
fn repeated_controls_name_the_row_they_act_on() {
    let (mut dom, rendered, _) = mount(snapshot(), configured());
    open(
        &mut dom,
        rendered,
        &["nav-settings", "nav-settings-vocabulary"],
    );
    let html = dioxus_ssr::render(&dom);
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
    let (mut dom, rendered, _) = mount(snapshot(), configured());
    let rendered = open(
        &mut dom,
        rendered,
        &["nav-settings", "nav-settings-vocabulary"],
    );
    let rendered = fire(&mut dom, rendered, "vocabulary-source", "input", "recieve");
    let rendered = fire(
        &mut dom,
        rendered,
        "vocabulary-replacement",
        "input",
        "receive",
    );
    assert!(sent_settings().is_empty());
    let html = dioxus_ssr::render(&dom);
    let form = tag(&html, "vocabulary-add-form").expect("add form");
    assert!(form.starts_with("<form"), "{form}");
    fire(&mut dom, rendered, "vocabulary-add-form", "submit", "");
    let sent = sent_settings();
    assert_eq!(sent.len(), 1);
    assert_eq!(
        sent[0].vocabulary_replacements.last(),
        Some(&VocabularyReplacement::new("recieve", "receive"))
    );
}

#[test]
fn typed_text_survives_a_refused_or_failed_save_until_the_user_reverts() {
    let (mut dom, rendered, probe) = mount(snapshot(), configured());
    let rendered = open(
        &mut dom,
        rendered,
        &["nav-settings", "nav-settings-ai-cleanup"],
    );
    let (_, management) = probe.signals();

    // Refused before it is sent.
    let rendered = fire(&mut dom, rendered, "llm-timeout", "input", "soon");
    let rendered = fire(&mut dom, rendered, "llm-timeout", "change", "soon");
    assert!(sent_settings().is_empty());
    let html = dioxus_ssr::render(&dom);
    let field = tag(&html, "llm-timeout").expect("time limit field");
    assert!(field.contains("value=\"soon\""), "{field}");
    assert!(field.contains("aria-invalid=\"true\""), "{field}");

    // Taken, then refused by the desktop.
    let rendered = fire(&mut dom, rendered, "provider-model", "input", "gpt-4o");
    let rendered = fire(&mut dom, rendered, "provider-model", "change", "gpt-4o");
    let sent = sent_settings();
    assert_eq!(sent.len(), 1);
    fake::fail("update_settings");
    let rendered = settle_with(&mut dom, rendered);
    let html = dioxus_ssr::render(&dom);
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

    let mut rendered = rendered;
    rendered.edits.extend(
        crate::test_support::key_rendered(&mut dom, &rendered, "provider-model", Key::Escape).edits,
    );
    settle_with(&mut dom, rendered);
    let html = dioxus_ssr::render(&dom);
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
    let (mut dom, rendered, probe) = mount(snapshot(), management);
    let rendered = open(
        &mut dom,
        rendered,
        &["nav-settings", "nav-settings-ai-cleanup"],
    );
    let (_, management) = probe.signals();
    let saved = |settings: Settings| {
        Ok::<_, ClientError>(SaveSettingsResponse {
            settings: SettingsView {
                settings,
                warnings: Vec::new(),
            },
            warnings: Vec::new(),
        })
    };

    let rendered = fire(&mut dom, rendered, "provider-name", "change", "anthropic");
    assert_eq!(sent_settings()[0].llm_provider, LlmProvider::Anthropic);
    fake::answer(
        "update_settings",
        Err::<(), _>(ClientError::Api(ApiErrorDto::new(
            ErrorCategoryDto::General,
            "invalid_settings",
            "AI cleanup needs a model for Anthropic",
        ))),
    );
    let rendered = settle_with(&mut dom, rendered);
    let html = dioxus_ssr::render(&dom);
    let error = tag(&html, "provider-name-error").expect("the desktop's refusal");
    assert!(error.contains("role=\"alert\""), "{error}");
    assert!(
        tag(&html, "provider-model")
            .expect("the chosen provider's model field")
            .contains("value=\"\""),
        "the fields shown are the chosen provider's"
    );

    let rendered = fire(
        &mut dom,
        rendered,
        "provider-model",
        "change",
        "claude-test",
    );
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
    let rendered = settle_with(&mut dom, rendered);
    let sent = sent_settings();
    assert_eq!(sent.len(), 3);
    assert_eq!(sent[2].llm_provider, LlmProvider::Anthropic);
    fake::answer("update_settings", saved(sent[2].clone()));
    settle_with(&mut dom, rendered);
    let html = dioxus_ssr::render(&dom);
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
    let (mut dom, rendered, _) = mount(snapshot(), management);
    let rendered = open(
        &mut dom,
        rendered,
        &["nav-settings", "nav-settings-ai-cleanup"],
    );
    let html = dioxus_ssr::render(&dom);
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

    let rendered = fire(
        &mut dom,
        rendered,
        "provider-cleanup-model-choice",
        "change",
        LARGE,
    );
    let sent = sent_settings();
    assert_eq!(sent.len(), 1);
    assert_eq!(
        sent[0].provider_settings(LlmProvider::LocalCleanup).model,
        LARGE
    );
    assert_eq!(sent[0].cleanup_model_id(), LARGE);
    settle_with(&mut dom, rendered);
    let html = dioxus_ssr::render(&dom);
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
    let (mut dom, rendered, _) = mount(snapshot(), management);
    let _rendered = open(&mut dom, rendered, &["nav-settings", "nav-settings-models"]);
    let html = dioxus_ssr::render(&dom);
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
    let (mut dom, rendered, _) = mount(snapshot(), configured());
    let rendered = open(
        &mut dom,
        rendered,
        &["nav-settings", "nav-settings-vocabulary"],
    );
    for id in ["vocabulary-rule-1", "vocabulary-add-form"] {
        mount_rendered(&mut dom, &rendered, id);
    }

    let rendered = open(&mut dom, rendered, &["vocabulary-rule-1-delete"]);
    settle(&mut dom);
    let sent = sent_settings();
    assert_eq!(sent.len(), 1);
    assert_eq!(
        sent[0].vocabulary_replacements,
        [VocabularyReplacement::new("whisper now", "WhisperNow")]
    );
    let html = dioxus_ssr::render(&dom);
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

    open(&mut dom, rendered, &["vocabulary-rule-1-delete"]);
    settle(&mut dom);
    assert_eq!(
        focused().as_deref(),
        Some("vocabulary-add-form"),
        "with no rule left, focus goes to the form that adds one"
    );
}

#[test]
fn prompt_fields_name_the_prompt_they_edit() {
    let (mut dom, rendered, _) = mount(snapshot(), configured());
    open(
        &mut dom,
        rendered,
        &["nav-settings", "nav-settings-ai-cleanup"],
    );
    let html = dioxus_ssr::render(&dom);
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
    let (mut dom, rendered, _) = mount(snapshot(), loaded());
    let rendered = open(
        &mut dom,
        rendered,
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
    let rendered = settle_with(&mut dom, rendered);
    assert!(dioxus_ssr::render(&dom).contains("/data/history.sqlite3"));

    let rendered = open(&mut dom, rendered, &["execute-clear-data"]);
    assert_eq!(fake::count("clear_data"), 0, "the first press only asks");
    let rendered = open(&mut dom, rendered, &["execute-clear-data-confirm"]);
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
    settle_with(&mut dom, rendered);
    let html = dioxus_ssr::render(&dom);
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
    let (dom, _, _) = mount(snapshot(), ManagementState::default());
    let html = dioxus_ssr::render(&dom);

    assert!(!html.contains("custom-chrome"), "{html}");
    assert!(!html.contains("window-close"), "{html}");
    assert!(!html.contains("data-tauri-drag-region"), "{html}");
}
