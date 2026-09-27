use std::{cell::RefCell, rc::Rc};

use dioxus::prelude::*;
use whispernow_api::{
    ApiErrorDto, AppStateDto, ErrorCategoryDto, EventPayload, MainWindowSectionDto, ModelStateDto,
    OpenMainWindowRequest, OverlayPosition, OverlayStyle, PhaseDto, RecordingMode,
    RecoveryActionDto,
};

use serde_json::Value;

use super::super::UiActions;
use super::{RecordingBar, meter_bars};
use crate::client::{ClientError, fake};
use crate::overlay::{OverlayAction, OverlayPhase, format_elapsed};
use crate::test_support::{click_rendered, focused, mount_rendered, use_test_context};
use crate::{AppShell, ManagementState, UiContext, UiState};

#[derive(Clone, Default)]
struct Probe(Rc<RefCell<Option<UiContext>>>);

impl PartialEq for Probe {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Probe {
    fn context(&self) -> UiContext {
        self.0.borrow().expect("surface rendered")
    }
}

#[component]
fn Surface(initial_state: UiState, settings: bool, probe: Probe) -> Element {
    let state = use_signal(move || initial_state);
    let management = use_signal(ManagementState::default);
    *probe.0.borrow_mut() = Some(use_test_context(state, management));
    if settings {
        rsx! { AppShell {} }
    } else {
        rsx! { RecordingBar {} }
    }
}

fn snapshot(phase: PhaseDto) -> AppStateDto {
    AppStateDto {
        phase,
        selected_model_id: whispernow_api::ModelId::new("sherpa-onnx-whisper-tiny")
            .expect("fixture model id"),
        model_state: ModelStateDto::Ready,
        last_transcript: None,
        error: None,
        active_downloads: Vec::new(),
        setup_test: None,
        overlay_style: OverlayStyle::WithLiveText,
        overlay_position: OverlayPosition::Bottom,
    }
}

fn phase_event(phase: PhaseDto) -> EventPayload {
    EventPayload::StateChanged {
        state: snapshot(phase),
    }
}

fn level(rms: f32, peak: f32, elapsed_ms: u64) -> EventPayload {
    EventPayload::AudioLevel {
        rms,
        peak,
        elapsed_ms,
    }
}

fn partial(session: u64, text: &str) -> EventPayload {
    EventPayload::PartialTranscript {
        session,
        committed_text: text.to_owned(),
    }
}

const fn listening(mode: RecordingMode) -> PhaseDto {
    PhaseDto::Recording { mode, session: 1 }
}

const TRANSCRIBING: PhaseDto = PhaseDto::Processing { cleaning_up: false };

/// A failed dictation whose error leads to `action`.
fn failed(action: Option<OverlayAction>) -> PhaseDto {
    let (category, recovery) = match action {
        None => (ErrorCategoryDto::General, None),
        Some(OverlayAction::CopyTranscript) => (
            ErrorCategoryDto::Output,
            Some(RecoveryActionDto::CopyManually),
        ),
        Some(OverlayAction::OpenHistory) => (ErrorCategoryDto::Recognition, None),
        Some(OverlayAction::OpenSettings) => (
            ErrorCategoryDto::General,
            Some(RecoveryActionDto::OpenSettings),
        ),
    };
    PhaseDto::Error {
        error: ApiErrorDto {
            category,
            code: "fixture".to_owned(),
            summary: "The microphone stopped responding".to_owned(),
            recovery,
        },
    }
}

/// The overlay in `phase`; a recording has delivered samples when `levels`.
fn state_with(phase: PhaseDto, levels: bool) -> UiState {
    let mut state = UiState::default();
    let recording = matches!(phase, PhaseDto::Recording { .. });
    state.hydrate(snapshot(phase));
    if recording && levels {
        state.apply_event(level(0.0, 0.0, 0));
    }
    state
}

fn state_in(phase: PhaseDto) -> UiState {
    state_with(phase, true)
}

fn render(state: UiState) -> String {
    dioxus_ssr::render_element(rsx! {
        Surface { initial_state: state, settings: false, probe: Probe::default() }
    })
}

/// The opening tag of the element rendered with `id`.
fn tag<'a>(html: &'a str, id: &str) -> Option<&'a str> {
    let start = html.find(&format!("id=\"{id}\""))?;
    let open = html[..start].rfind('<')?;
    let end = start + html[start..].find('>')?;
    Some(&html[open..=end])
}

/// The text of the one polite live region.
fn announcement(html: &str) -> &str {
    assert_eq!(
        html.matches("aria-live=").count(),
        1,
        "one live region: {html}"
    );
    let region = tag(html, "overlay-status").expect("status line");
    assert!(region.contains("role=\"status\""), "{region}");
    assert!(region.contains("aria-live=\"polite\""), "{region}");
    let start = html.find(region).expect("status line") + region.len();
    &html[start..start + html[start..].find("</p>").expect("closed status line")]
}

fn accessible_name(html: &str, id: &str) -> Option<String> {
    let tag = tag(html, id)?;
    let start = tag.find("aria-label=\"")? + "aria-label=\"".len();
    Some(tag[start..start + tag[start..].find('"')?].to_owned())
}

struct Expected {
    status: &'static str,
    close: Option<&'static str>,
    stop: bool,
    action: Option<&'static str>,
    timer: bool,
    meter: bool,
}

const CAPTURING: Expected = Expected {
    status: "Listening",
    close: Some("Cancel dictation"),
    stop: false,
    action: None,
    timer: true,
    meter: true,
};

const OUTCOME: Expected = Expected {
    status: "The microphone stopped responding",
    close: Some("Dismiss"),
    stop: false,
    action: None,
    timer: false,
    meter: false,
};

fn phase_cases() -> [(PhaseDto, bool, Expected); 10] {
    [
        (
            PhaseDto::Idle,
            true,
            Expected {
                status: "",
                close: None,
                ..OUTCOME
            },
        ),
        (
            listening(RecordingMode::Toggle),
            false,
            Expected {
                status: "Starting…",
                timer: false,
                ..CAPTURING
            },
        ),
        (listening(RecordingMode::Hold), true, CAPTURING),
        (
            listening(RecordingMode::Toggle),
            true,
            Expected {
                stop: true,
                ..CAPTURING
            },
        ),
        (
            TRANSCRIBING,
            true,
            Expected {
                status: "Transcribing…",
                close: Some("Cancel dictation"),
                ..OUTCOME
            },
        ),
        (
            PhaseDto::Processing { cleaning_up: true },
            true,
            Expected {
                status: "Cleaning up…",
                close: Some("Cancel dictation"),
                ..OUTCOME
            },
        ),
        (failed(None), true, OUTCOME),
        (
            failed(Some(OverlayAction::CopyTranscript)),
            true,
            Expected {
                action: Some("Copy"),
                ..OUTCOME
            },
        ),
        (
            failed(Some(OverlayAction::OpenHistory)),
            true,
            Expected {
                action: Some("Open history"),
                ..OUTCOME
            },
        ),
        (
            failed(Some(OverlayAction::OpenSettings)),
            true,
            Expected {
                action: Some("Fix"),
                ..OUTCOME
            },
        ),
    ]
}

#[test]
fn each_phase_renders_its_status_and_only_its_own_actions() {
    for (phase, levels, expected) in phase_cases() {
        let html = render(state_with(phase.clone(), levels));
        assert_eq!(announcement(&html), expected.status, "{phase:?}");
        assert_eq!(
            accessible_name(&html, "overlay-close").as_deref(),
            expected.close,
            "{phase:?}"
        );
        assert_eq!(
            accessible_name(&html, "overlay-stop").as_deref(),
            expected.stop.then_some("Stop and transcribe"),
            "{phase:?}"
        );
        let action = tag(&html, "overlay-action").map(|tag| {
            let start = html.find(tag).expect("action") + tag.len();
            html[start..start + html[start..].find("</button>").expect("closed button")].to_owned()
        });
        assert_eq!(action.as_deref(), expected.action, "{phase:?}");
        assert_eq!(html.contains("role=\"timer\""), expected.timer, "{phase:?}");
        assert_eq!(
            html.matches("<rect").count(),
            if expected.meter { 9 } else { 0 },
            "{phase:?}"
        );
        assert!(
            !html.contains(" hidden"),
            "nothing is hidden from view: {html}"
        );
        assert!(
            !html.contains("style="),
            "the CSP forbids inline styles: {html}"
        );
    }
}

#[test]
fn a_failure_is_visible_and_meter_updates_do_not_reannounce_it() {
    let mut state = state_in(listening(RecordingMode::Hold));
    let before = render(state.clone());
    state.apply_event(level(0.3, 0.6, 1_000));
    state.apply_event(level(0.1, 0.2, 2_000));
    let after = render(state.clone());
    assert_eq!(announcement(&before), announcement(&after));
    assert_ne!(before, after, "the meter and timer did update");

    state.apply_event(phase_event(failed(Some(OverlayAction::CopyTranscript))));
    let failed = render(state);
    assert_eq!(
        announcement(&failed),
        "The microphone stopped responding",
        "the message is the visible status line"
    );
}

#[test]
fn starting_arms_the_meter_until_the_first_samples_arrive() {
    let mut state = state_with(listening(RecordingMode::Toggle), false);
    assert!(render(state.clone()).contains("overlay-bars-arming"));
    assert!(!state.overlay.levels_seen);

    state.apply_event(level(0.2, 0.4, 33));
    assert!(state.overlay.levels_seen);
    assert!(!render(state).contains("overlay-bars-arming"));
}

#[test]
fn levels_jump_up_settle_down_and_stay_bounded_only_while_capturing() {
    let mut state = UiState::default();
    state.apply_event(level(1.0, 1.0, 10));
    assert!(!state.overlay.levels_seen, "no dictation is capturing");

    state.apply_event(phase_event(listening(RecordingMode::Hold)));
    state.apply_event(level(1.0, 1.0, 33));
    let first = state.overlay.levels[8];
    assert!(
        first > 0.75 && first < 1.0,
        "one loud reading nearly fills the bar: {first}"
    );
    for index in 0..12 {
        state.apply_event(level(5.0, 5.0, 66 + index * 33));
    }
    assert_eq!(state.overlay.levels.len(), 9);
    assert!(
        state
            .overlay
            .levels
            .iter()
            .all(|level| (0.0..=1.0).contains(level))
    );
    assert!(
        state.overlay.levels[8] > 0.99,
        "sustained loud speech fills the bar"
    );
    state.apply_event(level(0.0, 0.0, 500));
    let settling = state.overlay.levels[8];
    assert!(
        settling > 0.0 && settling < 0.5,
        "a pause brings the bar down within a reading: {settling}"
    );
    assert_eq!(format_elapsed(state.overlay.elapsed_ms), "0:00");
    state.apply_event(level(0.0, 0.0, 65_000));
    assert_eq!(format_elapsed(state.overlay.elapsed_ms), "1:05");

    state.apply_event(phase_event(TRANSCRIBING));
    let frozen = state.overlay.levels.map(f32::to_bits);
    state.apply_event(level(1.0, 1.0, 70_000));
    assert_eq!(
        state.overlay.levels.map(f32::to_bits),
        frozen,
        "levels stop with the recording"
    );
}

#[test]
fn ordinary_speech_scrolls_in_from_the_right() {
    let mut state = state_in(listening(RecordingMode::Hold));
    for index in 0..6 {
        // About -26 dBFS, a normal speaking level.
        state.apply_event(level(0.05, 0.2, 33 * index));
    }
    let bars = meter_bars(&state.overlay);
    let heights: Vec<f32> = bars.iter().map(|(_, _, height)| *height).collect();
    assert_eq!(heights.len(), 9);
    assert!(
        (0.35 * 20.0..0.65 * 20.0).contains(&heights[8]),
        "normal speech fills about half the bar: {heights:?}"
    );
    assert!(
        heights[..3].iter().all(|height| *height < heights[8]),
        "older bars still show the silence before it: {heights:?}"
    );
}

#[test]
fn a_late_hydration_never_replaces_a_newer_phase() {
    let mut state = UiState::default();
    state.apply_event(phase_event(TRANSCRIBING));
    state.hydrate(snapshot(PhaseDto::Idle));
    assert_eq!(state.overlay.phase, OverlayPhase::Transcribing);
}

#[test]
fn live_text_follows_the_style_and_resets_for_each_recording() {
    let mut state = state_in(listening(RecordingMode::Toggle));
    state.apply_event(partial(1, "first session words"));
    assert_eq!(state.overlay.live_text(), Some("first session words"));
    assert!(render(state.clone()).contains("first session words"));

    let mut minimal = state.clone();
    minimal.overlay.install(&AppStateDto {
        overlay_style: OverlayStyle::Minimal,
        ..snapshot(listening(RecordingMode::Toggle))
    });
    assert_eq!(
        minimal.overlay.live_text(),
        None,
        "the minimal style hides it"
    );

    state.apply_event(phase_event(TRANSCRIBING));
    assert_eq!(
        state.overlay.live_text(),
        None,
        "shown only while listening"
    );
    state.apply_event(phase_event(PhaseDto::Idle));
    state.apply_event(phase_event(PhaseDto::Recording {
        mode: RecordingMode::Toggle,
        session: 2,
    }));
    state.apply_event(partial(1, "a late partial of the previous recording"));
    assert_eq!(state.overlay.live_text(), None);

    let long = (0..100)
        .map(|word| format!("word{word}"))
        .collect::<Vec<_>>()
        .join(" ");
    state.apply_event(partial(2, &long));
    let shown = state.overlay.live_text().expect("live text");
    assert!(shown.len() <= 120, "only the last lines are kept");
    assert!(shown.ends_with("word99"));
    assert!(shown.starts_with("word"), "cut at a word boundary: {shown}");
}

/// A mounted recording bar whose controls can be clicked across re-renders.
struct Bar {
    dom: VirtualDom,
    probe: Probe,
    /// Every edit so far, newest first, so a click finds an element's current id.
    edits: dioxus::dioxus_core::Mutations,
}

impl Bar {
    fn mount(state: UiState) -> Self {
        fake::reset();
        let probe = Probe::default();
        let mut dom = VirtualDom::new_with_props(
            Surface,
            SurfaceProps {
                initial_state: state,
                settings: false,
                probe: probe.clone(),
            },
        );
        let edits = dom.rebuild_to_vec();
        Self { dom, probe, edits }
    }

    fn record(&mut self, mut edits: dioxus::dioxus_core::Mutations) {
        edits.edits.reverse();
        self.edits.edits.splice(0..0, edits.edits);
    }

    fn click(&mut self, target: &str) {
        let edits = click_rendered(&mut self.dom, &self.edits, target);
        self.record(edits);
    }

    fn show(&mut self, event: EventPayload) {
        let mut state = self.probe.context().state;
        state.write().apply_event(event);
        let edits = self.dom.render_immediate_to_vec();
        self.record(edits);
    }

    /// Runs `act` as a control of this bar would.
    fn act(&mut self, act: impl FnOnce(UiActions)) {
        let actions = UiActions::new(self.probe.context());
        self.dom.in_scope(ScopeId::ROOT, || act(actions));
        self.settle();
    }

    /// Answers the oldest waiting `command` and renders the result.
    fn answer(&mut self, command: &str, answer: Result<(), ClientError>) {
        fake::answer(command, answer);
        self.settle();
    }

    /// Runs the calls in flight and renders what they changed.
    fn settle(&mut self) {
        for _ in 0..4 {
            self.dom.process_events();
            let edits = self.dom.render_immediate_to_vec();
            self.record(edits);
        }
    }

    fn announcement(&self) -> String {
        announcement(&dioxus_ssr::render(&self.dom)).to_owned()
    }

    /// Whether the control rendered with `id` accepts a click.
    fn enabled(&self, id: &str) -> bool {
        let html = dioxus_ssr::render(&self.dom);
        !tag(&html, id)
            .unwrap_or_else(|| panic!("no {id} in {html}"))
            .contains("disabled")
    }
}

/// The request of the latest call of `command`, as JSON.
fn last_request(command: &str) -> Value {
    fake::requests::<Value>(command).pop().expect("a call")
}

#[test]
fn controls_send_the_overlay_commands_and_wait_for_them() {
    let open = |section| serde_json::to_value(OpenMainWindowRequest { section }).expect("json");
    let cases = [
        (
            listening(RecordingMode::Toggle),
            "overlay-stop",
            "stop_recording",
            Value::Null,
        ),
        (
            listening(RecordingMode::Hold),
            "overlay-close",
            "cancel_recording",
            Value::Null,
        ),
        (
            TRANSCRIBING,
            "overlay-close",
            "cancel_recording",
            Value::Null,
        ),
        (
            failed(None),
            "overlay-close",
            "dismiss_recording_overlay",
            Value::Null,
        ),
        (
            failed(Some(OverlayAction::CopyTranscript)),
            "overlay-action",
            "copy_last_transcript",
            Value::Null,
        ),
        (
            failed(Some(OverlayAction::OpenHistory)),
            "overlay-action",
            "open_main_window",
            open(MainWindowSectionDto::History),
        ),
        (
            failed(Some(OverlayAction::OpenSettings)),
            "overlay-action",
            "open_main_window",
            open(MainWindowSectionDto::Settings),
        ),
    ];
    for (phase, target, command, request) in cases {
        let mut bar = Bar::mount(state_in(phase.clone()));
        bar.click(target);
        assert_eq!(fake::commands(), [command], "{phase:?} {target}");
        assert_eq!(last_request(command), request, "{phase:?} {target}");
        assert!(
            !bar.enabled(target),
            "{phase:?}: a control waits for its own command"
        );
    }
}

#[test]
fn a_pending_stop_leaves_cancel_and_the_published_failure_usable() {
    let mut bar = Bar::mount(state_in(listening(RecordingMode::Toggle)));
    bar.click("overlay-stop");
    bar.show(phase_event(TRANSCRIBING));
    assert!(bar.enabled("overlay-close"), "cancel stays usable");
    bar.click("overlay-close");
    assert_eq!(fake::commands(), ["stop_recording", "cancel_recording"]);
    assert!(!bar.enabled("overlay-close"), "one cancel at a time");

    bar.show(phase_event(failed(Some(OverlayAction::CopyTranscript))));
    assert!(bar.enabled("overlay-action"));
    assert!(bar.enabled("overlay-close"));
    bar.click("overlay-action");
    assert_eq!(
        fake::commands(),
        ["stop_recording", "cancel_recording", "copy_last_transcript"]
    );
}

#[test]
fn a_late_stop_or_cancel_failure_never_replaces_the_published_outcome() {
    let no_speech = || {
        Err(ClientError::Api(ApiErrorDto {
            category: ErrorCategoryDto::Recognition,
            code: "no_speech".to_owned(),
            summary: "No speech was recognized".to_owned(),
            recovery: None,
        }))
    };
    let fail = |bar: &mut Bar, command: &str| {
        bar.act(|actions| {
            if command == "stop_recording" {
                actions.stop_recording();
            } else {
                actions.cancel_recording();
            }
        });
        bar.answer(command, no_speech());
        let management = bar.probe.context().management;
        assert_eq!(
            management.read().notice,
            None,
            "the overlay reports its own failures"
        );
    };
    for command in ["stop_recording", "cancel_recording"] {
        let mut bar = Bar::mount(state_in(failed(Some(OverlayAction::OpenHistory))));
        let published = bar.announcement();
        fail(&mut bar, command);
        assert_eq!(
            bar.announcement(),
            published,
            "{command}: the outcome is announced once"
        );

        // While the microphone is live, the failure is the only explanation.
        let mut bar = Bar::mount(state_in(listening(RecordingMode::Toggle)));
        let listening = bar.announcement();
        fail(&mut bar, command);
        assert_ne!(bar.announcement(), listening, "{command}");
    }
}

#[test]
fn listening_reads_as_starting_until_the_microphone_delivers_samples() {
    let mut state = UiState::default();
    state.apply_event(phase_event(listening(RecordingMode::Toggle)));
    let armed = render(state.clone());
    assert_eq!(announcement(&armed), "Starting…");
    assert!(!armed.contains("role=\"timer\""));
    assert!(tag(&armed, "overlay-stop").is_none());
    assert_eq!(
        accessible_name(&armed, "overlay-close").as_deref(),
        Some("Cancel dictation")
    );

    state.apply_event(level(0.2, 0.4, 33));
    let live = render(state);
    assert_eq!(
        announcement(&live),
        announcement(&render(state_in(listening(RecordingMode::Toggle))))
    );
    assert_ne!(announcement(&live), "Starting…");
    assert!(live.contains("role=\"timer\""));
    assert!(tag(&live, "overlay-stop").is_some());
}

#[test]
fn a_failed_overlay_action_is_announced_until_the_next_phase() {
    let mut bar = Bar::mount(state_in(failed(Some(OverlayAction::CopyTranscript))));
    bar.click("overlay-action");
    bar.answer(
        "copy_last_transcript",
        Err(ClientError::Api(ApiErrorDto {
            category: ErrorCategoryDto::Output,
            code: "copy_unavailable".to_owned(),
            summary: "The transcript could not be copied; open History to copy it".to_owned(),
            recovery: None,
        })),
    );
    let management = bar.probe.context().management;
    assert_eq!(
        management.read().notice,
        None,
        "the overlay reports its own failures"
    );
    assert_eq!(
        bar.announcement(),
        "The transcript could not be copied; open History to copy it"
    );

    bar.show(phase_event(listening(RecordingMode::Toggle)));
    assert_eq!(bar.announcement(), "Starting…");
}

#[test]
fn an_overlay_navigation_request_opens_its_route_and_moves_focus() {
    let probe = Probe::default();
    let mut idle = UiState::default();
    idle.hydrate(snapshot(PhaseDto::Idle));
    let mut dom = VirtualDom::new_with_props(
        Surface,
        SurfaceProps {
            initial_state: idle,
            settings: true,
            probe: probe.clone(),
        },
    );
    let rendered = dom.rebuild_to_vec();
    mount_rendered(&mut dom, &rendered, "page-heading");
    let selected = |dom: &VirtualDom, id: &str| {
        let html = dioxus_ssr::render(dom);
        tag(&html, &format!("nav-{id}"))
            .expect("navigation item")
            .contains("aria-current=\"page\"")
    };
    assert!(selected(&dom, "home"));
    assert_eq!(focused(), None);

    for (section, id, title) in [
        (MainWindowSectionDto::History, "history", "History"),
        (
            MainWindowSectionDto::Settings,
            "settings-general",
            "General &#38; Shortcuts",
        ),
        (MainWindowSectionDto::History, "history", "History"),
    ] {
        let actions = UiActions::new(probe.context());
        dom.in_scope(ScopeId::ROOT, || {
            actions.apply_event(EventPayload::NavigationRequested { section });
        });
        // Re-render, run the effect that follows the request, then the focus it moves.
        for _ in 0..3 {
            dom.process_events();
            dom.render_immediate_to_vec();
        }
        assert!(selected(&dom, id), "{section:?}");
        let html = dioxus_ssr::render(&dom);
        assert!(
            html.contains(&format!("id=\"page-heading\" tabindex=\"-1\">{title}</h1>")),
            "{section:?}: {html}"
        );
        assert_eq!(
            focused().as_deref(),
            Some("page-heading"),
            "{section:?}: each followed request moves focus to the heading"
        );
    }
}
