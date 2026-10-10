use dioxus::prelude::*;
use whispernow_api::{
    ApiErrorDto, ErrorCategoryDto, EventPayload, MainWindowSectionDto, OpenMainWindowRequest,
    PhaseDto, RecordingMode, RecoveryActionDto,
};

use serde_json::Value;

use super::{RecordingBar, meter_bars};
use crate::client::{ClientError, fake};
use crate::overlay::{OverlayAction, OverlayPhase, format_elapsed};
use crate::test_support::{Page, Probe, focused, snapshot, tag};
use crate::{AppShell, ManagementState, UiState};

#[component]
fn Surface(initial_state: UiState, settings: bool, probe: Probe) -> Element {
    let state = use_signal(move || initial_state);
    let management = use_signal(ManagementState::default);
    probe.provide(state, management);
    if settings {
        rsx! { AppShell {} }
    } else {
        rsx! { RecordingBar {} }
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
        Some(OverlayAction::OpenAiCleanup) => (
            ErrorCategoryDto::Cleanup,
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
    assert!(first > 0.0, "one loud reading shows at once: {first}");
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
    let loud = state.overlay.levels[8];
    assert!(loud >= first, "sustained loud speech raises the bar");
    state.apply_event(level(0.0, 0.0, 500));
    let settling = state.overlay.levels[8];
    assert!(
        settling > 0.0 && settling < loud,
        "a pause brings the bar down without dropping it: {settling}"
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
fn ordinary_speech_scrolls_in_from_the_right_below_loud_speech() {
    let heights = |rms: f32, peak: f32| {
        let mut state = state_in(listening(RecordingMode::Hold));
        for index in 0..6 {
            state.apply_event(level(rms, peak, 33 * index));
        }
        let heights: Vec<f32> = meter_bars(&state.overlay)
            .iter()
            .map(|bar| bar.height)
            .collect();
        heights
    };
    // About -26 dBFS, a normal speaking level.
    let ordinary = heights(0.05, 0.2);
    let loud = heights(1.0, 1.0);
    assert_eq!(ordinary.len(), 9);
    assert!(
        ordinary[..3].iter().all(|height| *height < ordinary[8]),
        "older bars still show the silence before it: {ordinary:?}"
    );
    assert!(
        ordinary[8] < loud[8],
        "normal speech leaves room above it: {ordinary:?} {loud:?}"
    );
}

#[test]
fn a_late_hydration_never_replaces_a_newer_phase() {
    let mut state = UiState::default();
    state.apply_event(phase_event(TRANSCRIBING));
    state.hydrate(snapshot(PhaseDto::Idle));
    assert_eq!(state.overlay.phase, OverlayPhase::Transcribing);
}

fn mount_bar(state: UiState) -> Page {
    Page::new(|probe| {
        VirtualDom::new_with_props(
            Surface,
            SurfaceProps {
                initial_state: state,
                settings: false,
                probe,
            },
        )
    })
}

/// Applies `event` to the bar's state, as the event listener does, and renders.
fn show(bar: &mut Page, event: EventPayload) {
    let mut state = bar.context().state;
    state.write().apply_event(event);
    bar.settle();
}

fn announced(bar: &Page) -> String {
    announcement(&bar.html()).to_owned()
}

fn enabled(bar: &Page, id: &str) -> bool {
    let html = bar.html();
    !tag(&html, id)
        .unwrap_or_else(|| panic!("no {id} in {html}"))
        .contains("disabled")
}

fn last_request(command: &str) -> Value {
    fake::requests::<Value>(command).pop().expect("a call")
}

#[test]
fn a_cleanup_failure_in_the_overlay_opens_ai_cleanup_settings() {
    let phase = PhaseDto::Error {
        error: ApiErrorDto::new(
            ErrorCategoryDto::Cleanup,
            "cleanup_failed",
            "AI cleanup was unavailable, so the transcript was kept as recognized",
        ),
    };
    let mut bar = mount_bar(state_in(phase));
    bar.click("overlay-action");
    assert_eq!(fake::commands(), ["open_main_window"]);
    assert_eq!(
        last_request("open_main_window"),
        serde_json::json!({ "section": "ai_cleanup" })
    );
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
        let mut bar = mount_bar(state_in(phase.clone()));
        bar.click(target);
        assert_eq!(fake::commands(), [command], "{phase:?} {target}");
        assert_eq!(last_request(command), request, "{phase:?} {target}");
        assert!(
            !enabled(&bar, target),
            "{phase:?}: a control waits for its own command"
        );
    }
}

#[test]
fn a_pending_stop_leaves_cancel_and_the_published_failure_usable() {
    let mut bar = mount_bar(state_in(listening(RecordingMode::Toggle)));
    bar.click("overlay-stop");
    show(&mut bar, phase_event(TRANSCRIBING));
    assert!(enabled(&bar, "overlay-close"), "cancel stays usable");
    bar.click("overlay-close");
    assert_eq!(fake::commands(), ["stop_recording", "cancel_recording"]);
    assert!(!enabled(&bar, "overlay-close"), "one cancel at a time");

    show(
        &mut bar,
        phase_event(failed(Some(OverlayAction::CopyTranscript))),
    );
    assert!(enabled(&bar, "overlay-action"));
    assert!(enabled(&bar, "overlay-close"));
    bar.click("overlay-action");
    assert_eq!(
        fake::commands(),
        ["stop_recording", "cancel_recording", "copy_last_transcript"]
    );
}

#[test]
fn a_late_stop_or_cancel_failure_never_replaces_the_published_outcome() {
    let no_speech = || {
        Err::<(), _>(ClientError::Api(ApiErrorDto {
            category: ErrorCategoryDto::Recognition,
            code: "no_speech".to_owned(),
            summary: "No speech was recognized".to_owned(),
            recovery: None,
        }))
    };
    let fail = |bar: &mut Page, command: &str| {
        bar.act(|actions| {
            if command == "stop_recording" {
                actions.stop_recording();
            } else {
                actions.cancel_recording();
            }
        });
        bar.answer(command, no_speech());
        let management = bar.context().management;
        assert_eq!(
            management.read().notice,
            None,
            "the overlay reports its own failures"
        );
    };
    for command in ["stop_recording", "cancel_recording"] {
        let mut bar = mount_bar(state_in(failed(Some(OverlayAction::OpenHistory))));
        let published = announced(&bar);
        fail(&mut bar, command);
        assert_eq!(
            announced(&bar),
            published,
            "{command}: the outcome is announced once"
        );

        // While the microphone is live, the failure is the only explanation.
        let mut bar = mount_bar(state_in(listening(RecordingMode::Toggle)));
        let listening = announced(&bar);
        fail(&mut bar, command);
        assert_ne!(announced(&bar), listening, "{command}");
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
    let mut bar = mount_bar(state_in(failed(Some(OverlayAction::CopyTranscript))));
    bar.click("overlay-action");
    bar.answer(
        "copy_last_transcript",
        Err::<(), _>(ClientError::Api(ApiErrorDto {
            category: ErrorCategoryDto::Output,
            code: "copy_unavailable".to_owned(),
            summary: "The transcript could not be copied; open History to copy it".to_owned(),
            recovery: None,
        })),
    );
    let management = bar.context().management;
    assert_eq!(
        management.read().notice,
        None,
        "the overlay reports its own failures"
    );
    assert_eq!(
        announced(&bar),
        "The transcript could not be copied; open History to copy it"
    );

    show(&mut bar, phase_event(listening(RecordingMode::Toggle)));
    assert_eq!(announced(&bar), "Starting…");
}

#[test]
fn an_overlay_navigation_request_opens_its_route_and_moves_focus() {
    let mut idle = UiState::default();
    idle.hydrate(snapshot(PhaseDto::Idle));
    let mut page = Page::new(|probe| {
        VirtualDom::new_with_props(
            Surface,
            SurfaceProps {
                initial_state: idle,
                settings: true,
                probe,
            },
        )
    });
    page.mount("page-heading");
    let selected = |page: &Page, id: &str| {
        let html = page.html();
        tag(&html, &format!("nav-{id}"))
            .expect("navigation item")
            .contains("aria-current=\"page\"")
    };
    assert!(selected(&page, "home"));
    assert_eq!(focused(), None);

    for (section, id, title) in [
        (MainWindowSectionDto::History, "history", "History"),
        (
            MainWindowSectionDto::Settings,
            "settings-general",
            "General &#38; Shortcuts",
        ),
        (
            MainWindowSectionDto::AiCleanup,
            "settings-ai-cleanup",
            "AI cleanup",
        ),
        (MainWindowSectionDto::History, "history", "History"),
    ] {
        page.publish(EventPayload::NavigationRequested { section });
        assert!(selected(&page, id), "{section:?}");
        let html = page.html();
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
