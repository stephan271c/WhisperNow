use dioxus::prelude::*;
use whispernow_domain::RecordingMode;
use whispernow_domain::ipc::{AppState, EventPayload, MainWindowSection, Phase, RecordingTrigger};
use whispernow_domain::{AppError, ErrorCategory};

use serde_json::Value;

use super::RecordingBar;
use crate::client::fake;
use crate::management::ManagementState;
use crate::overlay::format_elapsed;
use crate::state::UiState;
use crate::test_support::{Page, snapshot, tag};

fn state_changed(state: AppState) -> EventPayload {
    EventPayload::StateChanged { state }
}

fn level(rms: f32, peak: f32, elapsed_ms: u64) -> EventPayload {
    EventPayload::AudioLevel {
        rms,
        peak,
        elapsed_ms,
    }
}

fn listening(mode: RecordingMode) -> AppState {
    snapshot(Phase::Recording {
        trigger: match mode {
            RecordingMode::Hold => RecordingTrigger::Hold,
            RecordingMode::Toggle => RecordingTrigger::Toggle,
        },
        session: 1,
    })
}

const TRANSCRIBING: Phase = Phase::Processing { cleaning_up: false };

fn transcribing() -> AppState {
    snapshot(TRANSCRIBING)
}

fn failed(category: ErrorCategory) -> AppState {
    AppState {
        error: Some(AppError::new(
            category,
            "fixture",
            "The microphone stopped responding",
        )),
        ..snapshot(Phase::Error)
    }
}

/// The overlay showing `shown`; a recording has delivered samples when `levels`.
fn state_with(shown: AppState, levels: bool) -> UiState {
    let mut state = UiState {
        recording_bar: true,
        ..UiState::default()
    };
    let recording = matches!(shown.phase, Phase::Recording { .. });
    state.hydrate(shown);
    if recording && levels {
        state.apply_event(level(0.0, 0.0, 0));
    }
    state
}

fn state_in(shown: AppState) -> UiState {
    state_with(shown, true)
}

fn render(state: UiState) -> String {
    mount_bar(state).html()
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

#[derive(Debug)]
struct RenderedBar {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

fn rendered_bars(html: &str) -> Vec<RenderedBar> {
    let attribute = |tag: &str, name: &str| {
        tag.split(&format!(" {name}=\""))
            .nth(1)
            .expect("meter attribute")
            .split('"')
            .next()
            .expect("attribute value")
            .to_owned()
    };
    let number = |tag: &str, name: &str| {
        attribute(tag, name)
            .parse::<f32>()
            .expect("numeric meter attribute")
    };
    let first_rect = html.find("<rect").expect("meter bars");
    let svg_start = html[..first_rect].rfind("<svg").expect("meter SVG");
    let svg = &html[svg_start..svg_start + html[svg_start..].find('>').expect("SVG tag")];
    let bounds = attribute(svg, "viewBox")
        .split_whitespace()
        .map(|value| value.parse::<f32>().expect("SVG bound"))
        .collect::<Vec<_>>();
    assert_eq!(bounds.len(), 4, "{svg}");
    let bars = html
        .split("<rect")
        .skip(1)
        .map(|rest| {
            let rect = &rest[..rest.find('>').expect("bar tag")];
            RenderedBar {
                x: number(rect, "x"),
                y: number(rect, "y"),
                width: number(rect, "width"),
                height: number(rect, "height"),
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(bars.len(), 9, "{bars:?}");
    for bar in &bars {
        assert!(
            bar.width > 0.0
                && bar.height > 0.0
                && bar.x >= bounds[0]
                && bar.y >= bounds[1]
                && bar.x + bar.width <= bounds[0] + bounds[2]
                && bar.y + bar.height <= bounds[1] + bounds[3],
            "bar stays inside the rendered meter: {bar:?} {bounds:?}"
        );
    }
    assert!(
        bars.windows(2)
            .all(|pair| pair[0].x + pair[0].width <= pair[1].x),
        "bars run from left to right without overlapping: {bars:?}"
    );
    bars
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

fn phase_cases() -> [(AppState, bool, Expected); 9] {
    [
        (
            snapshot(Phase::Idle),
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
            snapshot(Phase::Recording {
                trigger: RecordingTrigger::SetupTest,
                session: 1,
            }),
            true,
            Expected {
                stop: true,
                ..CAPTURING
            },
        ),
        (
            listening(RecordingMode::Toggle),
            true,
            Expected {
                stop: true,
                ..CAPTURING
            },
        ),
        (
            transcribing(),
            true,
            Expected {
                status: "Transcribing…",
                close: Some("Cancel dictation"),
                ..OUTCOME
            },
        ),
        (
            snapshot(Phase::Processing { cleaning_up: true }),
            true,
            Expected {
                status: "Cleaning up…",
                close: Some("Cancel dictation"),
                ..OUTCOME
            },
        ),
        (failed(ErrorCategory::General), true, OUTCOME),
        (
            failed(ErrorCategory::Output),
            true,
            Expected {
                action: Some("Open history"),
                ..OUTCOME
            },
        ),
    ]
}

#[test]
fn each_phase_renders_its_status_and_only_its_own_actions() {
    for (shown, levels, expected) in phase_cases() {
        let phase = (shown.phase.clone(), shown.error.clone());
        let html = render(state_with(shown, levels));
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
            !html.contains("style="),
            "the CSP forbids inline styles: {html}"
        );
    }
}

#[test]
fn meter_updates_do_not_reannounce_the_status() {
    let mut state = state_in(listening(RecordingMode::Hold));
    state.apply_event(level(0.0, 0.0, 1_000));
    let before = render(state.clone());
    state.apply_event(level(0.3, 0.6, 1_000));
    state.apply_event(level(0.1, 0.2, 1_000));
    let after = render(state);
    assert_eq!(announcement(&before), announcement(&after));
    let heights = |html: &str| {
        rendered_bars(html)
            .iter()
            .map(|bar| bar.height.to_bits())
            .collect::<Vec<_>>()
    };
    assert_ne!(
        heights(&before),
        heights(&after),
        "the rendered bars respond while elapsed time is fixed"
    );
}

#[test]
fn levels_jump_up_settle_down_and_stay_bounded_only_while_capturing() {
    let mut state = UiState {
        recording_bar: true,
        ..UiState::default()
    };
    state.apply_event(level(1.0, 1.0, 10));
    assert!(!render(state.clone()).contains("role=\"timer\""));

    state.apply_event(state_changed(listening(RecordingMode::Hold)));
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

    state.apply_event(state_changed(transcribing()));
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
        let heights: Vec<f32> = rendered_bars(&render(state))
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
    let mut state = UiState {
        recording_bar: true,
        ..UiState::default()
    };
    state.apply_event(state_changed(transcribing()));
    state.hydrate(snapshot(Phase::Idle));
    assert_eq!(
        state.snapshot.as_ref().map(|snapshot| &snapshot.phase),
        Some(&TRANSCRIBING)
    );
}

fn mount_bar(state: UiState) -> Page {
    Page::host(state, ManagementState::default(), RecordingBar)
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
fn a_pending_settings_save_leaves_recording_cancellation_available() {
    let mut bar = mount_bar(state_in(snapshot(Phase::Idle)));
    bar.act(|mut actions| {
        actions
            .management
            .write()
            .install_settings(whispernow_domain::ipc::SettingsView::new(
                &crate::test_support::editable_settings(),
            ));
        actions.change_setting(whispernow_domain::SettingChange::Theme(
            whispernow_domain::ThemePreference::Dark,
        ));
    });
    show(&mut bar, state_changed(listening(RecordingMode::Toggle)));
    assert!(bar.context().management.read().settings_pending());
    assert!(enabled(&bar, "overlay-close"));
    bar.click("overlay-close");
    assert_eq!(fake::count("cancel_recording"), 1);
}

#[test]
fn controls_send_the_overlay_commands_and_wait_for_them() {
    let open = |section| serde_json::to_value(section).expect("json");
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
            transcribing(),
            "overlay-close",
            "cancel_recording",
            Value::Null,
        ),
        (
            failed(ErrorCategory::General),
            "overlay-close",
            "dismiss_recording_overlay",
            Value::Null,
        ),
        (
            failed(ErrorCategory::Output),
            "overlay-action",
            "open_main_window",
            open(MainWindowSection::History),
        ),
        (
            failed(ErrorCategory::Recognition),
            "overlay-action",
            "open_main_window",
            open(MainWindowSection::SpeechModels),
        ),
        (
            failed(ErrorCategory::Settings),
            "overlay-action",
            "open_main_window",
            open(MainWindowSection::Settings),
        ),
        (
            failed(ErrorCategory::Cleanup),
            "overlay-action",
            "open_main_window",
            open(MainWindowSection::AiCleanup),
        ),
    ];
    for (shown, target, command, request) in cases {
        let phase = (shown.phase.clone(), shown.error.clone());
        let mut bar = mount_bar(state_in(shown));
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
    show(&mut bar, state_changed(transcribing()));
    assert!(enabled(&bar, "overlay-close"), "cancel stays usable");
    bar.click("overlay-close");
    assert_eq!(fake::commands(), ["stop_recording", "cancel_recording"]);
    assert!(!enabled(&bar, "overlay-close"), "one cancel at a time");

    show(&mut bar, state_changed(failed(ErrorCategory::Output)));
    assert!(enabled(&bar, "overlay-action"));
    assert!(enabled(&bar, "overlay-close"));
    bar.click("overlay-action");
    assert_eq!(
        fake::commands(),
        ["stop_recording", "cancel_recording", "open_main_window"]
    );
}

#[test]
fn a_late_stop_or_cancel_failure_never_replaces_the_published_outcome() {
    let no_speech = || {
        Err::<(), _>(AppError {
            category: ErrorCategory::Recognition,
            code: "no_speech".to_owned(),
            summary: "No speech was recognized".to_owned(),
        })
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
        let mut bar = mount_bar(state_in(failed(ErrorCategory::Recognition)));
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
fn a_failed_overlay_action_is_announced_until_the_next_phase() {
    let mut bar = mount_bar(state_in(failed(ErrorCategory::Output)));
    bar.click("overlay-action");
    bar.answer(
        "open_main_window",
        Err::<(), _>(AppError {
            category: ErrorCategory::Output,
            code: "history_open_failed".to_owned(),
            summary: "History could not be opened".to_owned(),
        }),
    );
    let management = bar.context().management;
    assert_eq!(
        management.read().notice,
        None,
        "the overlay reports its own failures"
    );
    assert_eq!(announced(&bar), "History could not be opened");

    show(&mut bar, state_changed(listening(RecordingMode::Toggle)));
    assert_eq!(announced(&bar), "Starting…");
}

#[test]
fn a_failure_closes_on_its_own_unless_a_newer_phase_came_first() {
    let mut bar = mount_bar(state_in(failed(ErrorCategory::Audio)));
    bar.settle().wait();
    assert_eq!(fake::commands(), ["dismiss_recording_overlay"]);

    let mut bar = mount_bar(state_in(failed(ErrorCategory::General)));
    show(&mut bar, state_changed(listening(RecordingMode::Hold)));
    bar.wait();
    assert!(fake::commands().is_empty(), "a new recording keeps its bar");
    show(&mut bar, state_changed(failed(ErrorCategory::General)));
    bar.wait();
    assert_eq!(fake::commands(), ["dismiss_recording_overlay"]);
}

#[test]
fn a_failure_keeps_one_close_timer_until_it_ends() {
    let mut bar = mount_bar(state_in(listening(RecordingMode::Hold)));
    bar.settle();
    let idle = fake::live_delays();

    show(&mut bar, state_changed(failed(ErrorCategory::Audio)));
    assert_eq!(fake::live_delays(), idle + 1);
    // Losing its button shortens the wait, which starts again.
    show(&mut bar, state_changed(failed(ErrorCategory::General)));
    assert_eq!(fake::live_delays(), idle + 1);
    bar.wait();
    assert_eq!(fake::commands(), ["dismiss_recording_overlay"]);
    assert_eq!(fake::live_delays(), idle);

    show(&mut bar, state_changed(listening(RecordingMode::Hold)));
    show(&mut bar, state_changed(failed(ErrorCategory::General)));
    assert_eq!(fake::live_delays(), idle + 1);
    show(&mut bar, state_changed(listening(RecordingMode::Hold)));
    assert_eq!(fake::live_delays(), idle, "a newer phase cancels the timer");
}
