use dioxus::prelude::*;
use whispernow_domain::ipc::MicrophoneTestResponse;
use whispernow_domain::{RecordingMode, SetupStep, Shortcut};

use super::NoticeRegion;
use super::actions::DisabledReason;
use super::labels;
use super::ui::{
    Banner, Button, ButtonSize, ButtonVariant, EmptyState, Icon, IconName, ShortcutLabel, Tone,
};
use super::updates::UpdateStatus;
use crate::{
    HomeStatus, HomeStatusKind, NoticeScope, Route, SettingsRoute, UiContext, UiOperation,
};

/// The steps the setup checklist lists, each done once setup has moved past it.
const CHECKLIST: [(&str, SetupStep); 4] = [
    ("Allow microphone access", SetupStep::Permissions),
    ("Choose a speech model", SetupStep::Model),
    ("Set a recording shortcut", SetupStep::Shortcuts),
    ("Try a test dictation", SetupStep::TryIt),
];

#[component]
pub(super) fn HomePage() -> Element {
    let context = use_context::<UiContext>();
    let status = HomeStatus::from_state(&context.state.read());
    let transcript = context
        .state
        .read()
        .snapshot
        .as_ref()
        .and_then(|snapshot| snapshot.last_transcript.as_ref())
        .cloned();
    let management = context.management.read();
    let setup_step = management
        .setup
        .as_ref()
        .filter(|setup| !setup.first_run_complete)
        .map(|setup| setup.checkpoint.step);
    let shortcut = management.settings.as_ref().map(|view| {
        let settings = &view.settings;
        (settings.recording_mode, settings.recording_shortcut.clone())
    });
    let test = management.microphone_test.clone();
    drop(management);
    rsx! { div { class: "home-grid",
        if let Some(step) = setup_step {
            SetupChecklist { step }
        }
        UpdateStatus {}
        StatusCard { status }
        ShortcutsCard { shortcut }
        MicrophoneCard { test }
        LatestTranscript { transcript }
    } }
}

/// Setup waits here until the user continues it.
#[component]
fn SetupChecklist(step: SetupStep) -> Element {
    let context = use_context::<UiContext>();
    let navigator = context.navigator;
    rsx! {
        section { class: "setup-checklist-card", aria_labelledby: "setup-checklist-title",
            Banner { tone: Tone::Info,
                h2 { id: "setup-checklist-title", class: "banner-heading", "Finish setting up WhisperNow" }
                ol { id: "setup-checklist", class: "setup-checklist",
                    for (label, checklist_step) in CHECKLIST {
                        li { class: if step > checklist_step { "done" } else { "todo" },
                            Icon { name: if step > checklist_step { IconName::CheckCircle } else { IconName::ArrowRight } }
                            span { "{label}" }
                            span { class: "visually-hidden",
                                if step > checklist_step { " (done)" } else { " (to do)" }
                            }
                        }
                    }
                }
                Button {
                    id: "continue-setup",
                    variant: ButtonVariant::Primary,
                    size: ButtonSize::Sm,
                    onclick: move |_| navigator.go(Route::Setup),
                    "Continue setup"
                }
            }
        }
    }
}

#[component]
fn StatusCard(status: HomeStatus) -> Element {
    let context = use_context::<UiContext>();
    let navigator = context.navigator;
    let dot = match status.kind {
        HomeStatusKind::Ready => "status-dot",
        HomeStatusKind::Recording => "status-dot recording",
        HomeStatusKind::Problem => "status-dot error",
        HomeStatusKind::Paused | HomeStatusKind::Working | HomeStatusKind::Waiting => {
            "status-dot waiting"
        }
    };
    let live = if status.kind == HomeStatusKind::Problem {
        "alert"
    } else {
        "status"
    };
    rsx! {
        article { class: "card home-card", aria_labelledby: "home-status-title",
            h2 { id: "home-status-title", "Dictation" }
            div { id: "home-status", role: live, aria_atomic: "true",
                p { class: "status-line",
                    span { class: dot, aria_hidden: "true" }
                    strong { "{status.title}" }
                }
                if let Some(detail) = &status.detail {
                    p { class: "status-detail", "{detail}" }
                }
            }
            if let Some((label, section)) = status.fix {
                Button {
                    id: "home-status-fix",
                    size: ButtonSize::Sm,
                    onclick: move |_| navigator.go(Route::from_section(section)),
                    "{label}"
                }
            }
        }
    }
}

#[component]
fn ShortcutsCard(shortcut: Option<(RecordingMode, Shortcut)>) -> Element {
    let context = use_context::<UiContext>();
    let navigator = context.navigator;
    rsx! {
        article { class: "card home-card", aria_labelledby: "home-shortcuts-title",
            h2 { id: "home-shortcuts-title", "Shortcuts" }
            match shortcut {
                None => rsx! { p { class: "status-detail", "Loading your shortcuts…" } },
                Some((mode, shortcut)) => rsx! {
                    dl { class: "shortcut-list",
                        div { dt { "{labels::recording_mode(mode)}" } dd { ShortcutLabel { shortcut } } }
                    }
                },
            }
            Button {
                id: "home-change-shortcuts",
                variant: ButtonVariant::Ghost,
                size: ButtonSize::Sm,
                onclick: move |_| navigator.go(Route::Settings(SettingsRoute::General)),
                "Change shortcuts"
            }
        }
    }
}

#[component]
fn MicrophoneCard(test: Option<MicrophoneTestResponse>) -> Element {
    let context = use_context::<UiContext>();
    let blocked = context
        .blocked(UiOperation::TestMicrophone)
        .or(context.phase_blocked());
    let testing = context
        .management
        .read()
        .is_busy(UiOperation::TestMicrophone);
    rsx! {
        article { class: "card home-card", aria_labelledby: "home-microphone-title",
            h2 { id: "home-microphone-title", "Microphone" }
            p { "Listens for three seconds. Nothing is kept." }
            div { class: "microphone-test",
                Button {
                    id: "test-microphone",
                    disabled: blocked.is_some(),
                    aria_describedby: (blocked.is_some() && !testing).then_some("test-microphone-disabled-reason"),
                    onclick: move |_| context.test_microphone(),
                    Icon { name: IconName::Mic }
                    if testing { "Listening…" } else { "Test microphone" }
                }
                if testing {
                    p { class: "status-detail", role: "status", "Say a few words." }
                } else {
                    DisabledReason { id: "test-microphone-disabled-reason", reason: blocked }
                }
            }
            if let Some(test) = test {
                LevelMeter { test }
            }
            NoticeRegion { scope: NoticeScope::Home }
        }
    }
}

#[component]
pub(super) fn LevelMeter(test: MicrophoneTestResponse) -> Element {
    let peak = percent(test.peak);
    let bars = test
        .levels
        .iter()
        .enumerate()
        .map(|(index, level)| {
            // Speech is quiet on a linear scale; the square root keeps it visible.
            let height = (level.clamp(0.0, 1.0).sqrt() * 20.0).max(0.5);
            (index, 20.0 - height, height)
        })
        .collect::<Vec<_>>();
    let width = bars.len().max(1);
    rsx! {
        div {
            id: "microphone-level",
            class: "level-meter",
            role: "meter",
            aria_label: "Microphone level",
            aria_valuemin: "0",
            aria_valuemax: "100",
            aria_valuenow: "{peak}",
            aria_valuetext: "Loudest sound {peak}%",
            svg { class: "level-meter-bars", view_box: "0 0 {width} 20", preserve_aspect_ratio: "none",
                "aria-hidden": "true", "focusable": "false",
                for (index, y, height) in bars {
                    rect { x: "{index}", y: "{y}", width: "0.7", height: "{height}" }
                }
            }
            p { class: "level-meter-caption", "Loudest sound: {peak}%" }
        }
    }
}

pub(super) fn percent(level: f32) -> u8 {
    // Clamped to 0..=100 before the cast.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the value is clamped to 0..=100 first"
    )]
    let value = (level.clamp(0.0, 1.0) * 100.0).round() as u8;
    value
}

#[component]
fn LatestTranscript(transcript: Option<String>) -> Element {
    let context = use_context::<UiContext>();
    let navigator = context.navigator;
    rsx! {
        article { class: "card home-card home-transcript", aria_labelledby: "home-transcript-title",
            h2 { id: "home-transcript-title", "Latest transcript" }
            if let Some(text) = transcript {
                p { id: "latest-transcript", class: "transcript-text", "{text}" }
                div { class: "card-actions",
                    Button {
                        id: "copy-latest",
                        size: ButtonSize::Sm,
                        onclick: move |_| context.copy_text(text.clone(), NoticeScope::Home),
                        Icon { name: IconName::Copy }
                        "Copy"
                    }
                    Button {
                        variant: ButtonVariant::Ghost,
                        size: ButtonSize::Sm,
                        onclick: move |_| navigator.go(Route::History),
                        "Open history"
                    }
                }
            } else {
                EmptyState { icon: IconName::Wave, title: "No transcripts yet",
                    p { "Use your shortcut in any app and speak. Your latest transcript appears here." }
                }
            }
        }
    }
}
