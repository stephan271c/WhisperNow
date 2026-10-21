use dioxus::prelude::*;
use whispernow_domain::ipc::CleanupDevice;
use whispernow_domain::{RecordingMode, SetupStep, Shortcut};

use super::labels;
use super::microphone_test::MicrophoneTest;
use super::ui::{
    Banner, Button, ButtonSize, ButtonVariant, EmptyState, Icon, IconName, ShortcutLabel, Tone,
};
use super::updates::UpdateStatus;
use super::{CopyButton, NoticeRegion};
use crate::{CopyTarget, HomeStatus, HomeStatusKind, NoticeScope, Route, SettingsRoute, UiContext};

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
    let cleanup_device = context
        .state
        .read()
        .snapshot
        .as_ref()
        .and_then(|snapshot| snapshot.cleanup_device.clone());
    let management = context.management.read();
    let setup_step = management.unfinished_setup_step();
    let shortcut = management.settings.as_ref().map(|view| {
        let settings = &view.settings;
        (settings.recording_mode, settings.recording_shortcut.clone())
    });
    drop(management);
    rsx! { div { class: "home-grid",
        if let Some(step) = setup_step {
            SetupChecklist { step }
        }
        UpdateStatus {}
        StatusCard { status, cleanup_device }
        ShortcutsCard { shortcut }
        MicrophoneCard {}
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
fn StatusCard(status: HomeStatus, cleanup_device: Option<CleanupDevice>) -> Element {
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
                if let Some(device) = &cleanup_device {
                    p { id: "home-cleanup-device", class: "status-detail", "{labels::cleanup_device(device)}" }
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
fn MicrophoneCard() -> Element {
    rsx! {
        article { class: "card home-card home-microphone", aria_labelledby: "home-microphone-title",
            h2 { id: "home-microphone-title", "Microphone" }
            MicrophoneTest { id_prefix: "test-microphone" }
            NoticeRegion { scope: NoticeScope::Home }
        }
    }
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
                    CopyButton { id: "copy-latest", target: CopyTarget::LatestTranscript,
                        text, label: "Copy", size: ButtonSize::Sm, icon: true,
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
