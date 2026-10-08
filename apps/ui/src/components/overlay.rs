use dioxus::prelude::*;
use whispernow_api::RecordingMode;

use super::actions::UiActions;
use super::ui::{Button, ButtonSize, ButtonVariant, Icon, IconButton, IconName};
use crate::overlay::{LEVEL_BARS, OverlayAction, OverlayPhase, format_elapsed};
use crate::{OverlayView, UiContext, UiOperation};

/// Meter geometry in SVG units; bars are attributes because the CSP forbids inline styles.
const BAR_WIDTH: f32 = 4.0;
const BAR_GAP: f32 = 3.0;
const METER_HEIGHT: f32 = 20.0;
#[allow(clippy::cast_precision_loss, reason = "LEVEL_BARS is small")]
const METER_WIDTH: f32 = LEVEL_BARS as f32 * BAR_WIDTH + (LEVEL_BARS - 1) as f32 * BAR_GAP;
const MIN_BAR_HEIGHT: f32 = 3.0;

/// The status line, which is also what the polite live region announces.
fn status_label(phase: &OverlayPhase) -> &str {
    match phase {
        OverlayPhase::Hidden => "",
        OverlayPhase::Starting => "Starting…",
        OverlayPhase::Listening { .. } => "Listening",
        OverlayPhase::Transcribing => "Transcribing…",
        OverlayPhase::CleaningUp => "Cleaning up…",
        OverlayPhase::Failed { message, .. } => message,
    }
}

const fn phase_class(phase: &OverlayPhase) -> &'static str {
    match phase {
        OverlayPhase::Hidden => "overlay-idle",
        OverlayPhase::Starting => "overlay-starting",
        OverlayPhase::Listening { .. } => "overlay-listening",
        OverlayPhase::Transcribing => "overlay-transcribing",
        OverlayPhase::CleaningUp => "overlay-cleaning-up",
        OverlayPhase::Failed { .. } => "overlay-failed",
    }
}

const fn action_label(action: OverlayAction) -> &'static str {
    match action {
        OverlayAction::CopyTranscript => "Copy",
        OverlayAction::OpenHistory => "Open history",
        OverlayAction::OpenSettings => "Fix",
    }
}

#[derive(Clone, Copy)]
struct MeterBar {
    x: f32,
    y: f32,
    height: f32,
}

/// Oldest first.
fn meter_bars(view: &OverlayView) -> Vec<MeterBar> {
    view.levels
        .iter()
        .enumerate()
        .map(|(index, level)| {
            let level = if view.levels_seen { *level } else { 0.0 };
            let height = MIN_BAR_HEIGHT + (METER_HEIGHT - MIN_BAR_HEIGHT) * level.clamp(0.0, 1.0);
            #[allow(clippy::cast_precision_loss, reason = "the index is below LEVEL_BARS")]
            let x = index as f32 * (BAR_WIDTH + BAR_GAP);
            MeterBar {
                x,
                y: (METER_HEIGHT - height) / 2.0,
                height,
            }
        })
        .collect()
}

#[component]
pub fn RecordingBar() -> Element {
    let context = use_context::<UiContext>();
    let state = context.state.read();
    let view = &state.overlay;
    let phase = view.presented_phase();
    let status = view
        .action_error
        .clone()
        .unwrap_or_else(|| status_label(&phase).to_owned());
    let elapsed = format_elapsed(view.elapsed_ms);
    let bars = meter_bars(view);
    let arming = if view.levels_seen {
        ""
    } else {
        "overlay-bars-arming"
    };
    drop(state);

    let in_flight = matches!(
        phase,
        OverlayPhase::Starting
            | OverlayPhase::Listening { .. }
            | OverlayPhase::Transcribing
            | OverlayPhase::CleaningUp
    );
    let capturing = matches!(
        phase,
        OverlayPhase::Starting | OverlayPhase::Listening { .. }
    );
    let mode = match phase {
        OverlayPhase::Listening { mode } => Some(mode),
        _ => None,
    };
    let action = match phase {
        OverlayPhase::Failed { action, .. } => action,
        _ => None,
    };
    let actions = UiActions::new(context);
    let blocked = |operation| actions.blocked(operation).is_some();
    let stop_blocked = blocked(UiOperation::StopRecording);
    let close_blocked = if in_flight {
        blocked(UiOperation::CancelRecording)
    } else {
        blocked(UiOperation::OverlayAction)
    };
    let action_blocked = blocked(UiOperation::OverlayAction);
    let close_label = if in_flight {
        "Cancel dictation"
    } else {
        "Dismiss"
    };

    rsx! {
        main { class: "wn-app overlay",
            section { class: "overlay-pill {phase_class(&phase)}", aria_label: "Dictation",
                div { class: "overlay-row",
                    span { class: "overlay-indicator", aria_hidden: "true",
                        {match &phase {
                            OverlayPhase::Hidden => rsx! {},
                            OverlayPhase::Starting | OverlayPhase::Listening { .. } => rsx! {
                                span { class: "overlay-dot" }
                            },
                            OverlayPhase::Transcribing | OverlayPhase::CleaningUp => rsx! {
                                span { class: "overlay-spinner" }
                            },
                            OverlayPhase::Failed { .. } => rsx! { Icon { name: IconName::AlertCircle } },
                        }}
                    }
                    // The one live region: it changes only with the phase, so each outcome is
                    // announced once while the meter and timer update silently.
                    p { id: "overlay-status", class: "overlay-status", role: "status",
                        aria_live: "polite", aria_atomic: "true", "{status}"
                    }
                    if capturing {
                        svg { class: "overlay-bars {arming}", view_box: "0 0 {METER_WIDTH} {METER_HEIGHT}",
                            "aria-hidden": "true",
                            for MeterBar { x, y, height } in bars.iter().copied() {
                                rect { class: "overlay-bar", x: "{x}", y: "{y:.2}",
                                    width: "{BAR_WIDTH}", height: "{height:.2}", rx: "1"
                                }
                            }
                        }
                    }
                    if mode.is_some() {
                        span { class: "overlay-elapsed", role: "timer", aria_label: "Elapsed time",
                            "{elapsed}"
                        }
                    }
                    if phase != OverlayPhase::Hidden {
                        div { class: "overlay-actions",
                            if let Some(action) = action {
                                Button { id: "overlay-action", variant: ButtonVariant::Secondary,
                                    size: ButtonSize::Sm, disabled: action_blocked,
                                    onclick: move |_| actions.overlay_action(action),
                                    "{action_label(action)}"
                                }
                            }
                            if mode == Some(RecordingMode::Toggle) {
                                IconButton { id: "overlay-stop", icon: IconName::Stop,
                                    label: "Stop and transcribe", size: ButtonSize::Sm,
                                    disabled: stop_blocked,
                                    onclick: move |_| actions.stop_recording(),
                                }
                            }
                            IconButton { id: "overlay-close", icon: IconName::Close,
                                label: "{close_label}", size: ButtonSize::Sm, disabled: close_blocked,
                                onclick: move |_| if in_flight {
                                    actions.cancel_recording();
                                } else {
                                    actions.dismiss_overlay();
                                },
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
