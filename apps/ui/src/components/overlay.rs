use dioxus::core::Task;
use dioxus::prelude::*;
use whispernow_domain::ipc::{Phase, RecordingTrigger};

use super::ui::{Button, ButtonSize, ButtonVariant, Icon, IconButton, IconName};
use crate::overlay::{LEVEL_BARS, format_elapsed};
use crate::{Notice, OverlayView, UiContext, UiOperation, client};

/// A failure with a button stays longer than one without, so the button can be reached.
const FAILURE_WITH_ACTION_MS: u32 = 5_000;

/// Meter geometry in SVG units; bars are attributes because the CSP forbids inline styles.
const BAR_WIDTH: f32 = 4.0;
const BAR_GAP: f32 = 3.0;
const METER_HEIGHT: f32 = 20.0;
#[allow(clippy::cast_precision_loss, reason = "LEVEL_BARS is small")]
const METER_WIDTH: f32 = LEVEL_BARS as f32 * BAR_WIDTH + (LEVEL_BARS - 1) as f32 * BAR_GAP;
const MIN_BAR_HEIGHT: f32 = 3.0;

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
            let height = MIN_BAR_HEIGHT + (METER_HEIGHT - MIN_BAR_HEIGHT) * level;
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
    let phase = state
        .snapshot
        .as_ref()
        .map_or(&Phase::Idle, |snapshot| &snapshot.phase);
    let (phase_class, label, trigger, action) = match phase {
        Phase::Recording { trigger, .. } if view.levels_seen => {
            ("overlay-listening", "Listening", Some(*trigger), None)
        }
        Phase::Recording { .. } => ("overlay-starting", "Starting…", None, None),
        Phase::Processing { cleaning_up: false } => {
            ("overlay-transcribing", "Transcribing…", None, None)
        }
        Phase::Processing { cleaning_up: true } => {
            ("overlay-cleaning-up", "Cleaning up…", None, None)
        }
        Phase::Error => {
            let error = state
                .snapshot
                .as_ref()
                .and_then(|snapshot| snapshot.error.as_ref())
                .expect("a failed dictation shows its error");
            (
                "overlay-failed",
                error.summary.as_str(),
                None,
                crate::home_status::category_fix(error.category),
            )
        }
        Phase::Idle | Phase::Paused | Phase::ClearingData | Phase::Updating => {
            ("overlay-idle", "", None, None)
        }
    };
    let status = view
        .action_error
        .clone()
        .unwrap_or_else(|| label.to_owned());
    let elapsed = format_elapsed(view.elapsed_ms);
    let bars = meter_bars(view);
    let arming = if view.levels_seen {
        ""
    } else {
        "overlay-bars-arming"
    };
    let recording = matches!(phase, Phase::Recording { .. });
    let in_flight = phase.is_busy();
    let failed = *phase == Phase::Error;
    let visible = in_flight || failed;
    drop(state);

    let close_after = if action.is_some() {
        FAILURE_WITH_ACTION_MS
    } else {
        Notice::AUTO_DISMISS_MS
    };
    let mut closing = use_signal(|| None::<Task>);
    use_effect(use_reactive!(|failed, close_after| {
        if let Some(pending) = closing.write().take() {
            pending.cancel();
        }
        if failed {
            closing.set(Some(spawn(async move {
                client::delay(close_after).await;
                context.dismiss_overlay();
            })));
        }
    }));

    let blocked = |operation| context.blocked(operation).is_some();
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
            section { class: "overlay-pill {phase_class}", aria_label: "Dictation",
                div { class: "overlay-row",
                    span { class: "overlay-indicator", aria_hidden: "true",
                        if recording {
                            span { class: "overlay-dot" }
                        } else if in_flight {
                            span { class: "overlay-spinner" }
                        } else if failed {
                            Icon { name: IconName::AlertCircle }
                        }
                    }
                    // The one live region: it changes only with the phase, so each outcome is
                    // announced once while the meter and timer update silently.
                    p { id: "overlay-status", class: "overlay-status", role: "status",
                        aria_live: "polite", aria_atomic: "true", "{status}"
                    }
                    if recording {
                        svg { class: "overlay-bars {arming}", view_box: "0 0 {METER_WIDTH} {METER_HEIGHT}",
                            "aria-hidden": "true",
                            for MeterBar { x, y, height } in bars.iter().copied() {
                                rect { class: "overlay-bar", x: "{x}", y: "{y:.2}",
                                    width: "{BAR_WIDTH}", height: "{height:.2}", rx: "1"
                                }
                            }
                        }
                    }
                    if trigger.is_some() {
                        span { class: "overlay-elapsed", role: "timer", aria_label: "Elapsed time",
                            "{elapsed}"
                        }
                    }
                    if visible {
                        div { class: "overlay-actions",
                            if let Some((label, section)) = action {
                                Button { id: "overlay-action", variant: ButtonVariant::Secondary,
                                    size: ButtonSize::Sm, disabled: action_blocked,
                                    onclick: move |_| context.overlay_action(section),
                                    "{label}"
                                }
                            }
                            if matches!(trigger, Some(RecordingTrigger::Toggle | RecordingTrigger::SetupTest)) {
                                IconButton { id: "overlay-stop", icon: IconName::Stop,
                                    label: "Stop and transcribe", size: ButtonSize::Sm,
                                    disabled: stop_blocked,
                                    onclick: move |_| context.stop_recording(),
                                }
                            }
                            IconButton { id: "overlay-close", icon: IconName::Close,
                                label: "{close_label}", size: ButtonSize::Sm, disabled: close_blocked,
                                onclick: move |_| if in_flight {
                                    context.cancel_recording();
                                } else {
                                    context.dismiss_overlay();
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
