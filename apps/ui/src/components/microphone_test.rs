use dioxus::prelude::*;
use whispernow_domain::ipc::MicrophoneTestResponse;

use super::ui::{Icon, IconName};
use crate::blocked_reason::BlockedButton;
use crate::{UiContext, UiOperation};

#[component]
pub(super) fn MicrophoneTest(id_prefix: String) -> Element {
    let context = use_context::<UiContext>();
    let blocked = context
        .blocked(UiOperation::TestMicrophone)
        .or(context.phase_blocked());
    let management = context.management.read();
    let testing = management.is_busy(UiOperation::TestMicrophone);
    let test = management.microphone_test.clone();
    rsx! {
        div { class: "microphone-test",
            BlockedButton {
                id: id_prefix,
                blocked: blocked.filter(|_| !testing),
                disabled: testing,
                onclick: move |_| context.test_microphone(),
                Icon { name: IconName::Mic }
                if testing { "Listening…" } else { "Test microphone" }
            }
            if testing {
                p { class: "status-detail", role: "status", "Say a few words." }
            }
        }
        if let Some(test) = test {
            LevelMeter { test }
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
            let height = (level_fraction(*level) * 20.0).max(0.5);
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

/// How full a level meter is for `rms`. Speech is quiet on a linear scale; the square root
/// keeps it visible.
pub(super) fn level_fraction(rms: f32) -> f32 {
    rms.clamp(0.0, 1.0).sqrt()
}

pub(super) fn percent(level: f32) -> u8 {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the value is clamped to 0..=100 first"
    )]
    let value = (level.clamp(0.0, 1.0) * 100.0).round() as u8;
    value
}
