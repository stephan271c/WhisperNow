use dioxus::prelude::*;

use super::{Icon, IconName};

/// Severity of a banner or toast. Warnings and errors are announced assertively.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Tone {
    Info,
    Success,
    Warning,
    Error,
}

impl Tone {
    pub(crate) const fn class(self) -> &'static str {
        match self {
            Self::Info => "tone-info",
            Self::Success => "tone-success",
            Self::Warning => "tone-warning",
            Self::Error => "tone-error",
        }
    }

    pub(crate) const fn icon(self) -> IconName {
        match self {
            Self::Info => IconName::Info,
            Self::Success => IconName::CheckCircle,
            Self::Warning => IconName::Warning,
            Self::Error => IconName::AlertCircle,
        }
    }

    pub(crate) const fn role(self) -> &'static str {
        match self {
            Self::Info | Self::Success => "status",
            Self::Warning | Self::Error => "alert",
        }
    }
}

#[component]
pub(crate) fn Banner(tone: Tone, title: Option<String>, children: Element) -> Element {
    rsx! {
        div { class: "banner {tone.class()}", role: tone.role(),
            span { class: "banner-icon", Icon { name: tone.icon() } }
            div { class: "banner-body",
                if let Some(title) = title {
                    p { class: "banner-title", "{title}" }
                }
                div { class: "banner-message", {children} }
            }
        }
    }
}

#[component]
pub(crate) fn ToastRegion(children: Element) -> Element {
    rsx! { section { class: "toast-region", aria_label: "Notifications", {children} } }
}

/// A determinate progress bar. `label` is required so the bar always has an accessible name.
#[component]
pub(crate) fn ProgressBar(
    label: String,
    value: u64,
    max: u64,
    value_text: Option<String>,
) -> Element {
    let max = max.max(1);
    let percent = value.min(max).saturating_mul(100) / max;
    rsx! {
        div {
            class: "progress",
            role: "progressbar",
            aria_label: "{label}",
            aria_valuemin: "0",
            aria_valuemax: "100",
            aria_valuenow: "{percent}",
            aria_valuetext: value_text,
            svg { class: "progress-track", view_box: "0 0 100 4", preserve_aspect_ratio: "none",
                "aria-hidden": "true", "focusable": "false",
                rect { class: "progress-fill", x: "0", y: "0", width: "{percent}", height: "4" }
            }
        }
    }
}
