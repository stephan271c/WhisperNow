use dioxus::prelude::*;

use super::{ButtonSize, Icon, IconButton, IconName};

/// Severity of a banner or toast. Warnings and errors are announced assertively.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Tone {
    Info,
    Success,
    Warning,
    Error,
}

impl Tone {
    const fn class(self) -> &'static str {
        match self {
            Self::Info => "tone-info",
            Self::Success => "tone-success",
            Self::Warning => "tone-warning",
            Self::Error => "tone-error",
        }
    }

    const fn icon(self) -> IconName {
        match self {
            Self::Info => IconName::Info,
            Self::Success => IconName::CheckCircle,
            Self::Warning => IconName::Warning,
            Self::Error => IconName::AlertCircle,
        }
    }

    const fn role(self) -> &'static str {
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

/// A dismissible message: transient in the `ToastRegion`, or inline beside the control it is
/// about when `placement` says so. `action` renders between the message and the close button.
#[component]
pub(crate) fn Toast(
    tone: Tone,
    message: String,
    on_dismiss: EventHandler<()>,
    id: Option<String>,
    placement: Option<String>,
    action: Option<Element>,
) -> Element {
    let dismiss_id = id.as_ref().map(|id| format!("{id}-dismiss"));
    let class = placement.map_or_else(
        || format!("toast {}", tone.class()),
        |placement| format!("toast {} {placement}", tone.class()),
    );
    rsx! {
        div { id, class, role: tone.role(),
            span { class: "toast-icon", Icon { name: tone.icon() } }
            p { class: "toast-message", "{message}" }
            if let Some(action) = action {
                {action}
            }
            IconButton {
                id: dismiss_id,
                icon: IconName::Close,
                label: "Dismiss notification",
                size: ButtonSize::Sm,
                onclick: move |_| on_dismiss.call(()),
            }
        }
    }
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
