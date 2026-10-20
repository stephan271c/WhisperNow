use dioxus::prelude::*;

use super::{Icon, IconName};

/// Visual weight of a button. Exactly one primary action per group.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum ButtonVariant {
    Primary,
    #[default]
    Secondary,
    Ghost,
}

impl ButtonVariant {
    const fn class(self) -> &'static str {
        match self {
            Self::Primary => "btn-primary",
            Self::Secondary => "btn-secondary",
            Self::Ghost => "btn-ghost",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum ButtonSize {
    Sm,
    #[default]
    Md,
}

impl ButtonSize {
    const fn class(self) -> &'static str {
        match self {
            Self::Sm => "btn-sm",
            Self::Md => "btn-md",
        }
    }
}

/// A `type="button"` control; element attributes such as `id`, `disabled` and `title` pass through.
#[component]
pub(crate) fn Button(
    #[props(default)] variant: ButtonVariant,
    #[props(default)] size: ButtonSize,
    onclick: EventHandler<MouseEvent>,
    #[props(extends = GlobalAttributes, extends = button)] attributes: Vec<Attribute>,
    children: Element,
) -> Element {
    rsx! {
        button {
            r#type: "button",
            class: "btn {variant.class()} {size.class()}",
            onclick: move |event| onclick.call(event),
            ..attributes,
            {children}
        }
    }
}

/// An icon-only button whose required `label` is its accessible name and tooltip.
#[component]
pub(crate) fn IconButton(
    icon: IconName,
    label: String,
    #[props(default = ButtonVariant::Ghost)] variant: ButtonVariant,
    #[props(default)] size: ButtonSize,
    onclick: EventHandler<MouseEvent>,
    #[props(extends = GlobalAttributes, extends = button)] attributes: Vec<Attribute>,
) -> Element {
    rsx! {
        button {
            r#type: "button",
            class: "btn btn-icon {variant.class()} {size.class()}",
            aria_label: "{label}",
            title: "{label}",
            onclick: move |event| onclick.call(event),
            ..attributes,
            Icon { name: icon }
        }
    }
}

/// A destructive action that asks for a second, explicit confirmation.
///
/// The first press reveals `confirm_label` (id `{id}-confirm`) and a cancel button
/// (id `{id}-cancel`); focus moves to the confirmation and returns to the trigger once the
/// choice is made, so keyboard users keep their place. In a repeated row, `target` names what
/// the buttons act on; it follows the visible label in their accessible names.
#[component]
pub(crate) fn ConfirmButton(
    id: String,
    label: String,
    confirm_label: String,
    onconfirm: EventHandler<()>,
    target: Option<String>,
    #[props(default)] disabled: bool,
    #[props(default)] size: ButtonSize,
) -> Element {
    let mut armed = use_signal(|| false);
    let mut refocus_trigger = use_signal(|| false);
    let mut disarm = move || {
        armed.set(false);
        refocus_trigger.set(true);
    };
    let named = |visible: &str| target.as_ref().map(|target| format!("{visible} {target}"));
    if !armed() {
        return rsx! {
            button {
                r#type: "button",
                id: "{id}",
                class: "btn btn-danger {size.class()}",
                aria_label: named(&label),
                disabled,
                onmounted: move |element| async move {
                    if refocus_trigger() {
                        refocus_trigger.set(false);
                        let _ = element.set_focus(true).await;
                    }
                },
                onclick: move |_| armed.set(true),
                "{label}"
            }
        };
    }
    rsx! {
        span {
            class: "confirm-button",
            role: "group",
            aria_label: named(&label).unwrap_or_else(|| label.clone()),
            onkeydown: move |event| {
                if event.key() == Key::Escape {
                    disarm();
                }
            },
            button {
                r#type: "button",
                id: "{id}-confirm",
                class: "btn btn-danger {size.class()}",
                aria_label: named(&confirm_label),
                disabled,
                onmounted: move |element| async move {
                    let _ = element.set_focus(true).await;
                },
                onclick: move |_| {
                    disarm();
                    onconfirm.call(());
                },
                "{confirm_label}"
            }
            button {
                r#type: "button",
                id: "{id}-cancel",
                class: "btn btn-ghost {size.class()}",
                aria_label: named("Cancel"),
                onclick: move |_| disarm(),
                "Cancel"
            }
        }
    }
}
