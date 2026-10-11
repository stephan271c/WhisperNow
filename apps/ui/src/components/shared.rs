use crate::UiContext;
use std::collections::BTreeSet;

use dioxus::prelude::*;
use whispernow_domain::ipc::PasteHelperSetup;
use whispernow_domain::{Shortcut, ShortcutKey, ShortcutModifier};

use super::ui::{Banner, Button, Kbd, Tone, shortcut_parts};
use crate::NoticeScope;

#[derive(Clone, Debug, Eq, PartialEq)]
enum CaptureStatus {
    Idle,
    Listening,
    ModifierOnly,
    MissingModifier,
    Unsupported,
    Captured(Shortcut),
}

impl CaptureStatus {
    fn is_capturing(&self) -> bool {
        matches!(
            self,
            Self::Listening | Self::ModifierOnly | Self::MissingModifier | Self::Unsupported
        )
    }
}

/// Shown while Wayland paste goes through the `RemoteDesktop` portal because the
/// paste helper is not set up: why the desktop shows its indicator, and how to
/// set the helper up.
#[component]
pub(super) fn PasteHelperNotice(setup: PasteHelperSetup, scope: NoticeScope) -> Element {
    let context = use_context::<UiContext>();
    rsx! { div { id: "paste-helper-notice",
        Banner { tone: Tone::Warning, title: "Your desktop shows a remote control indicator",
            p { "WhisperNow's paste helper is not set up, so your desktop presses the paste keys for it and shows its remote control indicator while WhisperNow runs." }
            match setup.command {
                Some(command) => rsx! {
                    p { "To paste without it, run this command in a terminal, then restart WhisperNow:" }
                    code { id: "paste-helper-command", class: "command", "{command}" }
                    div { class: "card-actions",
                        Button { id: "paste-helper-copy",
                            onclick: move |_| context.copy_text(command.clone(), scope),
                            "Copy command"
                        }
                    }
                },
                None => rsx! {
                    p { "Install WhisperNow from its .deb or .rpm package, which sets the helper up, then restart WhisperNow." }
                },
            }
        }
    } }
}

/// The button's accessible name is its visible text followed by `label`; the current keys
/// and the capture guidance describe it. With `lone_keys`, Escape or a function key may be
/// pressed without a modifier (as the cancel shortcut allows), so Escape is captured and
/// only Tab leaves the capture.
#[component]
pub(super) fn ShortcutCapture(
    id: String,
    label: String,
    shortcut: Option<Shortcut>,
    oncapture: EventHandler<Shortcut>,
    #[props(default)] disabled: bool,
    #[props(default)] lone_keys: bool,
) -> Element {
    let mut status = use_signal(|| CaptureStatus::Idle);
    // While listening, the desktop's global shortcuts are suspended so pressing
    // the current shortcut is captured instead of starting a recording.
    use_effect(move || crate::client::request_shortcut_capture(status().is_capturing()));
    use_drop(|| crate::client::request_shortcut_capture(false));
    let parts = shortcut.as_ref().map_or_else(Vec::new, shortcut_parts);
    let help_id = format!("{id}-help");
    let keys_id = format!("{id}-keys");
    let guidance = match status() {
        CaptureStatus::Idle => "",
        CaptureStatus::Listening if lone_keys => {
            "Listening… Press Escape, F1–F24, or hold a modifier and press a key. Tab cancels."
        }
        CaptureStatus::Listening => {
            "Listening… Hold Control, Alt, Shift, or Meta and press a key. Escape cancels."
        }
        CaptureStatus::ModifierOnly => "Keep the modifier held and press the shortcut key.",
        CaptureStatus::MissingModifier if lone_keys => {
            "Only Escape and F1–F24 work alone; hold Control, Alt, Shift, or Meta for other keys."
        }
        CaptureStatus::MissingModifier => "Include at least one of Control, Alt, Shift, or Meta.",
        CaptureStatus::Unsupported => {
            "That key is not supported. Try a letter, number, navigation key, or F1–F24."
        }
        CaptureStatus::Captured(_) => "Shortcut captured.",
    };
    let capturing = status().is_capturing();
    let visible = if capturing {
        "Press shortcut…"
    } else if parts.is_empty() {
        "Choose shortcut"
    } else {
        "Change"
    };
    rsx! {
        div { class: "shortcut-capture-field",
            span { id: "{keys_id}", class: "shortcut-capture-keys",
                if parts.is_empty() {
                    span { class: "shortcut-capture-empty", "Not set" }
                } else {
                    for part in parts { Kbd { "{part}" } }
                }
            }
            button {
                id: "{id}",
                r#type: "button",
                class: if capturing { "shortcut-capture capturing" } else { "shortcut-capture" },
                aria_label: "{visible} {label}",
                aria_describedby: "{keys_id} {help_id}",
                aria_pressed: capturing,
                disabled,
                onclick: move |_| {
                    status.set(CaptureStatus::Listening);
                },
                onblur: move |_| {
                    if !matches!(status(), CaptureStatus::Captured(_)) {
                        status.set(CaptureStatus::Idle);
                    }
                },
                onkeydown: move |event| {
                    if !status().is_capturing() {
                        return;
                    }
                    let modifiers = event.modifiers();
                    let code = event.code().to_string();
                    if code == "Tab" && modifiers.is_empty() {
                        status.set(CaptureStatus::Idle);
                        return;
                    }
                    event.prevent_default();
                    if event.is_auto_repeating() || event.is_composing() {
                        return;
                    }
                    if code == "Escape" && modifiers.is_empty() && !lone_keys {
                        status.set(CaptureStatus::Idle);
                        return;
                    }
                    let captured = captured_shortcut(&code, modifiers, lone_keys);
                    status.set(captured.clone());
                    if let CaptureStatus::Captured(shortcut) = captured {
                        oncapture.call(shortcut);
                    }
                },
                "{visible}"
            }
        }
        p {
            id: help_id,
            class: if matches!(status(), CaptureStatus::MissingModifier | CaptureStatus::Unsupported) { "shortcut-capture-help error" } else { "shortcut-capture-help" },
            role: "status",
            aria_live: "polite",
            "{guidance}"
        }
    }
}

fn captured_shortcut(
    code: &str,
    modifiers: dioxus::prelude::keyboard_types::Modifiers,
    lone_keys: bool,
) -> CaptureStatus {
    if is_modifier_code(code) {
        return CaptureStatus::ModifierOnly;
    }
    let Some(key) = ShortcutKey::from_code(code) else {
        return CaptureStatus::Unsupported;
    };
    let shortcut = Shortcut {
        modifiers: captured_modifiers(modifiers),
        key,
    };
    let valid = if lone_keys {
        shortcut.is_cancel_shortcut()
    } else {
        shortcut.is_recording_shortcut()
    };
    if !valid {
        return CaptureStatus::MissingModifier;
    }
    CaptureStatus::Captured(shortcut)
}

fn captured_modifiers(
    modifiers: dioxus::prelude::keyboard_types::Modifiers,
) -> BTreeSet<ShortcutModifier> {
    [
        (modifiers.ctrl(), ShortcutModifier::Control),
        (modifiers.alt(), ShortcutModifier::Alt),
        (modifiers.shift(), ShortcutModifier::Shift),
        (modifiers.meta(), ShortcutModifier::Meta),
    ]
    .into_iter()
    .filter_map(|(held, modifier)| held.then_some(modifier))
    .collect()
}

fn is_modifier_code(code: &str) -> bool {
    matches!(
        code,
        "ControlLeft"
            | "ControlRight"
            | "AltLeft"
            | "AltRight"
            | "ShiftLeft"
            | "ShiftRight"
            | "MetaLeft"
            | "MetaRight"
            | "OSLeft"
            | "OSRight"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use dioxus::prelude::keyboard_types::Modifiers;

    #[test]
    fn capture_records_the_complete_modifier_chord_in_canonical_order() {
        let captured = captured_shortcut(
            "Space",
            Modifiers::META | Modifiers::SHIFT | Modifiers::CONTROL,
            false,
        );
        assert_eq!(
            captured,
            CaptureStatus::Captured(Shortcut {
                modifiers: [
                    ShortcutModifier::Control,
                    ShortcutModifier::Shift,
                    ShortcutModifier::Meta,
                ]
                .into(),
                key: ShortcutKey::Space,
            })
        );
    }

    #[test]
    fn capture_waits_for_a_primary_key_and_requires_a_modifier() {
        assert_eq!(
            captured_shortcut("ControlLeft", Modifiers::CONTROL, false),
            CaptureStatus::ModifierOnly
        );
        assert_eq!(
            captured_shortcut("KeyR", Modifiers::empty(), false),
            CaptureStatus::MissingModifier
        );
        assert_eq!(
            captured_shortcut("Escape", Modifiers::empty(), false),
            CaptureStatus::MissingModifier
        );
        assert_eq!(
            captured_shortcut("Semicolon", Modifiers::CONTROL, false),
            CaptureStatus::Unsupported
        );
    }

    #[test]
    fn lone_key_capture_accepts_only_non_typing_keys_without_a_modifier() {
        for (code, key) in [("Escape", ShortcutKey::Escape), ("F13", ShortcutKey::F13)] {
            assert_eq!(
                captured_shortcut(code, Modifiers::empty(), true),
                CaptureStatus::Captured(Shortcut {
                    modifiers: [].into(),
                    key,
                })
            );
        }
        for code in ["KeyR", "Space", "Enter"] {
            assert_eq!(
                captured_shortcut(code, Modifiers::empty(), true),
                CaptureStatus::MissingModifier,
                "{code}"
            );
        }
        assert_eq!(
            captured_shortcut("KeyR", Modifiers::CONTROL, true),
            CaptureStatus::Captured(Shortcut {
                modifiers: [ShortcutModifier::Control].into(),
                key: ShortcutKey::R,
            })
        );
    }
}
