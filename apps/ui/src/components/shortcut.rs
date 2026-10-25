use std::collections::BTreeSet;

use dioxus::prelude::*;
use whispernow_domain::{RecordingMode, Shortcut, ShortcutKey, ShortcutModifier};

use super::labels;
use super::ui::focus::take_focus;
use super::ui::{Kbd, ShortcutLabel, shortcut_parts};

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

/// The button's accessible name is its visible text followed by `label`; the current keys
/// and the capture guidance describe it. With `lone_keys`, Escape or a function key may be
/// pressed without a modifier (as the cancel shortcut allows), so Escape is captured and
/// only Tab leaves the capture.
#[component]
pub(super) fn ShortcutCapture(
    id: String,
    label: String,
    shortcut: Shortcut,
    oncapture: EventHandler<Shortcut>,
    #[props(default)] disabled: bool,
    #[props(default)] lone_keys: bool,
) -> Element {
    let mut status = use_signal(|| CaptureStatus::Idle);
    let mut button = use_signal(|| None::<std::rc::Rc<MountedData>>);
    // While listening, the desktop's global shortcuts are suspended so pressing
    // the current shortcut is captured instead of starting a recording.
    use_effect(use_reactive!(|disabled| {
        let capturing = status().is_capturing();
        if disabled && capturing {
            status.set(CaptureStatus::Idle);
        }
        crate::client::request_shortcut_capture(capturing && !disabled);
    }));
    use_drop(|| crate::client::request_shortcut_capture(false));
    let parts = shortcut_parts(&shortcut);
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
    // Only a refused key is shown; guidance for the capture in progress is for screen readers,
    // since showing it would shift the page whenever the user starts or ends a capture.
    let refused = matches!(
        status(),
        CaptureStatus::MissingModifier | CaptureStatus::Unsupported
    );
    let capturing = status().is_capturing();
    let visible = if capturing {
        "Press shortcut…"
    } else {
        "Change"
    };
    rsx! {
        div { class: "shortcut-capture-field",
            span { id: "{keys_id}", class: "shortcut-capture-keys",
                for part in parts { Kbd { "{part}" } }
            }
            button {
                id: "{id}",
                r#type: "button",
                class: if capturing { "shortcut-capture capturing" } else { "shortcut-capture" },
                aria_label: "{visible} {label}",
                aria_describedby: "{keys_id} {help_id}",
                aria_pressed: capturing,
                disabled,
                onmounted: move |event| button.set(Some(event.data())),
                // Safari does not focus a button on click, and the keys arrive at the focused one.
                onclick: move |_| {
                    if !disabled {
                        status.set(CaptureStatus::Listening);
                        take_focus(button);
                    }
                },
                onblur: move |_| {
                    if !matches!(status(), CaptureStatus::Captured(_)) {
                        status.set(CaptureStatus::Idle);
                    }
                },
                onkeydown: move |event| {
                    if disabled || !status().is_capturing() {
                        return;
                    }
                    let modifiers = event.modifiers();
                    let code = event.code();
                    if code == Code::Tab && modifiers.is_empty() {
                        status.set(CaptureStatus::Idle);
                        return;
                    }
                    event.prevent_default();
                    if event.is_auto_repeating() || event.is_composing() {
                        return;
                    }
                    if code == Code::Escape && modifiers.is_empty() && !lone_keys {
                        status.set(CaptureStatus::Idle);
                        return;
                    }
                    let captured = captured_shortcut(code, modifiers, lone_keys);
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
            class: if refused { "shortcut-capture-help error" } else { "shortcut-capture-help" },
            role: "status",
            aria_live: "polite",
            if refused {
                "{guidance}"
            } else {
                span { class: "visually-hidden", "{guidance}" }
            }
        }
    }
}

fn captured_shortcut(code: Code, modifiers: Modifiers, lone_keys: bool) -> CaptureStatus {
    if is_modifier_code(code) {
        return CaptureStatus::ModifierOnly;
    }
    let Some(key) = ShortcutKey::from_code(&code.to_string()) else {
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

fn captured_modifiers(modifiers: Modifiers) -> BTreeSet<ShortcutModifier> {
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

fn is_modifier_code(code: Code) -> bool {
    matches!(
        code,
        Code::ControlLeft
            | Code::ControlRight
            | Code::AltLeft
            | Code::AltRight
            | Code::ShiftLeft
            | Code::ShiftRight
            | Code::MetaLeft
            | Code::MetaRight
    )
}

#[component]
pub(super) fn ShortcutSummary(mode: RecordingMode, shortcut: Shortcut) -> Element {
    rsx! { div { dt { "{labels::recording_mode(mode)}" } dd { ShortcutLabel { shortcut } } } }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_records_the_complete_modifier_chord_in_canonical_order() {
        let captured = captured_shortcut(
            Code::Space,
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
        for code in [
            Code::ControlLeft,
            Code::ControlRight,
            Code::AltLeft,
            Code::AltRight,
            Code::ShiftLeft,
            Code::ShiftRight,
            Code::MetaLeft,
            Code::MetaRight,
        ] {
            assert_eq!(
                captured_shortcut(code, Modifiers::CONTROL, false),
                CaptureStatus::ModifierOnly,
                "{code}"
            );
        }
        assert_eq!(
            captured_shortcut(Code::KeyR, Modifiers::empty(), false),
            CaptureStatus::MissingModifier
        );
        assert_eq!(
            captured_shortcut(Code::Escape, Modifiers::empty(), false),
            CaptureStatus::MissingModifier
        );
        assert_eq!(
            captured_shortcut(Code::Semicolon, Modifiers::CONTROL, false),
            CaptureStatus::Unsupported
        );
    }

    #[test]
    fn lone_key_capture_accepts_only_non_typing_keys_without_a_modifier() {
        for (code, key) in [
            (Code::Escape, ShortcutKey::Escape),
            (Code::F13, ShortcutKey::F13),
        ] {
            assert_eq!(
                captured_shortcut(code, Modifiers::empty(), true),
                CaptureStatus::Captured(Shortcut {
                    modifiers: [].into(),
                    key,
                })
            );
        }
        for code in [Code::KeyR, Code::Space, Code::Enter] {
            assert_eq!(
                captured_shortcut(code, Modifiers::empty(), true),
                CaptureStatus::MissingModifier,
                "{code}"
            );
        }
        assert_eq!(
            captured_shortcut(Code::KeyR, Modifiers::CONTROL, true),
            CaptureStatus::Captured(Shortcut {
                modifiers: [ShortcutModifier::Control].into(),
                key: ShortcutKey::R,
            })
        );
    }
}
