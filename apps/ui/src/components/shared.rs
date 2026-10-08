use std::collections::BTreeSet;

use dioxus::prelude::*;
use whispernow_api::{
    PasteHelperSetupDto, PermissionStatusDto, Shortcut, ShortcutKey, ShortcutModifier,
};

use super::actions::use_actions;
use super::ui::{Banner, Button, Kbd, Tone, shortcut_parts};
use crate::NoticeScope;

#[derive(Clone, Copy, PartialEq)]
enum CaptureStatus {
    Idle,
    Listening,
    ModifierOnly,
    MissingModifier,
    Unsupported,
    Captured,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CaptureError {
    MissingModifier,
    UnsupportedKey,
}

#[component]
pub(super) fn MicrophonePermissionNotice(status: Option<PermissionStatusDto>) -> Element {
    let (label, guidance) = match status {
        Some(PermissionStatusDto::Granted) => (
            "Microphone access is available",
            "Starting a recording remains the final permission check.",
        ),
        Some(PermissionStatusDto::Denied) => (
            "Microphone access is denied",
            "Allow microphone access in system privacy settings, then reopen or retry setup.",
        ),
        _ => (
            "Microphone permission is unknown",
            "Your system will request access when capture starts. If it fails, enable microphone access in system privacy settings.",
        ),
    };
    rsx! { div { class: "permission-status", role: "status",
        strong { "{label}" }
        p { "{guidance}" }
    } }
}

/// Shown while Wayland paste goes through the `RemoteDesktop` portal because the
/// paste helper is not set up: why the desktop shows its indicator, and how to
/// set the helper up.
#[component]
pub(super) fn PasteHelperNotice(setup: PasteHelperSetupDto, scope: NoticeScope) -> Element {
    let actions = use_actions();
    rsx! { div { id: "paste-helper-notice",
        Banner { tone: Tone::Warning, title: "Your desktop shows a remote control indicator",
            p { "WhisperNow's paste helper is not set up, so your desktop presses the paste keys for it and shows its remote control indicator while WhisperNow runs." }
            match setup.command {
                Some(command) => rsx! {
                    p { "To paste without it, run this command in a terminal, then restart WhisperNow:" }
                    code { id: "paste-helper-command", class: "command", "{command}" }
                    div { class: "card-actions",
                        Button { id: "paste-helper-copy",
                            onclick: move |_| actions.copy_text(command.clone(), scope),
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
    let mut capturing = use_signal(|| false);
    let mut status = use_signal(|| CaptureStatus::Idle);
    // While listening, the desktop's global shortcuts are suspended so pressing
    // the current shortcut is captured instead of starting a recording.
    use_effect(move || sync_shortcut_capture(capturing()));
    use_drop(|| sync_shortcut_capture(false));
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
        CaptureStatus::Captured => "Shortcut captured.",
    };
    let visible = if capturing() {
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
                class: if capturing() { "shortcut-capture capturing" } else { "shortcut-capture" },
                aria_label: "{visible} {label}",
                aria_describedby: "{keys_id} {help_id}",
                aria_pressed: capturing(),
                disabled,
                onclick: move |_| {
                    capturing.set(true);
                    status.set(CaptureStatus::Listening);
                },
                onblur: move |_| {
                    capturing.set(false);
                    if status() != CaptureStatus::Captured {
                        status.set(CaptureStatus::Idle);
                    }
                },
                onkeydown: move |event| {
                    if !capturing() {
                        return;
                    }
                    let modifiers = event.modifiers();
                    let code = event.code().to_string();
                    if code == "Tab" && modifiers.is_empty() {
                        capturing.set(false);
                        status.set(CaptureStatus::Idle);
                        return;
                    }
                    event.prevent_default();
                    if event.is_auto_repeating() || event.is_composing() {
                        return;
                    }
                    if code == "Escape" && modifiers.is_empty() && !lone_keys {
                        capturing.set(false);
                        status.set(CaptureStatus::Idle);
                        return;
                    }
                    match captured_shortcut(&code, modifiers, lone_keys) {
                        Ok(Some(shortcut)) => {
                            capturing.set(false);
                            status.set(CaptureStatus::Captured);
                            oncapture.call(shortcut);
                        }
                        Ok(None) => status.set(CaptureStatus::ModifierOnly),
                        Err(CaptureError::MissingModifier) => {
                            status.set(CaptureStatus::MissingModifier);
                        }
                        Err(CaptureError::UnsupportedKey) => {
                            status.set(CaptureStatus::Unsupported);
                        }
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

fn sync_shortcut_capture(capturing: bool) {
    #[cfg(target_arch = "wasm32")]
    crate::client::request_shortcut_capture(capturing);
    #[cfg(not(target_arch = "wasm32"))]
    let _ = capturing;
}

fn captured_shortcut(
    code: &str,
    modifiers: dioxus::prelude::keyboard_types::Modifiers,
    lone_keys: bool,
) -> Result<Option<Shortcut>, CaptureError> {
    if is_modifier_code(code) {
        return Ok(None);
    }
    let key = ShortcutKey::from_code(code).ok_or(CaptureError::UnsupportedKey)?;
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
        return Err(CaptureError::MissingModifier);
    }
    Ok(Some(shortcut))
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
        )
        .expect("supported chord")
        .expect("primary key");
        assert_eq!(
            captured,
            Shortcut {
                modifiers: [
                    ShortcutModifier::Control,
                    ShortcutModifier::Shift,
                    ShortcutModifier::Meta,
                ]
                .into(),
                key: ShortcutKey::Space,
            }
        );
    }

    #[test]
    fn capture_waits_for_a_primary_key_and_requires_a_modifier() {
        assert_eq!(
            captured_shortcut("ControlLeft", Modifiers::CONTROL, false),
            Ok(None)
        );
        assert_eq!(
            captured_shortcut("KeyR", Modifiers::empty(), false),
            Err(CaptureError::MissingModifier)
        );
        assert_eq!(
            captured_shortcut("Escape", Modifiers::empty(), false),
            Err(CaptureError::MissingModifier)
        );
        assert_eq!(
            captured_shortcut("Semicolon", Modifiers::CONTROL, false),
            Err(CaptureError::UnsupportedKey)
        );
    }

    #[test]
    fn lone_key_capture_accepts_only_non_typing_keys_without_a_modifier() {
        for (code, key) in [("Escape", ShortcutKey::Escape), ("F13", ShortcutKey::F13)] {
            assert_eq!(
                captured_shortcut(code, Modifiers::empty(), true),
                Ok(Some(Shortcut {
                    modifiers: [].into(),
                    key,
                }))
            );
        }
        for code in ["KeyR", "Space", "Enter"] {
            assert_eq!(
                captured_shortcut(code, Modifiers::empty(), true),
                Err(CaptureError::MissingModifier),
                "{code}"
            );
        }
        assert_eq!(
            captured_shortcut("KeyR", Modifiers::CONTROL, true),
            Ok(Some(Shortcut {
                modifiers: [ShortcutModifier::Control].into(),
                key: ShortcutKey::R,
            }))
        );
    }
}
