//! macOS Accessibility revalidation and balanced Core Graphics Command-V.

#[cfg(target_os = "macos")]
use super::{NativePasteError, NativePasteInjection, wait_until_modifiers_released};
#[cfg(target_os = "macos")]
use crate::destination::{NativeDestination, still_focused};

const COMMAND_KEYCODE: u16 = 55;
const V_KEYCODE: u16 = 9;
const COMMAND_FLAG: u64 = 1_048_576;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MacKeyStroke {
    keycode: u16,
    key_down: bool,
    flags: u64,
}

const COMMAND_V: [MacKeyStroke; 4] = [
    MacKeyStroke {
        keycode: COMMAND_KEYCODE,
        key_down: true,
        flags: COMMAND_FLAG,
    },
    MacKeyStroke {
        keycode: V_KEYCODE,
        key_down: true,
        flags: COMMAND_FLAG,
    },
    MacKeyStroke {
        keycode: V_KEYCODE,
        key_down: false,
        flags: COMMAND_FLAG,
    },
    MacKeyStroke {
        keycode: COMMAND_KEYCODE,
        key_down: false,
        flags: 0,
    },
];

#[cfg(target_os = "macos")]
pub(super) fn inject_if_current(destination: NativeDestination) -> NativePasteInjection {
    match wait_until_modifiers_released(|| Ok(modifier_held()))
        .and_then(|()| still_focused(destination))
    {
        Ok(()) => post_command_v(),
        Err(error) => NativePasteInjection::NotSent(error),
    }
}

#[cfg(target_os = "macos")]
fn modifier_held() -> bool {
    use objc2_core_graphics::{CGEventFlags, CGEventSource, CGEventSourceStateID};

    let held = CGEventFlags::MaskCommand
        | CGEventFlags::MaskControl
        | CGEventFlags::MaskAlternate
        | CGEventFlags::MaskShift;
    CGEventSource::flags_state(CGEventSourceStateID::HIDSystemState).intersects(held)
}

#[cfg(target_os = "macos")]
fn post_command_v() -> NativePasteInjection {
    use objc2_core_graphics::{
        CGEvent, CGEventFlags, CGEventSource, CGEventSourceStateID, CGEventTapLocation,
    };

    let Some(source) = CGEventSource::new(CGEventSourceStateID::HIDSystemState) else {
        return NativePasteInjection::NotSent(NativePasteError::InjectionFailed);
    };
    let mut events = Vec::with_capacity(COMMAND_V.len());
    for stroke in COMMAND_V {
        let Some(event) =
            CGEvent::new_keyboard_event(Some(&source), stroke.keycode, stroke.key_down)
        else {
            return NativePasteInjection::NotSent(NativePasteError::InjectionFailed);
        };
        CGEvent::set_flags(Some(&event), CGEventFlags(stroke.flags));
        events.push(event);
    }
    for event in &events {
        CGEvent::post(CGEventTapLocation::HIDEventTap, Some(event));
    }
    NativePasteInjection::Sent
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_v_is_one_balanced_down_up_sequence() {
        assert_eq!(COMMAND_V.len(), 4);
        assert_eq!(COMMAND_V[0].keycode, COMMAND_KEYCODE);
        assert!(COMMAND_V[0].key_down);
        assert_eq!(COMMAND_V[1].keycode, V_KEYCODE);
        assert!(COMMAND_V[1].key_down);
        assert_eq!(COMMAND_V[2].keycode, V_KEYCODE);
        assert!(!COMMAND_V[2].key_down);
        assert_eq!(COMMAND_V[3].keycode, COMMAND_KEYCODE);
        assert!(!COMMAND_V[3].key_down);
    }

    #[test]
    fn every_v_event_carries_command_and_final_release_clears_it() {
        assert_eq!(COMMAND_V[1].flags, COMMAND_FLAG);
        assert_eq!(COMMAND_V[2].flags, COMMAND_FLAG);
        assert_eq!(COMMAND_V[3].flags, 0);
    }
}
