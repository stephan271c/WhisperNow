//! macOS paste: waits for the modifiers, re-checks Accessibility trust and
//! focus, then posts a balanced Core Graphics Command-V.

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

    let modifier_mask = CGEventFlags::MaskCommand
        | CGEventFlags::MaskControl
        | CGEventFlags::MaskAlternate
        | CGEventFlags::MaskShift;
    CGEventSource::flags_state(CGEventSourceStateID::HIDSystemState).intersects(modifier_mask)
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
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn command_v_is_one_balanced_down_up_sequence() {
        let pressed: Vec<_> = COMMAND_V
            .iter()
            .filter(|stroke| stroke.key_down)
            .map(|stroke| stroke.keycode)
            .collect();
        assert_eq!(pressed, [55, 9], "paste must press Command then V");
        for stroke in COMMAND_V.iter().filter(|stroke| stroke.keycode == 9) {
            assert_eq!(
                stroke.flags & 1_048_576,
                1_048_576,
                "V must carry the Command modifier on press and release"
            );
        }
        let mut held = HashSet::new();
        for stroke in COMMAND_V {
            assert!(if stroke.key_down {
                held.insert(stroke.keycode)
            } else {
                held.remove(&stroke.keycode)
            });
        }
        assert!(held.is_empty());
        assert_eq!(COMMAND_V.last().expect("paste sequence").flags, 0);
    }
}
