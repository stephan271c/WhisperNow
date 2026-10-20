//! macOS paste: waits for the modifiers, re-checks Accessibility trust and
//! focus, then posts a balanced Core Graphics Command-V.

use objc2_core_graphics::{
    CGEvent, CGEventFlags, CGEventSource, CGEventSourceStateID, CGEventTapLocation,
};

use super::{NativePasteError, NativePasteInjection, wait_until_modifiers_released};
use crate::destination::{NativeDestination, still_focused};

const COMMAND_KEYCODE: u16 = 55;
const V_KEYCODE: u16 = 9;

pub(super) fn inject_if_current(destination: NativeDestination) -> NativePasteInjection {
    match wait_until_modifiers_released(|| Ok(modifier_held()))
        .and_then(|()| still_focused(destination))
    {
        Ok(()) => post_command_v(),
        Err(error) => NativePasteInjection::NotSent(error),
    }
}

fn modifier_held() -> bool {
    let modifier_mask = CGEventFlags::MaskCommand
        | CGEventFlags::MaskControl
        | CGEventFlags::MaskAlternate
        | CGEventFlags::MaskShift;
    CGEventSource::flags_state(CGEventSourceStateID::HIDSystemState).intersects(modifier_mask)
}

fn post_command_v() -> NativePasteInjection {
    let Some(source) = CGEventSource::new(CGEventSourceStateID::HIDSystemState) else {
        return NativePasteInjection::NotSent(NativePasteError::InjectionFailed);
    };
    let pressed = [COMMAND_KEYCODE, V_KEYCODE];
    let downs = pressed.into_iter().map(|keycode| (keycode, true));
    let ups = pressed.into_iter().rev().map(|keycode| (keycode, false));
    let mut events = Vec::with_capacity(2 * pressed.len());
    for (keycode, key_down) in downs.chain(ups) {
        let Some(event) = CGEvent::new_keyboard_event(Some(&source), keycode, key_down) else {
            return NativePasteInjection::NotSent(NativePasteError::InjectionFailed);
        };
        let flags = if keycode == COMMAND_KEYCODE && !key_down {
            CGEventFlags::empty()
        } else {
            CGEventFlags::MaskCommand
        };
        CGEvent::set_flags(Some(&event), flags);
        events.push(event);
    }
    for event in &events {
        CGEvent::post(CGEventTapLocation::HIDEventTap, Some(event));
    }
    NativePasteInjection::Sent
}
