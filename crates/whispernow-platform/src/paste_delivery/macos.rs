//! macOS destination capture and paste: waits for the modifiers, re-checks Accessibility
//! trust and focus, then posts a balanced Core Graphics Command-V.

use core::ptr::{NonNull, null};

use objc2_app_kit::NSWorkspace;
use objc2_application_services::{AXError, AXUIElement};
use objc2_core_foundation::{CFHash, CFRetained, CFString, CFType};
use objc2_core_graphics::{
    CGEvent, CGEventFlags, CGEventSource, CGEventSourceStateID, CGEventTapLocation,
};

use super::{
    NativePasteError, NativePasteInjection, accessibility_trusted, still_focused,
    wait_until_modifiers_released,
};
use crate::{PasteKeys, destination::NativeDestination};

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

#[allow(unsafe_code, reason = "the Accessibility API is FFI")]
pub(super) fn capture_system_destination() -> Result<Option<NativeDestination>, NativePasteError> {
    if !accessibility_trusted() {
        return Err(NativePasteError::PermissionDenied);
    }
    let application = NSWorkspace::sharedWorkspace()
        .frontmostApplication()
        .ok_or(NativePasteError::Unavailable)?;
    let pid = application.processIdentifier();
    let process_id = u32::try_from(pid).map_err(|_| NativePasteError::Unavailable)?;
    if process_id == 0 {
        return Err(NativePasteError::Unavailable);
    }

    // SAFETY: the PID came from NSWorkspace's current frontmost application.
    let application_element = unsafe { AXUIElement::new_application(pid) };
    let window = copy_ax_identity(&application_element, "AXFocusedWindow")?;
    let focused_element = copy_ax_identity(&application_element, "AXFocusedUIElement")?;
    Ok(Some(NativeDestination::new(
        window,
        process_id,
        focused_element,
        PasteKeys::ControlV,
    )))
}

#[allow(unsafe_code, reason = "AXUIElementCopyAttributeValue is FFI")]
fn copy_ax_identity(
    element: &AXUIElement,
    attribute: &'static str,
) -> Result<u64, NativePasteError> {
    let attribute = CFString::from_static_str(attribute);
    let mut raw: *const CFType = null();
    let output = NonNull::from(&mut raw);
    // SAFETY: `output` points to a writable CFTypeRef slot. A successful Copy
    // call returns a +1 object owned by the CFRetained wrapper below.
    let result = unsafe { element.copy_attribute_value(&attribute, output) };
    if result != AXError::Success {
        return Err(NativePasteError::Unavailable);
    }
    let raw = NonNull::new(raw.cast_mut()).ok_or(NativePasteError::Unavailable)?;
    // SAFETY: AXUIElementCopyAttributeValue returned this object at +1.
    let value: CFRetained<CFType> = unsafe { CFRetained::from_raw(raw) };
    let identity =
        u64::try_from(CFHash(Some(&value))).map_err(|_| NativePasteError::Unavailable)?;
    if identity == 0 {
        Err(NativePasteError::Unavailable)
    } else {
        Ok(identity)
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
