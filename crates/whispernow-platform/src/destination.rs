//! Captures the focused window immediately before a delivery; nothing is retained from the
//! start of the recording.

use std::process;

#[cfg(target_os = "linux")]
use x11rb::{
    connection::Connection as _,
    cookie::Cookie,
    errors::ConnectionError,
    protocol::xproto::{AtomEnum, ConnectionExt as _, InputFocus, Window},
    rust_connection::RustConnection,
    x11_utils::TryParse,
};

#[cfg(target_os = "macos")]
use crate::paste_delivery::accessibility_trusted;
use crate::{PasteKeys, paste_delivery::NativePasteError};

#[cfg(target_os = "windows")]
type NativeWindow = usize;
#[cfg(target_os = "linux")]
type NativeWindow = u32;
#[cfg(not(any(target_os = "windows", target_os = "linux")))]
type NativeWindow = u64;

/// Opaque identity held only for the duration of one output operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct NativeDestination {
    window: NativeWindow,
    process_id: u32,
    focused_element: u64,
    keys: PasteKeys,
}

impl NativeDestination {
    #[cfg(target_os = "windows")]
    pub(crate) const fn process_id(self) -> u32 {
        self.process_id
    }

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub(crate) const fn keys(self) -> PasteKeys {
        self.keys
    }
}

pub(crate) fn capture() -> Result<NativeDestination, NativePasteError> {
    foreign(capture_system_destination()?, process::id())
}

/// `found`, unless nothing has focus or this application, `own_process_id`,
/// has it: a transcript is never pasted into the app itself.
fn foreign(
    found: Option<NativeDestination>,
    own_process_id: u32,
) -> Result<NativeDestination, NativePasteError> {
    found
        .filter(|destination| destination.process_id != own_process_id)
        .ok_or(NativePasteError::Unavailable)
}

/// Whether focus is still where `expected` was captured.
#[cfg(any(target_os = "windows", target_os = "macos"))]
pub(crate) fn still_focused(expected: NativeDestination) -> Result<(), NativePasteError> {
    if capture_system_destination()? == Some(expected) {
        Ok(())
    } else {
        Err(NativePasteError::Unavailable)
    }
}

#[cfg(target_os = "macos")]
fn capture_system_destination() -> Result<Option<NativeDestination>, NativePasteError> {
    use objc2_app_kit::NSWorkspace;
    use objc2_application_services::AXUIElement;

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
    Ok(Some(NativeDestination {
        window,
        process_id,
        focused_element,
        keys: PasteKeys::ControlV,
    }))
}

#[cfg(target_os = "macos")]
fn copy_ax_identity(
    element: &objc2_application_services::AXUIElement,
    attribute: &'static str,
) -> Result<u64, NativePasteError> {
    use core::ptr::{NonNull, null};

    use objc2_core_foundation::{CFHash, CFRetained, CFString, CFType};

    let attribute = CFString::from_static_str(attribute);
    let mut raw: *const CFType = null();
    let output = NonNull::from(&mut raw);
    // SAFETY: `output` points to a writable CFTypeRef slot. A successful Copy
    // call returns a +1 object owned by the CFRetained wrapper below.
    let result = unsafe { element.copy_attribute_value(&attribute, output) };
    if result.0 != 0 {
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

#[cfg(target_os = "windows")]
fn capture_system_destination() -> Result<Option<NativeDestination>, NativePasteError> {
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GUITHREADINFO, GetClassNameW, GetForegroundWindow, GetGUIThreadInfo,
        GetWindowThreadProcessId,
    };

    // SAFETY: these calls only query opaque window identities owned by User32.
    let window = unsafe { GetForegroundWindow() };
    if window.is_null() {
        return Ok(None);
    }
    let mut process_id = 0_u32;
    // SAFETY: `window` came from User32 and the process-id pointer is valid.
    let thread_id = unsafe { GetWindowThreadProcessId(window, &raw mut process_id) };
    if thread_id == 0 || process_id == 0 {
        return Err(NativePasteError::Unavailable);
    }
    let mut gui = GUITHREADINFO {
        cbSize: u32::try_from(core::mem::size_of::<GUITHREADINFO>())
            .map_err(|_| NativePasteError::Unavailable)?,
        ..GUITHREADINFO::default()
    };
    // SAFETY: `gui` has the documented size and remains writable for the call.
    if unsafe { GetGUIThreadInfo(thread_id, &raw mut gui) } == 0
        || gui.hwndActive != window
        || gui.hwndFocus.is_null()
    {
        return Err(NativePasteError::Unavailable);
    }
    let mut class_name = [0_u16; 128];
    // SAFETY: the buffer is writable and the HWND came from User32.
    let class_length = unsafe {
        GetClassNameW(
            window,
            class_name.as_mut_ptr(),
            i32::try_from(class_name.len()).unwrap_or(i32::MAX),
        )
    };
    let keys = if class_length > 0 {
        windows_paste_keys(&class_name[..usize::try_from(class_length).unwrap_or(0)])
    } else {
        PasteKeys::ControlV
    };
    Ok(Some(NativeDestination {
        window: window as usize,
        process_id,
        focused_element: gui.hwndFocus as usize as u64,
        keys,
    }))
}

#[cfg(target_os = "windows")]
fn windows_paste_keys(class_name: &[u16]) -> PasteKeys {
    if class_name
        == "CASCADIA_HOSTING_WINDOW_CLASS"
            .encode_utf16()
            .collect::<Vec<_>>()
    {
        PasteKeys::ControlShiftV
    } else if class_name == "ConsoleWindowClass".encode_utf16().collect::<Vec<_>>() {
        PasteKeys::ShiftInsert
    } else {
        PasteKeys::ControlV
    }
}

#[cfg(target_os = "linux")]
fn capture_system_destination() -> Result<Option<NativeDestination>, NativePasteError> {
    let (connection, root) = connect_x11()?;
    capture_on(&connection, root)
}

#[cfg(target_os = "linux")]
pub(crate) fn connect_x11() -> Result<(RustConnection, Window), NativePasteError> {
    let (connection, screen_number) =
        x11rb::connect(None).map_err(|_| NativePasteError::Unavailable)?;
    let root = connection
        .setup()
        .roots
        .get(screen_number)
        .ok_or(NativePasteError::Unavailable)?
        .root;
    Ok((connection, root))
}

#[cfg(target_os = "linux")]
pub(crate) fn reply<R: TryParse>(
    request: Result<Cookie<'_, RustConnection, R>, ConnectionError>,
) -> Result<R, NativePasteError> {
    request
        .ok()
        .and_then(|cookie| cookie.reply().ok())
        .ok_or(NativePasteError::Unavailable)
}

#[cfg(target_os = "linux")]
pub(crate) fn capture_on(
    connection: &RustConnection,
    root: Window,
) -> Result<Option<NativeDestination>, NativePasteError> {
    let window = x11_property_u32(connection, root, b"_NET_ACTIVE_WINDOW", AtomEnum::WINDOW)?;
    if window == 0 || window == root {
        return Ok(None);
    }
    let process_id = x11_property_u32(connection, window, b"_NET_WM_PID", AtomEnum::CARDINAL)?;
    if process_id == 0 {
        return Err(NativePasteError::Unavailable);
    }
    let focus = reply(connection.get_input_focus())?.focus;
    if focus == 0 || focus == u32::from(InputFocus::POINTER_ROOT) || focus == root {
        return Err(NativePasteError::Unavailable);
    }
    Ok(Some(NativeDestination {
        window,
        process_id,
        focused_element: u64::from(focus),
        keys: x11_paste_keys(connection, window).unwrap_or(PasteKeys::ControlV),
    }))
}

/// A window property holding exactly one 32-bit value of `expected_type`.
#[cfg(target_os = "linux")]
fn x11_property_u32(
    connection: &RustConnection,
    window: Window,
    name: &[u8],
    expected_type: AtomEnum,
) -> Result<u32, NativePasteError> {
    let property = reply(connection.intern_atom(false, name))?.atom;
    let expected_type = u32::from(expected_type);
    let value = reply(connection.get_property(false, window, property, expected_type, 0, 1))?;
    if value.type_ != expected_type
        || value.format != 32
        || value.value_len != 1
        || value.bytes_after != 0
    {
        return Err(NativePasteError::Unavailable);
    }
    value
        .value32()
        .and_then(|mut values| values.next())
        .ok_or(NativePasteError::Unavailable)
}

/// The keys for the window's `WM_CLASS`, whose instance and class names are
/// separated by NUL bytes.
#[cfg(target_os = "linux")]
fn x11_paste_keys(connection: &RustConnection, window: Window) -> Option<PasteKeys> {
    let class =
        reply(connection.get_property(false, window, AtomEnum::WM_CLASS, AtomEnum::STRING, 0, 64))
            .ok()?;
    if class.type_ != u32::from(AtomEnum::STRING) || class.format != 8 || class.bytes_after != 0 {
        return None;
    }
    let keys = String::from_utf8_lossy(&class.value)
        .split('\0')
        .map(PasteKeys::for_application)
        .find(|keys| *keys != PasteKeys::ControlV)
        .unwrap_or(PasteKeys::ControlV);
    Some(keys)
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn capture_system_destination() -> Result<Option<NativeDestination>, NativePasteError> {
    Err(NativePasteError::Unavailable)
}

#[cfg(all(
    test,
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
mod tests {
    use super::*;

    fn destination(process_id: u32) -> NativeDestination {
        NativeDestination {
            window: 10,
            process_id,
            focused_element: 30,
            keys: PasteKeys::ControlV,
        }
    }

    #[test]
    fn another_application_with_focus_is_the_target() {
        assert_eq!(foreign(Some(destination(7)), 8), Ok(destination(7)));
    }

    #[test]
    fn this_application_or_no_focus_is_not_a_target() {
        assert_eq!(
            foreign(Some(destination(8)), 8),
            Err(NativePasteError::Unavailable)
        );
        assert_eq!(foreign(None, 8), Err(NativePasteError::Unavailable));
    }
}
