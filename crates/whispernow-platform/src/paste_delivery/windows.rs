//! Windows destination capture and paste: one fixed, balanced `SendInput` batch, sent only
//! after the target's integrity level is no higher than ours, the physical modifiers are
//! released, and the capture-time window, process and focused child match.
#![allow(
    unsafe_code,
    reason = "SendInput, token queries and User32 window queries are Win32 FFI"
)]

use std::{
    io,
    mem::size_of,
    os::windows::io::{AsRawHandle as _, FromRawHandle as _, OwnedHandle},
    ptr,
};

use super::{NativePasteError, NativePasteInjection, still_focused, wait_until_modifiers_released};
use windows_sys::Win32::{
    Foundation::{ERROR_ACCESS_DENIED, ERROR_INSUFFICIENT_BUFFER, HANDLE},
    Security::{
        GetSidSubAuthority, GetSidSubAuthorityCount, GetTokenInformation, IsValidSid,
        TOKEN_MANDATORY_LABEL, TOKEN_QUERY, TokenIntegrityLevel,
    },
    System::Threading::{
        GetCurrentProcess, OpenProcess, OpenProcessToken, PROCESS_QUERY_LIMITED_INFORMATION,
    },
    UI::Input::KeyboardAndMouse::{
        GetAsyncKeyState, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYEVENTF_EXTENDEDKEY,
        KEYEVENTF_KEYUP, SendInput, VIRTUAL_KEY, VK_CONTROL, VK_INSERT, VK_LCONTROL, VK_LMENU,
        VK_LSHIFT, VK_LWIN, VK_RCONTROL, VK_RMENU, VK_RSHIFT, VK_RWIN, VK_SHIFT, VK_V,
    },
    UI::WindowsAndMessaging::{
        GUITHREADINFO, GetClassNameW, GetForegroundWindow, GetGUIThreadInfo,
        GetWindowThreadProcessId,
    },
};

use crate::{PasteKeys, destination::NativeDestination};

const ACCESS_DENIED: Option<i32> = Some(ERROR_ACCESS_DENIED.cast_signed());
const INSUFFICIENT_BUFFER: Option<i32> = Some(ERROR_INSUFFICIENT_BUFFER.cast_signed());

const MODIFIER_KEYS: [VIRTUAL_KEY; 8] = [
    VK_LCONTROL,
    VK_RCONTROL,
    VK_LSHIFT,
    VK_RSHIFT,
    VK_LMENU,
    VK_RMENU,
    VK_LWIN,
    VK_RWIN,
];

#[derive(Clone, Copy)]
struct Key {
    virtual_key: VIRTUAL_KEY,
    extended: bool,
}

const fn plain(virtual_key: VIRTUAL_KEY) -> Key {
    Key {
        virtual_key,
        extended: false,
    }
}

const CONTROL_V: [Key; 2] = [plain(VK_CONTROL), plain(VK_V)];
const CONTROL_SHIFT_V: [Key; 3] = [plain(VK_CONTROL), plain(VK_SHIFT), plain(VK_V)];
const SHIFT_INSERT: [Key; 2] = [
    plain(VK_SHIFT),
    Key {
        virtual_key: VK_INSERT,
        extended: true,
    },
];

pub(super) fn inject_if_current(destination: NativeDestination) -> NativePasteInjection {
    let checked = ensure_target_integrity(destination.process_id())
        .and_then(|()| wait_until_modifiers_released(|| Ok(physical_modifier_is_down())))
        .and_then(|()| still_focused(destination));
    match checked {
        Ok(()) => send_shortcut(destination.keys()),
        Err(failure) => NativePasteInjection::NotSent(failure),
    }
}

pub(super) fn capture_system_destination() -> Result<Option<NativeDestination>, NativePasteError> {
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
        cbSize: u32::try_from(size_of::<GUITHREADINFO>())
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
        PasteKeys::for_window_class(&class_name[..usize::try_from(class_length).unwrap_or(0)])
    } else {
        PasteKeys::ControlV
    };
    Ok(Some(NativeDestination::new(
        window as usize as u64,
        process_id,
        gui.hwndFocus as usize as u64,
        keys,
    )))
}

fn ensure_target_integrity(process_id: u32) -> Result<(), NativePasteError> {
    // SAFETY: the process identifier came from the foreground HWND. The handle
    // is opened query-only.
    let target_process =
        unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false.into(), process_id) };
    if target_process.is_null() {
        return Err(last_query_failure());
    }
    // SAFETY: `OpenProcess` returned a new open handle that nothing else owns.
    let target_process = unsafe { OwnedHandle::from_raw_handle(target_process) };

    // SAFETY: `GetCurrentProcess` returns a process pseudo-handle that must not
    // be closed and remains valid for this query.
    let current_integrity = token_integrity_level(unsafe { GetCurrentProcess() })?;
    let target_integrity = token_integrity_level(target_process.as_raw_handle())?;
    classify_integrity_levels(current_integrity, target_integrity)
}

fn token_integrity_level(process: HANDLE) -> Result<u32, NativePasteError> {
    let mut token = ptr::null_mut();
    // SAFETY: `process` is a valid process or current-process pseudo-handle;
    // `token` remains writable for the call and requests query access only.
    if unsafe { OpenProcessToken(process, TOKEN_QUERY, &raw mut token) } == 0 {
        return Err(last_query_failure());
    }
    // SAFETY: `OpenProcessToken` succeeded, so `token` is a new open handle that nothing
    // else owns.
    let token = unsafe { OwnedHandle::from_raw_handle(token) };

    let mut required_bytes = 0_u32;
    // SAFETY: the first call intentionally supplies no buffer to obtain the
    // required byte count documented for `GetTokenInformation`.
    let first_result = unsafe {
        GetTokenInformation(
            token.as_raw_handle(),
            TokenIntegrityLevel,
            ptr::null_mut(),
            0,
            &raw mut required_bytes,
        )
    };
    let size_error = io::Error::last_os_error().raw_os_error();
    let required = usize::try_from(required_bytes).map_err(|_| NativePasteError::Unavailable)?;
    if first_result != 0
        || size_error != INSUFFICIENT_BUFFER
        || required < size_of::<TOKEN_MANDATORY_LABEL>()
        || required > 64 * 1024
    {
        return Err(if size_error == ACCESS_DENIED {
            NativePasteError::PermissionDenied
        } else {
            NativePasteError::Unavailable
        });
    }

    let words = required.div_ceil(size_of::<usize>());
    let mut buffer = vec![0_usize; words];
    // SAFETY: the allocation is at least `required_bytes`, aligned for the
    // pointer-containing label structure, and remains writable for the call.
    if unsafe {
        GetTokenInformation(
            token.as_raw_handle(),
            TokenIntegrityLevel,
            buffer.as_mut_ptr().cast(),
            required_bytes,
            &raw mut required_bytes,
        )
    } == 0
    {
        return Err(last_query_failure());
    }

    // SAFETY: the successful API call initialized a `TOKEN_MANDATORY_LABEL` at
    // the aligned start of `buffer`; its SID pointer remains owned by that
    // buffer until this function returns.
    let label = unsafe { &*buffer.as_ptr().cast::<TOKEN_MANDATORY_LABEL>() };
    let sid = label.Label.Sid;
    // SAFETY: the SID came from the validated token-information buffer.
    if sid.is_null() || unsafe { IsValidSid(sid) } == 0 {
        return Err(NativePasteError::Unavailable);
    }
    // SAFETY: a valid SID exposes its sub-authority count and corresponding
    // sub-authorities for the lifetime of the token-information buffer.
    let count = unsafe { GetSidSubAuthorityCount(sid) };
    if count.is_null() {
        return Err(NativePasteError::Unavailable);
    }
    // SAFETY: `count` belongs to the validated SID and was checked non-null.
    let count = unsafe { *count };
    if count == 0 {
        return Err(NativePasteError::Unavailable);
    }
    // SAFETY: the last sub-authority is the mandatory integrity RID and the
    // validated SID contains at least one sub-authority.
    let rid = unsafe { GetSidSubAuthority(sid, u32::from(count - 1)) };
    if rid.is_null() {
        return Err(NativePasteError::Unavailable);
    }
    // SAFETY: `rid` points inside the live validated SID buffer.
    Ok(unsafe { *rid })
}

const fn classify_integrity_levels(
    current_integrity: u32,
    target_integrity: u32,
) -> Result<(), NativePasteError> {
    if target_integrity > current_integrity {
        Err(NativePasteError::PermissionDenied)
    } else {
        Ok(())
    }
}

/// Callers invoke this immediately after a failing Win32 query, before any other call
/// overwrites the thread's last error.
fn last_query_failure() -> NativePasteError {
    if io::Error::last_os_error().raw_os_error() == ACCESS_DENIED {
        NativePasteError::PermissionDenied
    } else {
        NativePasteError::Unavailable
    }
}

fn physical_modifier_is_down() -> bool {
    MODIFIER_KEYS.into_iter().any(|virtual_key| {
        // SAFETY: every value is a documented virtual-key constant. Only the
        // high bit is inspected because it represents the current physical
        // key state; the unreliable transition bit is deliberately ignored.
        let state = unsafe { GetAsyncKeyState(i32::from(virtual_key)) };
        state < 0
    })
}

fn send_shortcut(keys: PasteKeys) -> NativePasteInjection {
    let inputs = shortcut_inputs(shortcut_keys(keys));
    let requested = u32::try_from(inputs.len()).expect("shortcut length fits u32");
    let input_size = i32::try_from(size_of::<INPUT>()).expect("INPUT size fits i32");

    // SAFETY: the slice contains exactly `requested` initialized INPUT values,
    // and `input_size` is the ABI size of INPUT. This is the only SendInput
    // call, keeping every shortcut in one balanced batch.
    let sent = unsafe { SendInput(requested, inputs.as_ptr(), input_size) };
    map_send_result(sent, requested)
}

const fn shortcut_keys(keys: PasteKeys) -> &'static [Key] {
    match keys {
        PasteKeys::ControlV => &CONTROL_V,
        PasteKeys::ControlShiftV => &CONTROL_SHIFT_V,
        PasteKeys::ShiftInsert => &SHIFT_INSERT,
    }
}

fn shortcut_inputs(pressed: &[Key]) -> Vec<INPUT> {
    let downs = pressed.iter().map(|key| native_input(*key, false));
    let ups = pressed.iter().rev().map(|key| native_input(*key, true));
    downs.chain(ups).collect()
}

const fn native_input(key: Key, key_up: bool) -> INPUT {
    let mut flags = if key_up { KEYEVENTF_KEYUP } else { 0 };
    if key.extended {
        flags |= KEYEVENTF_EXTENDEDKEY;
    }
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: key.virtual_key,
                wScan: 0,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: 0,
            },
        },
    }
}

const fn map_send_result(sent: u32, requested: u32) -> NativePasteInjection {
    if sent == requested {
        NativePasteInjection::Sent
    } else if sent == 0 {
        NativePasteInjection::NotSent(NativePasteError::InjectionFailed)
    } else {
        NativePasteInjection::Uncertain(NativePasteError::InjectionFailed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn send_input_count_has_explicit_mutation_semantics() {
        assert_eq!(map_send_result(4, 4), NativePasteInjection::Sent);
        assert_eq!(
            map_send_result(0, 4),
            NativePasteInjection::NotSent(NativePasteError::InjectionFailed)
        );
        assert_eq!(
            map_send_result(2, 4),
            NativePasteInjection::Uncertain(NativePasteError::InjectionFailed)
        );
    }

    #[test]
    fn insert_events_carry_the_extended_key_flag() {
        let insert_flags: Vec<_> = shortcut_inputs(&SHIFT_INSERT)
            .iter()
            // SAFETY: `native_input` initialized the active union member as `ki`.
            .filter(|input| unsafe { input.Anonymous.ki.wVk } == VK_INSERT)
            // SAFETY: `native_input` initialized the active union member as `ki`.
            .map(|input| unsafe { input.Anonymous.ki.dwFlags })
            .collect();
        assert_eq!(
            insert_flags,
            [
                KEYEVENTF_EXTENDEDKEY,
                KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP
            ]
        );
    }

    #[test]
    fn higher_integrity_targets_are_typed_as_permission_denied() {
        assert_eq!(classify_integrity_levels(0x2000, 0x2000), Ok(()));
        assert_eq!(classify_integrity_levels(0x3000, 0x2000), Ok(()));
        assert_eq!(
            classify_integrity_levels(0x2000, 0x3000),
            Err(NativePasteError::PermissionDenied)
        );
    }
}
