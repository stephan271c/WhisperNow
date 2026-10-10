//! Windows target revalidation and keyboard-only paste injection.
//!
//! This boundary accepts no clipboard or transcript data. It emits one fixed,
//! balanced `SendInput` batch only after all physical modifiers have been
//! observed released and the capture-time top-level window, process, and
//! focused child have been checked again.

use std::{mem::size_of, ptr};

use super::{NativePasteError, NativePasteInjection, wait_until_modifiers_released};
use windows_sys::Win32::{
    Foundation::{
        CloseHandle, ERROR_ACCESS_DENIED, ERROR_INSUFFICIENT_BUFFER, GetLastError, HANDLE,
    },
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
};

use crate::{
    PasteKeys,
    destination::{NativeDestination, still_focused},
};

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
const MAX_SHORTCUT_EVENTS: usize = 6;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KeyStroke {
    virtual_key: VIRTUAL_KEY,
    key_up: bool,
    extended: bool,
}

impl KeyStroke {
    const fn down(virtual_key: VIRTUAL_KEY) -> Self {
        Self {
            virtual_key,
            key_up: false,
            extended: false,
        }
    }

    const fn down_extended(virtual_key: VIRTUAL_KEY) -> Self {
        Self {
            virtual_key,
            key_up: false,
            extended: true,
        }
    }

    const fn up(virtual_key: VIRTUAL_KEY) -> Self {
        Self {
            virtual_key,
            key_up: true,
            extended: false,
        }
    }

    const fn up_extended(virtual_key: VIRTUAL_KEY) -> Self {
        Self {
            virtual_key,
            key_up: true,
            extended: true,
        }
    }
}

const CONTROL_V: [KeyStroke; 4] = [
    KeyStroke::down(VK_CONTROL),
    KeyStroke::down(VK_V),
    KeyStroke::up(VK_V),
    KeyStroke::up(VK_CONTROL),
];
const CONTROL_SHIFT_V: [KeyStroke; 6] = [
    KeyStroke::down(VK_CONTROL),
    KeyStroke::down(VK_SHIFT),
    KeyStroke::down(VK_V),
    KeyStroke::up(VK_V),
    KeyStroke::up(VK_SHIFT),
    KeyStroke::up(VK_CONTROL),
];
const SHIFT_INSERT: [KeyStroke; 4] = [
    KeyStroke::down(VK_SHIFT),
    KeyStroke::down_extended(VK_INSERT),
    KeyStroke::up_extended(VK_INSERT),
    KeyStroke::up(VK_SHIFT),
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

struct OwnedHandle(HANDLE);

impl Drop for OwnedHandle {
    fn drop(&mut self) {
        // SAFETY: this wrapper is created only for real handles returned by
        // `OpenProcess` or `OpenProcessToken` and owns each handle exactly once.
        let _ = unsafe { CloseHandle(self.0) };
    }
}

pub(super) fn ensure_target_integrity(process_id: u32) -> Result<(), NativePasteError> {
    // SAFETY: the process identifier came from the foreground HWND. The handle
    // is opened query-only and immediately wrapped for deterministic closure.
    let target_process =
        unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false.into(), process_id) };
    if target_process.is_null() {
        return Err(last_query_failure());
    }
    let target_process = OwnedHandle(target_process);

    // SAFETY: `GetCurrentProcess` returns a process pseudo-handle that must not
    // be closed and remains valid for this query.
    let current_integrity = token_integrity_level(unsafe { GetCurrentProcess() })?;
    let target_integrity = token_integrity_level(target_process.0)?;
    classify_integrity_levels(current_integrity, target_integrity)
}

fn token_integrity_level(process: HANDLE) -> Result<u32, NativePasteError> {
    let mut token = ptr::null_mut();
    // SAFETY: `process` is a valid process or current-process pseudo-handle;
    // `token` remains writable for the call and requests query access only.
    if unsafe { OpenProcessToken(process, TOKEN_QUERY, &raw mut token) } == 0 {
        return Err(last_query_failure());
    }
    let token = OwnedHandle(token);

    let mut required_bytes = 0_u32;
    // SAFETY: the first call intentionally supplies no buffer to obtain the
    // required byte count documented for `GetTokenInformation`.
    let first_result = unsafe {
        GetTokenInformation(
            token.0,
            TokenIntegrityLevel,
            ptr::null_mut(),
            0,
            &raw mut required_bytes,
        )
    };
    // SAFETY: `GetLastError` reads thread-local error state immediately after
    // the failed size query.
    let size_error = unsafe { GetLastError() };
    if first_result != 0
        || size_error != ERROR_INSUFFICIENT_BUFFER
        || usize::try_from(required_bytes).map_or(true, |size| {
            size < size_of::<TOKEN_MANDATORY_LABEL>() || size > 64 * 1024
        })
    {
        return Err(if size_error == ERROR_ACCESS_DENIED {
            NativePasteError::PermissionDenied
        } else {
            NativePasteError::Unavailable
        });
    }

    let required = usize::try_from(required_bytes).map_err(|_| NativePasteError::Unavailable)?;
    let words = required
        .checked_add(size_of::<usize>() - 1)
        .map(|size| size / size_of::<usize>())
        .ok_or(NativePasteError::Unavailable)?;
    let mut buffer = vec![0_usize; words];
    // SAFETY: the allocation is at least `required_bytes`, aligned for the
    // pointer-containing label structure, and remains writable for the call.
    if unsafe {
        GetTokenInformation(
            token.0,
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

fn last_query_failure() -> NativePasteError {
    // SAFETY: callers invoke this immediately after a failing Win32 query.
    if unsafe { GetLastError() } == ERROR_ACCESS_DENIED {
        NativePasteError::PermissionDenied
    } else {
        NativePasteError::Unavailable
    }
}

pub(super) fn physical_modifier_is_down() -> bool {
    MODIFIER_KEYS.into_iter().any(|virtual_key| {
        // SAFETY: every value is a documented virtual-key constant. Only the
        // high bit is inspected because it represents the current physical
        // key state; the unreliable transition bit is deliberately ignored.
        let state = unsafe { GetAsyncKeyState(i32::from(virtual_key)) };
        (u16::from_ne_bytes(state.to_ne_bytes()) & 0x8000) != 0
    })
}

fn send_shortcut(keys: PasteKeys) -> NativePasteInjection {
    let strokes = shortcut_strokes(keys);
    let mut inputs = [INPUT::default(); MAX_SHORTCUT_EVENTS];
    for (input, stroke) in inputs.iter_mut().zip(strokes) {
        *input = native_input(*stroke);
    }

    let requested = u32::try_from(strokes.len()).unwrap_or(0);
    let input_size = i32::try_from(size_of::<INPUT>()).unwrap_or(0);
    if requested == 0 || input_size == 0 {
        return NativePasteInjection::NotSent(NativePasteError::InjectionFailed);
    }

    // SAFETY: the slice contains exactly `requested` initialized INPUT values,
    // and `input_size` is the ABI size of INPUT. This is the only SendInput
    // call, keeping every shortcut in one balanced batch.
    let sent = unsafe { SendInput(requested, inputs.as_ptr(), input_size) };
    map_send_result(sent, requested)
}

const fn shortcut_strokes(keys: PasteKeys) -> &'static [KeyStroke] {
    match keys {
        PasteKeys::ControlV => &CONTROL_V,
        PasteKeys::ControlShiftV => &CONTROL_SHIFT_V,
        PasteKeys::ShiftInsert => &SHIFT_INSERT,
    }
}

const fn native_input(stroke: KeyStroke) -> INPUT {
    let mut flags = if stroke.key_up { KEYEVENTF_KEYUP } else { 0 };
    if stroke.extended {
        flags |= KEYEVENTF_EXTENDEDKEY;
    }
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: stroke.virtual_key,
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
    fn fixed_shortcuts_are_balanced_and_ordered() {
        assert_eq!(
            shortcut_strokes(PasteKeys::ControlV),
            [
                KeyStroke::down(VK_CONTROL),
                KeyStroke::down(VK_V),
                KeyStroke::up(VK_V),
                KeyStroke::up(VK_CONTROL),
            ]
        );
        assert_eq!(
            shortcut_strokes(PasteKeys::ControlShiftV),
            [
                KeyStroke::down(VK_CONTROL),
                KeyStroke::down(VK_SHIFT),
                KeyStroke::down(VK_V),
                KeyStroke::up(VK_V),
                KeyStroke::up(VK_SHIFT),
                KeyStroke::up(VK_CONTROL),
            ]
        );
        assert_eq!(
            shortcut_strokes(PasteKeys::ShiftInsert),
            [
                KeyStroke::down(VK_SHIFT),
                KeyStroke::down_extended(VK_INSERT),
                KeyStroke::up_extended(VK_INSERT),
                KeyStroke::up(VK_SHIFT),
            ]
        );
    }

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
        let down = native_input(KeyStroke::down_extended(VK_INSERT));
        let up = native_input(KeyStroke::up_extended(VK_INSERT));

        // SAFETY: `native_input` initialized the active union member as `ki`.
        let down_flags = unsafe { down.Anonymous.ki.dwFlags };
        // SAFETY: `native_input` initialized the active union member as `ki`.
        let up_flags = unsafe { up.Anonymous.ki.dwFlags };
        assert_eq!(down_flags, KEYEVENTF_EXTENDEDKEY);
        assert_eq!(up_flags, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP);
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
