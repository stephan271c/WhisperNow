//! X11 paste injection through the server-provided XTEST extension.
//!
//! Keycodes come from the live core keyboard and modifier maps. The adapter
//! never releases a key the user owns: it waits until no modifier key is
//! physically down, then grabs the server, re-checks the destination and the
//! modifiers, and sends one balanced shortcut.

use x11rb::{
    connection::Connection,
    cookie::Cookie,
    errors::ConnectionError,
    protocol::{
        xkb::{self, ConnectionExt as _},
        xproto::{
            ConnectionExt as _, GetKeyboardMappingReply, GetModifierMappingReply, KEY_PRESS_EVENT,
            KEY_RELEASE_EVENT, Window,
        },
        xtest::ConnectionExt as _,
    },
    rust_connection::RustConnection,
    x11_utils::TryParse,
};

use super::{NativePasteError, NativePasteInjection, wait_until_modifiers_released};
use crate::{
    PasteKeys,
    destination::{NativeDestination, capture_on, connect_x11},
};

const XK_V_LOWER: u32 = 0x0076;
const SHIFT_MODIFIER_ROW: usize = 0;
const CONTROL_MODIFIER_ROW: usize = 2;

#[derive(Debug, Eq, PartialEq)]
struct Keys {
    control: u8,
    shift: u8,
    v: u8,
    /// Every keycode in the modifier map, including locks and `AltGr`.
    modifiers: Vec<u8>,
}

pub(super) fn inject_if_current(target: NativeDestination) -> NativePasteInjection {
    let (connection, root) = match connect_x11() {
        Ok(connected) => connected,
        Err(error) => return NativePasteInjection::NotSent(error),
    };
    let prepared = reply(connection.xkb_use_extension(1, 0))
        .and_then(|_| read_keys(&connection))
        .and_then(|keys| {
            wait_until_modifiers_released(|| any_pressed(&connection, &keys.modifiers))
        });
    if let Err(error) = prepared {
        return NativePasteInjection::NotSent(error);
    }
    if connection.grab_server().is_err() {
        return NativePasteInjection::NotSent(NativePasteError::InjectionFailed);
    }
    let outcome = match revalidate(&connection, root, target) {
        Ok(keys) => send_shortcut(&connection, root, &key_sequence(target.keys(), &keys)),
        Err(error) => NativePasteInjection::NotSent(error),
    };
    let released = connection
        .ungrab_server()
        .is_ok_and(|cookie| cookie.check().is_ok());
    match outcome {
        NativePasteInjection::Sent if !released => {
            NativePasteInjection::Uncertain(NativePasteError::InjectionFailed)
        }
        outcome => outcome,
    }
}

/// Whether the X server offers XTEST 2.1 or later; `None` when no X server
/// could be reached to ask.
pub(super) fn xtest_available() -> Option<bool> {
    x11rb::connect(None)
        .ok()
        .map(|(connection, _)| reply(connection.xtest_get_version(2, 1)).is_ok())
}

fn reply<R: TryParse>(
    request: Result<Cookie<'_, RustConnection, R>, ConnectionError>,
) -> Result<R, NativePasteError> {
    request
        .ok()
        .and_then(|cookie| cookie.reply().ok())
        .ok_or(NativePasteError::Unavailable)
}

fn read_keys(connection: &RustConnection) -> Result<Keys, NativePasteError> {
    let state = reply(connection.xkb_get_state(xkb::ID::USE_CORE_KBD.into()))?;
    if u8::from(state.group) != 0 {
        // Core keyboard mappings do not describe other layout groups well
        // enough to find their `V`, so refuse rather than guess.
        return Err(NativePasteError::Unavailable);
    }
    let setup = connection.setup();
    let count = setup
        .max_keycode
        .saturating_sub(setup.min_keycode)
        .saturating_add(1);
    let keyboard = reply(connection.get_keyboard_mapping(setup.min_keycode, count))?;
    let modifiers = reply(connection.get_modifier_mapping())?;
    resolve_keys(setup.min_keycode, &keyboard, &modifiers).ok_or(NativePasteError::Unavailable)
}

fn resolve_keys(
    min_keycode: u8,
    keyboard: &GetKeyboardMappingReply,
    modifiers: &GetModifierMappingReply,
) -> Option<Keys> {
    let per_keycode = usize::from(keyboard.keysyms_per_keycode);
    let width = usize::from(modifiers.keycodes_per_modifier());
    if per_keycode == 0 || width == 0 {
        return None;
    }
    // Only the unshifted first-group keysym counts, so a `V` on another layout
    // or level is never chosen.
    let keycode_for = |keysym: u32| {
        let offset = keyboard
            .keysyms
            .chunks_exact(per_keycode)
            .position(|keysyms| keysyms[0] == keysym)?;
        min_keycode.checked_add(u8::try_from(offset).ok()?)
    };
    let first_in_row = |row: usize| {
        modifiers
            .keycodes
            .chunks_exact(width)
            .nth(row)?
            .iter()
            .copied()
            .find(|keycode| *keycode != 0)
    };
    let mut modifier_keycodes: Vec<u8> = modifiers
        .keycodes
        .iter()
        .copied()
        .filter(|keycode| *keycode != 0)
        .collect();
    modifier_keycodes.sort_unstable();
    modifier_keycodes.dedup();
    Some(Keys {
        control: first_in_row(CONTROL_MODIFIER_ROW)?,
        shift: first_in_row(SHIFT_MODIFIER_ROW)?,
        v: keycode_for(XK_V_LOWER)?,
        modifiers: modifier_keycodes,
    })
}

fn any_pressed(connection: &RustConnection, keycodes: &[u8]) -> Result<bool, NativePasteError> {
    let bitmap = reply(connection.query_keymap())?.keys;
    Ok(keycodes
        .iter()
        .any(|keycode| bitmap[usize::from(keycode / 8)] & (1 << (keycode % 8)) != 0))
}

/// Runs under the server grab: the destination must be unchanged and no
/// modifier may be down.
fn revalidate(
    connection: &RustConnection,
    root: Window,
    expected: NativeDestination,
) -> Result<Keys, NativePasteError> {
    if capture_on(connection, root)? != Some(expected) {
        return Err(NativePasteError::Unavailable);
    }
    let keys = read_keys(connection)?;
    if any_pressed(connection, &keys.modifiers)? {
        return Err(NativePasteError::ModifierStillHeld);
    }
    Ok(keys)
}

/// The keys to press in order; they are released in reverse order.
fn key_sequence(paste_keys: PasteKeys, keys: &Keys) -> Vec<u8> {
    match paste_keys {
        PasteKeys::ControlV => vec![keys.control, keys.v],
        PasteKeys::ControlShiftV => vec![keys.control, keys.shift, keys.v],
    }
}

fn send_shortcut(
    connection: &RustConnection,
    root: Window,
    presses: &[u8],
) -> NativePasteInjection {
    for (index, keycode) in presses.iter().enumerate() {
        if let Err(failure) = fake_key(connection, root, KEY_PRESS_EVENT, *keycode) {
            let possibly_sent = failure == KeyFailure::MaybeSent;
            release(
                connection,
                root,
                &presses[..index + usize::from(possibly_sent)],
            );
            return if possibly_sent && index + 1 == presses.len() {
                NativePasteInjection::Uncertain(NativePasteError::InjectionFailed)
            } else {
                NativePasteInjection::NotSent(NativePasteError::InjectionFailed)
            };
        }
    }
    if release(connection, root, presses) {
        NativePasteInjection::Sent
    } else {
        NativePasteInjection::Uncertain(NativePasteError::InjectionFailed)
    }
}

/// Releases `pressed` in reverse order, attempting every key.
fn release(connection: &RustConnection, root: Window, pressed: &[u8]) -> bool {
    let mut released = true;
    for keycode in pressed.iter().rev() {
        released &= fake_key(connection, root, KEY_RELEASE_EVENT, *keycode).is_ok();
    }
    released
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum KeyFailure {
    NotSent,
    /// The request went out but the server reported an error, so it may have acted on it.
    MaybeSent,
}

fn fake_key(
    connection: &RustConnection,
    root: Window,
    event: u8,
    keycode: u8,
) -> Result<(), KeyFailure> {
    const CURRENT_TIME: u32 = 0;
    const CORE_KEYBOARD: u8 = 0;
    connection
        .xtest_fake_input(event, keycode, CURRENT_TIME, root, 0, 0, CORE_KEYBOARD)
        .map_err(|_| KeyFailure::NotSent)?
        .check()
        .map_err(|_| KeyFailure::MaybeSent)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIN_KEYCODE: u8 = 8;
    const KEYSYMS_PER_KEYCODE: usize = 2;
    const XK_V_UPPER: u32 = 0x0056;
    const XK_SHIFT_L: u32 = 0xffe1;
    const XK_CONTROL_L: u32 = 0xffe3;
    const XK_ALT_L: u32 = 0xffe9;

    fn mapping_fixture() -> (GetKeyboardMappingReply, GetModifierMappingReply) {
        let mut keysyms = vec![0; 128 * KEYSYMS_PER_KEYCODE];
        for (keycode, values) in [
            (37, [XK_CONTROL_L, 0]),
            (50, [XK_SHIFT_L, 0]),
            (55, [XK_V_LOWER, XK_V_UPPER]),
            (64, [XK_ALT_L, 0]),
        ] {
            let offset = usize::from(keycode - MIN_KEYCODE) * KEYSYMS_PER_KEYCODE;
            keysyms[offset..offset + KEYSYMS_PER_KEYCODE].copy_from_slice(&values);
        }
        let keyboard = GetKeyboardMappingReply {
            keysyms_per_keycode: 2,
            sequence: 0,
            keysyms,
        };
        let mut keycodes = vec![0; 8 * 2];
        keycodes[SHIFT_MODIFIER_ROW * 2] = 50;
        keycodes[CONTROL_MODIFIER_ROW * 2] = 37;
        keycodes[3 * 2] = 64;
        let modifiers = GetModifierMappingReply {
            sequence: 0,
            length: 4,
            keycodes,
        };
        (keyboard, modifiers)
    }

    #[test]
    fn resolves_shortcut_keys_and_every_modifier_from_server_maps() {
        let (keyboard, modifiers) = mapping_fixture();

        assert_eq!(
            resolve_keys(MIN_KEYCODE, &keyboard, &modifiers),
            Some(Keys {
                control: 37,
                shift: 50,
                v: 55,
                modifiers: vec![37, 50, 64],
            })
        );
    }

    #[test]
    fn refuses_a_keymap_without_a_control_modifier() {
        let (keyboard, mut modifiers) = mapping_fixture();
        modifiers.keycodes[CONTROL_MODIFIER_ROW * 2] = 0;

        assert_eq!(resolve_keys(MIN_KEYCODE, &keyboard, &modifiers), None);
    }

    #[test]
    fn a_v_outside_the_first_group_is_never_the_paste_key() {
        let (mut keyboard, modifiers) = mapping_fixture();
        let offset = usize::from(55 - MIN_KEYCODE) * KEYSYMS_PER_KEYCODE;
        keyboard.keysyms[offset..offset + 2].copy_from_slice(&[0x0061, XK_V_LOWER]);

        assert_eq!(resolve_keys(MIN_KEYCODE, &keyboard, &modifiers), None);
    }

    #[test]
    fn shortcuts_press_modifiers_before_the_key() {
        let (keyboard, modifiers) = mapping_fixture();
        let keys = resolve_keys(MIN_KEYCODE, &keyboard, &modifiers).expect("fixture keys");

        assert_eq!(key_sequence(PasteKeys::ControlV, &keys), [37, 55]);
        assert_eq!(key_sequence(PasteKeys::ControlShiftV, &keys), [37, 50, 55]);
    }

    #[test]
    #[ignore = "needs an X11 server with XTEST, such as Xvfb"]
    fn a_live_x11_server_exposes_xtest_and_the_paste_keys() {
        let (connection, _) = x11rb::connect(None).expect("connect to X11");

        assert_eq!(
            xtest_available(),
            Some(true),
            "the setup paste check sees XTEST"
        );
        reply(connection.xtest_get_version(2, 1)).expect("XTEST 2.1+");
        reply(connection.xkb_use_extension(1, 0)).expect("XKB");
        let keys = read_keys(&connection).expect("paste keys are mapped");
        assert!(keys.modifiers.contains(&keys.control));
        assert!(keys.modifiers.contains(&keys.shift));
    }
}
