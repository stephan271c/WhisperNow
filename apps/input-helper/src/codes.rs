//! Linux input codes from `linux/input-event-codes.h`.

pub const EV_SYN: u16 = 0;
pub const EV_KEY: u16 = 1;
pub const SYN_DROPPED: u16 = 3;
pub const KEY_LEFTCTRL: u16 = 29;
pub const KEY_LEFTSHIFT: u16 = 42;
/// The V key's position. QWERTY, QWERTZ, AZERTY and Colemak keep V there.
pub const KEY_V: u16 = 47;
pub const KEY_RIGHTSHIFT: u16 = 54;
pub const KEY_LEFTALT: u16 = 56;
pub const KEY_RIGHTCTRL: u16 = 97;
pub const KEY_RIGHTALT: u16 = 100;
pub const KEY_LEFTMETA: u16 = 125;
pub const KEY_RIGHTMETA: u16 = 126;
pub const MAX_KEY_CODE: u16 = 0x2ff;

/// Ctrl, Alt and Super, left and right. A chord must hold one of these, so the
/// helper can never report plain or shifted typing.
pub const COMMAND_MODIFIERS: [u16; 6] = [
    KEY_LEFTCTRL,
    KEY_RIGHTCTRL,
    KEY_LEFTALT,
    KEY_RIGHTALT,
    KEY_LEFTMETA,
    KEY_RIGHTMETA,
];

/// Shift, Ctrl, Alt and Super, left and right. A paste waits until none is
/// physically held, because a held one would combine with the shortcut.
pub const MODIFIERS: [u16; 8] = [
    KEY_LEFTCTRL,
    KEY_RIGHTCTRL,
    KEY_LEFTSHIFT,
    KEY_RIGHTSHIFT,
    KEY_LEFTALT,
    KEY_RIGHTALT,
    KEY_LEFTMETA,
    KEY_RIGHTMETA,
];
