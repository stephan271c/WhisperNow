//! A virtual keyboard that can press only the paste shortcuts.

use std::{
    fs::{File, OpenOptions},
    io::{self, Write as _},
    os::{fd::AsRawFd, unix::fs::OpenOptionsExt as _},
    thread,
    time::Duration,
};

/// The device name, which the key readers skip so its own keys are not mistaken
/// for the user's.
pub const NAME: &str = "WhisperNow paste keyboard";

const KEY_LEFTCTRL: u16 = 29;
const KEY_LEFTSHIFT: u16 = 42;
/// The V key's position. QWERTY, QWERTZ, AZERTY and Colemak keep V there.
const KEY_V: u16 = 47;
const EV_SYN: u16 = 0;
const EV_KEY: u16 = 1;
const BUS_VIRTUAL: u16 = 0x06;

// `linux/uinput.h` requests: `_IOW('U', 100, int)`, `_IOW('U', 101, int)`,
// `_IOW('U', 3, struct uinput_setup)` and `_IO('U', 1)`, `_IO('U', 2)`.
const UI_SET_EVBIT: u64 = 0x4004_5564;
const UI_SET_KEYBIT: u64 = 0x4004_5565;
const UI_DEV_SETUP: u64 = 0x405c_5503;
const UI_DEV_CREATE: u64 = 0x5501;
const UI_DEV_DESTROY: u64 = 0x5502;

/// Lets the compositor add the new device before its first key.
const SETTLE: Duration = Duration::from_millis(300);
/// Between key events, so every application sees each one.
const KEY_GAP: Duration = Duration::from_millis(8);

/// `struct uinput_setup`.
#[repr(C)]
struct UinputSetup {
    bustype: u16,
    vendor: u16,
    product: u16,
    version: u16,
    name: [u8; 80],
    ff_effects_max: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PasteShortcut {
    ControlV,
    ControlShiftV,
}

impl PasteShortcut {
    pub fn parse(command: &str) -> Option<Self> {
        match command {
            "ctrl-v" => Some(Self::ControlV),
            "ctrl-shift-v" => Some(Self::ControlShiftV),
            _ => None,
        }
    }

    const fn keys(self) -> &'static [u16] {
        match self {
            Self::ControlV => &[KEY_LEFTCTRL, KEY_V],
            Self::ControlShiftV => &[KEY_LEFTCTRL, KEY_LEFTSHIFT, KEY_V],
        }
    }
}

pub struct VirtualKeyboard {
    file: File,
}

impl VirtualKeyboard {
    pub fn create() -> io::Result<Self> {
        let file = OpenOptions::new()
            .write(true)
            .custom_flags(libc::O_NONBLOCK)
            .open("/dev/uinput")?;
        let keyboard = Self { file };
        keyboard.request(UI_SET_EVBIT, libc::c_ulong::from(EV_KEY))?;
        for key in [KEY_LEFTCTRL, KEY_LEFTSHIFT, KEY_V] {
            keyboard.request(UI_SET_KEYBIT, libc::c_ulong::from(key))?;
        }
        let mut setup = UinputSetup {
            bustype: BUS_VIRTUAL,
            vendor: 0,
            product: 0,
            version: 1,
            name: [0; 80],
            ff_effects_max: 0,
        };
        setup.name[..NAME.len()].copy_from_slice(NAME.as_bytes());
        keyboard.request(UI_DEV_SETUP, std::ptr::from_ref(&setup) as libc::c_ulong)?;
        keyboard.request(UI_DEV_CREATE, 0)?;
        thread::sleep(SETTLE);
        Ok(keyboard)
    }

    /// Presses the shortcut's keys in order and releases them in reverse.
    pub fn press(&mut self, shortcut: PasteShortcut) -> io::Result<()> {
        let keys = shortcut.keys();
        let mut pressed = 0;
        let mut result = Ok(());
        for key in keys {
            result = self.key(*key, 1);
            if result.is_err() {
                break;
            }
            pressed += 1;
        }
        // Every pressed key is released, even after a failure, so no modifier
        // stays held.
        for key in keys[..pressed].iter().rev() {
            let released = self.key(*key, 0);
            result = result.and(released);
        }
        result
    }

    fn key(&mut self, code: u16, value: i32) -> io::Result<()> {
        self.event(EV_KEY, code, value)?;
        self.event(EV_SYN, 0, 0)?;
        thread::sleep(KEY_GAP);
        Ok(())
    }

    /// Writes one `struct input_event`; the kernel stamps its time.
    fn event(&mut self, kind: u16, code: u16, value: i32) -> io::Result<()> {
        let event = [
            &0_i64.to_ne_bytes()[..],
            &0_i64.to_ne_bytes(),
            &kind.to_ne_bytes(),
            &code.to_ne_bytes(),
            &value.to_ne_bytes(),
        ]
        .concat();
        self.file.write_all(&event)
    }

    fn request(&self, request: u64, argument: libc::c_ulong) -> io::Result<()> {
        // SAFETY: each request takes an integer or, for UI_DEV_SETUP, a
        // pointer to a live `uinput_setup` the caller keeps for the call.
        #[allow(unsafe_code, reason = "uinput is configured only by ioctl")]
        let result = unsafe { libc::ioctl(self.file.as_raw_fd(), request as _, argument) };
        if result < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
}

impl Drop for VirtualKeyboard {
    fn drop(&mut self) {
        let _destroyed = self.request(UI_DEV_DESTROY, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_paste_shortcuts_are_accepted() {
        assert_eq!(
            PasteShortcut::parse("ctrl-v"),
            Some(PasteShortcut::ControlV)
        );
        assert_eq!(
            PasteShortcut::parse("ctrl-shift-v"),
            Some(PasteShortcut::ControlShiftV)
        );
        for other in ["", "ctrl-w", "ctrl-v ", "alt-f4"] {
            assert_eq!(PasteShortcut::parse(other), None, "{other}");
        }
        assert_eq!(PasteShortcut::ControlShiftV.keys(), [29, 42, 47]);
    }

    #[test]
    fn the_setup_struct_matches_the_kernel_layout() {
        assert_eq!(std::mem::size_of::<UinputSetup>(), 92);
    }
}
