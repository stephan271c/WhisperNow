//! A virtual keyboard that can press only the paste shortcuts.

use std::{
    fs::{File, OpenOptions},
    io::{self, Write as _},
    os::{fd::AsRawFd, unix::fs::OpenOptionsExt as _},
    thread,
    time::Duration,
};

use crate::{
    codes::{EV_KEY, KEY_LEFTCTRL, KEY_LEFTSHIFT, KEY_V},
    event::{InputEvent, KeyValue},
};

/// The device name, which the key readers skip so its own keys are not mistaken
/// for the user's.
pub const NAME: &str = "WhisperNow paste keyboard";

const BUS_VIRTUAL: u16 = 0x06;

const UINPUT_IOCTL_BASE: u32 = b'U' as u32;
const UI_SET_EVBIT: libc::Ioctl = libc::_IOW::<libc::c_int>(UINPUT_IOCTL_BASE, 100);
const UI_SET_KEYBIT: libc::Ioctl = libc::_IOW::<libc::c_int>(UINPUT_IOCTL_BASE, 101);
const UI_DEV_SETUP: libc::Ioctl = libc::_IOW::<libc::uinput_setup>(UINPUT_IOCTL_BASE, 3);
const UI_DEV_CREATE: libc::Ioctl = libc::_IO(UINPUT_IOCTL_BASE, 1);
const UI_DEV_DESTROY: libc::Ioctl = libc::_IO(UINPUT_IOCTL_BASE, 2);

/// Lets the compositor add the new device before its first key.
const SETTLE: Duration = Duration::from_millis(300);
/// Between key events, so every application sees each one.
const KEY_GAP: Duration = Duration::from_millis(8);

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
        let mut setup = libc::uinput_setup {
            id: libc::input_id {
                bustype: BUS_VIRTUAL,
                vendor: 0,
                product: 0,
                version: 1,
            },
            name: [0; libc::UINPUT_MAX_NAME_SIZE],
            ff_effects_max: 0,
        };
        for (slot, byte) in setup.name.iter_mut().zip(NAME.bytes()) {
            *slot = libc::c_char::from_ne_bytes([byte]);
        }
        keyboard.setup(&setup)?;
        keyboard.request(UI_DEV_CREATE, 0)?;
        thread::sleep(SETTLE);
        Ok(keyboard)
    }

    pub fn press(&mut self, shortcut: PasteShortcut) -> io::Result<()> {
        let keys = shortcut.keys();
        let mut pressed_count = 0;
        let mut result = Ok(());
        for key in keys {
            result = self.key(*key, KeyValue::Press);
            if result.is_err() {
                break;
            }
            pressed_count += 1;
        }
        // Every pressed key is released, even after a failure, so no modifier
        // stays held.
        for key in keys[..pressed_count].iter().rev() {
            let released = self.key(*key, KeyValue::Release);
            result = result.and(released);
        }
        result
    }

    fn key(&mut self, code: u16, value: KeyValue) -> io::Result<()> {
        self.write(InputEvent::key(code, value))?;
        self.write(InputEvent::report())?;
        thread::sleep(KEY_GAP);
        Ok(())
    }

    fn write(&mut self, event: InputEvent) -> io::Result<()> {
        self.file.write_all(&event.to_bytes())
    }

    /// For the requests that take an integer or nothing.
    fn request(&self, request: libc::Ioctl, argument: libc::c_ulong) -> io::Result<()> {
        // SAFETY: the requests passed here read no memory.
        #[allow(unsafe_code, reason = "uinput is configured only by ioctl")]
        let result = unsafe { libc::ioctl(self.file.as_raw_fd(), request, argument) };
        checked(result)
    }

    fn setup(&self, setup: &libc::uinput_setup) -> io::Result<()> {
        // SAFETY: UI_DEV_SETUP reads one `uinput_setup` through the pointer,
        // which is valid for the call.
        #[allow(unsafe_code, reason = "uinput is configured only by ioctl")]
        let result = unsafe {
            libc::ioctl(
                self.file.as_raw_fd(),
                UI_DEV_SETUP,
                std::ptr::from_ref(setup),
            )
        };
        checked(result)
    }
}

fn checked(result: libc::c_int) -> io::Result<()> {
    if result < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
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
}
