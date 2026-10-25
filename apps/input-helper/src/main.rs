//! Reports when a held recording shortcut is released.
//!
//! GNOME's `GlobalShortcuts` portal reports a shortcut's press but sometimes
//! drops its release, and Wayland gives applications no other view of the
//! keyboard. This helper reads `/dev/input` directly. It is installed setgid
//! `input` so that only it, not `WhisperNow` or anything else the user runs, can
//! read keyboards, and it reports only what a recording shortcut needs:
//!
//! - It watches only the chord given as an argument, which must hold Ctrl,
//!   Alt or Super, so typed text is never observable through it.
//! - It writes `ready` after reading the devices' initial key states,
//!   `held <CLOCK_MONOTONIC milliseconds>` when the chord is held,
//!   `<CLOCK_MONOTONIC milliseconds>` when it is released, and
//!   `unavailable` when the chord can no longer be monitored.
//! - It exits when standard input closes, so it never outlives `WhisperNow`.
//!
//! Usage: `whispernow-input-helper <code>[,<code>]+<code>...`, where codes
//! are Linux input key codes, for example `29,97+57` for Ctrl+Space.
//!
//! With `--paste` it instead creates a virtual keyboard that can press only
//! Ctrl+V and Ctrl+Shift+V, so `WhisperNow` can paste on Wayland without a
//! remote desktop session. It prints `ready` once the keyboard exists, then
//! answers each `ctrl-v` or `ctrl-shift-v` line with `sent`, `held` when the
//! user kept a modifier held, or `failed`; any other line, or one sent while
//! a paste is pending, gets `refused`. It reads keyboards only to see held
//! modifiers.

#[cfg(target_os = "linux")]
mod chord;
#[cfg(target_os = "linux")]
mod codes;
#[cfg(target_os = "linux")]
mod devices;
#[cfg(target_os = "linux")]
mod event;
#[cfg(target_os = "linux")]
mod virtual_keyboard;

use std::process::ExitCode;

#[cfg(target_os = "linux")]
fn main() -> ExitCode {
    let specs = std::env::args().skip(1).collect::<Vec<_>>();
    if specs == ["--paste"] {
        return paste();
    }
    let [spec] = specs.as_slice() else {
        eprintln!("expected one recording shortcut chord");
        return ExitCode::from(64);
    };
    match spec.parse::<chord::Chord>() {
        Ok(chord) => devices::run(devices::Mode::Watch(chord)),
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(64)
        }
    }
}

#[cfg(target_os = "linux")]
fn paste() -> ExitCode {
    match virtual_keyboard::VirtualKeyboard::create() {
        Ok(keyboard) => devices::run(devices::Mode::Paste(keyboard)),
        Err(error) => {
            eprintln!("the paste keyboard could not be created: {error}");
            ExitCode::from(3)
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn main() -> ExitCode {
    eprintln!("only Linux keyboards are supported");
    ExitCode::FAILURE
}
