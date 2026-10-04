//! Reports when a held recording shortcut is released.
//!
//! GNOME's `GlobalShortcuts` portal reports a shortcut's press but sometimes
//! drops its release, and Wayland gives applications no other view of the
//! keyboard. This helper reads `/dev/input` directly. It is installed setgid
//! `input` so that only it, not `WhisperNow` or anything else the user runs, can
//! read keyboards, and it reports only what a recording shortcut needs:
//!
//! - It watches only the chords given as arguments, and each must hold Ctrl,
//!   Alt or Super, so typed text is never observable through it.
//! - It writes one line, `<chord id hex> <CLOCK_MONOTONIC milliseconds>`,
//!   when a fully held chord stops being held, and nothing else.
//! - It exits when standard input closes, so it never outlives `WhisperNow`.
//!
//! Usage: `whispernow-input-helper <id>=<code>[,<code>]+<code>...`, where codes
//! are Linux input key codes, for example `8003e=29,97+57` for Ctrl+Space.
//!
//! With `--paste` it instead creates a virtual keyboard that can press only
//! Ctrl+V and Ctrl+Shift+V, so `WhisperNow` can paste on Wayland without a
//! remote desktop session. It prints `ready` once the keyboard exists, then
//! answers each `ctrl-v` or `ctrl-shift-v` line with `sent`, `held` when the
//! user kept a modifier held, or `failed`. It reads keyboards only to see
//! held modifiers.

mod chord;
#[cfg(target_os = "linux")]
mod devices;
#[cfg(target_os = "linux")]
mod virtual_keyboard;

use std::process::ExitCode;

fn main() -> ExitCode {
    let specs = std::env::args().skip(1).collect::<Vec<_>>();
    if specs == ["--paste"] {
        return paste();
    }
    let chords = match chord::parse_chords(&specs) {
        Ok(chords) => chords,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(64);
        }
    };
    run(chords)
}

#[cfg(target_os = "linux")]
fn run(chords: Vec<chord::Chord>) -> ExitCode {
    devices::run(chord::ChordTracker::new(chords), None)
}

#[cfg(target_os = "linux")]
fn paste() -> ExitCode {
    match virtual_keyboard::VirtualKeyboard::create() {
        Ok(keyboard) => devices::run(chord::ChordTracker::new(Vec::new()), Some(keyboard)),
        Err(error) => {
            eprintln!("the paste keyboard could not be created: {error}");
            ExitCode::from(3)
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn run(_chords: Vec<chord::Chord>) -> ExitCode {
    eprintln!("only Linux keyboards are supported");
    ExitCode::FAILURE
}

#[cfg(not(target_os = "linux"))]
fn paste() -> ExitCode {
    run(Vec::new())
}
