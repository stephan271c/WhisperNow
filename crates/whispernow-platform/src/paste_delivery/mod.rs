//! The native paste transaction: `arboard` clipboard plus a fixed keystroke.
//!
//! Native Wayland sessions use the portal path in [`crate::wayland`] instead.

use std::{
    thread,
    time::{Duration, Instant},
};

use whispernow_core::text_output::{KeystrokeOutcome, PastePlatform};

use crate::destination::{self, NativeDestination};

mod clipboard;

#[cfg(target_os = "windows")]
mod windows;

#[cfg(target_os = "linux")]
mod x11;

#[cfg(any(target_os = "macos", test))]
mod macos;

pub use clipboard::{ClipboardContents, NativeClipboard};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativePasteError {
    PermissionDenied,
    /// No clipboard, no focused application, or focus moved before the keystroke.
    Unavailable,
    ModifierStillHeld,
    InjectionFailed,
}

type NativePasteInjection = KeystrokeOutcome<NativePasteError>;

/// How long a paste waits for the user to release the physical modifier keys;
/// the adapters never synthesize releases for keys the user still owns.
const MODIFIER_RELEASE_TIMEOUT: Duration = Duration::from_millis(400);
const MODIFIER_POLL_INTERVAL: Duration = Duration::from_millis(5);

pub(crate) fn wait_until_modifiers_released(
    mut poll_modifiers: impl FnMut() -> Result<bool, NativePasteError>,
) -> Result<(), NativePasteError> {
    let deadline = Instant::now() + MODIFIER_RELEASE_TIMEOUT;
    while poll_modifiers()? {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(NativePasteError::ModifierStillHeld);
        }
        thread::sleep(MODIFIER_POLL_INTERVAL.min(remaining));
    }
    Ok(())
}

/// The clipboard plus the paste keystroke for whichever application has focus
/// when the keystroke is sent.
pub struct NativePastePlatform {
    clipboard: NativeClipboard,
}

impl NativePastePlatform {
    /// Connects to the clipboard of a Windows, macOS or native X11 session.
    /// Native Wayland sessions, including their `XWayland`, paste another way.
    pub fn connect() -> Result<Self, NativePasteError> {
        Ok(Self {
            clipboard: NativeClipboard::connect()?,
        })
    }
}

/// Whether macOS trusts this process with Accessibility, which the paste
/// keystroke needs. It never prompts.
#[cfg(target_os = "macos")]
#[must_use]
pub fn accessibility_trusted() -> bool {
    // SAFETY: this query has no pointer preconditions and does not prompt.
    unsafe { objc2_application_services::AXIsProcessTrusted() }
}

/// Whether the X server offers XTEST, the paste keystroke's only injection
/// path on native X11. The server's answer holds for the session, so it is
/// asked once; a server that could not be reached is asked again next time.
#[cfg(target_os = "linux")]
#[must_use]
pub fn x11_paste_injection_available() -> bool {
    static XTEST: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if let Some(available) = XTEST.get() {
        return *available;
    }
    x11::xtest_available().is_some_and(|available| *XTEST.get_or_init(|| available))
}

impl PastePlatform for NativePastePlatform {
    type Snapshot = ClipboardContents;
    type Error = NativePasteError;

    fn snapshot_clipboard(&mut self) -> Result<Option<ClipboardContents>, NativePasteError> {
        self.clipboard.snapshot()
    }

    fn write_transcript(&mut self, text: &str) -> Result<(), NativePasteError> {
        self.clipboard.write_text(text)
    }

    fn clipboard_holds(&mut self, text: &str) -> Result<bool, NativePasteError> {
        self.clipboard.holds_text(text)
    }

    fn restore_clipboard(&mut self, snapshot: ClipboardContents) -> Result<(), NativePasteError> {
        self.clipboard.restore(&snapshot)
    }

    fn send_paste_shortcut(&mut self) -> NativePasteInjection {
        match destination::capture() {
            Ok(target) => inject_if_current(target),
            Err(error) => NativePasteInjection::NotSent(error),
        }
    }
}

fn inject_if_current(target: NativeDestination) -> NativePasteInjection {
    #[cfg(target_os = "windows")]
    {
        windows::inject_if_current(target)
    }
    #[cfg(target_os = "linux")]
    {
        x11::inject_if_current(target)
    }
    #[cfg(target_os = "macos")]
    {
        macos::inject_if_current(target)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        let _ = target;
        NativePasteInjection::NotSent(NativePasteError::Unavailable)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    #[test]
    fn a_modifier_released_in_time_lets_the_paste_go_on() {
        let polls = Cell::new(0);
        let poll_modifier = || {
            polls.set(polls.get() + 1);
            Ok(polls.get() < 3)
        };
        assert_eq!(wait_until_modifiers_released(poll_modifier), Ok(()));
        assert_eq!(polls.get(), 3);
    }

    #[test]
    fn a_modifier_held_too_long_or_an_unreadable_keyboard_stops_the_paste() {
        assert_eq!(
            wait_until_modifiers_released(|| Ok(true)),
            Err(NativePasteError::ModifierStillHeld)
        );
        assert_eq!(
            wait_until_modifiers_released(|| Err(NativePasteError::Unavailable)),
            Err(NativePasteError::Unavailable)
        );
    }
}
