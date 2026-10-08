//! Opens a local directory in the desktop's file manager, or a system settings page.

use std::{
    ffi::OsStr,
    io,
    path::Path,
    process::{Command, Stdio},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrivacySettingsPage {
    Microphone,
    /// macOS only.
    Accessibility,
}

/// Opens `directory` in the desktop's file manager without waiting for it.
pub fn open_in_file_manager(directory: &Path) -> io::Result<()> {
    let program = if cfg!(target_os = "macos") {
        "open"
    } else if cfg!(target_os = "windows") {
        "explorer"
    } else {
        "xdg-open"
    };
    spawn_opener(program, directory.as_os_str())
}

/// Opens the system settings `page` without waiting for it.
///
/// # Errors
///
/// Returns [`io::ErrorKind::Unsupported`] where the platform has no such page (Linux, and
/// Accessibility outside macOS), and the spawn error when the opener cannot be started.
pub fn open_privacy_settings(page: PrivacySettingsPage) -> io::Result<()> {
    let (program, uri) = privacy_settings_uri(page).ok_or(io::ErrorKind::Unsupported)?;
    spawn_opener(program, OsStr::new(uri))
}

/// The opener and URI of `page` on this platform.
const fn privacy_settings_uri(page: PrivacySettingsPage) -> Option<(&'static str, &'static str)> {
    if cfg!(target_os = "macos") {
        Some((
            "open",
            match page {
                PrivacySettingsPage::Microphone => {
                    "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"
                }
                PrivacySettingsPage::Accessibility => {
                    "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
                }
            },
        ))
    } else if cfg!(target_os = "windows") {
        match page {
            PrivacySettingsPage::Microphone => Some(("explorer", "ms-settings:privacy-microphone")),
            PrivacySettingsPage::Accessibility => None,
        }
    } else {
        None
    }
}

fn spawn_opener(program: &str, argument: &OsStr) -> io::Result<()> {
    let mut opener = Command::new(program)
        .arg(argument)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    // Reap the opener so it never lingers as a zombie. Its exit status is not
    // meaningful: `explorer` reports failure even after opening the folder.
    std::thread::Builder::new()
        .name("whispernow-opener".to_owned())
        .spawn(move || {
            let _status = opener.wait();
        })?;
    Ok(())
}
