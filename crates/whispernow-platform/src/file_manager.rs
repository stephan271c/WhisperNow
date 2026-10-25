//! Opens a local directory, a web page or a system settings page with the desktop's opener.

use std::{
    ffi::OsStr,
    io,
    path::Path,
    process::{Command, Stdio},
};

use whispernow_domain::ipc::PermissionKind;

const OPENER: &str = if cfg!(target_os = "macos") {
    "open"
} else if cfg!(target_os = "windows") {
    "explorer"
} else {
    "xdg-open"
};

pub fn open_in_file_manager(directory: &Path) -> io::Result<()> {
    spawn_opener(directory.as_os_str())
}

pub fn open_url(url: &str) -> io::Result<()> {
    spawn_opener(OsStr::new(url))
}

/// Fails with [`io::ErrorKind::Unsupported`] where the platform has no such page (Linux, and
/// Accessibility outside macOS).
pub fn open_privacy_settings(kind: PermissionKind) -> io::Result<()> {
    let uri = privacy_settings_uri(kind).ok_or(io::ErrorKind::Unsupported)?;
    spawn_opener(OsStr::new(uri))
}

const fn privacy_settings_uri(kind: PermissionKind) -> Option<&'static str> {
    if cfg!(target_os = "macos") {
        match kind {
            PermissionKind::Microphone => {
                Some("x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")
            }
            PermissionKind::Accessibility => Some(
                "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
            ),
            PermissionKind::GlobalShortcuts
            | PermissionKind::RemoteDesktop
            | PermissionKind::PasteSupport => None,
        }
    } else if cfg!(target_os = "windows") {
        match kind {
            PermissionKind::Microphone => Some("ms-settings:privacy-microphone"),
            PermissionKind::Accessibility
            | PermissionKind::GlobalShortcuts
            | PermissionKind::RemoteDesktop
            | PermissionKind::PasteSupport => None,
        }
    } else {
        None
    }
}

fn spawn_opener(argument: &OsStr) -> io::Result<()> {
    let mut opener = Command::new(OPENER)
        .arg(argument)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    // Reap the opener so it never lingers as a zombie. `explorer` reports failure even
    // after opening the folder, so its exit status is ignored.
    std::thread::Builder::new()
        .name("whispernow-opener".to_owned())
        .spawn(move || {
            let _status = opener.wait();
        })?;
    Ok(())
}
