//! Gives a native Wayland process the app id that XDG portals require.
//!
//! Portals identify an unsandboxed caller by the systemd scope its launcher
//! created. A process started from a terminal, or from the `WhisperNow.desktop`
//! autostart entry, has no valid id, and GNOME then refuses `GlobalShortcuts`
//! with "An app id is required". The host `Registry` portal assigns an id
//! explicitly, but only one with an installed `<id>.desktop` entry, so that
//! entry is written first and hidden from menus.

use std::{
    fs, io,
    path::{Path, PathBuf},
};

pub(crate) struct PortalIdentity {
    app_id: String,
    name: String,
    applications_dir: PathBuf,
}

impl PortalIdentity {
    pub(crate) fn new(app_id: String, name: String, data_dir: &Path) -> Self {
        Self {
            app_id,
            name,
            applications_dir: data_dir.join("applications"),
        }
    }

    /// Must run before any other portal call in this process: the portal binds
    /// the shared D-Bus connection to the first app id it sees.
    pub(crate) async fn register(&self) {
        if let Err(error) = self.install_desktop_entry() {
            tracing::warn!(%error, "portal desktop entry could not be written");
        }
        let registered = match self.app_id.parse::<ashpd::AppID>() {
            Ok(app_id) => ashpd::register_host_app(app_id).await,
            Err(error) => Err(error),
        };
        if let Err(error) = registered {
            tracing::warn!(%error, app_id = %self.app_id, "portal app id could not be registered");
        }
    }

    fn install_desktop_entry(&self) -> io::Result<()> {
        let exec = std::env::var_os("APPIMAGE")
            .map(PathBuf::from)
            .map_or_else(std::env::current_exe, Ok)?;
        let entry = desktop_entry(&self.name, &exec);
        let path = self
            .applications_dir
            .join(format!("{}.desktop", self.app_id));
        if fs::read_to_string(&path).is_ok_and(|existing| existing == entry) {
            return Ok(());
        }
        fs::create_dir_all(&self.applications_dir)?;
        fs::write(path, entry)
    }
}

fn desktop_entry(name: &str, exec: &Path) -> String {
    format!(
        "[Desktop Entry]\nType=Application\nName={name}\nExec={}\nTerminal=false\nNoDisplay=true\n",
        quoted_exec_argument(&exec.to_string_lossy())
    )
}

/// Quotes one `Exec` argument. The string-value escape applies before the
/// quoting rule, so a quoted `\` is written as four backslashes.
fn quoted_exec_argument(argument: &str) -> String {
    let mut quoted = String::from("\"");
    for character in argument.chars() {
        match character {
            '\\' => quoted.push_str(r"\\\\"),
            '"' | '`' | '$' => {
                quoted.push_str(r"\\");
                quoted.push(character);
            }
            '%' => quoted.push_str("%%"),
            _ => quoted.push(character),
        }
    }
    quoted.push('"');
    quoted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exec_paths_are_quoted_with_desktop_entry_escapes() {
        assert_eq!(
            quoted_exec_argument("/opt/My Apps/whisper$now"),
            r#""/opt/My Apps/whisper\\$now""#
        );
        assert_eq!(quoted_exec_argument(r"a\b%c"), r#""a\\\\b%%c""#);
    }

    #[test]
    fn the_entry_is_hidden_from_menus() {
        let entry = desktop_entry("WhisperNow", Path::new("/usr/bin/whispernow-desktop"));
        assert!(entry.lines().any(|line| line == "NoDisplay=true"));
    }

    #[test]
    fn installing_twice_leaves_one_entry_named_after_the_app_id() {
        let data = tempfile::tempdir().expect("data dir");
        let identity = PortalIdentity::new(
            "com.whispernow.desktop".to_owned(),
            "WhisperNow".to_owned(),
            data.path(),
        );
        identity.install_desktop_entry().expect("first install");
        identity.install_desktop_entry().expect("second install");
        let entries = fs::read_dir(data.path().join("applications"))
            .expect("applications dir")
            .map(|entry| entry.expect("entry").file_name())
            .collect::<Vec<_>>();
        assert_eq!(entries, ["com.whispernow.desktop.desktop"]);
    }
}
