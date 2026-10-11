use std::{
    fs, io,
    path::{Path, PathBuf},
};

use directories::BaseDirs;
use thiserror::Error;

const APP_DIRECTORY: &str = "WhisperNow";
const SETTINGS_FILE: &str = "settings.json";
const UNREADABLE_SETTINGS_MARKER: &str = ".bad-";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AppPaths {
    config_dir: PathBuf,
    data_dir: PathBuf,
}

impl AppPaths {
    /// Resolves the same config/data roots used by the Python `platformdirs`
    /// release and retains its capitalized application directory.
    pub fn discover() -> Result<Self, AppPathError> {
        let base = BaseDirs::new().ok_or(AppPathError::Unavailable)?;
        #[cfg(target_os = "windows")]
        {
            // platformdirs uses non-roaming LocalAppData for both calls when
            // `roaming` is omitted, as it is in the Python release.
            Ok(Self::from_windows_local_data_dir(base.data_local_dir()))
        }
        #[cfg(not(target_os = "windows"))]
        {
            Ok(Self::from_base_dirs(base.config_dir(), base.data_dir()))
        }
    }

    /// Builds paths from explicit roots. This is the test seam; callers do not
    /// need to mutate global environment variables or touch real user data.
    #[must_use]
    pub fn from_base_dirs(config_base: impl AsRef<Path>, data_base: impl AsRef<Path>) -> Self {
        Self {
            config_dir: config_base.as_ref().join(APP_DIRECTORY),
            data_dir: data_base.as_ref().join(APP_DIRECTORY),
        }
    }

    /// Reproduces Python `platformdirs` defaults on Windows, where config and
    /// data both live below non-roaming `LocalAppData`.
    #[must_use]
    pub fn from_windows_local_data_dir(local_data: impl AsRef<Path>) -> Self {
        Self::from_base_dirs(&local_data, &local_data)
    }

    #[must_use]
    pub fn config_dir(&self) -> &Path {
        &self.config_dir
    }

    #[must_use]
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    #[must_use]
    pub fn settings_file(&self) -> PathBuf {
        self.config_dir.join(SETTINGS_FILE)
    }

    /// Where an unreadable `settings.json` is moved: `settings.json.bad-<stamp>`.
    #[must_use]
    pub fn unreadable_settings_file(&self, stamp: &str) -> PathBuf {
        self.config_dir.join(format!(
            "{SETTINGS_FILE}{UNREADABLE_SETTINGS_MARKER}{stamp}"
        ))
    }

    /// The Python release's `settings.json`, kept after it was imported.
    #[must_use]
    pub fn python_settings_file(&self) -> PathBuf {
        self.config_dir.join(format!("{SETTINGS_FILE}.python"))
    }

    /// Every existing `settings.json.bad-*` entry and the kept Python settings, sorted by
    /// path. A missing config directory has none.
    pub fn settings_backup_files(&self) -> io::Result<Vec<PathBuf>> {
        let unreadable = format!("{SETTINGS_FILE}{UNREADABLE_SETTINGS_MARKER}");
        let entries = match fs::read_dir(&self.config_dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error),
        };
        let python = self.python_settings_file();
        let mut copies = Vec::new();
        for entry in entries {
            let entry = entry?;
            if entry
                .file_name()
                .as_encoded_bytes()
                .starts_with(unreadable.as_bytes())
                || entry.path() == python
            {
                copies.push(entry.path());
            }
        }
        copies.sort();
        Ok(copies)
    }

    #[must_use]
    pub fn history_file(&self) -> PathBuf {
        self.config_dir.join("history.json")
    }

    #[must_use]
    pub fn migrated_history_file(&self) -> PathBuf {
        self.config_dir.join("history.json.migrated")
    }

    /// Collision-safe quarantine for malformed legacy history documents.
    #[must_use]
    pub fn history_quarantine_dir(&self) -> PathBuf {
        self.config_dir.join("history-quarantine")
    }

    #[must_use]
    pub fn history_database_file(&self) -> PathBuf {
        self.data_dir.join("history.sqlite3")
    }

    #[must_use]
    pub fn history_database_wal_file(&self) -> PathBuf {
        self.data_dir.join("history.sqlite3-wal")
    }

    #[must_use]
    pub fn history_database_shm_file(&self) -> PathBuf {
        self.data_dir.join("history.sqlite3-shm")
    }

    #[must_use]
    pub fn recordings_dir(&self) -> PathBuf {
        self.data_dir.join("recordings")
    }

    #[must_use]
    pub fn models_dir(&self) -> PathBuf {
        self.data_dir.join("models")
    }

    #[must_use]
    pub fn cuda_pack_dir(&self) -> PathBuf {
        self.data_dir.join("llama-cuda")
    }

    #[must_use]
    pub fn logs_dir(&self) -> PathBuf {
        self.data_dir.join("logs")
    }

    /// Portal-issued `RemoteDesktop` restore token. This is deliberately kept
    /// separate from user-editable settings because the token is a capability,
    /// not a preference.
    #[must_use]
    pub fn wayland_remote_desktop_token_file(&self) -> PathBuf {
        self.config_dir.join("wayland-remote-desktop-token")
    }

    /// Where an `AppImage` copies the Wayland paste helper and its install script,
    /// because root cannot read the `AppImage`'s own files.
    #[must_use]
    pub fn paste_helper_setup_dir(&self) -> PathBuf {
        self.data_dir.join("helper-setup")
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum AppPathError {
    #[error("the operating system did not provide user config/data directories")]
    Unavailable,
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::AppPaths;

    #[test]
    fn explicit_roots_preserve_python_compatible_layout() {
        let paths = AppPaths::from_base_dirs("test-config", "test-data");

        assert_eq!(
            paths.settings_file(),
            Path::new("test-config/WhisperNow/settings.json")
        );
        assert_eq!(
            paths.history_file(),
            Path::new("test-config/WhisperNow/history.json")
        );
        assert_eq!(paths.models_dir(), Path::new("test-data/WhisperNow/models"));
    }

    #[test]
    fn windows_layout_uses_non_roaming_local_app_data_for_both_roots() {
        let paths = AppPaths::from_windows_local_data_dir("LocalAppData");

        assert_eq!(
            paths.settings_file(),
            Path::new("LocalAppData/WhisperNow/settings.json")
        );
        assert_eq!(
            paths.models_dir(),
            Path::new("LocalAppData/WhisperNow/models")
        );
    }
}
