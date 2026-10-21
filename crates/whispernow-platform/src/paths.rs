use std::{
    fs, io,
    path::{Path, PathBuf},
};

use directories::BaseDirs;
use time::OffsetDateTime;

const APP_DIRECTORY: &str = "WhisperNow";
const SETTINGS_FILE: &str = "settings.json";
const HISTORY_FILE: &str = "history.json";
const UNREADABLE_MARKER: &str = ".bad-";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AppPaths {
    config_dir: PathBuf,
    data_dir: PathBuf,
}

impl AppPaths {
    /// Resolves the same config/data roots used by the Python `platformdirs`
    /// release and retains its capitalized application directory.
    pub fn discover() -> io::Result<Self> {
        let base = BaseDirs::new().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "the operating system did not provide user config/data directories",
            )
        })?;
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

    /// Builds paths from explicit roots, so tests never touch the real user directories.
    #[must_use]
    pub fn from_base_dirs(config_base: impl AsRef<Path>, data_base: impl AsRef<Path>) -> Self {
        Self {
            config_dir: config_base.as_ref().join(APP_DIRECTORY),
            data_dir: data_base.as_ref().join(APP_DIRECTORY),
        }
    }

    #[doc(hidden)]
    #[must_use]
    pub fn under(root: impl AsRef<Path>) -> Self {
        let root = root.as_ref();
        Self::from_base_dirs(root.join("config"), root.join("data"))
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

    pub(crate) fn unreadable_stamp() -> String {
        let now = OffsetDateTime::now_utc();
        format!(
            "{:04}{:02}{:02}T{:02}{:02}{:02}Z",
            now.year(),
            u8::from(now.month()),
            now.day(),
            now.hour(),
            now.minute(),
            now.second()
        )
    }

    /// Where an unreadable `settings.json` is moved: `settings.json.bad-<stamp>`.
    #[must_use]
    pub fn unreadable_settings_file(&self, stamp: &str) -> PathBuf {
        self.config_dir
            .join(format!("{SETTINGS_FILE}{UNREADABLE_MARKER}{stamp}"))
    }

    /// The Python release's `settings.json`, kept after it was imported.
    #[must_use]
    pub fn python_settings_file(&self) -> PathBuf {
        self.config_dir.join(format!("{SETTINGS_FILE}.python"))
    }

    /// Every existing `settings.json.bad-*` entry and the kept Python settings, sorted by
    /// path. A missing config directory has none.
    pub fn settings_backup_files(&self) -> io::Result<Vec<PathBuf>> {
        self.backup_files(SETTINGS_FILE, &self.python_settings_file())
    }

    fn backup_files(&self, name: &str, imported: &Path) -> io::Result<Vec<PathBuf>> {
        let unreadable_prefix = format!("{name}{UNREADABLE_MARKER}");
        let entries = match fs::read_dir(&self.config_dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error),
        };
        let mut backups = Vec::new();
        for entry in entries {
            let entry = entry?;
            if entry
                .file_name()
                .as_encoded_bytes()
                .starts_with(unreadable_prefix.as_bytes())
                || entry.path() == imported
            {
                backups.push(entry.path());
            }
        }
        backups.sort();
        Ok(backups)
    }

    #[must_use]
    pub fn history_file(&self) -> PathBuf {
        self.config_dir.join(HISTORY_FILE)
    }

    #[must_use]
    pub fn migrated_history_file(&self) -> PathBuf {
        self.config_dir.join(format!("{HISTORY_FILE}.migrated"))
    }

    #[must_use]
    pub fn unreadable_history_file(&self, stamp: &str) -> PathBuf {
        self.config_dir
            .join(format!("{HISTORY_FILE}{UNREADABLE_MARKER}{stamp}"))
    }

    pub fn history_backup_files(&self) -> io::Result<Vec<PathBuf>> {
        self.backup_files(HISTORY_FILE, &self.migrated_history_file())
    }

    #[must_use]
    pub fn history_database_file(&self) -> PathBuf {
        self.data_dir.join("history.sqlite3")
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
    pub fn python_logs_dir(&self) -> PathBuf {
        self.data_dir.join("logs")
    }

    /// Portal-issued `RemoteDesktop` restore token. It is kept out of the user-editable
    /// settings because it is a capability, not a preference.
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
    fn unreadable_stamps_are_compact_utc_times() {
        let stamp = AppPaths::unreadable_stamp();

        let (date, time) = stamp.split_once('T').expect("date and time");
        let time = time.strip_suffix('Z').expect("UTC marker");
        assert!(date.len() == 8 && time.len() == 6);
        assert!(
            date.bytes()
                .chain(time.bytes())
                .all(|byte| byte.is_ascii_digit())
        );
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
