//! Locating and deleting the fixed application-data targets in [`AppPaths`].

use std::{
    fs, io,
    path::{Path, PathBuf},
};

use thiserror::Error;
pub use whispernow_domain::ClearDataCategory;

use crate::AppPaths;

/// One existing application-data target.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DataLocation {
    pub category: ClearDataCategory,
    pub path: PathBuf,
}

/// What happened to one target; a failure carries the I/O error kind.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClearDataResultEntry {
    pub category: ClearDataCategory,
    pub path: PathBuf,
    pub outcome: Result<(), io::ErrorKind>,
}

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[error("could not inspect {category:?} data: {kind}")]
pub struct ClearDataError {
    pub category: ClearDataCategory,
    pub kind: io::ErrorKind,
}

/// Lists and deletes only the fixed targets represented by [`AppPaths`].
#[derive(Debug)]
pub struct DataCleaner {
    paths: AppPaths,
}

impl DataCleaner {
    #[must_use]
    pub fn new(paths: AppPaths) -> Self {
        Self { paths }
    }

    /// The fixed targets that currently exist.
    ///
    /// # Errors
    ///
    /// Returns an error when a target's metadata cannot be inspected.
    pub fn locations(&self) -> Result<Vec<DataLocation>, ClearDataError> {
        let mut locations = Vec::new();
        for (category, path) in self.fixed_targets()? {
            match fs::symlink_metadata(&path) {
                Ok(_) => locations.push(DataLocation { category, path }),
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(ClearDataError {
                        category,
                        kind: error.kind(),
                    });
                }
            }
        }
        Ok(locations)
    }

    /// Deletes the existing targets in `categories` and reports each one.
    ///
    /// Retained recordings are removed before the `SQLite` lifecycle database; if they cannot
    /// be removed, the database and its sidecars are kept so the recordings stay recoverable.
    ///
    /// # Errors
    ///
    /// Returns an error when the targets cannot be listed; nothing is deleted then.
    pub fn clear(
        &self,
        categories: &[ClearDataCategory],
    ) -> Result<Vec<ClearDataResultEntry>, ClearDataError> {
        let targets = self
            .locations()?
            .into_iter()
            .filter(|location| categories.contains(&location.category));
        let mut retained_audio_removed = true;
        let mut entries = Vec::new();
        for DataLocation { category, path } in targets {
            let outcome = if !retained_audio_removed && self.is_lifecycle_database(&path) {
                Err(io::ErrorKind::Other)
            } else {
                delete(&path)
            };
            if path == self.paths.recordings_dir() && outcome.is_err() {
                retained_audio_removed = false;
            }
            entries.push(ClearDataResultEntry {
                category,
                path,
                outcome,
            });
        }
        Ok(entries)
    }

    fn is_lifecycle_database(&self, path: &Path) -> bool {
        [
            self.paths.history_database_file(),
            self.paths.history_database_wal_file(),
            self.paths.history_database_shm_file(),
        ]
        .iter()
        .any(|database| database == path)
    }

    /// Fixed [`AppPaths`] targets plus every settings file moved aside.
    fn fixed_targets(&self) -> Result<Vec<(ClearDataCategory, PathBuf)>, ClearDataError> {
        let backups = self
            .paths
            .settings_backup_files()
            .map_err(|error| ClearDataError {
                category: ClearDataCategory::Settings,
                kind: error.kind(),
            })?;
        let mut targets = vec![(ClearDataCategory::Settings, self.paths.settings_file())];
        targets.extend(
            backups
                .into_iter()
                .map(|path| (ClearDataCategory::Settings, path)),
        );
        targets.extend([
            (ClearDataCategory::History, self.paths.history_file()),
            (
                ClearDataCategory::History,
                self.paths.migrated_history_file(),
            ),
            (
                ClearDataCategory::History,
                self.paths.history_quarantine_dir(),
            ),
            // Retained recordings are removed before the SQLite lifecycle authority. Desktop
            // preflight has already tombstoned and deleted referenced audio while the database is
            // open; this ordering also preserves recovery metadata if directory deletion fails.
            (ClearDataCategory::History, self.paths.recordings_dir()),
            (
                ClearDataCategory::History,
                self.paths.history_database_file(),
            ),
            (
                ClearDataCategory::History,
                self.paths.history_database_wal_file(),
            ),
            (
                ClearDataCategory::History,
                self.paths.history_database_shm_file(),
            ),
            (ClearDataCategory::Models, self.paths.models_dir()),
            (ClearDataCategory::Logs, self.paths.logs_dir()),
        ]);
        Ok(targets)
    }
}

fn delete(path: &Path) -> Result<(), io::ErrorKind> {
    let result = match fs::symlink_metadata(path) {
        Err(error) => Err(error),
        // `symlink_metadata` ensures a symlink is unlinked itself instead of
        // following its destination.
        Ok(metadata) if metadata.file_type().is_dir() => fs::remove_dir_all(path),
        Ok(_) => fs::remove_file(path),
    };
    match result {
        Err(error) if error.kind() != io::ErrorKind::NotFound => Err(error.kind()),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use tempfile::TempDir;

    use super::{ClearDataCategory, DataCleaner};
    use crate::AppPaths;

    const EVERYTHING: [ClearDataCategory; 4] = [
        ClearDataCategory::Settings,
        ClearDataCategory::History,
        ClearDataCategory::Models,
        ClearDataCategory::Logs,
    ];

    struct Fixture {
        temp: TempDir,
        paths: AppPaths,
    }

    impl Fixture {
        fn separate_roots() -> Self {
            let temp = tempfile::tempdir().expect("temporary directory");
            let paths =
                AppPaths::from_base_dirs(temp.path().join("config"), temp.path().join("data"));
            Self { temp, paths }
        }

        fn windows_layout() -> Self {
            let temp = tempfile::tempdir().expect("temporary directory");
            let paths = AppPaths::from_windows_local_data_dir(temp.path().join("local-data"));
            Self { temp, paths }
        }

        fn create_every_target(&self) {
            create_file(&self.paths.settings_file());
            create_file(&self.paths.history_file());
            create_file(&self.paths.migrated_history_file());
            create_dir(&self.paths.history_quarantine_dir());
            create_file(&self.paths.history_database_file());
            create_file(&self.paths.history_database_wal_file());
            create_file(&self.paths.history_database_shm_file());
            create_dir(&self.paths.recordings_dir());
            create_dir(&self.paths.models_dir());
            create_dir(&self.paths.logs_dir());
        }
    }

    fn create_file(path: &Path) {
        fs::create_dir_all(path.parent().expect("file parent")).expect("create file parent");
        fs::write(path, b"test data").expect("create file");
    }

    fn create_dir(path: &Path) {
        fs::create_dir_all(path).expect("create directory");
        fs::write(path.join("content"), b"test data").expect("create directory content");
    }

    #[test]
    fn nothing_to_clear_lists_and_deletes_nothing() {
        let fixture = Fixture::separate_roots();
        let manager = DataCleaner::new(fixture.paths);

        assert_eq!(manager.locations(), Ok(Vec::new()));
        assert_eq!(manager.clear(&EVERYTHING), Ok(Vec::new()));
    }

    #[test]
    fn clear_lists_and_deletes_only_exact_fixed_targets() {
        let fixture = Fixture::separate_roots();
        fixture.create_every_target();
        let config_sentinel = fixture.paths.config_dir().join("keep.txt");
        let data_sentinel = fixture.paths.data_dir().join("keep.txt");
        create_file(&config_sentinel);
        create_file(&data_sentinel);
        let manager = DataCleaner::new(fixture.paths.clone());

        let categories = manager
            .locations()
            .expect("locations")
            .into_iter()
            .map(|location| location.category)
            .collect::<Vec<_>>();
        assert_eq!(
            categories,
            [
                [ClearDataCategory::Settings].as_slice(),
                &[ClearDataCategory::History; 7],
                &[ClearDataCategory::Models, ClearDataCategory::Logs],
            ]
            .concat()
        );

        let report = manager.clear(&EVERYTHING).expect("clear");
        assert_eq!(report.len(), 10);
        assert!(report.iter().all(|entry| entry.outcome.is_ok()));
        assert_eq!(manager.locations(), Ok(Vec::new()));
        assert!(config_sentinel.exists());
        assert!(data_sentinel.exists());
        assert!(fixture.paths.config_dir().is_dir());
        assert!(fixture.paths.data_dir().is_dir());
    }

    #[test]
    fn only_the_requested_categories_are_deleted() {
        let fixture = Fixture::separate_roots();
        fixture.create_every_target();
        let manager = DataCleaner::new(fixture.paths.clone());

        let report = manager
            .clear(&[ClearDataCategory::Logs])
            .expect("clear logs");

        assert_eq!(report.len(), 1);
        assert!(!fixture.paths.logs_dir().exists());
        assert!(fixture.paths.settings_file().exists());
        assert!(fixture.paths.models_dir().exists());
        assert!(fixture.paths.history_database_file().exists());
    }

    #[test]
    fn settings_moved_aside_are_listed_and_deleted_with_settings() {
        let fixture = Fixture::separate_roots();
        create_file(&fixture.paths.settings_file());
        let first = fixture.paths.unreadable_settings_file("20260925T120000Z");
        let second = fixture.paths.python_settings_file();
        create_file(&first);
        create_file(&second);
        let unrelated = fixture.paths.config_dir().join("settings.json.bak");
        create_file(&unrelated);
        let manager = DataCleaner::new(fixture.paths.clone());

        let settings_paths = manager
            .locations()
            .expect("locations")
            .into_iter()
            .filter(|location| location.category == ClearDataCategory::Settings)
            .map(|location| location.path)
            .collect::<Vec<_>>();
        assert_eq!(
            settings_paths,
            vec![fixture.paths.settings_file(), first.clone(), second.clone()]
        );
        manager.clear(&EVERYTHING).expect("clear");
        assert!(!first.exists());
        assert!(!second.exists());
        assert!(!fixture.paths.settings_file().exists());
        assert!(unrelated.exists());
    }

    #[cfg(unix)]
    #[test]
    fn recordings_failure_preserves_lifecycle_database_and_sidecars() {
        use std::os::unix::fs::PermissionsExt as _;

        let fixture = Fixture::separate_roots();
        let recordings = fixture.paths.recordings_dir();
        create_file(&recordings.join("retained.wav"));
        // Without write permission its recording cannot be deleted, so neither can it.
        fs::set_permissions(&recordings, fs::Permissions::from_mode(0o500))
            .expect("read-only recordings");
        for path in [
            fixture.paths.history_database_file(),
            fixture.paths.history_database_wal_file(),
            fixture.paths.history_database_shm_file(),
        ] {
            create_file(&path);
        }
        let manager = DataCleaner::new(fixture.paths.clone());
        let report = manager.clear(&EVERYTHING).expect("clear");
        fs::set_permissions(&recordings, fs::Permissions::from_mode(0o700))
            .expect("writable recordings");

        let history_failures = report
            .iter()
            .filter(|entry| entry.category == ClearDataCategory::History && entry.outcome.is_err())
            .count();
        assert_eq!(
            history_failures, 4,
            "recordings plus three SQLite files fail closed"
        );
        assert!(fixture.paths.recordings_dir().exists());
        assert!(fixture.paths.history_database_file().exists());
        assert!(fixture.paths.history_database_wal_file().exists());
        assert!(fixture.paths.history_database_shm_file().exists());
    }

    #[test]
    fn shared_windows_config_and_data_root_is_never_deleted() {
        let fixture = Fixture::windows_layout();
        fixture.create_every_target();
        let sentinel = fixture.paths.config_dir().join("unrelated.db");
        create_file(&sentinel);
        let manager = DataCleaner::new(fixture.paths.clone());

        let report = manager.clear(&EVERYTHING).expect("clear");

        assert!(report.iter().all(|entry| entry.outcome.is_ok()));
        assert_eq!(fixture.paths.config_dir(), fixture.paths.data_dir());
        assert!(fixture.paths.config_dir().is_dir());
        assert!(sentinel.exists());
    }

    #[cfg(unix)]
    #[test]
    fn symlink_targets_and_nested_symlinks_never_delete_outside_content() {
        use std::os::unix::fs::symlink;

        let fixture = Fixture::separate_roots();
        let outside = fixture.temp.path().join("outside");
        create_dir(&outside);
        fs::create_dir_all(fixture.paths.data_dir()).expect("create data root");
        symlink(&outside, fixture.paths.models_dir()).expect("models symlink");
        create_dir(&fixture.paths.logs_dir());
        symlink(&outside, fixture.paths.logs_dir().join("outside-link")).expect("nested symlink");
        let manager = DataCleaner::new(fixture.paths.clone());

        let report = manager.clear(&EVERYTHING).expect("clear");

        assert!(report.iter().all(|entry| entry.outcome.is_ok()));
        assert!(outside.join("content").exists());
        assert!(!fixture.paths.models_dir().exists());
        assert!(!fixture.paths.logs_dir().exists());
    }
}
