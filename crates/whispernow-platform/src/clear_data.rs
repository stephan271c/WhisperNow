//! Locating and deleting the fixed application-data targets in [`AppPaths`].

use std::{
    fs, io,
    path::{Path, PathBuf},
};

use thiserror::Error;
pub use whispernow_domain::ClearDataCategory;
use whispernow_domain::ipc::DataLocation;

use crate::AppPaths;

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

impl AppPaths {
    /// The fixed targets that currently exist.
    pub fn data_locations(&self) -> Result<Vec<DataLocation>, ClearDataError> {
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

    /// Retained recordings are removed before the `SQLite` lifecycle database; if they cannot
    /// be removed, the database and its sidecars are kept so the recordings stay recoverable.
    pub fn clear_data(
        &self,
        categories: &[ClearDataCategory],
    ) -> Result<Vec<ClearDataResultEntry>, ClearDataError> {
        let targets = self
            .data_locations()?
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
            if path == self.recordings_dir() && outcome.is_err() {
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
            self.history_database_file(),
            self.history_database_wal_file(),
            self.history_database_shm_file(),
        ]
        .iter()
        .any(|database| database == path)
    }

    /// Fixed [`AppPaths`] targets plus every settings file moved aside.
    fn fixed_targets(&self) -> Result<Vec<(ClearDataCategory, PathBuf)>, ClearDataError> {
        let backups = self
            .settings_backup_files()
            .map_err(|error| ClearDataError {
                category: ClearDataCategory::Settings,
                kind: error.kind(),
            })?;
        let mut targets = vec![(ClearDataCategory::Settings, self.settings_file())];
        targets.extend(
            backups
                .into_iter()
                .map(|path| (ClearDataCategory::Settings, path)),
        );
        targets.extend([
            (ClearDataCategory::History, self.history_file()),
            (ClearDataCategory::History, self.migrated_history_file()),
            (ClearDataCategory::History, self.history_quarantine_dir()),
            // Before the database, which `clear_data` keeps when the recordings cannot be removed.
            (ClearDataCategory::History, self.recordings_dir()),
            (ClearDataCategory::History, self.history_database_file()),
            (ClearDataCategory::History, self.history_database_wal_file()),
            (ClearDataCategory::History, self.history_database_shm_file()),
            (ClearDataCategory::Models, self.models_dir()),
            (ClearDataCategory::Logs, self.logs_dir()),
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

    use super::ClearDataCategory;
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
        let paths = fixture.paths;

        assert_eq!(paths.data_locations(), Ok(Vec::new()));
        assert_eq!(paths.clear_data(&EVERYTHING), Ok(Vec::new()));
    }

    #[test]
    fn clear_lists_and_deletes_only_exact_fixed_targets() {
        let fixture = Fixture::separate_roots();
        fixture.create_every_target();
        let config_sentinel = fixture.paths.config_dir().join("keep.txt");
        let data_sentinel = fixture.paths.data_dir().join("keep.txt");
        create_file(&config_sentinel);
        create_file(&data_sentinel);
        let paths = fixture.paths.clone();

        let categories = paths
            .data_locations()
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

        let report = paths.clear_data(&EVERYTHING).expect("clear");
        assert_eq!(report.len(), 10);
        assert!(report.iter().all(|entry| entry.outcome.is_ok()));
        assert_eq!(paths.data_locations(), Ok(Vec::new()));
        assert!(config_sentinel.exists());
        assert!(data_sentinel.exists());
        assert!(fixture.paths.config_dir().is_dir());
        assert!(fixture.paths.data_dir().is_dir());
    }

    #[test]
    fn only_the_requested_categories_are_deleted() {
        let fixture = Fixture::separate_roots();
        fixture.create_every_target();
        let paths = fixture.paths.clone();

        let report = paths
            .clear_data(&[ClearDataCategory::Logs])
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
        let later = fixture.paths.unreadable_settings_file("20260926T120000Z");
        let earlier = fixture.paths.unreadable_settings_file("20260925T120000Z");
        let python = fixture.paths.python_settings_file();
        let unrelated = [
            fixture.paths.config_dir().join("settings.json.bak"),
            fixture.paths.config_dir().join("other.json.bad-1"),
        ];
        for path in [&later, &earlier, &python, &unrelated[0], &unrelated[1]] {
            create_file(path);
        }
        let paths = fixture.paths.clone();

        let settings_paths = paths
            .data_locations()
            .expect("locations")
            .into_iter()
            .filter(|location| location.category == ClearDataCategory::Settings)
            .map(|location| location.path)
            .collect::<Vec<_>>();
        assert_eq!(
            settings_paths,
            vec![
                fixture.paths.settings_file(),
                earlier.clone(),
                later.clone(),
                python.clone()
            ]
        );
        paths.clear_data(&EVERYTHING).expect("clear");
        for removed in [&earlier, &later, &python, &fixture.paths.settings_file()] {
            assert!(!removed.exists(), "{removed:?}");
        }
        assert!(unrelated.iter().all(|path| path.exists()));
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
        let paths = fixture.paths.clone();
        let report = paths.clear_data(&EVERYTHING).expect("clear");
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
        let paths = fixture.paths.clone();

        let report = paths.clear_data(&EVERYTHING).expect("clear");

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
        let paths = fixture.paths.clone();

        let report = paths.clear_data(&EVERYTHING).expect("clear");

        assert!(report.iter().all(|entry| entry.outcome.is_ok()));
        assert!(outside.join("content").exists());
        assert!(!fixture.paths.models_dir().exists());
        assert!(!fixture.paths.logs_dir().exists());
    }
}
