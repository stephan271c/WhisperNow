//! Locating and deleting the fixed application-data targets in [`AppPaths`].

use std::{
    fs, io,
    path::{Path, PathBuf},
};

use thiserror::Error;
use whispernow_domain::ClearDataCategory;
use whispernow_domain::ipc::{ClearDataItem, DataLocation};

use crate::AppPaths;

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[error("could not inspect {category:?} data: {kind}")]
pub struct ClearDataError {
    pub category: ClearDataCategory,
    pub kind: io::ErrorKind,
}

impl AppPaths {
    /// The fixed targets that currently exist.
    pub fn data_locations(&self) -> Result<Vec<DataLocation>, ClearDataError> {
        self.data_locations_for(&ClearDataCategory::ALL)
    }

    pub fn data_locations_for(
        &self,
        categories: &[ClearDataCategory],
    ) -> Result<Vec<DataLocation>, ClearDataError> {
        self.inspect_targets(categories)
    }

    /// The history database and recordings belong to the open history repository, and the
    /// CUDA pack to the model store; they are listed but left for those owners to clear.
    pub fn clear_data(
        &self,
        categories: &[ClearDataCategory],
    ) -> Result<Vec<ClearDataItem>, ClearDataError> {
        let cuda_pack = self.cuda_pack_dir();
        let targets = self
            .inspect_targets(categories)?
            .into_iter()
            .filter(|location| {
                location.path != cuda_pack
                    && location.path != self.history_database_file()
                    && location.path != self.recordings_dir()
            });
        let mut entries = Vec::new();
        for DataLocation { category, path } in targets {
            let deleted = remove_path(&path)
                .map_or_else(|error| error.kind() == io::ErrorKind::NotFound, |()| true);
            entries.push(ClearDataItem {
                category,
                path: path.to_string_lossy().into_owned(),
                deleted,
            });
        }
        Ok(entries)
    }

    fn inspect_targets(
        &self,
        categories: &[ClearDataCategory],
    ) -> Result<Vec<DataLocation>, ClearDataError> {
        let mut locations = Vec::new();
        for (category, path) in self.fixed_targets(categories)? {
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

    fn fixed_targets(
        &self,
        categories: &[ClearDataCategory],
    ) -> Result<Vec<(ClearDataCategory, PathBuf)>, ClearDataError> {
        let mut targets = Vec::new();
        if categories.contains(&ClearDataCategory::Settings) {
            let backups = self
                .settings_backup_files()
                .map_err(|error| ClearDataError {
                    category: ClearDataCategory::Settings,
                    kind: error.kind(),
                })?;
            targets.push((ClearDataCategory::Settings, self.settings_file()));
            targets.extend(
                backups
                    .into_iter()
                    .map(|path| (ClearDataCategory::Settings, path)),
            );
            targets.push((ClearDataCategory::Settings, self.python_logs_dir()));
        }
        if categories.contains(&ClearDataCategory::History) {
            let backups = self
                .history_backup_files()
                .map_err(|error| ClearDataError {
                    category: ClearDataCategory::History,
                    kind: error.kind(),
                })?;
            targets.push((ClearDataCategory::History, self.history_file()));
            targets.extend(
                backups
                    .into_iter()
                    .map(|path| (ClearDataCategory::History, path)),
            );
        }
        targets.extend(
            [
                (ClearDataCategory::History, self.recordings_dir()),
                (ClearDataCategory::History, self.history_database_file()),
                (ClearDataCategory::Models, self.models_dir()),
                (ClearDataCategory::Models, self.cuda_pack_dir()),
            ]
            .into_iter()
            .filter(|(category, _)| categories.contains(category)),
        );
        Ok(targets)
    }
}

pub(crate) fn remove_path(path: &Path) -> io::Result<()> {
    // `symlink_metadata` keeps deletion from following the target of a symlink.
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_dir() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use tempfile::TempDir;

    use super::ClearDataCategory;
    use crate::AppPaths;

    const EVERYTHING: [ClearDataCategory; 3] = [
        ClearDataCategory::Settings,
        ClearDataCategory::History,
        ClearDataCategory::Models,
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
            create_file(&self.paths.unreadable_history_file("20261015T120000Z"));
            create_file(&self.paths.history_database_file());
            create_dir(&self.paths.recordings_dir());
            create_dir(&self.paths.models_dir());
            create_dir(&self.paths.cuda_pack_dir());
            create_dir(&self.paths.python_logs_dir());
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
                [ClearDataCategory::Settings; 2].as_slice(),
                &[ClearDataCategory::History; 5],
                &[ClearDataCategory::Models; 2],
            ]
            .concat()
        );

        let report = paths.clear_data(&EVERYTHING).expect("clear");
        assert_eq!(report.len(), 6);
        assert!(report.iter().all(|entry| entry.deleted));
        let remaining = paths.data_locations().expect("remaining targets");
        assert_eq!(remaining.len(), 3);
        assert!(
            remaining
                .iter()
                .any(|item| item.path == paths.history_database_file())
        );
        assert!(
            remaining
                .iter()
                .any(|item| item.path == paths.recordings_dir())
        );
        assert!(
            remaining
                .iter()
                .any(|item| item.path == paths.cuda_pack_dir())
        );
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
            .clear_data(&[ClearDataCategory::Settings])
            .expect("clear settings");

        assert_eq!(report.len(), 2);
        assert!(!fixture.paths.python_logs_dir().exists());
        assert!(!fixture.paths.settings_file().exists());
        assert!(fixture.paths.models_dir().exists());
        assert!(fixture.paths.cuda_pack_dir().exists());
        assert!(fixture.paths.history_database_file().exists());
    }

    #[test]
    fn inspection_failure_happens_before_any_target_is_deleted() {
        let fixture = Fixture::separate_roots();
        create_file(fixture.paths.config_dir());
        create_file(&fixture.paths.history_database_file());
        create_dir(&fixture.paths.recordings_dir());
        create_dir(&fixture.paths.models_dir());
        create_dir(&fixture.paths.cuda_pack_dir());
        create_dir(&fixture.paths.python_logs_dir());

        let error = fixture
            .paths
            .clear_data(&EVERYTHING)
            .expect_err("inspection failure");

        assert_eq!(error.category, ClearDataCategory::Settings);
        for path in [
            fixture.paths.history_database_file(),
            fixture.paths.recordings_dir().join("content"),
            fixture.paths.models_dir().join("content"),
            fixture.paths.cuda_pack_dir().join("content"),
            fixture.paths.python_logs_dir().join("content"),
        ] {
            assert_eq!(fs::read(path).expect("data preserved"), b"test data");
        }
    }

    #[test]
    fn settings_moved_aside_are_listed_and_deleted_with_settings() {
        let fixture = Fixture::separate_roots();
        create_file(&fixture.paths.settings_file());
        create_dir(&fixture.paths.python_logs_dir());
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
                python.clone(),
                fixture.paths.python_logs_dir(),
            ]
        );
        paths.clear_data(&EVERYTHING).expect("clear");
        assert!(!fixture.paths.python_logs_dir().exists());
        for removed in [&earlier, &later, &python, &fixture.paths.settings_file()] {
            assert!(!removed.exists(), "{removed:?}");
        }
        assert!(unrelated.iter().all(|path| path.exists()));
    }

    #[test]
    fn malformed_history_backups_are_listed_and_deleted_only_with_history() {
        let fixture = Fixture::separate_roots();
        let later = fixture.paths.unreadable_history_file("20261015T120001Z");
        let earlier = fixture.paths.unreadable_history_file("20261015T120000Z");
        let imported = fixture.paths.migrated_history_file();
        let unrelated = fixture.paths.config_dir().join("history.json.bak");
        for path in [&later, &earlier, &imported, &unrelated] {
            create_file(path);
        }
        let locations = fixture.paths.data_locations().expect("locations");
        assert_eq!(
            locations
                .into_iter()
                .map(|location| location.path)
                .collect::<Vec<_>>(),
            vec![earlier.clone(), later.clone(), imported.clone()]
        );

        fixture
            .paths
            .clear_data(&[ClearDataCategory::Settings])
            .expect("clear settings");
        for path in [&later, &earlier, &imported] {
            assert!(path.exists());
        }
        let report = fixture
            .paths
            .clear_data(&[ClearDataCategory::History])
            .expect("clear history");
        assert_eq!(report.len(), 3);
        assert!(
            report
                .iter()
                .all(|item| item.category == ClearDataCategory::History && item.deleted)
        );
        for path in [&later, &earlier, &imported] {
            assert!(!path.exists());
        }
        assert!(unrelated.exists());
    }

    #[test]
    fn shared_windows_config_and_data_root_is_never_deleted() {
        let fixture = Fixture::windows_layout();
        fixture.create_every_target();
        let sentinel = fixture.paths.config_dir().join("unrelated.db");
        create_file(&sentinel);
        let paths = fixture.paths.clone();

        let report = paths.clear_data(&EVERYTHING).expect("clear");

        assert!(report.iter().all(|entry| entry.deleted));
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
        symlink(&outside, fixture.paths.cuda_pack_dir()).expect("CUDA pack symlink");
        create_dir(&fixture.paths.python_logs_dir());
        symlink(
            &outside,
            fixture.paths.python_logs_dir().join("outside-link"),
        )
        .expect("nested symlink");
        let paths = fixture.paths.clone();

        let report = paths.clear_data(&EVERYTHING).expect("clear");

        assert!(report.iter().all(|entry| entry.deleted));
        assert!(outside.join("content").exists());
        assert!(!fixture.paths.models_dir().exists());
        assert!(fixture.paths.cuda_pack_dir().exists());
        assert!(!fixture.paths.python_logs_dir().exists());
    }
}
