//! Locating and deleting the fixed application-data targets in [`AppPaths`].

use std::{fs, io, iter, path::PathBuf};

use whispernow_domain::ClearDataCategory;
use whispernow_domain::ipc::{ClearDataItem, DataLocation, DataPlace};

use crate::{AppPaths, atomic_file::remove_path};

#[derive(Debug)]
struct Target {
    category: ClearDataCategory,
    path: PathBuf,
    removable: bool,
}

impl Target {
    fn new(category: ClearDataCategory, path: PathBuf) -> Self {
        Self {
            category,
            path,
            removable: true,
        }
    }

    fn location(&self) -> DataLocation {
        DataLocation {
            category: self.category,
            place: DataPlace::Path(self.path.clone()),
        }
    }

    fn listed(category: ClearDataCategory, path: PathBuf) -> Self {
        Self {
            removable: false,
            ..Self::new(category, path)
        }
    }
}

#[derive(Debug)]
pub struct ClearPlan {
    targets: Vec<Target>,
}

impl ClearPlan {
    #[must_use]
    pub fn locations(&self) -> Vec<DataLocation> {
        self.targets.iter().map(Target::location).collect()
    }

    /// The history database and recordings belong to the open history repository; they are
    /// listed but left for it to clear.
    #[must_use]
    pub fn remove(&self, categories: &[ClearDataCategory]) -> Vec<ClearDataItem> {
        self.targets
            .iter()
            .filter(|target| target.removable && categories.contains(&target.category))
            .map(|target| {
                let deleted = match remove_path(&target.path) {
                    Ok(()) => true,
                    Err(error) if error.kind() == io::ErrorKind::NotFound => true,
                    Err(error) => {
                        tracing::warn!(
                            category = ?target.category,
                            kind = ?error.kind(),
                            "application data could not be deleted"
                        );
                        false
                    }
                };
                ClearDataItem {
                    location: target.location(),
                    deleted,
                }
            })
            .collect()
    }
}

impl AppPaths {
    pub fn data_locations(&self) -> io::Result<Vec<DataLocation>> {
        Ok(self.clear_plan(&ClearDataCategory::ALL)?.locations())
    }

    pub fn clear_plan(&self, categories: &[ClearDataCategory]) -> io::Result<ClearPlan> {
        let mut targets = Vec::new();
        for target in self.fixed_targets(categories)? {
            match fs::symlink_metadata(&target.path) {
                Ok(_) => targets.push(target),
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
        }
        Ok(ClearPlan { targets })
    }

    fn fixed_targets(&self, categories: &[ClearDataCategory]) -> io::Result<Vec<Target>> {
        let mut targets = Vec::new();
        if categories.contains(&ClearDataCategory::Settings) {
            let backups = self.settings_backup_files()?;
            targets.extend(
                iter::once(self.settings_file())
                    .chain(backups)
                    .chain(iter::once(self.python_logs_dir()))
                    .map(|path| Target::new(ClearDataCategory::Settings, path)),
            );
        }
        if categories.contains(&ClearDataCategory::History) {
            let backups = self.history_backup_files()?;
            targets.extend(
                iter::once(self.history_file())
                    .chain(backups)
                    .map(|path| Target::new(ClearDataCategory::History, path)),
            );
            targets.push(Target::listed(
                ClearDataCategory::History,
                self.recordings_dir(),
            ));
            targets.push(Target::listed(
                ClearDataCategory::History,
                self.history_database_file(),
            ));
        }
        if categories.contains(&ClearDataCategory::Models) {
            targets.push(Target::new(ClearDataCategory::Models, self.models_dir()));
        }
        Ok(targets)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
    };

    use tempfile::TempDir;

    use super::ClearDataCategory;
    use crate::AppPaths;
    use whispernow_domain::ipc::{ClearDataItem, DataLocation, DataPlace};

    const EVERYTHING: [ClearDataCategory; 3] = [
        ClearDataCategory::Settings,
        ClearDataCategory::History,
        ClearDataCategory::Models,
    ];

    struct Fixture {
        #[cfg_attr(not(unix), expect(dead_code))]
        temp: TempDir,
        paths: AppPaths,
    }

    impl Fixture {
        fn separate_roots() -> Self {
            let temp = tempfile::tempdir().expect("temporary directory");
            let paths = AppPaths::under(temp.path());
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
            create_dir(&self.paths.python_logs_dir());
        }
    }

    fn clear(paths: &AppPaths, categories: &[ClearDataCategory]) -> Vec<ClearDataItem> {
        paths
            .clear_plan(categories)
            .expect("clear plan")
            .remove(categories)
    }

    fn path_of(location: DataLocation) -> PathBuf {
        match location.place {
            DataPlace::Path(path) => path,
            place => panic!("{place:?} is not a path"),
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

        assert!(paths.data_locations().expect("locations").is_empty());
        assert!(clear(&paths, &EVERYTHING).is_empty());
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
                &[ClearDataCategory::Models],
            ]
            .concat()
        );

        let report = clear(&paths, &EVERYTHING);
        assert_eq!(report.len(), 6);
        assert!(report.iter().all(|entry| entry.deleted));
        let remaining = paths.data_locations().expect("remaining targets");
        assert_eq!(remaining.len(), 2);
        let remaining: Vec<_> = remaining.into_iter().map(path_of).collect();
        assert!(remaining.contains(&paths.history_database_file()));
        assert!(remaining.contains(&paths.recordings_dir()));
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
            .clear_plan(&EVERYTHING)
            .expect("clear plan")
            .remove(&[ClearDataCategory::Settings]);

        assert_eq!(report.len(), 2);
        assert!(!fixture.paths.python_logs_dir().exists());
        assert!(!fixture.paths.settings_file().exists());
        assert!(fixture.paths.models_dir().exists());
        assert!(fixture.paths.history_database_file().exists());
    }

    #[test]
    fn inspection_failure_happens_before_any_target_is_deleted() {
        let fixture = Fixture::separate_roots();
        create_file(fixture.paths.config_dir());
        create_file(&fixture.paths.history_database_file());
        create_dir(&fixture.paths.recordings_dir());
        create_dir(&fixture.paths.models_dir());
        create_dir(&fixture.paths.python_logs_dir());

        fixture
            .paths
            .clear_plan(&EVERYTHING)
            .expect_err("inspection failure");

        for path in [
            fixture.paths.history_database_file(),
            fixture.paths.recordings_dir().join("content"),
            fixture.paths.models_dir().join("content"),
            fixture.paths.python_logs_dir().join("content"),
        ] {
            assert_eq!(fs::read(path).expect("data preserved"), b"test data");
        }
    }

    #[test]
    fn a_plan_removes_only_the_targets_it_inspected() {
        let fixture = Fixture::separate_roots();
        create_file(&fixture.paths.settings_file());
        let plan = fixture.paths.clear_plan(&EVERYTHING).expect("clear plan");
        let later = fixture.paths.unreadable_settings_file("20260926T120000Z");
        create_file(&later);

        let report = plan.remove(&EVERYTHING);

        assert_eq!(report.len(), 1);
        assert!(!fixture.paths.settings_file().exists());
        assert!(later.exists());
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
            .map(path_of)
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
        clear(&paths, &EVERYTHING);
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
            locations.into_iter().map(path_of).collect::<Vec<_>>(),
            vec![earlier.clone(), later.clone(), imported.clone()]
        );

        clear(&fixture.paths, &[ClearDataCategory::Settings]);
        for path in [&later, &earlier, &imported] {
            assert!(path.exists());
        }
        let report = clear(&fixture.paths, &[ClearDataCategory::History]);
        assert_eq!(report.len(), 3);
        assert!(
            report
                .iter()
                .all(|item| item.location.category == ClearDataCategory::History && item.deleted)
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

        let report = clear(&paths, &EVERYTHING);

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
        create_dir(&fixture.paths.python_logs_dir());
        symlink(
            &outside,
            fixture.paths.python_logs_dir().join("outside-link"),
        )
        .expect("nested symlink");
        let paths = fixture.paths.clone();

        let report = clear(&paths, &EVERYTHING);

        assert!(report.iter().all(|entry| entry.deleted));
        assert!(outside.join("content").exists());
        assert!(!fixture.paths.models_dir().exists());
        assert!(!fixture.paths.python_logs_dir().exists());
    }
}
