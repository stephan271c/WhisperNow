use std::{
    path::PathBuf,
    sync::{Mutex, MutexGuard},
};
use whispernow_domain::AppError;

use whispernow_domain::{ClearDataCategory, ErrorCategory, ipc::ClearDataItem};
use whispernow_domain::{EntryId, HistoryLimit, HistoryPage, HistoryPageRequest, NewHistoryEntry};
use whispernow_platform::{AppPaths, SqliteHistoryRepository};

pub(crate) struct HistoryStore {
    paths: AppPaths,
    repository: Mutex<Result<SqliteHistoryRepository, AppError>>,
}

impl HistoryStore {
    pub(super) fn open(paths: AppPaths) -> (Self, Option<AppError>) {
        let (repository, diagnostic) = match SqliteHistoryRepository::open(&paths) {
            Ok((repository, diagnostic)) => (Ok(repository), diagnostic),
            Err(error) => (Err(error.clone()), Some(error)),
        };
        (
            Self {
                paths,
                repository: Mutex::new(repository),
            },
            diagnostic,
        )
    }

    /// Removes the entries beyond a new `limit`; returns whether any were removed.
    pub(super) fn prune(&self, limit: HistoryLimit) -> Result<bool, AppError> {
        self.lock()
            .as_ref()
            .map_err(Clone::clone)?
            .prune(limit)
            .map(|removed| removed > 0)
    }

    /// Saves one dictation, keeping its audio when `keep_audio` or its status ask for it, and
    /// at most `limit` entries.
    pub(super) fn insert(
        &self,
        entry: NewHistoryEntry,
        audio: &[f32],
        keep_audio: bool,
        limit: HistoryLimit,
    ) -> Result<EntryId, AppError> {
        self.lock()
            .as_mut()
            .map_err(|error| error.clone())?
            .insert(entry, audio, keep_audio, limit)
    }

    /// Records that an entry's text never reached the focused application.
    pub(super) fn mark_not_delivered(&self, id: EntryId, error: &str) -> Result<(), AppError> {
        self.lock()
            .as_ref()
            .map_err(Clone::clone)?
            .mark_not_delivered(id, error)
    }

    pub(super) fn list(&self, request: &HistoryPageRequest) -> Result<HistoryPage, AppError> {
        self.lock().as_ref().map_err(Clone::clone)?.list(request)
    }

    /// One entry's kept audio as WAV bytes.
    pub(crate) fn audio(&self, id: EntryId) -> Result<Vec<u8>, AppError> {
        self.lock()
            .as_ref()
            .map_err(Clone::clone)?
            .audio(id)?
            .ok_or_else(|| {
                AppError::new(
                    ErrorCategory::NotFound,
                    "history_audio_not_kept",
                    "The audio for this dictation was not kept",
                )
            })
    }

    /// Deletes one entry; returns whether it existed.
    pub(super) fn delete(&self, id: EntryId) -> Result<bool, AppError> {
        self.lock().as_ref().map_err(Clone::clone)?.delete(id)
    }

    /// Deletes every entry; returns whether anything was removed.
    pub(super) fn clear(&self) -> Result<bool, AppError> {
        self.lock().as_ref().map_err(Clone::clone)?.clear()
    }

    /// The folder of kept audio, created when missing so it can be opened before the first
    /// recording is kept.
    pub(super) fn recordings_dir(&self) -> Result<PathBuf, AppError> {
        let repository = self.lock();
        repository.as_ref().map_err(Clone::clone)?;
        let directory = self.paths.recordings_dir();
        std::fs::create_dir_all(&directory).map_err(|_| {
            AppError::new(
                ErrorCategory::General,
                "recordings_folder_unavailable",
                "The recordings folder could not be created",
            )
        })?;
        Ok(directory)
    }

    pub(super) fn clear_data(&self) -> ClearDataItem {
        let repository = self.lock();
        let cleared = repository
            .as_ref()
            .map_err(Clone::clone)
            .and_then(SqliteHistoryRepository::clear)
            .inspect_err(|error| {
                tracing::warn!(
                    code = error.code,
                    "transcription history could not be cleared"
                );
            });
        let deleted = cleared.is_ok()
            && self
                .paths
                .clear_data(&[ClearDataCategory::History])
                .is_ok_and(|items| items.iter().all(|item| item.deleted));
        ClearDataItem {
            category: ClearDataCategory::History,
            path: "History entries, recordings, and legacy files".to_owned(),
            deleted,
        }
    }

    fn lock(&self) -> MutexGuard<'_, Result<SqliteHistoryRepository, AppError>> {
        self.repository
            .lock()
            .expect("history repository lock poisoned")
    }
}

#[cfg(test)]
mod tests {
    use whispernow_domain::{EntryStatus, HistoryFilter, UtcMillis};

    use super::*;

    fn store() -> (tempfile::TempDir, HistoryStore) {
        let dir = tempfile::tempdir().expect("temp dir");
        let paths = AppPaths::from_base_dirs(dir.path().join("config"), dir.path().join("data"));
        (dir, HistoryStore::open(paths).0)
    }

    fn entry(text: &str) -> NewHistoryEntry {
        NewHistoryEntry {
            created_at: UtcMillis::now(),
            duration_millis: Some(1_000),
            raw_text: Some(text.to_owned()),
            cleaned_text: None,
            status: EntryStatus::Ok,
            error: None,
            asr_model_id: None,
        }
    }

    #[test]
    fn pruning_reports_removals_and_missing_audio_reports_not_kept() {
        let (_dir, store) = store();
        let first = store
            .insert(entry("first"), &[], false, HistoryLimit::Unlimited)
            .expect("insert");
        assert!(!store.prune(HistoryLimit::Last20).expect("nothing to prune"));
        for _ in 0..20 {
            store
                .insert(entry("later"), &[], false, HistoryLimit::Unlimited)
                .expect("insert");
        }
        assert!(store.prune(HistoryLimit::Last20).expect("prune"));
        let error = store.audio(first).expect_err("missing audio");
        assert_eq!(error.code, "history_audio_not_kept");
    }

    #[test]
    fn failed_startup_open_is_not_retried_after_the_path_is_repaired() {
        let dir = tempfile::tempdir().expect("temp dir");
        let data = dir.path().join("data");
        std::fs::write(&data, b"obstructed").expect("block data directory");
        let paths = AppPaths::from_base_dirs(dir.path().join("config"), &data);
        let (store, diagnostic) = HistoryStore::open(paths.clone());
        let error = diagnostic.expect("startup error");
        std::fs::remove_file(data).expect("repair data directory");
        let request = HistoryPageRequest::first_page(HistoryFilter::default());
        assert_eq!(store.list(&request).expect_err("not reopened"), error);
        assert_eq!(
            store
                .insert(entry("later"), &[], false, HistoryLimit::Unlimited)
                .expect_err("not reopened"),
            error
        );
        assert!(!paths.history_database_file().exists());
    }

    #[test]
    fn malformed_legacy_history_is_reported_only_at_startup() {
        let dir = tempfile::tempdir().expect("temp dir");
        let paths = AppPaths::from_base_dirs(dir.path().join("config"), dir.path().join("data"));
        std::fs::create_dir_all(paths.config_dir()).expect("config directory");
        std::fs::write(paths.history_file(), b"malformed").expect("legacy history");
        let (store, diagnostic) = HistoryStore::open(paths.clone());
        assert!(diagnostic.is_some());
        std::fs::write(paths.history_file(), b"also malformed").expect("new legacy history");
        store.clear().expect("clear");
        store
            .list(&HistoryPageRequest::first_page(HistoryFilter::default()))
            .expect("same connection");
        assert!(paths.history_file().exists());
    }
}
