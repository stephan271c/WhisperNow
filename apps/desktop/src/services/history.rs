use std::{
    path::PathBuf,
    sync::{Mutex, MutexGuard},
};

use whispernow_api::{ApiErrorDto, ErrorCategoryDto};
use whispernow_domain::{
    AppError, EntryId, HistoryEntry, HistoryLimit, HistoryPage, HistoryPageRequest, NewHistoryEntry,
};
use whispernow_platform::{AppPaths, SqliteHistoryRepository};

const UNAVAILABLE_CODE: &str = "history_unavailable";
const UNAVAILABLE_SUMMARY: &str = "Transcription history is unavailable";

struct HistoryState {
    /// `None` after a failed open or a data clear; the next use opens it again.
    repository: Option<SqliteHistoryRepository>,
    /// A problem found while opening, reported once.
    warning: Option<AppError>,
}

/// Transcription history: one database file and a folder of WAV files. Every use holds the lock, so a
/// data clear never races a write.
pub(crate) struct HistoryStore {
    paths: AppPaths,
    state: Mutex<HistoryState>,
}

impl HistoryStore {
    pub(super) fn open(paths: AppPaths) -> Self {
        let store = Self {
            paths,
            state: Mutex::new(HistoryState {
                repository: None,
                warning: None,
            }),
        };
        let mut state = store.lock();
        if let Err(error) = store.repository(&mut state) {
            state.warning = Some(error);
        }
        drop(state);
        store
    }

    /// Takes the problem found while opening history, if any.
    pub(super) fn take_warning(&self) -> Option<ApiErrorDto> {
        self.lock()
            .warning
            .take()
            .map(|error| storage_error(&error))
    }

    /// Removes the entries beyond a new `limit`; returns whether any were removed.
    pub(super) fn prune(&self, limit: HistoryLimit) -> Result<bool, ApiErrorDto> {
        let mut state = self.lock();
        let repository = self.repository(&mut state).map_err(|e| storage_error(&e))?;
        repository
            .prune(limit)
            .map(|removed| removed > 0)
            .map_err(|error| storage_error(&error))
    }

    /// Saves one dictation, keeping its audio when `keep_audio` or its status ask for it, and
    /// at most `limit` entries.
    pub(super) fn insert(
        &self,
        entry: NewHistoryEntry,
        audio: &[f32],
        keep_audio: bool,
        limit: HistoryLimit,
    ) -> Result<HistoryEntry, AppError> {
        let mut state = self.lock();
        self.repository(&mut state)?
            .insert(entry, audio, keep_audio, limit)
    }

    /// Records that an entry's text never reached the focused application.
    pub(super) fn mark_not_delivered(&self, id: EntryId, error: &str) -> Result<(), AppError> {
        let mut state = self.lock();
        self.repository(&mut state)?.mark_not_delivered(id, error)
    }

    pub(super) fn list(&self, request: &HistoryPageRequest) -> Result<HistoryPage, ApiErrorDto> {
        let mut state = self.lock();
        self.repository(&mut state)
            .and_then(|repository| repository.list(request))
            .map_err(|error| storage_error(&error))
    }

    /// One entry's kept audio as WAV bytes.
    pub(crate) fn audio(&self, id: EntryId) -> Result<Vec<u8>, ApiErrorDto> {
        let mut state = self.lock();
        self.repository(&mut state)
            .and_then(|repository| repository.audio(id))
            .map_err(|error| storage_error(&error))?
            .ok_or_else(|| {
                ApiErrorDto::new(
                    ErrorCategoryDto::NotFound,
                    "history_audio_not_kept",
                    "The audio for this dictation was not kept",
                )
            })
    }

    /// Deletes one entry; returns whether it existed.
    pub(super) fn delete(&self, id: EntryId) -> Result<bool, ApiErrorDto> {
        let mut state = self.lock();
        self.repository(&mut state)
            .and_then(|repository| repository.delete(id))
            .map_err(|error| storage_error(&error))
    }

    /// Deletes every entry; returns whether anything was removed.
    pub(super) fn clear(&self) -> Result<bool, ApiErrorDto> {
        let mut state = self.lock();
        self.repository(&mut state)
            .and_then(SqliteHistoryRepository::clear)
            .map_err(|error| storage_error(&error))
    }

    /// The folder of kept audio, created when missing so it can be opened before the first
    /// recording is kept.
    pub(super) fn recordings_dir(&self) -> Result<PathBuf, ApiErrorDto> {
        let directory = self.paths.recordings_dir();
        std::fs::create_dir_all(&directory).map_err(|_| {
            ApiErrorDto::new(
                ErrorCategoryDto::General,
                "recordings_folder_unavailable",
                "The recordings folder could not be created",
            )
        })?;
        Ok(directory)
    }

    pub(super) fn close(&self) -> Result<(), AppError> {
        match self.lock().repository.take() {
            Some(repository) => repository.close(),
            None => Ok(()),
        }
    }

    /// Clears and closes history, then runs `operation` while still holding the lock, so the
    /// files it deletes are not reopened meanwhile. The next use opens history again.
    pub(super) fn with_repository_closed<T>(
        &self,
        operation: impl FnOnce() -> T,
    ) -> Result<T, ApiErrorDto> {
        let mut state = self.lock();
        if let Some(repository) = state.repository.take() {
            repository.clear().map_err(|error| storage_error(&error))?;
            repository.close().map_err(|error| storage_error(&error))?;
        }
        state.warning = None;
        Ok(operation())
    }

    fn repository<'a>(
        &self,
        state: &'a mut HistoryState,
    ) -> Result<&'a SqliteHistoryRepository, AppError> {
        if state.repository.is_none() {
            let (repository, diagnostic) =
                SqliteHistoryRepository::open(&self.paths).map_err(|error| {
                    tracing::warn!(code = error.code, "transcription history did not open");
                    AppError::redacted(UNAVAILABLE_CODE, UNAVAILABLE_SUMMARY)
                })?;
            if let Some(diagnostic) = diagnostic {
                tracing::warn!(
                    code = diagnostic.code,
                    "transcription history opened with a problem"
                );
                state.warning = Some(diagnostic);
            }
            state.repository = Some(repository);
        }
        Ok(state.repository.as_ref().expect("history was just opened"))
    }

    fn lock(&self) -> MutexGuard<'_, HistoryState> {
        self.state.lock().expect("history state lock poisoned")
    }
}

fn storage_error(error: &AppError) -> ApiErrorDto {
    let category = if error.code == UNAVAILABLE_CODE {
        ErrorCategoryDto::HistoryUnavailable
    } else {
        ErrorCategoryDto::General
    };
    ApiErrorDto::new(category, &error.code, &error.summary)
}

#[cfg(test)]
mod tests {
    use whispernow_domain::{EntryStatus, HistoryFilter, UtcMillis};

    use super::*;

    fn store() -> (tempfile::TempDir, HistoryStore) {
        let dir = tempfile::tempdir().expect("temp dir");
        let paths = AppPaths::from_base_dirs(dir.path().join("config"), dir.path().join("data"));
        (dir, HistoryStore::open(paths))
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
            llm_model: None,
        }
    }

    #[test]
    fn a_smaller_limit_prunes_and_kept_audio_plays_back() {
        let (_dir, store) = store();
        for index in 0..25 {
            store
                .insert(
                    entry(&format!("entry {index}")),
                    &[0.0; 160],
                    true,
                    HistoryLimit::Unlimited,
                )
                .expect("insert");
        }
        assert!(store.prune(HistoryLimit::Last20).expect("prune"));
        let page = store
            .list(&HistoryPageRequest::first_page(HistoryFilter::default()))
            .expect("list");
        assert_eq!(page.entries.len(), 20);
        let wav = store.audio(page.entries[0].id).expect("kept audio");
        assert!(wav.starts_with(b"RIFF"));
    }

    #[test]
    fn history_reopens_after_a_data_clear() {
        let (_dir, store) = store();
        store
            .insert(entry("before"), &[], false, HistoryLimit::Last20)
            .expect("insert");
        store.with_repository_closed(|| ()).expect("closed");
        let page = store
            .list(&HistoryPageRequest::first_page(HistoryFilter::default()))
            .expect("reopened");
        assert!(page.entries.is_empty());
        assert!(store.audio(EntryId(1)).is_err());
    }
}
