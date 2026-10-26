use std::{
    cell::RefCell,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Mutex, MutexGuard},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
use whispernow_domain::{AppError, ErrorCategory};

use whispernow_core::{CancellationToken, ResolvedModel};
use whispernow_domain::ipc::ModelSummary;
use whispernow_domain::{DownloadPhase, ModelAvailability};
use whispernow_domain::{ModelFamily, ModelId};
use whispernow_platform::{
    DownloadProgress, HeldUnloaded, LocalCleanupService, ModelStore, ModelStoreError,
};

use crate::progress_throttle::ProgressThrottle;

use super::{
    Backend, lock,
    state::{DownloadTask, StateCell},
    unknown_model,
};

/// The dictation worker owns the recognizer; selecting a model asks it to load the model.
pub(crate) struct ModelLibrary {
    store: Arc<ModelStore>,
    state: Arc<StateCell>,
    /// Freed before its model file is deleted.
    local_cleanup: Arc<LocalCleanupService>,
    /// Held for a whole model download start, selection, deletion or data clear, so each sees
    /// the models as it left them.
    transition: Mutex<()>,
    /// Download workers, joined at shutdown. Each one's download is in
    /// `state.active_downloads` until it ends.
    workers: Mutex<Vec<JoinHandle<()>>>,
}

impl ModelLibrary {
    pub(super) fn new(
        store: Arc<ModelStore>,
        state: Arc<StateCell>,
        local_cleanup: Arc<LocalCleanupService>,
    ) -> Self {
        Self {
            store,
            state,
            local_cleanup,
            transition: Mutex::new(()),
            workers: Mutex::new(Vec::new()),
        }
    }

    #[cfg(all(test, unix))]
    pub(crate) fn model_store(&self) -> &ModelStore {
        self.store.as_ref()
    }

    pub(super) fn transition(&self) -> MutexGuard<'_, ()> {
        lock(&self.transition)
    }

    pub(crate) fn list(&self) -> Result<Vec<ModelSummary>, AppError> {
        let models = self
            .store
            .list()
            .map_err(|error| model_list_error(&error))?
            .into_iter()
            .map(|entry| ModelSummary::new(entry.definition, entry.availability))
            .collect();
        Ok(models)
    }

    pub(crate) fn download(&self, model_id: &ModelId) -> Result<(), AppError> {
        let _transition = self.transition();
        let total_bytes = self
            .store
            .definition(model_id)
            .map_err(|_| unknown_model())?
            .source
            .archive_size_bytes;
        let token = CancellationToken::default();
        self.state.update(|state| {
            state.require_not_in_maintenance()?;
            if state.active_downloads.contains_key(model_id) {
                return Err(AppError::new(
                    ErrorCategory::General,
                    "download_already_active",
                    "This model download is already in progress",
                ));
            }
            if state.failed_download.as_ref() == Some(model_id) {
                state.failed_download = None;
            }
            state.active_downloads.insert(
                model_id.clone(),
                DownloadTask::new(model_id.clone(), total_bytes, token.clone()),
            );
            Ok(())
        })?;
        let store = Arc::clone(&self.store);
        let state = Arc::clone(&self.state);
        let worker_model_id = model_id.clone();
        let worker = thread::Builder::new()
            .name(format!("model-download-{model_id}"))
            .spawn(move || {
                run_model_download(&state, &worker_model_id, |observer| {
                    store.download(&worker_model_id, &token, observer)
                });
            })
            .inspect_err(
                |error| tracing::warn!(%error, "the model download worker could not be started"),
            );
        let Ok(worker) = worker else {
            self.state.apply(|state| {
                state.active_downloads.remove(model_id);
                state.failed_download = Some(model_id.clone());
            });
            return Err(AppError::new(
                ErrorCategory::General,
                "download_worker",
                "The model download could not be started",
            ));
        };
        let mut workers = lock(&self.workers);
        workers.retain(|worker| !worker.is_finished());
        workers.push(worker);
        Ok(())
    }

    pub(crate) fn cancel_download(&self, model_id: &ModelId) -> Result<(), AppError> {
        let state = self.state.lock();
        let download = state.active_downloads.get(model_id).ok_or_else(|| {
            AppError::new(
                ErrorCategory::General,
                "download_inactive",
                "The download is no longer active",
            )
        })?;
        download.cancellation.cancel();
        Ok(())
    }

    /// The caller holds the model transition, so nothing changes before the file is deleted.
    fn require_deletable(&self, model_id: &ModelId) -> Result<ModelFamily, AppError> {
        let state = self.state.lock();
        if state.selected_model_id == *model_id {
            return Err(AppError::new(
                ErrorCategory::General,
                "active_model",
                "Switch models before deleting the active model",
            ));
        }
        state.require_no_downloads()?;
        drop(state);
        let models = self
            .store
            .list()
            .map_err(|error| model_list_error(&error))?;
        let target = models
            .iter()
            .find(|entry| entry.definition.id == *model_id)
            .ok_or_else(unknown_model)?;
        if target.availability != ModelAvailability::Downloaded {
            return Err(AppError::new(
                ErrorCategory::Model,
                "model_not_deletable",
                "Only downloaded models can be deleted",
            ));
        }
        let usable = models
            .iter()
            .filter(|entry| {
                entry.definition.family().is_speech()
                    && entry.availability != ModelAvailability::NotInstalled
            })
            .count();
        if target.definition.family().is_speech() && usable <= 1 {
            return Err(AppError::new(
                ErrorCategory::General,
                "last_usable_model",
                "Keep at least one usable model installed",
            ));
        }
        Ok(target.definition.family())
    }

    pub(crate) fn delete(&self, model_id: &ModelId) -> Result<(), AppError> {
        let _transition = self.transition();
        self.state.lock().phase.require_idle()?;
        let family = self.require_deletable(model_id)?;
        // A cleanup model is freed, and kept from loading, until its file is gone.
        let _unloaded = if family == ModelFamily::Cleanup {
            Some(hold_cleanup_unloaded(&self.local_cleanup)?)
        } else {
            None
        };
        self.store
            .remove(model_id)
            .inspect_err(
                |error| tracing::warn!(%error, %model_id, "the model could not be deleted"),
            )
            .map_err(|_| {
                AppError::new(
                    ErrorCategory::Model,
                    "model_delete",
                    "The model could not be deleted",
                )
            })?;
        self.state.events().models_changed();
        Ok(())
    }

    pub(super) fn request_shutdown(&self) {
        for download in self.state.lock().active_downloads.values() {
            download.cancellation.cancel();
        }
    }

    /// Cancels every download, including one started after [`Self::request_shutdown`], and
    /// waits for the workers.
    pub(super) fn shutdown_downloads(&self) {
        self.request_shutdown();
        for worker in std::mem::take(&mut *lock(&self.workers)) {
            if worker.join().is_err() {
                tracing::error!("a model download worker panicked while ending");
            }
        }
    }
}

impl Backend {
    /// Loads the model and then saves it as the selected one. A model that cannot load leaves
    /// the saved selection as it was.
    pub(crate) fn select_model(&self, model_id: ModelId) -> Result<(), AppError> {
        let _transition = self.models.transition();
        self.state.lock().require_quiescent()?;
        let resolved = installed(&self.models.store, &model_id)?;
        if !resolved.definition.family().is_speech() {
            return Err(AppError::new(
                ErrorCategory::Model,
                "model_not_speech",
                "This model cleans up transcripts; choose a speech model",
            ));
        }
        self.dictation.load_model(resolved)?;
        self.settings.edit(|saved| {
            saved.model_id.clone_from(&model_id);
            Ok(())
        })?;
        self.state.apply(|state| state.selected_model_id = model_id);
        Ok(())
    }
}

pub(super) fn installed(store: &ModelStore, id: &ModelId) -> Result<ResolvedModel, AppError> {
    store
        .resolve(id)
        .map_err(|error| {
            tracing::warn!(%error, model_id = %id, "the model could not be inspected");
            AppError::new(
                ErrorCategory::Model,
                "model_resolve",
                "The model could not be inspected",
            )
        })?
        .ok_or_else(|| {
            AppError::new(
                ErrorCategory::Model,
                "model_not_installed",
                "Install the model before using it",
            )
        })
}

fn model_list_error(error: &ModelStoreError) -> AppError {
    tracing::warn!(%error, "the models could not be listed");
    AppError::new(
        ErrorCategory::Model,
        "model_list",
        "Models could not be listed",
    )
}

/// A download that panics ends as a failed one, so its entry cannot block later model changes.
fn run_model_download(
    state: &StateCell,
    model_id: &ModelId,
    download: impl FnOnce(&dyn Fn(DownloadProgress)) -> Result<ResolvedModel, ModelStoreError>,
) {
    let started = Instant::now();
    let throttle = RefCell::new(ProgressThrottle::default());
    let observer = |progress: DownloadProgress| {
        let phase = progress.phase;
        let downloading = matches!(phase, DownloadPhase::Downloading);
        if downloading
            && !throttle.borrow_mut().poll_due(
                Instant::now(),
                progress.downloaded_bytes,
                Some(progress.total_bytes),
            )
        {
            return;
        }
        let bytes_per_second = downloading
            .then(|| transfer_rate(progress.downloaded_bytes, started.elapsed()))
            .flatten();
        state.apply(|current| {
            let download = current
                .active_downloads
                .get_mut(model_id)
                .expect("download worker owns its registered progress until completion");
            download.progress.phase = phase;
            download.progress.downloaded_bytes = progress.downloaded_bytes;
            download.progress.total_bytes = progress.total_bytes;
            download.progress.bytes_per_second = bytes_per_second;
        });
    };
    let failed = match catch_unwind(AssertUnwindSafe(|| download(&observer))) {
        Ok(Ok(_)) => false,
        Ok(Err(ModelStoreError::Cancelled)) => {
            tracing::info!(model_id = %model_id, "model download cancelled");
            false
        }
        Ok(Err(error)) => {
            tracing::warn!(%error, model_id = %model_id, "the model could not be installed");
            true
        }
        Err(_) => {
            tracing::error!(model_id = %model_id, "the model download panicked");
            true
        }
    };
    state.apply(|current| {
        current.active_downloads.remove(model_id);
        if failed {
            current.failed_download = Some(model_id.clone());
        }
    });
    state.events().models_changed();
}

const MIN_RATE_INTERVAL: Duration = Duration::from_millis(500);

/// The average transfer rate over `elapsed`, or `None` before the interval is long enough.
fn transfer_rate(bytes: u64, elapsed: Duration) -> Option<u64> {
    (elapsed >= MIN_RATE_INTERVAL).then(|| {
        let rate = u128::from(bytes) * 1_000 / elapsed.as_millis();
        u64::try_from(rate).unwrap_or(u64::MAX)
    })
}

pub(super) fn hold_cleanup_unloaded(
    cleanup: &LocalCleanupService,
) -> Result<HeldUnloaded<'_>, AppError> {
    cleanup
        .hold_unloaded(Duration::from_secs(10))
        .ok_or_else(|| {
            AppError::new(
                ErrorCategory::General,
                "model_in_use",
                "The cleanup model is busy; try again when the current cleanup finishes",
            )
        })
}

#[cfg(test)]
mod tests {
    use std::{
        io::{self, Write},
        net::{TcpListener, TcpStream},
        path::PathBuf,
        sync::{Arc, mpsc},
        thread,
        time::{Duration, Instant},
    };

    use whispernow_core::{CancellationToken, ResolvedModel};
    use whispernow_domain::ipc::{EventPayload, ModelState};
    use whispernow_domain::{DEFAULT_CLEANUP_MODEL_ID, DownloadPhase, ModelId, ModelRegistry};
    use whispernow_platform::{
        AppPaths, DownloadProgress, DownloadTimeouts, LocalCleanupService, ModelStore,
    };

    use crate::services::{
        Fixture, Phase, RecordingTrigger,
        state::{DownloadTask, Events, Published, State, StateCell},
    };

    use super::{ModelLibrary, run_model_download, transfer_rate};

    struct DownloadFixture {
        models: Arc<ModelLibrary>,
        listener: TcpListener,
        model_id: ModelId,
        _root: tempfile::TempDir,
    }

    impl DownloadFixture {
        fn new(state: &Arc<StateCell>) -> Self {
            let root = tempfile::tempdir().expect("download root");
            let listener = TcpListener::bind("127.0.0.1:0").expect("download listener");
            listener
                .set_nonblocking(true)
                .expect("nonblocking listener");
            let registry = ModelRegistry::builtin();
            let mut definition = registry
                .get(DEFAULT_CLEANUP_MODEL_ID)
                .expect("cleanup model")
                .clone();
            definition.source.url = format!(
                "http://{}/model.gguf",
                listener.local_addr().expect("listener address")
            );
            definition.source.archive_size_bytes = 7;
            definition.source.sha256 =
                "f16d05ec6b29248d2c61adb1e9263f78e4f7bace1b955014a2d17872cfe4064d".into();
            let model_id = definition.id.clone();
            let paths = AppPaths::under(root.path());
            let store = Arc::new(ModelStore::with_registry_and_timeouts(
                ModelRegistry::new(vec![definition]),
                paths,
                DownloadTimeouts {
                    connect: Duration::from_secs(10),
                    receive_body: Duration::from_secs(10),
                    total: Duration::from_secs(10),
                },
            ));
            let local_cleanup = Arc::new(LocalCleanupService::new(
                root.path().join("llama"),
                Arc::clone(&store),
                |_| {},
            ));
            Self {
                models: Arc::new(ModelLibrary::new(store, Arc::clone(state), local_cleanup)),
                listener,
                model_id,
                _root: root,
            }
        }

        fn request(&self) -> TcpStream {
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                match self.listener.accept() {
                    Ok((stream, _)) => return stream,
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                        assert!(
                            Instant::now() < deadline,
                            "download never requested its file"
                        );
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => panic!("download request failed: {error}"),
                }
            }
        }
    }

    impl Drop for DownloadFixture {
        fn drop(&mut self) {
            self.models.shutdown_downloads();
        }
    }

    #[test]
    fn update_admission_rejects_a_download_waiting_for_the_model_transition() {
        let fixture = Fixture::new();
        fixture.flush();
        let download = DownloadFixture::new(&fixture.state);
        fixture
            .state
            .apply(|state| state.failed_download = Some(download.model_id.clone()));
        let transition = download.models.transition();
        let (started, starting) = mpsc::channel();

        // Holding the existing transition lock orders the admissions without a runtime hook.
        thread::scope(|scope| {
            let downloading = scope.spawn(|| {
                started.send(()).expect("download-start observer");
                download.models.download(&download.model_id)
            });
            starting
                .recv_timeout(Duration::from_secs(10))
                .expect("download call started");
            fixture
                .dictation
                .begin_update()
                .expect("update admitted first");
            drop(transition);
            let error = downloading
                .join()
                .expect("download admission thread")
                .expect_err("update blocks the waiting download");
            assert_eq!(error.code, "update_installing");
        });

        assert_eq!(fixture.state.snapshot().phase, Phase::Updating);
        assert!(fixture.state.lock().active_downloads.is_empty());
        assert!(fixture.state.lock().failed_download.is_some());
        assert!(super::lock(&download.models.workers).is_empty());
    }

    #[test]
    fn registered_download_blocks_update_and_duplicate_admission_until_cancelled() {
        let fixture = Fixture::new();
        fixture.flush();
        let download = DownloadFixture::new(&fixture.state);
        fixture
            .state
            .apply(|state| state.failed_download = Some(download.model_id.clone()));
        download
            .models
            .download(&download.model_id)
            .expect("download admitted first");
        let mut connection = download.request();

        assert!(fixture.state.lock().failed_download.is_none());
        assert_eq!(fixture.state.snapshot().active_downloads.len(), 1);
        assert_eq!(
            fixture
                .dictation
                .begin_update()
                .expect_err("download blocks update")
                .code,
            "model_transition_busy"
        );
        assert_eq!(fixture.state.snapshot().phase, Phase::Idle);
        assert_eq!(
            download
                .models
                .download(&download.model_id)
                .expect_err("duplicate download refused")
                .code,
            "download_already_active"
        );
        assert_eq!(super::lock(&download.models.workers).len(), 1);

        download
            .models
            .cancel_download(&download.model_id)
            .expect("cancel download");
        connection
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 7\r\nConnection: close\r\n\r\nfixture")
            .expect("release the pending response");
        download.models.shutdown_downloads();
        assert_eq!(fixture.state.snapshot().active_downloads, []);
        assert!(fixture.state.lock().failed_download.is_none());
        assert!(fixture.state.snapshot().error.is_none());
        fixture
            .dictation
            .begin_update()
            .expect("update admitted after cancellation");
    }

    #[test]
    fn a_download_remains_allowed_during_recording_and_completes_normally() {
        let fixture = Fixture::new();
        fixture.flush();
        let download = DownloadFixture::new(&fixture.state);
        fixture
            .dictation
            .start(RecordingTrigger::Toggle)
            .expect("start recording");
        download
            .models
            .download(&download.model_id)
            .expect("download during recording");
        let mut connection = download.request();
        connection
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 7\r\nConnection: close\r\n\r\nfixture")
            .expect("model response");
        for worker in std::mem::take(&mut *super::lock(&download.models.workers)) {
            worker.join().expect("successful download worker");
        }

        assert!(
            download
                .models
                .store
                .resolve(&download.model_id)
                .expect("resolve model")
                .is_some()
        );
        assert!(matches!(
            fixture.state.snapshot().phase,
            Phase::Recording { .. }
        ));
        assert_eq!(fixture.state.snapshot().active_downloads, []);
        assert!(fixture.state.lock().failed_download.is_none());
        assert!(fixture.state.snapshot().error.is_none());
        fixture.dictation.cancel().expect("cancel recording");
        fixture.flush();
    }

    #[test]
    fn a_failed_download_marks_its_model_without_becoming_the_applications_error() {
        let fixture = Fixture::new();
        fixture.flush();
        let download = DownloadFixture::new(&fixture.state);
        download
            .models
            .download(&download.model_id)
            .expect("download admitted");
        let mut connection = download.request();
        connection
            .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .expect("failing response");
        download.models.shutdown_downloads();

        let snapshot = fixture.state.snapshot();
        assert_eq!(snapshot.active_downloads, []);
        assert_eq!(snapshot.failed_download, Some(download.model_id.clone()));
        assert!(snapshot.error.is_none());
    }

    #[test]
    fn a_panicking_download_leaves_no_active_entry_and_shows_as_failed() {
        let model_id = ModelId::default();
        let (events, published) = Events::channel();
        let state = StateCell::new(State::new(model_id.clone()), events);
        state.apply(|state| {
            state.active_downloads.insert(
                model_id.clone(),
                DownloadTask::new(model_id.clone(), 1_024, CancellationToken::default()),
            );
        });

        run_model_download(&state, &model_id, |_| panic!("the download panicked"));

        let snapshot = state.snapshot();
        assert_eq!(snapshot.active_downloads, []);
        assert_eq!(snapshot.failed_download, Some(model_id));
        assert_eq!(snapshot.model_state, ModelState::Failed);
        assert!(snapshot.error.is_none());
        state
            .lock()
            .require_no_downloads()
            .expect("later model changes are not blocked");
        assert!(
            published
                .try_iter()
                .any(|event| matches!(event, Published::Event(EventPayload::ModelsChanged)))
        );
    }

    #[test]
    fn thousands_of_tiny_reports_publish_a_bounded_number_of_states_and_every_phase() {
        let definition = ModelRegistry::builtin()
            .get(DEFAULT_CLEANUP_MODEL_ID)
            .expect("cleanup model")
            .clone();
        let model_id = definition.id.clone();
        let (events, published) = Events::channel();
        let state = StateCell::new(State::new(ModelId::default()), events);
        state.apply(|state| {
            state.active_downloads.insert(
                model_id.clone(),
                DownloadTask::new(model_id.clone(), 0, CancellationToken::default()),
            );
        });
        published.try_iter().for_each(drop);
        let later_phases = [
            DownloadPhase::ValidatingArchive,
            DownloadPhase::Extracting,
            DownloadPhase::Installing,
            DownloadPhase::Complete,
        ];

        let started = Instant::now();
        let (chunk, total) = (256_u64, 1024 * 1024_u64);
        run_model_download(&state, &model_id, |observer| {
            for downloaded in (0..=total).step_by(256) {
                observer(DownloadProgress {
                    phase: DownloadPhase::Downloading,
                    downloaded_bytes: downloaded,
                    total_bytes: total,
                });
            }
            for phase in later_phases {
                observer(DownloadProgress {
                    phase,
                    downloaded_bytes: total,
                    total_bytes: total,
                });
            }
            Ok(ResolvedModel {
                definition,
                root: PathBuf::new(),
            })
        });
        let elapsed = started.elapsed();

        let downloads: Vec<_> = published
            .try_iter()
            .filter_map(|event| match event {
                Published::Event(EventPayload::StateChanged { state }) => Some(
                    state
                        .active_downloads
                        .into_iter()
                        .find(|download| download.model_id == model_id),
                ),
                _ => None,
            })
            .collect();
        let (removal, reported) = downloads.split_last().expect("published states");
        assert!(removal.is_none(), "the last state drops the download");
        let phases: Vec<_> = reported
            .iter()
            .map(|download| download.as_ref().expect("reported download").phase)
            .collect();
        let downloading = phases.len() - later_phases.len();
        assert_eq!(phases[downloading..], later_phases);
        assert!(
            phases[..downloading]
                .iter()
                .all(|phase| *phase == DownloadPhase::Downloading)
        );
        // The first report, 100 whole percents, and one per 100 ms the loop took.
        let timed = usize::try_from(elapsed.as_millis() / 100).expect("short test") + 1;
        assert!(
            (101..=101 + timed).contains(&downloading),
            "{downloading} of {} reports",
            total / chunk + 1
        );
    }

    #[test]
    fn transfer_rate_waits_for_a_measurable_interval_then_averages() {
        assert_eq!(transfer_rate(4_000_000, Duration::from_millis(100)), None);
        assert_eq!(
            transfer_rate(4_000_000, Duration::from_secs(2)),
            Some(2_000_000)
        );
        assert_eq!(transfer_rate(0, Duration::from_secs(1)), Some(0));
    }
}
