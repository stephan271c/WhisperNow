use std::{
    sync::{Arc, Mutex, MutexGuard},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
use whispernow_domain::AppError;

use whispernow_core::{CancellationToken, ResolvedModel};
use whispernow_domain::ipc::{ActiveDownload, ErrorCategory, ModelSummary, Phase};
use whispernow_domain::{DownloadPhase, ModelAvailability};
use whispernow_domain::{ModelFamily, ModelId, ModelProfile};
use whispernow_platform::{
    CudaPackState, DownloadProgress, Llama, LocalCleanupService, ModelStore, ModelStoreError,
};

use super::{
    Dictation, lock,
    settings::SettingsCoordinator,
    state::{DownloadTask, StateCell},
};

/// The dictation worker owns the recognizer; selecting a model asks it to load the model.
pub(crate) struct ModelCoordinator {
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

impl ModelCoordinator {
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
            .map_err(model_list_error)?
            .into_iter()
            .map(|entry| ModelSummary {
                recommended: entry.definition.id == ModelId::default(),
                family: entry.definition.family(),
                id: entry.definition.id,
                display_name: entry.definition.display_name,
                availability: entry.availability,
                languages: entry.definition.languages,
                size_bytes: entry.definition.source.archive_size_bytes,
                speed: entry.definition.profile.speed(),
                accuracy: entry.definition.profile.accuracy(),
                benchmark: entry.definition.profile.benchmark(),
            })
            .chain(self.cuda_pack())
            .collect();
        Ok(models)
    }

    /// The CUDA pack, listed with the models it serves, where the platform has one.
    fn cuda_pack(&self) -> Option<ModelSummary> {
        let availability = match self.store.cuda_pack_state() {
            CudaPackState::Unsupported => return None,
            CudaPackState::NotInstalled => ModelAvailability::NotInstalled,
            CudaPackState::Installed => ModelAvailability::Downloaded,
        };
        Some(ModelSummary {
            id: ModelId::cuda_pack(),
            display_name: "NVIDIA CUDA for cleanup models".to_owned(),
            family: ModelFamily::CudaPack,
            recommended: false,
            availability,
            languages: Vec::new(),
            size_bytes: self.store.cuda_pack_size(),
            speed: CUDA_PACK_PROFILE.speed(),
            accuracy: CUDA_PACK_PROFILE.accuracy(),
            benchmark: None,
        })
    }

    pub(crate) fn download(&self, model_id: &ModelId) -> Result<(), AppError> {
        let _transition = self.transition();
        let token = CancellationToken::default();
        self.state.update(|state| {
            if matches!(state.phase, Phase::ClearingData | Phase::Updating) {
                state.phase.require_idle()?;
            }
            if state.active_downloads.contains_key(model_id) {
                return Err(AppError::new(
                    ErrorCategory::General,
                    "download_already_active",
                    "This model download is already in progress",
                ));
            }
            if state.active_downloads.is_empty() {
                state.download_failures = 0;
            }
            state.active_downloads.insert(
                model_id.clone(),
                DownloadTask {
                    progress: ActiveDownload {
                        model_id: model_id.clone(),
                        phase: DownloadPhase::Downloading,
                        downloaded_bytes: 0,
                        total_bytes: None,
                        bytes_per_second: None,
                    },
                    cancellation: token.clone(),
                },
            );
            Ok(())
        })?;
        let store = Arc::clone(&self.store);
        let state = Arc::clone(&self.state);
        let local_cleanup = Arc::clone(&self.local_cleanup);
        let worker_model_id = model_id.clone();
        let worker = thread::Builder::new()
            .name(format!("model-download-{model_id}"))
            .spawn(move || {
                run_model_download(store, &state, &local_cleanup, worker_model_id, token);
            });
        let Ok(worker) = worker else {
            self.state.apply(|state| {
                state.active_downloads.remove(model_id);
                state.download_failures = state.download_failures.saturating_add(1);
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

    /// Loads the model and then saves it as the selected one. A model that cannot load leaves
    /// the saved selection as it was.
    pub(crate) fn select(
        &self,
        model_id: ModelId,
        settings: &SettingsCoordinator,
        dictation: &Dictation,
    ) -> Result<(), AppError> {
        let _transition = self.transition();
        {
            let state = self.state.lock();
            state.phase.require_idle()?;
            state.require_no_downloads()?;
        }
        let resolved = installed(&self.store, &model_id)?;
        if !resolved.definition.family().is_speech() {
            return Err(AppError::new(
                ErrorCategory::Model,
                "model_not_speech",
                "This model cleans up transcripts; choose a speech model",
            ));
        }
        dictation.load_model(resolved)?;
        let mut current = settings.lock();
        let mut saved = current.clone();
        if saved.model_id != model_id {
            saved.model_id = model_id.clone();
            settings.save(&mut current, saved)?;
        }
        self.state.apply(|state| state.selected_model_id = model_id);
        Ok(())
    }

    /// Checks that the requested model is a downloaded, inactive model and not the last usable
    /// one. The caller holds the model transition, so nothing changes that
    /// before the file is deleted.
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
        let models = self.store.list().map_err(model_list_error)?;
        let target = models
            .iter()
            .find(|entry| entry.definition.id == *model_id)
            .ok_or_else(|| {
                AppError::new(
                    ErrorCategory::Model,
                    "unknown_model",
                    "The model is not supported",
                )
            })?;
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
        if model_id.is_cuda_pack() {
            return self.delete_cuda_pack();
        }
        let family = self.require_deletable(model_id)?;
        // A cleanup model is freed, and kept from loading, until its file is gone.
        let _unloaded = if family == ModelFamily::Cleanup {
            Some(hold_cleanup_unloaded(&self.local_cleanup)?)
        } else {
            None
        };
        self.store.remove(model_id).map_err(|_| {
            AppError::new(
                ErrorCategory::Model,
                "model_delete",
                "The model could not be deleted",
            )
        })?;
        Ok(())
    }

    /// Removes the CUDA pack. Once loaded it stays in use until the app restarts, so it is
    /// then only marked, and deleted at the next start.
    fn delete_cuda_pack(&self) -> Result<(), AppError> {
        self.state.lock().require_no_downloads()?;
        self.store
            .remove_cuda_pack(Llama::cuda_loaded())
            .map_err(|_| {
                AppError::new(
                    ErrorCategory::Model,
                    "model_delete",
                    "The CUDA download could not be deleted",
                )
            })?;
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
            let _worker_result = worker.join();
        }
    }
}

pub(super) fn installed(store: &ModelStore, id: &ModelId) -> Result<ResolvedModel, AppError> {
    store
        .resolve(id)
        .map_err(|_| {
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

pub(super) fn model_list_error<E>(_error: E) -> AppError {
    AppError::new(
        ErrorCategory::Model,
        "model_list",
        "Models could not be listed",
    )
}

#[allow(clippy::needless_pass_by_value)]
fn run_model_download(
    store: Arc<ModelStore>,
    state: &StateCell,
    local_cleanup: &LocalCleanupService,
    model_id: ModelId,
    cancellation: CancellationToken,
) {
    let started = Instant::now();
    let observer = |progress: DownloadProgress| {
        let phase = progress.phase;
        let bytes_per_second = matches!(phase, DownloadPhase::Downloading)
            .then(|| transfer_rate(progress.downloaded_bytes, started.elapsed()))
            .flatten();
        state.apply(|current| {
            let download = current
                .active_downloads
                .get_mut(&model_id)
                .expect("download worker owns its registered progress until completion");
            download.progress.phase = phase;
            download.progress.downloaded_bytes = progress.downloaded_bytes;
            download.progress.total_bytes = progress.total_bytes;
            download.progress.bytes_per_second = bytes_per_second;
        });
    };
    let result = if model_id.is_cuda_pack() {
        let installed = store.download_cuda_pack(&cancellation, &observer);
        if installed.is_ok() {
            // The loaded cleanup model, if any, moves to CUDA now.
            local_cleanup.reload();
        }
        installed
    } else {
        store
            .download(&model_id, &cancellation, &observer)
            .map(drop)
    };
    let completion_error = match result {
        Ok(()) => None,
        Err(ModelStoreError::Cancelled) => {
            tracing::info!(model_id = %model_id, "model download cancelled");
            None
        }
        Err(_) => Some(AppError::new(
            ErrorCategory::Model,
            "model_download",
            "The model could not be installed",
        )),
    };
    state.apply(|current| {
        current.active_downloads.remove(&model_id);
        if let Some(error) = completion_error {
            current.download_failures = current.download_failures.saturating_add(1);
            current.last_error = Some(error);
        }
    });
}

const CUDA_PACK_PROFILE: ModelProfile = ModelProfile::rated(5, 3);

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
) -> Result<impl Drop + '_, AppError> {
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
        sync::{Arc, mpsc},
        thread,
        time::{Duration, Instant},
    };

    use whispernow_domain::{DEFAULT_CLEANUP_MODEL_ID, ModelId, ModelRegistry};
    use whispernow_platform::{AppPaths, DownloadTimeouts, LocalCleanupService, ModelStore};

    use crate::services::{Fixture, RecordingTrigger, state::StateCell};

    use super::{ModelCoordinator, Phase, transfer_rate};

    struct DownloadFixture {
        models: Arc<ModelCoordinator>,
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
            let paths =
                AppPaths::from_base_dirs(root.path().join("config"), root.path().join("data"));
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
                Arc::new(|_| None),
                Arc::new(|| None),
            ));
            Self {
                models: Arc::new(ModelCoordinator::new(
                    store,
                    Arc::clone(state),
                    local_cleanup,
                )),
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
        fixture.state.lock().download_failures = 3;
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
        assert_eq!(fixture.state.lock().download_failures, 3);
        assert!(super::lock(&download.models.workers).is_empty());
    }

    #[test]
    fn registered_download_blocks_update_and_duplicate_admission_until_cancelled() {
        let fixture = Fixture::new();
        fixture.flush();
        let download = DownloadFixture::new(&fixture.state);
        fixture.state.lock().download_failures = 3;
        download
            .models
            .download(&download.model_id)
            .expect("download admitted first");
        let mut connection = download.request();

        assert_eq!(fixture.state.lock().download_failures, 0);
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
        assert!(fixture.state.snapshot().active_downloads.is_empty());
        assert_eq!(fixture.state.lock().download_failures, 0);
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
        assert!(fixture.state.snapshot().active_downloads.is_empty());
        assert_eq!(fixture.state.lock().download_failures, 0);
        assert!(fixture.state.snapshot().error.is_none());
        fixture.dictation.cancel().expect("cancel recording");
        fixture.flush();
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
