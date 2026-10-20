use std::{
    sync::{Arc, Mutex, MutexGuard},
    time::Duration,
};
use whispernow_domain::AppError;

use whispernow_core::audio::{AudioDeviceChoice, AudioInput, AudioInputError};
use whispernow_domain::ipc::{
    AudioDevice, ErrorCategory, MicrophoneTestResponse, PermissionKind, PermissionStatus,
    PermissionsResponse, SaveSettingsResponse, SettingsView,
};
use whispernow_domain::{
    CleanupTarget, EntryId, HistoryFilter, HistoryPage, HistoryPageRequest, Settings,
    SetupCheckpoint, SetupStep,
};
use whispernow_platform::{
    AppPaths, LocalCleanupService, open_in_file_manager, open_privacy_settings,
};

pub(crate) use dictation::Dictation;
#[cfg(test)]
pub(crate) use dictation::tests::Fixture;
use history::HistoryStore;
use models::ModelCoordinator;
use output::OutputCoordinator;
use permissions::{MicrophoneAccess, SessionFacts, permission_states};
use settings::{SettingsCoordinator, device_choice, require_connected_device, settings_error};
pub(crate) use state::{Events, StateCell};
pub(crate) use whispernow_domain::ipc::{Phase, RecordingTrigger};

mod clear_data;
mod compose;
mod dictation;
mod history;
mod microphone_test;
mod models;
mod output;
mod permissions;
mod settings;
mod state;
#[cfg(test)]
mod test_support;

/// Tauri commands call its coordinators directly, and its own methods for work that spans
/// several coordinators.
pub(crate) struct NativeServices {
    pub(crate) settings: Arc<SettingsCoordinator>,
    pub(crate) history: Arc<HistoryStore>,
    audio: Arc<dyn AudioInput>,
    pub(crate) models: Arc<ModelCoordinator>,
    pub(crate) state: Arc<StateCell>,
    pub(crate) dictation: Dictation,
    paths: AppPaths,
    pub(crate) output: Arc<OutputCoordinator>,
    microphone_access: MicrophoneAccess,
    local_cleanup: Arc<LocalCleanupService>,
}

impl NativeServices {
    /// A change or a failure broadcasts `HistoryChanged` so every listener re-fetches. The
    /// history store's lock orders it against dictations.
    fn run_history_mutation(
        &self,
        operation: impl FnOnce() -> Result<bool, AppError>,
    ) -> Result<(), AppError> {
        let result = operation();
        if !matches!(result, Ok(false)) {
            self.state.events().history_changed();
        }
        result.map(|_| ())
    }

    /// Deletes aged-out history on its own: once now, then daily, so entries expire while the
    /// app sits minimized for days.
    pub(crate) fn spawn_retention_checks(services: Arc<Self>) {
        tauri::async_runtime::spawn(async move {
            services.prune_expired_history();
            let mut interval = tokio::time::interval(Duration::from_hours(24));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            interval.tick().await;
            loop {
                interval.tick().await;
                services.prune_expired_history();
            }
        });
    }

    /// The settings lock is dropped before the history lock is taken, per their lock order.
    fn prune_expired_history(&self) {
        let (limit, retention) = {
            let settings = self.settings.lock();
            (settings.history_limit, settings.history_retention)
        };
        if let Err(error) = self.run_history_mutation(|| self.history.prune(limit, retention)) {
            tracing::warn!(code = error.code, "aged-out history could not be pruned");
        }
    }

    /// Persists the edited settings and applies the resulting history limit. The
    /// caller applies native side effects after this returns, from the settings
    /// before the edit that come back with the response. A pruning failure after
    /// the save is a warning next to the saved settings.
    pub(crate) fn update_settings(
        &self,
        edited: Settings,
    ) -> Result<(Settings, SaveSettingsResponse), AppError> {
        let (before, settings) = self.settings.update(edited, self.audio.as_ref())?;
        follow_cleanup_provider(&self.local_cleanup, &settings.settings);
        let limit = settings.settings.history_limit;
        let retention = settings.settings.history_retention;
        let pruning = self.run_history_mutation(|| {
            if limit == before.history_limit && retention == before.history_retention {
                Ok(false)
            } else {
                self.history.prune(limit, retention)
            }
        });
        let warnings = pruning
            .err()
            .map(|warning| warning.summary)
            .into_iter()
            .collect();
        Ok((before, SaveSettingsResponse { settings, warnings }))
    }

    pub(crate) fn copy_last_transcript(&self) -> Result<(), AppError> {
        let text = self.state.lock().last_transcript.clone().ok_or_else(|| {
            AppError::new(
                ErrorCategory::NotFound,
                "no_transcript",
                "There is no transcript to copy yet",
            )
        })?;
        self.output.copy(&text)
    }

    pub(crate) fn list_audio_devices(&self) -> Result<Vec<AudioDevice>, AppError> {
        self.audio
            .list_input_devices()
            .map_err(|_| audio_devices_unavailable())
    }

    /// Listens to the selected microphone for a few seconds and reports its levels. No audio
    /// is kept.
    pub(crate) fn test_microphone(&self) -> Result<MicrophoneTestResponse, AppError> {
        let device = self.settings.input_device()?;
        self.listen(device, microphone_test::TEST_DURATION)
    }

    /// Captures on the dictation worker and records the permission answer the capture showed.
    fn listen(
        &self,
        device: AudioDeviceChoice,
        duration: Duration,
    ) -> Result<MicrophoneTestResponse, AppError> {
        let result = self.dictation.test_microphone(device, duration);
        self.microphone_access.observe(&result);
        result
    }

    /// The shortcut portal's consent is the shortcut coordinator's to report, so the caller
    /// passes it in.
    pub(crate) fn permissions(&self, global_shortcuts: PermissionStatus) -> PermissionsResponse {
        permission_states(
            self.microphone_access.status(),
            SessionFacts::collect(&self.output, &self.paths, global_shortcuts),
        )
    }

    /// Asks for one first-run permission where the desktop services can.
    pub(crate) fn request_permission(&self, kind: PermissionKind) -> Result<(), AppError> {
        match kind {
            PermissionKind::Microphone => self.request_microphone(),
            PermissionKind::RemoteDesktop => self.request_remote_desktop(),
            PermissionKind::Accessibility
            | PermissionKind::GlobalShortcuts
            | PermissionKind::PasteSupport => Err(permission_not_requestable()),
        }
    }

    pub(crate) fn open_permission_settings(kind: PermissionKind) -> Result<(), AppError> {
        open_privacy_settings(kind).map_err(|_| no_permission_settings())
    }

    /// Listens to the default microphone briefly, which is how each platform
    /// asks for it. A refusal is an answer, shown on the permission card; any
    /// other failure is returned.
    fn request_microphone(&self) -> Result<(), AppError> {
        match self.listen(
            AudioDeviceChoice::SystemDefault,
            microphone_test::PERMISSION_CHECK_DURATION,
        ) {
            Err(error) if error.category != ErrorCategory::Permission => Err(error),
            Ok(_) | Err(_) => Ok(()),
        }
    }

    /// Asks the compositor for the paste keystroke's consent and waits for the
    /// user's answer, which the permission card then shows.
    fn request_remote_desktop(&self) -> Result<(), AppError> {
        self.state.lock().phase.require_idle()?;
        self.output
            .request_remote_desktop_consent()
            .map(drop)
            .ok_or_else(permission_not_requestable)
    }

    pub(crate) fn save_setup_checkpoint(
        &self,
        checkpoint: SetupCheckpoint,
    ) -> Result<SettingsView, AppError> {
        let model_supported = self
            .models
            .model_store()
            .definition(&checkpoint.model_id)
            .is_ok();
        self.settings
            .save_setup_checkpoint(checkpoint, model_supported)
    }

    /// Saves setup's choices and loads the chosen model. A model that cannot load leaves the
    /// saved settings and setup as they were.
    pub(crate) fn complete_setup(
        &self,
        checkpoint: &SetupCheckpoint,
    ) -> Result<SettingsView, AppError> {
        if checkpoint.step != SetupStep::TryIt {
            return Err(AppError::new(
                ErrorCategory::General,
                "setup_incomplete",
                "Review every setup step before finishing",
            ));
        }
        let before = self.settings.load();
        {
            let state = self.state.lock();
            if before.first_run_complete {
                return Err(setup_already_complete());
            }
            state.phase.require_idle()?;
        }
        let device = device_choice(checkpoint.input_device.as_deref())?;
        require_connected_device(self.audio.as_ref(), &device)?;
        let mut edited = before.clone();
        apply_checkpoint(&mut edited, checkpoint);
        edited
            .validate_update(&before)
            .map_err(|error| settings_error(&error))?;
        self.select_model(checkpoint.model_id.clone())?;
        self.settings.edit(|settings| {
            apply_checkpoint(settings, checkpoint);
            Ok(SettingsView::new(settings))
        })
    }

    /// Reads one history page in any phase; only the history store's own lock
    /// orders it against changes and data clearing. A search that is too long is the search
    /// field's problem; a store that cannot be read, or a problem found while opening it, is
    /// the application's.
    pub(crate) fn history(&self, request: &HistoryPageRequest) -> Result<HistoryPage, AppError> {
        if request.filter.query.len() > HistoryFilter::MAX_QUERY_BYTES {
            return Err(AppError::new(
                ErrorCategory::General,
                "text_too_large",
                "This search is too long. Shorten it and try again.",
            ));
        }
        self.history
            .list(request)
            .inspect_err(|error| self.state.publish_error(error.clone()))
    }

    /// The path never crosses the webview boundary.
    pub(crate) fn open_recordings_folder(&self) -> Result<(), AppError> {
        let directory = self.history.recordings_dir()?;
        open_in_file_manager(&directory).map_err(|_| {
            AppError::new(
                ErrorCategory::General,
                "recordings_folder_open_failed",
                "The recordings folder could not be opened",
            )
        })
    }

    pub(crate) fn delete_history_entry(&self, request: EntryId) -> Result<(), AppError> {
        self.run_history_mutation(|| self.history.delete(request))
    }

    pub(crate) fn clear_history(&self) -> Result<(), AppError> {
        self.run_history_mutation(|| self.history.clear())
    }

    /// Stops accepting work and ends the dictation in flight, keeping its audio.
    pub(crate) fn request_shutdown(&self) {
        self.output.request_shutdown();
        self.models.request_shutdown();
        self.dictation.request_shutdown();
    }

    pub(crate) fn shutdown(&self) {
        self.models.shutdown_downloads();
        self.dictation.join();
        // Output admission is closed by request_shutdown; joining its producer
        // before draining also covers the helper's early keystroke response.
        self.output.shutdown();
        self.state.lock().active_downloads.clear();
    }
}

/// Locks service state. A poisoned lock means another thread panicked mid-update, which is a
/// bug; it is not reported to the user as a recoverable error.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().expect("service state lock poisoned")
}

fn permission_not_requestable() -> AppError {
    AppError::new(
        ErrorCategory::General,
        "permission_not_requestable",
        "This permission cannot be requested here",
    )
}

/// This platform has no system page for the permission, or it could not be opened.
fn no_permission_settings() -> AppError {
    AppError::new(
        ErrorCategory::General,
        "permission_settings_unavailable",
        "The system settings page could not be opened",
    )
}

pub(crate) fn shutting_down() -> AppError {
    AppError::new(
        ErrorCategory::General,
        "application_shutting_down",
        "The application is shutting down",
    )
}

pub(crate) fn provider_models_unavailable() -> AppError {
    AppError::new(
        ErrorCategory::Cleanup,
        "provider_models_unavailable",
        "The provider's models could not be listed",
    )
}

fn audio_devices_unavailable() -> AppError {
    AppError::new(
        ErrorCategory::Audio,
        "audio_devices",
        "Audio input devices could not be listed",
    )
}

fn unknown_model() -> AppError {
    AppError::new(
        ErrorCategory::Model,
        "unknown_model",
        "The model is not supported",
    )
}

fn setup_already_complete() -> AppError {
    AppError::new(
        ErrorCategory::General,
        "setup_already_complete",
        "Setup has already finished",
    )
}

fn capture_start_error(error: &AudioInputError) -> AppError {
    if matches!(error, AudioInputError::PermissionDenied) {
        AppError::new(
            ErrorCategory::Permission,
            "microphone_permission_denied",
            "Microphone access is disabled. Enable it in system privacy settings and retry.",
        )
    } else {
        AppError::new(
            ErrorCategory::Audio,
            "audio_start",
            "Audio capture could not be started",
        )
    }
}

/// Loads the chosen local cleanup model in the background when AI cleanup uses it, and frees it
/// otherwise.
fn follow_cleanup_provider(local_cleanup: &LocalCleanupService, settings: &Settings) {
    local_cleanup.set_use_gpu(settings.cleanup_uses_gpu);
    match settings.active_cleanup_target() {
        Some(CleanupTarget::Local { model }) => local_cleanup.preload(model),
        _ => local_cleanup.unload(),
    }
}

fn apply_checkpoint(settings: &mut Settings, checkpoint: &SetupCheckpoint) {
    settings.input_device.clone_from(&checkpoint.input_device);
    settings
        .recording_shortcut
        .clone_from(&checkpoint.recording_shortcut);
    settings.recording_mode = checkpoint.recording_mode;
    settings.first_run_complete = true;
    settings.setup = None;
}
