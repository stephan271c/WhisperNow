use std::{
    path::PathBuf,
    sync::{Arc, Mutex, MutexGuard},
    time::Duration,
};

use whispernow_api::{
    ApiErrorDto, AudioDeviceDto, DeleteHistoryEntryRequest, ErrorCategoryDto, EventPayload,
    HistoryFilter, HistoryPage, HistoryPageRequest, ListAudioDevicesResponse,
    MicrophoneTestResponse, PermissionKindDto, PermissionStatusDto, PermissionsResponse,
    SaveSettingsResponse, Settings, SettingsView, SetupCheckpoint, SetupStep, TestPromptRequest,
    TestPromptResponse,
};
use whispernow_core::{
    CancellationToken,
    audio::{AudioDeviceChoice, AudioInput, AudioInputError},
};
use whispernow_domain::LlmProvider;
use whispernow_platform::{
    DataCleaner, LocalCleanupService, PrivacySettingsPage, open_in_file_manager,
    open_privacy_settings,
};

use dictation::AiCleanup;
pub(crate) use dictation::Dictation;
use history::HistoryStore;
use models::ModelCoordinator;
use output::OutputCoordinator;
use permissions::{MicrophoneAccess, SystemPermissions, permission_states};
pub(crate) use settings::PersistedShortcuts;
use settings::{SettingsCoordinator, device_choice, require_connected_device, settings_error};
pub(crate) use state::{Events, Phase, RecordingTrigger, StateCell};

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

/// The native services. Tauri commands call its coordinators directly, and its own
/// methods for work that spans several coordinators.
pub(crate) struct NativeServices {
    pub(crate) settings: Arc<SettingsCoordinator>,
    pub(crate) history: Arc<HistoryStore>,
    audio: Arc<dyn AudioInput>,
    pub(crate) models: Arc<ModelCoordinator>,
    pub(crate) state: Arc<StateCell>,
    pub(crate) dictation: Dictation,
    llm: Arc<AiCleanup>,
    clear_data: DataCleaner,
    output: Arc<OutputCoordinator>,
    paste_helper_setup_dir: PathBuf,
    microphone_access: MicrophoneAccess,
    local_cleanup: Arc<LocalCleanupService>,
}

impl NativeServices {
    fn publish_history_changed(&self) {
        self.state.events().publish(EventPayload::HistoryChanged);
    }

    /// Runs one history change. A change or a failure broadcasts `HistoryChanged` so every
    /// listener re-fetches. The history store's lock orders it against dictations.
    fn run_history_mutation(
        &self,
        operation: impl FnOnce() -> Result<bool, ApiErrorDto>,
    ) -> Result<(), ApiErrorDto> {
        let result = operation();
        if !matches!(result, Ok(false)) {
            self.publish_history_changed();
        }
        result.map(|_| ())
    }

    /// Persists the edited settings and applies the resulting history limit. The
    /// caller applies native side effects after this returns, from the settings
    /// before the edit that come back with the response. A pruning failure after
    /// the save is a warning next to the saved settings.
    pub(crate) fn update_settings(
        &self,
        edited: Settings,
    ) -> Result<(Settings, SaveSettingsResponse), ApiErrorDto> {
        self.state.lock().phase.require_idle()?;
        let (before, settings) = self.settings.update(edited, self.audio.as_ref())?;
        self.follow_cleanup_provider(&settings.settings);
        let limit = settings.settings.history_limit;
        let pruning = if limit == before.history_limit {
            Ok(false)
        } else {
            self.history.prune(limit)
        };
        if !matches!(pruning, Ok(false)) {
            self.publish_history_changed();
        }
        let warnings = pruning.err().into_iter().collect();
        Ok((before, SaveSettingsResponse { settings, warnings }))
    }

    /// Loads the chosen local cleanup model in the background when AI cleanup uses it, and
    /// frees it otherwise.
    fn follow_cleanup_provider(&self, settings: &Settings) {
        if settings.llm_provider == LlmProvider::LocalCleanup && settings.active_prompt_id.is_some()
        {
            self.local_cleanup.preload(settings.cleanup_model_id());
        } else {
            self.local_cleanup.unload();
        }
    }

    /// Runs a saved prompt on sample text with the saved provider settings.
    pub(crate) fn test_prompt(
        &self,
        request: &TestPromptRequest,
        shutdown: &CancellationToken,
    ) -> Result<TestPromptResponse, ApiErrorDto> {
        self.state.refuse_while_updating()?;
        let (plan, timeout) = self.settings.prompt_test_plan(&request.prompt_id)?;
        let text = self
            .llm
            .run_prompt(&plan, &request.sample_text, timeout, shutdown)?;
        Ok(TestPromptResponse { text })
    }

    /// Whether [`Self::copy_last_transcript`] can work in this session; native Wayland output
    /// has no clipboard access of its own.
    pub(crate) fn can_copy_transcripts(&self) -> bool {
        self.output.can_copy()
    }

    /// Copies the last dictation's final text for a manual paste.
    pub(crate) fn copy_last_transcript(&self) -> Result<(), ApiErrorDto> {
        let text = self
            .state
            .lock()
            .last_transcript
            .as_ref()
            .map(|transcript| transcript.final_text.clone())
            .filter(|text| !text.is_empty())
            .ok_or_else(|| {
                ApiErrorDto::new(
                    ErrorCategoryDto::NotFound,
                    "no_transcript",
                    "There is no transcript to copy yet",
                )
            })?;
        self.output.copy(&text).map_err(|_| {
            ApiErrorDto::new(
                ErrorCategoryDto::Output,
                "copy_unavailable",
                "The transcript could not be copied; open History to copy it",
            )
        })
    }

    pub(crate) fn list_audio_devices(&self) -> ListAudioDevicesResponse {
        let devices = self
            .audio
            .list_input_devices()
            .unwrap_or_default()
            .into_iter()
            // The windows offer the system default themselves.
            .filter_map(|device| match device.choice {
                AudioDeviceChoice::SystemDefault => None,
                AudioDeviceChoice::Device(id) => Some(AudioDeviceDto {
                    id: id.as_str().to_owned(),
                    display_name: device.display_name,
                    is_default: device.is_current_default,
                }),
            })
            .collect();
        ListAudioDevicesResponse { devices }
    }

    /// Listens to the selected microphone for a few seconds and reports its levels. No audio
    /// is kept.
    pub(crate) fn test_microphone(&self) -> Result<MicrophoneTestResponse, ApiErrorDto> {
        let device = device_choice(self.settings.recording_settings()?.input_device.as_deref());
        self.listen(device, microphone_test::TEST_DURATION)
    }

    /// Captures on the dictation worker and records the permission answer the capture showed.
    fn listen(
        &self,
        device: AudioDeviceChoice,
        duration: Duration,
    ) -> Result<MicrophoneTestResponse, ApiErrorDto> {
        let result = self.dictation.test_microphone(device, duration);
        self.microphone_access.observe(&result);
        result
    }

    /// The command that sets the Wayland paste helper up, when this build
    /// carries its install script.
    fn paste_helper_setup_command(&self) -> Option<String> {
        #[cfg(target_os = "linux")]
        {
            crate::input_helper::setup_command(&self.paste_helper_setup_dir)
        }
        #[cfg(not(target_os = "linux"))]
        {
            None
        }
    }

    /// The first-run permission cards. The shortcut portal's consent is the
    /// shortcut coordinator's to report, so the caller passes it in.
    pub(crate) fn permissions(&self, global_shortcuts: PermissionStatusDto) -> PermissionsResponse {
        permission_states(&SystemPermissions {
            services: self,
            global_shortcuts,
        })
    }

    /// Asks for one first-run permission where the desktop services can. The
    /// shortcut portal's consent belongs to the shortcut coordinator.
    pub(crate) fn request_permission(&self, kind: PermissionKindDto) -> Result<(), ApiErrorDto> {
        match kind {
            PermissionKindDto::Microphone => self.request_microphone(),
            PermissionKindDto::RemoteDesktop => self.request_remote_desktop(),
            PermissionKindDto::Accessibility
            | PermissionKindDto::GlobalShortcuts
            | PermissionKindDto::PasteSupport => Err(permission_not_requestable()),
        }
    }

    /// Opens the system settings page that grants `kind`, where the platform has one.
    pub(crate) fn open_permission_settings(kind: PermissionKindDto) -> Result<(), ApiErrorDto> {
        let page = match kind {
            PermissionKindDto::Microphone => PrivacySettingsPage::Microphone,
            PermissionKindDto::Accessibility => PrivacySettingsPage::Accessibility,
            PermissionKindDto::GlobalShortcuts
            | PermissionKindDto::RemoteDesktop
            | PermissionKindDto::PasteSupport => return Err(no_permission_settings()),
        };
        open_privacy_settings(page).map_err(|_| no_permission_settings())
    }

    /// Listens to the default microphone briefly, which is how each platform
    /// asks for it. A refusal is an answer, shown on the permission card; any
    /// other failure is returned.
    fn request_microphone(&self) -> Result<(), ApiErrorDto> {
        match self.listen(
            AudioDeviceChoice::SystemDefault,
            microphone_test::PERMISSION_CHECK_DURATION,
        ) {
            Err(error) if error.code != "microphone_permission_denied" => Err(error),
            Ok(_) | Err(_) => Ok(()),
        }
    }

    /// Asks the compositor for the paste keystroke's consent and waits for the
    /// user's answer, which the permission card then shows.
    fn request_remote_desktop(&self) -> Result<(), ApiErrorDto> {
        self.state.lock().phase.require_idle()?;
        self.output
            .request_remote_desktop_consent()
            .map(drop)
            .ok_or_else(permission_not_requestable)
    }

    /// Saves setup's choices and loads the chosen model. A model that cannot load leaves the
    /// saved settings and setup as they were.
    pub(crate) fn complete_setup(
        &self,
        checkpoint: &SetupCheckpoint,
    ) -> Result<SettingsView, ApiErrorDto> {
        if checkpoint.step != SetupStep::TryIt {
            return Err(ApiErrorDto::new(
                ErrorCategoryDto::General,
                "setup_incomplete",
                "Review every setup step before finishing",
            ));
        }
        {
            let state = self.state.lock();
            if state.setup_complete {
                return Err(ApiErrorDto::new(
                    ErrorCategoryDto::General,
                    "setup_already_complete",
                    "Setup has already finished",
                ));
            }
            state.phase.require_idle()?;
        }
        require_connected_device(self.audio.as_ref(), checkpoint.input_device.as_deref())?;
        let resolved = self
            .models
            .resolve(&checkpoint.model_id)
            .map_err(|_| {
                ApiErrorDto::new(
                    ErrorCategoryDto::Model,
                    "model_resolve",
                    "The model could not be inspected",
                )
            })?
            .ok_or_else(|| {
                ApiErrorDto::new(
                    ErrorCategoryDto::Model,
                    "model_not_installed",
                    "Install the model before finishing setup",
                )
            })?;
        self.dictation.load_model(resolved)?;
        let transaction = self.settings.transaction();
        let mut settings = transaction.load()?;
        let before = settings.clone();
        settings.model_id = checkpoint.model_id.clone();
        settings.input_device.clone_from(&checkpoint.input_device);
        settings.hold_shortcut = checkpoint.hold_shortcut.clone();
        settings.hold_shortcut_enabled = checkpoint.hold_shortcut_enabled;
        settings
            .toggle_shortcut
            .clone_from(&checkpoint.toggle_shortcut);
        settings.first_run_complete = true;
        settings.setup = None;
        settings
            .validate_update(&before)
            .map_err(|error| settings_error(&error))?;
        transaction.save(&settings)?;
        self.state.apply(|state| {
            state.setup_complete = true;
            state.selected_model_id.clone_from(&checkpoint.model_id);
        });
        Ok(SettingsView::new(&settings))
    }

    /// Reads one history page in any phase; only the history store's own lock
    /// orders it against changes and data clearing. A search that is too long is the search
    /// field's problem; a store that cannot be read, or a problem found while opening it, is
    /// the application's.
    pub(crate) fn history(&self, request: &HistoryPageRequest) -> Result<HistoryPage, ApiErrorDto> {
        if request.filter.query.len() > HistoryFilter::MAX_QUERY_BYTES {
            return Err(ApiErrorDto::new(
                ErrorCategoryDto::General,
                "text_too_large",
                "This search is too long. Shorten it and try again.",
            ));
        }
        let page = self
            .history
            .list(request)
            .inspect_err(|error| self.state.publish_error(error.clone()))?;
        if let Some(warning) = self.history.take_warning() {
            self.state.publish_error(warning);
        }
        Ok(page)
    }

    /// Opens the retained-recordings directory in the desktop file manager. The
    /// path never crosses the webview boundary.
    pub(crate) fn open_recordings_folder(&self) -> Result<(), ApiErrorDto> {
        let directory = self.history.recordings_dir()?;
        open_in_file_manager(&directory).map_err(|_| {
            ApiErrorDto::new(
                ErrorCategoryDto::General,
                "recordings_folder_open_failed",
                "The recordings folder could not be opened",
            )
        })
    }

    pub(crate) fn delete_history_entry(
        &self,
        request: DeleteHistoryEntryRequest,
    ) -> Result<(), ApiErrorDto> {
        self.run_history_mutation(|| self.history.delete(request.id))
    }

    pub(crate) fn clear_history(&self) -> Result<(), ApiErrorDto> {
        self.run_history_mutation(|| self.history.clear())
    }

    /// Stops accepting work and ends the dictation in flight, keeping its audio.
    pub(crate) fn request_shutdown(&self) {
        self.output.request_shutdown();
        self.models.request_shutdown();
        self.dictation.request_shutdown();
    }

    pub(crate) fn shutdown(&self) {
        self.output.shutdown();
        self.models.shutdown_downloads();
        self.dictation.join();
        self.state.lock().active_downloads.clear();
        if let Err(error) = self.history.close() {
            tracing::warn!(
                code = error.code,
                "history repository could not close cleanly"
            );
        }
    }
}

/// Locks service state. A poisoned lock means another thread panicked mid-update, which is a
/// bug; it is not reported to the user as a recoverable error.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().expect("service state lock poisoned")
}

/// This platform has no in-app way to ask for the permission.
fn permission_not_requestable() -> ApiErrorDto {
    ApiErrorDto::new(
        ErrorCategoryDto::General,
        "permission_not_requestable",
        "This permission cannot be requested here",
    )
}

/// This platform has no system page for the permission, or it could not be opened.
fn no_permission_settings() -> ApiErrorDto {
    ApiErrorDto::new(
        ErrorCategoryDto::General,
        "permission_settings_unavailable",
        "The system settings page could not be opened",
    )
}

fn shutting_down() -> ApiErrorDto {
    ApiErrorDto::new(
        ErrorCategoryDto::General,
        "application_shutting_down",
        "The application is shutting down",
    )
}

fn capture_start_error(error: &AudioInputError) -> ApiErrorDto {
    if matches!(error, AudioInputError::PermissionDenied) {
        ApiErrorDto::new(
            ErrorCategoryDto::Permission,
            "microphone_permission_denied",
            "Microphone access is disabled. Enable it in system privacy settings and retry.",
        )
    } else {
        ApiErrorDto::new(
            ErrorCategoryDto::Audio,
            "audio_start",
            "Audio capture could not be started",
        )
    }
}
