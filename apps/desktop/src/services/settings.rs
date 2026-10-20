use std::sync::{Arc, Mutex, MutexGuard};
use whispernow_domain::{AppError, ErrorCategory};

use whispernow_core::audio::{AudioDeviceChoice, AudioDeviceId, AudioInput};
use whispernow_domain::ipc::{SetApiKeyRequest, SettingsView};
use whispernow_domain::{ApiKey, LlmProvider, RemoteProvider, SettingsError};
use whispernow_domain::{ModelId, Settings, SetupCheckpoint, SetupStep, WindowGeometry};
use whispernow_platform::{SettingsNotice, SettingsStore};

use super::{
    audio_devices_unavailable,
    dictation::{CleanupPlan, DictationSettings, RecordingSettings},
    lock, provider_models_unavailable, setup_already_complete,
    state::StateCell,
    unknown_model,
};

/// Owns settings persistence and serializes its writers.
pub(crate) struct SettingsCoordinator {
    persistence: Arc<SettingsStore>,
    state: Arc<StateCell>,
    current: Mutex<Settings>,
}

impl SettingsCoordinator {
    pub(super) fn new(
        persistence: Arc<SettingsStore>,
        state: Arc<StateCell>,
        settings: Settings,
    ) -> Self {
        Self {
            persistence,
            state,
            current: Mutex::new(settings),
        }
    }

    /// Lock order: model-transition -> settings -> history -> state.
    pub(super) fn lock(&self) -> MutexGuard<'_, Settings> {
        lock(&self.current)
    }

    /// Applies `change` to a copy of the saved settings. An edit that changed them is saved and
    /// becomes the saved settings; a failed edit or save leaves them as they were.
    pub(super) fn edit<R>(
        &self,
        change: impl FnOnce(&mut Settings) -> Result<R, AppError>,
    ) -> Result<R, AppError> {
        let mut saved = self.lock();
        let mut edited = saved.clone();
        let result = change(&mut edited)?;
        if edited != *saved {
            self.persistence.save(&edited)?;
            *saved = edited;
        }
        Ok(result)
    }

    pub(super) fn api_key(&self, provider: LlmProvider) -> Result<Option<ApiKey>, AppError> {
        self.persistence.api_key(provider)
    }

    /// `provider` at its saved endpoint, with its API key when one is saved.
    pub(crate) fn provider_and_key(
        &self,
        provider: LlmProvider,
    ) -> Result<(RemoteProvider, Option<ApiKey>), AppError> {
        let settings = self.load();
        let endpoint = settings.provider_settings(provider).api_base.clone();
        let remote =
            RemoteProvider::new(provider, endpoint).ok_or_else(provider_models_unavailable)?;
        let api_key = if settings.api_keys.contains(&provider) {
            self.api_key(provider)?
        } else {
            None
        };
        Ok((remote, api_key))
    }

    /// Deletes every provider's API key from the credential store. Returns whether the
    /// store accepted every deletion; it stops at the first refusal, because an
    /// unavailable store typically refuses, or blocks on, every entry.
    pub(super) fn delete_api_keys(&self) -> bool {
        LlmProvider::ALL
            .into_iter()
            .all(|provider| self.persistence.delete_api_key(provider).is_ok())
    }

    /// Publishes, once, what happened to the settings file as a state warning.
    fn report_notice(&self, notice: Option<SettingsNotice>) {
        let (code, summary) = match notice {
            None => return,
            Some(SettingsNotice::Unreadable { path }) => (
                "settings_unreadable",
                format!(
                    "Settings could not be read and were reset to defaults. The original file was moved to {}",
                    path.display()
                ),
            ),
            Some(SettingsNotice::ApiKeysNotImported { path }) => (
                "api_keys_not_imported",
                format!(
                    "The OS credential store is unavailable, so the earlier version's API keys were not imported; enter them again. They remain in {}",
                    path.display()
                ),
            ),
        };
        self.warn(code, &summary);
    }

    /// Publishes what happened to the settings file as it loaded, and warns when the saved
    /// settings break a rule, such as after a hand edit. They stay in effect, and an edit that
    /// leaves that rule alone still saves.
    pub(super) fn report_loaded(&self, saved: &Settings, notice: Option<SettingsNotice>) {
        self.report_notice(notice);
        if let Err(error) = saved.validate() {
            self.warn(
                "settings_break_a_rule",
                &format!("A saved setting needs attention: {error}"),
            );
        }
    }

    fn warn(&self, code: &str, summary: &str) {
        self.state
            .publish_error(AppError::new(ErrorCategory::General, code, summary));
    }

    pub(super) fn input_device(&self) -> Result<AudioDeviceChoice, AppError> {
        device_choice(self.lock().input_device.as_deref())
    }

    /// Provider configurations in the snapshot carry no API key.
    pub(super) fn recording_settings(&self) -> Result<RecordingSettings, AppError> {
        let settings = self.load();
        let input_device = device_choice(settings.input_device.as_deref())?;
        let cleanup = cleanup_plan(&settings);
        let model_id = settings.model_id.clone();
        Ok(recording(settings, model_id, input_device, cleanup))
    }

    /// The snapshot a first-run test dictation records with: the model and microphone setup
    /// has chosen so far, or the saved ones once setup has finished. It never runs AI cleanup,
    /// so the test never reaches a provider.
    pub(super) fn setup_test_settings(&self) -> Result<RecordingSettings, AppError> {
        let settings = self.load();
        let (model_id, selected_device) = if settings.first_run_complete {
            (settings.model_id.clone(), settings.input_device.as_deref())
        } else if let Some(checkpoint) = settings.setup.as_ref() {
            (
                checkpoint.model_id.clone(),
                checkpoint.input_device.as_deref(),
            )
        } else {
            (ModelId::default(), None)
        };
        let input_device = device_choice(selected_device)?;
        Ok(recording(settings, model_id, input_device, None))
    }

    pub(crate) fn load(&self) -> Settings {
        self.lock().clone()
    }

    pub(crate) fn get(&self) -> SettingsView {
        SettingsView::new(&self.load())
    }

    pub(crate) fn save_window_geometry(&self, geometry: WindowGeometry) -> Result<(), AppError> {
        self.edit(|settings| {
            settings.window_geometry = Some(geometry);
            Ok(())
        })
    }

    /// Saves the settings the window edited, keeping what it does not edit: the providers
    /// with API keys, the model, setup's progress and the window geometry.
    /// The edit must not break a rule the saved settings kept, and a newly chosen microphone
    /// must be connected. Returns the settings before the edit with the saved view.
    pub(super) fn update(
        &self,
        edited: Settings,
        audio: &dyn AudioInput,
    ) -> Result<(Settings, SettingsView), AppError> {
        let choice = device_choice(edited.input_device.as_deref())?;
        self.edit(|settings| {
            let before = std::mem::replace(settings, edited);
            settings.model_id.clone_from(&before.model_id);
            settings.first_run_complete = before.first_run_complete;
            settings.setup.clone_from(&before.setup);
            settings.window_geometry = before.window_geometry;
            settings.api_keys.clone_from(&before.api_keys);
            settings
                .validate_update(&before)
                .map_err(|error| settings_error(&error))?;
            if settings.input_device != before.input_device {
                require_connected_device(audio, &choice)?;
            }
            Ok((before, SettingsView::new(settings)))
        })
    }

    pub(crate) fn set_api_key(&self, request: &SetApiKeyRequest) -> Result<SettingsView, AppError> {
        self.persistence
            .store_api_key(request.provider, &request.key)
            .map_err(|_| {
                AppError::new(
                    ErrorCategory::General,
                    "credential_store_unavailable",
                    "The OS credential store is unavailable, so the API key was not saved",
                )
            })?;
        self.edit(|settings| {
            settings.api_keys.insert(request.provider);
            Ok(SettingsView::new(settings))
        })
    }

    pub(crate) fn clear_api_key(&self, provider: LlmProvider) -> Result<SettingsView, AppError> {
        self.persistence.delete_api_key(provider).map_err(|_| {
            AppError::new(
                ErrorCategory::General,
                "credential_store_unavailable",
                "The OS credential store is unavailable, so the API key was not removed",
            )
        })?;
        self.edit(|settings| {
            settings.api_keys.remove(&provider);
            Ok(SettingsView::new(settings))
        })
    }

    pub(super) fn save_setup_checkpoint(
        &self,
        checkpoint: SetupCheckpoint,
        model_supported: bool,
    ) -> Result<SettingsView, AppError> {
        self.edit(|settings| {
            if settings.first_run_complete {
                return Err(setup_already_complete());
            }
            if !model_supported {
                return Err(unknown_model());
            }
            device_choice(checkpoint.input_device.as_deref())?;
            settings.setup = Some(checkpoint);
            Ok(SettingsView::new(settings))
        })
    }

    /// Starts first-run setup again from its first step. The checkpoint begins
    /// with the saved choices, and the saved settings stay in effect until setup
    /// finishes again.
    pub(crate) fn restart_setup(&self) -> Result<SettingsView, AppError> {
        self.edit(|settings| {
            if !settings.first_run_complete {
                return Err(AppError::new(
                    ErrorCategory::General,
                    "setup_incomplete",
                    "Finish the remaining setup steps first",
                ));
            }
            self.state.lock().phase.require_idle()?;
            let checkpoint = SetupCheckpoint {
                step: SetupStep::Welcome,
                model_id: settings.model_id.clone(),
                input_device: settings.input_device.clone(),
                recording_shortcut: settings.recording_shortcut.clone(),
                recording_mode: settings.recording_mode,
            };
            settings.setup = Some(checkpoint);
            settings.first_run_complete = false;
            Ok(SettingsView::new(settings))
        })
    }
}

/// A settings rule an edit broke, worded for the control that made it.
pub(crate) fn settings_error(error: &SettingsError) -> AppError {
    AppError::new(
        ErrorCategory::General,
        "invalid_settings",
        error.to_string(),
    )
}

/// What a dictation records with: `settings`' vocabulary and history rules, with the given
/// model, microphone and cleanup.
fn recording(
    settings: Settings,
    model_id: ModelId,
    input_device: AudioDeviceChoice,
    cleanup: Option<CleanupPlan>,
) -> RecordingSettings {
    RecordingSettings {
        input_device,
        dictation: DictationSettings {
            model_id,
            vocabulary: settings.vocabulary_replacements,
            cleanup,
            history_limit: settings.history_limit,
            history_retention: settings.history_retention,
            keep_audio: settings.keep_audio,
        },
    }
}

/// What capture opens for the saved input device: the system default when none is chosen.
pub(super) fn device_choice(selected: Option<&str>) -> Result<AudioDeviceChoice, AppError> {
    match selected {
        None => Ok(AudioDeviceChoice::SystemDefault),
        Some(id) => AudioDeviceId::new(id)
            .map(AudioDeviceChoice::Device)
            .ok_or_else(|| {
                AppError::new(
                    ErrorCategory::Audio,
                    "audio_device_invalid",
                    "Choose an available microphone or the system default",
                )
            }),
    }
}

/// Refuses a chosen input device that is not connected; the system default always is.
pub(super) fn require_connected_device(
    audio: &dyn AudioInput,
    choice: &AudioDeviceChoice,
) -> Result<(), AppError> {
    let AudioDeviceChoice::Device(id) = choice else {
        return Ok(());
    };
    let devices = audio
        .list_input_devices()
        .map_err(|_| audio_devices_unavailable())?;
    if devices.iter().any(|device| device.id == id.as_str()) {
        Ok(())
    } else {
        Err(AppError::new(
            ErrorCategory::Audio,
            "audio_device_missing",
            "Choose an available microphone",
        ))
    }
}

fn cleanup_plan(settings: &Settings) -> Option<CleanupPlan> {
    settings.active_cleanup_target().map(|target| CleanupPlan {
        key_saved: settings.api_keys.contains(&settings.llm_provider),
        target,
    })
}

#[cfg(test)]
mod tests {
    use std::{fs, sync::Arc};

    use whispernow_core::audio::{AudioDeviceChoice, AudioDeviceId};
    use whispernow_domain::ipc::{AudioDevice, SetApiKeyRequest};
    use whispernow_domain::{
        ApiKey, AppError, ErrorCategory, LlmProvider, SetupCheckpoint, ThemePreference,
        VocabularyReplacement,
    };
    use whispernow_platform::SettingsStore;

    use whispernow_domain::WindowGeometry;

    use super::SettingsCoordinator;
    use crate::services::test_support::{FakeAudioInput, SettingsHarness};

    fn fixture() -> SettingsHarness {
        SettingsHarness::new(whispernow_domain::Settings::default())
    }

    impl SettingsHarness {
        fn restart(&mut self) {
            let loaded = self.persistence.load().expect("startup load");
            self.settings = Arc::new(SettingsCoordinator::new(
                Arc::clone(&self.persistence),
                Arc::clone(&self.state),
                loaded.settings.clone(),
            ));
            self.settings.report_loaded(&loaded.settings, loaded.notice);
        }

        fn warning_code(&self) -> Option<String> {
            let state = self.state.lock();
            state.last_error.as_ref().map(|error| error.code.clone())
        }

        fn set_key(&self, secret: &str) -> Result<(), whispernow_domain::AppError> {
            self.settings
                .set_api_key(&SetApiKeyRequest {
                    provider: LlmProvider::OpenAi,
                    key: ApiKey::new(secret).expect("a key"),
                })
                .map(|_| ())
        }

        fn stored_key(&self) -> Option<String> {
            self.settings
                .api_key(LlmProvider::OpenAi)
                .expect("readable key")
                .map(|key| key.expose_secret().to_owned())
        }
    }

    fn geometry() -> WindowGeometry {
        WindowGeometry {
            x: 10,
            y: 20,
            width: 640,
            height: 480,
        }
    }

    #[test]
    fn recording_settings_does_not_wait_for_a_keyring_write() {
        use std::sync::{Mutex, mpsc};
        use std::time::Duration;
        use whispernow_platform::{CredentialStore, CredentialStoreUnavailable};

        struct BlockingStore {
            entered: mpsc::Sender<()>,
            release: Mutex<mpsc::Receiver<()>>,
        }

        impl CredentialStore for BlockingStore {
            fn read(&self, _: &str) -> Result<Option<ApiKey>, CredentialStoreUnavailable> {
                Ok(None)
            }

            fn write(&self, _: &str, _: &ApiKey) -> Result<(), CredentialStoreUnavailable> {
                self.entered.send(()).expect("write entered");
                self.release
                    .lock()
                    .expect("release lock")
                    .recv()
                    .expect("released");
                Ok(())
            }

            fn delete(&self, _: &str) -> Result<(), CredentialStoreUnavailable> {
                Ok(())
            }
        }

        let fixture = fixture();
        let (entered, waiting) = mpsc::channel();
        let (release, blocked) = mpsc::channel();
        let persistence = Arc::new(SettingsStore::with_store(
            fixture.paths.clone(),
            Arc::new(BlockingStore {
                entered,
                release: Mutex::new(blocked),
            }),
        ));
        let coordinator = SettingsCoordinator::new(
            persistence,
            Arc::clone(&fixture.state),
            fixture.settings.load(),
        );
        std::thread::scope(|scope| {
            let write = scope.spawn(|| {
                coordinator.set_api_key(&SetApiKeyRequest {
                    provider: LlmProvider::OpenAi,
                    key: ApiKey::new("blocked-key").expect("key"),
                })
            });
            waiting
                .recv_timeout(Duration::from_secs(5))
                .expect("keyring entered");
            let (ready, snapshot) = mpsc::channel();
            let coordinator = &coordinator;
            scope.spawn(move || {
                ready
                    .send(coordinator.recording_settings().is_ok())
                    .expect("snapshot");
            });
            let result = snapshot.recv_timeout(Duration::from_secs(5));
            release.send(()).expect("release keyring before asserting");
            assert!(result.expect("recording settings returned while keyring blocked"));
            write.join().expect("writer thread").expect("key saved");
        });
    }

    fn connected_microphone() -> FakeAudioInput {
        FakeAudioInput::listing(Ok(vec![AudioDevice {
            id: "connected microphone".to_owned(),
            display_name: "Connected microphone".to_owned(),
            is_default: true,
        }]))
    }

    #[test]
    fn malformed_explicit_microphone_updates_and_checkpoints_are_not_saved() {
        let fixture = fixture();
        let before = fixture.settings.get();
        let mut edited = before.settings.clone();
        edited.input_device = Some(" ".to_owned());
        assert_eq!(
            fixture
                .settings
                .update(edited, &connected_microphone())
                .expect_err("invalid explicit microphone")
                .code,
            "audio_device_invalid"
        );
        let checkpoint = SetupCheckpoint {
            input_device: Some(" ".to_owned()),
            ..SetupCheckpoint::default()
        };
        assert_eq!(
            fixture
                .settings
                .save_setup_checkpoint(checkpoint, true)
                .expect_err("invalid checkpoint microphone")
                .code,
            "audio_device_invalid"
        );
        assert_eq!(fixture.settings.get(), before);
        assert!(!fixture.paths.settings_file().exists());
    }

    #[test]
    fn connected_and_default_microphone_selections_persist_and_disconnected_ones_are_refused() {
        let fixture = fixture();
        let mut edited = fixture.settings.get().settings;
        edited.input_device = Some("connected microphone".to_owned());
        let (_, saved) = fixture
            .settings
            .update(edited, &connected_microphone())
            .expect("connected microphone");
        let choice = AudioDeviceChoice::Device(
            AudioDeviceId::new("connected microphone").expect("device id"),
        );
        assert_eq!(
            fixture
                .settings
                .recording_settings()
                .expect("snapshot")
                .input_device,
            choice
        );
        let before = fs::read(fixture.paths.settings_file()).expect("saved settings");
        let mut disconnected = saved.settings.clone();
        disconnected.input_device = Some("disconnected microphone".to_owned());
        assert_eq!(
            fixture
                .settings
                .update(disconnected, &connected_microphone())
                .expect_err("disconnected microphone")
                .code,
            "audio_device_missing"
        );
        assert_eq!(
            fs::read(fixture.paths.settings_file()).expect("unchanged settings"),
            before
        );
        let mut default = saved.settings;
        default.input_device = None;
        let (_, saved) = fixture
            .settings
            .update(default, &connected_microphone())
            .expect("system default");
        assert_eq!(saved.settings.input_device, None);
        assert_eq!(
            fixture
                .settings
                .recording_settings()
                .expect("snapshot")
                .input_device,
            AudioDeviceChoice::SystemDefault
        );
        assert_eq!(
            fixture.settings.input_device().expect("input device"),
            AudioDeviceChoice::SystemDefault
        );
    }

    #[test]
    fn malformed_saved_microphones_cannot_become_default_recording_snapshots() {
        for setup in [false, true] {
            let fixture = fixture();
            let mut saved = fixture.settings.get().settings;
            if setup {
                saved.setup = Some(SetupCheckpoint {
                    input_device: Some(" ".to_owned()),
                    ..SetupCheckpoint::default()
                });
            } else {
                saved.input_device = Some(" ".to_owned());
            }
            fixture
                .settings
                .edit(|settings| {
                    *settings = saved;
                    Ok(())
                })
                .expect("hand-edited settings");
            let snapshot = if setup {
                fixture.settings.setup_test_settings()
            } else {
                fixture.settings.recording_settings()
            };
            let Err(error) = snapshot else {
                panic!("invalid saved microphone accepted");
            };
            assert_eq!(error.code, "audio_device_invalid");
            if !setup {
                assert_eq!(
                    fixture
                        .settings
                        .input_device()
                        .expect_err("invalid saved microphone")
                        .code,
                    "audio_device_invalid"
                );
            }
        }
    }

    #[test]
    fn an_edit_is_saved_only_when_it_changes_the_settings() {
        let fixture = fixture();
        fixture.settings.edit(|_| Ok(())).expect("no change");
        assert!(!fixture.paths.settings_file().exists());

        fixture
            .settings
            .edit(|settings| {
                settings.theme = ThemePreference::Dark;
                Ok(())
            })
            .expect("change");
        assert!(fixture.paths.settings_file().exists());
        assert_eq!(fixture.settings.load().theme, ThemePreference::Dark);
    }

    #[test]
    fn a_refused_edit_leaves_the_settings_and_the_file_alone() {
        let fixture = fixture();
        let error = fixture
            .settings
            .edit(|settings| {
                settings.theme = ThemePreference::Dark;
                Err::<(), _>(AppError::new(ErrorCategory::General, "refused", "refused"))
            })
            .expect_err("refused");

        assert_eq!(error.code, "refused");
        assert_eq!(fixture.settings.load().theme, ThemePreference::System);
        assert!(!fixture.paths.settings_file().exists());
    }

    #[test]
    fn an_update_saves_the_edit_and_keeps_what_the_window_does_not_own() {
        let mut fixture = fixture();
        fixture
            .settings
            .save_window_geometry(geometry())
            .expect("geometry save");
        fixture.set_key("kept-key").expect("store key");
        let saved = fixture.settings.get().settings;
        let mut edited = saved.clone();
        edited.theme = ThemePreference::Dark;
        edited.first_run_complete = !saved.first_run_complete;
        edited.window_geometry = None;
        edited.api_keys.clear();

        let (before, view) = fixture
            .settings
            .update(edited, &whispernow_platform::CpalAudioInput::new())
            .expect("theme change");

        assert_eq!(before, saved);
        let mut expected = saved;
        expected.theme = ThemePreference::Dark;
        assert_eq!(view.settings, expected);
        fixture.restart();
        assert_eq!(fixture.settings.get().settings, expected);
        assert_eq!(fixture.stored_key().as_deref(), Some("kept-key"));
    }

    #[test]
    fn recording_settings_enable_cleanup_only_after_the_provider_is_repaired() {
        let fixture = fixture();
        let mut edited = fixture.settings.get().settings;
        edited.cleanup_enabled = true;
        edited.llm_provider = LlmProvider::Custom;
        edited
            .provider_settings_mut(LlmProvider::Custom)
            .model
            .clear();
        let audio = whispernow_platform::CpalAudioInput::new();
        let (_, view) = fixture
            .settings
            .update(edited, &audio)
            .expect("save provider");
        let recording = fixture.settings.recording_settings().expect("recording");
        assert!(recording.dictation.cleanup.is_none());

        let mut edited = view.settings;
        let provider = edited.provider_settings_mut(LlmProvider::Custom);
        provider.model = "local".to_owned();
        provider.api_base = Some(
            whispernow_domain::ProviderEndpoint::parse("http://localhost:8080/v1")
                .expect("endpoint"),
        );
        fixture
            .settings
            .update(edited, &audio)
            .expect("repair provider");
        let recording = fixture.settings.recording_settings().expect("recording");
        assert!(recording.dictation.cleanup.is_some());
    }

    #[test]
    fn saved_settings_that_break_a_rule_load_with_a_warning_and_other_edits_save() {
        let mut fixture = fixture();
        fs::write(
            fixture.paths.settings_file(),
            r#"{"settings_version": 2, "vocabulary_replacements": [[" ", "x"]]}"#,
        )
        .expect("hand-edited settings");

        fixture.restart();
        let saved = fixture.settings.get().settings;
        assert_eq!(saved.vocabulary_replacements[0].source, " ");
        assert_eq!(
            fixture.warning_code().as_deref(),
            Some("settings_break_a_rule")
        );

        let saved_rules = saved.vocabulary_replacements.clone();
        let mut edited = saved;
        edited.theme = ThemePreference::Dark;
        fixture
            .settings
            .update(edited, &whispernow_platform::CpalAudioInput::new())
            .expect("an edit that leaves the broken rule alone saves");
        let reloaded = fixture.settings.get().settings;
        assert_eq!(
            (reloaded.theme, reloaded.vocabulary_replacements),
            (ThemePreference::Dark, saved_rules)
        );
    }

    #[test]
    fn settings_reads_ignore_disk_changes_until_restart() {
        let fixture = fixture();
        let before = fixture.settings.load();
        let malformed = "{ unreadable hand edit";
        fs::write(fixture.paths.settings_file(), malformed).expect("hand edit");

        assert_eq!(fixture.settings.get().settings, before);
        assert!(!fixture.settings.load().first_run_complete);
        assert!(fixture.settings.recording_settings().is_ok());
        assert!(fixture.settings.setup_test_settings().is_ok());
        assert_eq!(
            fs::read_to_string(fixture.paths.settings_file()).expect("original file"),
            malformed
        );
        assert!(
            fixture
                .paths
                .settings_backup_files()
                .expect("backups")
                .is_empty()
        );
        assert_eq!(fixture.warning_code(), None);
    }

    #[test]
    fn a_failed_save_keeps_the_last_settings_in_memory() {
        let fixture = fixture();
        let before = fixture.settings.load();
        fs::create_dir(fixture.paths.settings_file()).expect("block settings replacement");
        let mut edited = before.clone();
        edited.theme = ThemePreference::Dark;

        let error = fixture
            .settings
            .update(edited, &connected_microphone())
            .expect_err("settings write refused");
        assert_eq!(error.code, "settings_write");
        assert_eq!(fixture.settings.load(), before);
    }

    #[test]
    fn an_update_that_breaks_a_rule_is_refused_and_nothing_is_saved() {
        let fixture = fixture();
        let mut edited = fixture.settings.get().settings;
        edited.vocabulary_replacements = vec![VocabularyReplacement::new(" ", "x")];

        let error = fixture
            .settings
            .update(edited, &whispernow_platform::CpalAudioInput::new())
            .expect_err("a replacement needs text to replace");

        assert_eq!(error.code, "invalid_settings");
        assert_eq!(error.category, ErrorCategory::General);
        assert!(!fixture.paths.settings_file().exists());
    }

    #[test]
    fn an_api_key_is_kept_only_in_the_store_and_can_be_cleared() {
        let fixture = fixture();
        let secret = ["kept", "-in-", "store"].concat();
        fixture.set_key(&secret).expect("store key");

        let saved = fs::read_to_string(fixture.paths.settings_file()).expect("settings");
        assert!(!saved.contains(&secret));
        let view = fixture.settings.get();
        assert!(view.settings.api_keys.contains(&LlmProvider::OpenAi));
        assert_eq!(fixture.stored_key(), Some(secret));

        fixture
            .settings
            .clear_api_key(LlmProvider::OpenAi)
            .expect("clear key");
        assert_eq!(fixture.stored_key(), None);
        assert!(
            !fixture
                .settings
                .get()
                .settings
                .api_keys
                .contains(&LlmProvider::OpenAi)
        );
    }

    #[test]
    fn an_unavailable_store_refuses_key_changes_and_keeps_settings() {
        let fixture = fixture();
        fixture.set_key("existing").expect("store key");
        let before = fs::read(fixture.paths.settings_file()).expect("settings");
        fixture.credentials.set_available(false);

        let error = fixture.set_key("fresh").expect_err("store unavailable");
        assert_eq!(error.code, "credential_store_unavailable");
        let error = fixture
            .settings
            .clear_api_key(LlmProvider::OpenAi)
            .expect_err("clearing needs the store");
        assert_eq!(error.code, "credential_store_unavailable");

        assert_eq!(
            fs::read(fixture.paths.settings_file()).expect("settings"),
            before
        );
        assert!(!fixture.settings.delete_api_keys());
    }

    #[test]
    fn unreadable_settings_are_reported_once_with_where_they_went() {
        let mut fixture = fixture();
        let truncated = r#"{"settings_version": 2, "vocabulary_replacements": [["teh", "#;
        fs::write(fixture.paths.settings_file(), truncated).expect("truncated settings");

        fixture.restart();
        let moved = fixture.paths.settings_backup_files().expect("list");
        let warning = fixture
            .state
            .lock()
            .last_error
            .clone()
            .expect("settings warning");
        assert_eq!(warning.code, "settings_unreadable");
        assert!(warning.summary.contains(&moved[0].display().to_string()));

        fixture.state.lock().last_error = None;
        fixture.settings.get();
        assert_eq!(fixture.warning_code(), None);
    }
}
