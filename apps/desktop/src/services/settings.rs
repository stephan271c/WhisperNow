use std::sync::{Arc, Mutex, MutexGuard};
use whispernow_domain::{AppError, ErrorCategory};

use whispernow_core::audio::AudioInput;
use whispernow_domain::ipc::{SetApiKeyRequest, SettingsView};
use whispernow_domain::{ApiKey, AudioDeviceId, LlmProvider, RemoteProvider, SettingsError};
use whispernow_domain::{
    ModelId, SettingChange, Settings, SetupCheckpoint, SetupProgress, SetupStep, WindowGeometry,
};
use whispernow_platform::{SettingsNotice, SettingsStore};

use super::{
    audio_devices_unavailable,
    dictation::{CleanupPlan, DictationSettings, RecordingSettings},
    lock, provider_models_unavailable, setup_already_complete,
    state::StateCell,
};

/// Owns settings persistence and serializes its writers.
pub(crate) struct SavedSettings {
    persistence: Arc<SettingsStore>,
    state: Arc<StateCell>,
    current: Mutex<Settings>,
}

impl SavedSettings {
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

    pub(crate) fn provider_and_key(
        &self,
        provider: LlmProvider,
    ) -> Result<(RemoteProvider, Option<ApiKey>), AppError> {
        let settings = self.snapshot();
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

    /// Returns whether the store accepted every deletion. It stops at the first refusal, because
    /// an unavailable store typically refuses, or blocks on, every entry.
    pub(super) fn delete_api_keys(&self) -> bool {
        LlmProvider::ALL
            .into_iter()
            .all(|provider| self.persistence.delete_api_key(provider).is_ok())
    }

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

    /// Saved settings that break a rule, such as after a hand edit, stay in effect, and an edit
    /// that leaves that rule alone still saves.
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

    pub(super) fn input_device(&self) -> Option<AudioDeviceId> {
        self.lock().input_device.clone()
    }

    /// Provider configurations in the snapshot carry no API key.
    pub(super) fn recording_settings(&self) -> RecordingSettings {
        let settings = self.snapshot();
        let input_device = settings.input_device.clone();
        let cleanup = cleanup_plan(&settings);
        let model_id = settings.model_id.clone();
        recording(settings, model_id, input_device, cleanup)
    }

    /// The snapshot a first-run test dictation records with: the model and microphone setup
    /// has chosen so far, or the saved ones once setup has finished. It never runs AI cleanup,
    /// so the test never reaches a provider.
    pub(super) fn setup_test_settings(&self) -> RecordingSettings {
        let settings = self.snapshot();
        let (model_id, input_device) = match &settings.setup {
            SetupProgress::Complete => (settings.model_id.clone(), settings.input_device.clone()),
            SetupProgress::Unfinished(checkpoint) => {
                (checkpoint.model_id.clone(), checkpoint.input_device.clone())
            }
        };
        recording(settings, model_id, input_device, None)
    }

    pub(crate) fn snapshot(&self) -> Settings {
        self.lock().clone()
    }

    pub(crate) fn view(&self) -> SettingsView {
        SettingsView::new(&self.snapshot())
    }

    pub(crate) fn save_window_geometry(&self, geometry: WindowGeometry) -> Result<(), AppError> {
        self.edit(|settings| {
            settings.window_geometry = Some(geometry);
            Ok(())
        })
    }

    pub(super) fn update(
        &self,
        change: &SettingChange,
        audio: &dyn AudioInput,
    ) -> Result<SettingsView, AppError> {
        if let SettingChange::InputDevice(device) = &change {
            require_connected_device(audio, device.as_ref())?;
        }
        self.edit(|settings| {
            change.clone().apply(settings);
            change
                .check(settings)
                .map_err(|error| settings_error(&error))?;
            Ok(SettingsView::new(settings))
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
    ) -> Result<SettingsView, AppError> {
        self.edit(|settings| {
            if settings.setup.is_complete() {
                return Err(setup_already_complete());
            }
            settings.setup = SetupProgress::Unfinished(checkpoint);
            Ok(SettingsView::new(settings))
        })
    }

    /// The checkpoint begins with the saved choices, and the saved settings stay in effect until
    /// setup finishes again.
    pub(crate) fn restart_setup(&self) -> Result<SettingsView, AppError> {
        self.edit(|settings| {
            if !settings.setup.is_complete() {
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
            settings.setup = SetupProgress::Unfinished(checkpoint);
            Ok(SettingsView::new(settings))
        })
    }
}

pub(crate) fn settings_error(error: &SettingsError) -> AppError {
    AppError::new(
        ErrorCategory::General,
        "invalid_settings",
        error.to_string(),
    )
}

fn recording(
    settings: Settings,
    model_id: ModelId,
    input_device: Option<AudioDeviceId>,
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

/// The system default (`None`) is always connected.
pub(super) fn require_connected_device(
    audio: &dyn AudioInput,
    device: Option<&AudioDeviceId>,
) -> Result<(), AppError> {
    let Some(id) = device else {
        return Ok(());
    };
    let devices = audio
        .list_input_devices()
        .map_err(|error| audio_devices_unavailable(&error))?;
    if devices.iter().any(|device| device.id == *id) {
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

    use whispernow_domain::ipc::{AudioDevice, SetApiKeyRequest};
    use whispernow_domain::{
        ApiKey, AppError, AudioDeviceId, ErrorCategory, LlmProvider, SettingChange,
        ThemePreference, VocabularyReplacement,
    };
    use whispernow_platform::SettingsStore;

    use super::SavedSettings;
    use crate::services::test_support::{FakeAudioInput, SettingsHarness};

    fn fixture() -> SettingsHarness {
        SettingsHarness::new(whispernow_domain::Settings::default())
    }

    impl SettingsHarness {
        fn restart(&mut self) {
            let loaded = self.persistence.load().expect("startup load");
            self.settings = Arc::new(SavedSettings::new(
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
            fn read(&self, _: LlmProvider) -> Result<Option<ApiKey>, CredentialStoreUnavailable> {
                Ok(None)
            }

            fn write(&self, _: LlmProvider, _: &ApiKey) -> Result<(), CredentialStoreUnavailable> {
                self.entered.send(()).expect("write entered");
                self.release
                    .lock()
                    .expect("release lock")
                    .recv()
                    .expect("released");
                Ok(())
            }

            fn delete(&self, _: LlmProvider) -> Result<(), CredentialStoreUnavailable> {
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
        let settings = SavedSettings::new(
            persistence,
            Arc::clone(&fixture.state),
            fixture.settings.snapshot(),
        );
        std::thread::scope(|scope| {
            let write = scope.spawn(|| {
                settings.set_api_key(&SetApiKeyRequest {
                    provider: LlmProvider::OpenAi,
                    key: ApiKey::new("blocked-key").expect("key"),
                })
            });
            waiting
                .recv_timeout(Duration::from_secs(5))
                .expect("keyring entered");
            let (ready, snapshot) = mpsc::channel();
            let settings = &settings;
            scope.spawn(move || {
                settings.recording_settings();
                ready.send(()).expect("snapshot");
            });
            let result = snapshot.recv_timeout(Duration::from_secs(5));
            release.send(()).expect("release keyring before asserting");
            result.expect("recording settings returned while keyring blocked");
            write.join().expect("writer thread").expect("key saved");
        });
    }

    fn device(id: &str) -> AudioDeviceId {
        AudioDeviceId::new(id).expect("device id")
    }

    fn connected_microphone() -> FakeAudioInput {
        FakeAudioInput::listing(Ok(vec![AudioDevice {
            id: device("connected microphone"),
            display_name: "Connected microphone".to_owned(),
        }]))
    }

    #[test]
    fn only_a_microphone_change_checks_the_microphone() {
        let fixture = fixture();
        fixture
            .settings
            .edit(|settings| {
                settings.input_device = Some(device("unplugged microphone"));
                Ok(())
            })
            .expect("saved microphone");
        fixture
            .settings
            .update(
                &SettingChange::Theme(ThemePreference::Dark),
                &connected_microphone(),
            )
            .expect("a theme change ignores the unplugged microphone");
        let error = fixture
            .settings
            .update(
                &SettingChange::InputDevice(Some(device("unplugged microphone"))),
                &connected_microphone(),
            )
            .expect_err("choosing it checks it");
        assert_eq!(error.code, "audio_device_missing");
    }

    #[test]
    fn connected_and_default_microphone_selections_persist_and_disconnected_ones_are_refused() {
        let fixture = fixture();
        fixture
            .settings
            .update(
                &SettingChange::InputDevice(Some(device("connected microphone"))),
                &connected_microphone(),
            )
            .expect("connected microphone");
        assert_eq!(
            fixture.settings.recording_settings().input_device,
            Some(device("connected microphone"))
        );
        let before = fs::read(fixture.paths.settings_file()).expect("saved settings");
        assert_eq!(
            fixture
                .settings
                .update(
                    &SettingChange::InputDevice(Some(device("disconnected microphone"))),
                    &connected_microphone()
                )
                .expect_err("disconnected microphone")
                .code,
            "audio_device_missing"
        );
        assert_eq!(
            fs::read(fixture.paths.settings_file()).expect("unchanged settings"),
            before
        );
        let saved = fixture
            .settings
            .update(&SettingChange::InputDevice(None), &connected_microphone())
            .expect("system default");
        assert_eq!(saved.settings.input_device, None);
        assert_eq!(fixture.settings.recording_settings().input_device, None);
        assert_eq!(fixture.settings.input_device(), None);
    }

    #[test]
    fn a_blank_saved_microphone_reads_as_the_system_default() {
        let mut fixture = fixture();
        fs::write(
            fixture.paths.settings_file(),
            r#"{"settings_version": 2, "input_device": " ", "setup": {"unfinished": {"input_device": ""}}}"#,
        )
        .expect("hand-edited settings");

        fixture.restart();
        assert_eq!(fixture.settings.input_device(), None);
        assert_eq!(fixture.settings.setup_test_settings().input_device, None);
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
        assert_eq!(fixture.settings.snapshot().theme, ThemePreference::Dark);
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
        assert_eq!(fixture.settings.snapshot().theme, ThemePreference::System);
        assert!(!fixture.paths.settings_file().exists());
    }

    #[test]
    fn recording_settings_enable_cleanup_only_after_the_provider_is_repaired() {
        let fixture = fixture();
        let audio = whispernow_platform::CpalAudioInput::new();
        for change in [
            SettingChange::CleanupEnabled(true),
            SettingChange::ActiveProvider(LlmProvider::Custom),
        ] {
            fixture
                .settings
                .update(&change, &audio)
                .expect("save provider");
        }
        let recording = fixture.settings.recording_settings();
        assert!(recording.dictation.cleanup.is_none());

        for change in [
            SettingChange::ChooseProviderModel {
                provider: LlmProvider::Custom,
                model: "local".to_owned(),
            },
            SettingChange::ProviderEndpoint {
                provider: LlmProvider::Custom,
                api_base: Some(
                    whispernow_domain::ProviderEndpoint::parse("http://localhost:8080/v1")
                        .expect("endpoint"),
                ),
            },
        ] {
            fixture
                .settings
                .update(&change, &audio)
                .expect("repair provider");
        }
        let recording = fixture.settings.recording_settings();
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
        let saved = fixture.settings.view().settings;
        assert_eq!(saved.vocabulary_replacements[0].source, " ");
        assert_eq!(
            fixture.warning_code().as_deref(),
            Some("settings_break_a_rule")
        );

        let saved_rules = saved.vocabulary_replacements.clone();
        fixture
            .settings
            .update(
                &SettingChange::Theme(ThemePreference::Dark),
                &whispernow_platform::CpalAudioInput::new(),
            )
            .expect("an edit that leaves the broken rule alone saves");
        let reloaded = fixture.settings.view().settings;
        assert_eq!(
            (reloaded.theme, reloaded.vocabulary_replacements),
            (ThemePreference::Dark, saved_rules)
        );
    }

    #[test]
    fn settings_reads_ignore_disk_changes_until_restart() {
        let fixture = fixture();
        let before = fixture.settings.snapshot();
        let malformed = "{ unreadable hand edit";
        fs::write(fixture.paths.settings_file(), malformed).expect("hand edit");

        assert_eq!(fixture.settings.view().settings, before);
        assert!(!fixture.settings.snapshot().setup.is_complete());
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
        let before = fixture.settings.snapshot();
        fs::create_dir(fixture.paths.settings_file()).expect("block settings replacement");
        let error = fixture
            .settings
            .update(
                &SettingChange::Theme(ThemePreference::Dark),
                &connected_microphone(),
            )
            .expect_err("settings write refused");
        assert_eq!(error.code, "settings_write");
        assert_eq!(fixture.settings.snapshot(), before);
    }

    #[test]
    fn an_update_that_breaks_a_rule_is_refused_and_nothing_is_saved() {
        let fixture = fixture();
        let error = fixture
            .settings
            .update(
                &SettingChange::Vocabulary(vec![VocabularyReplacement::new(" ", "x")]),
                &whispernow_platform::CpalAudioInput::new(),
            )
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
        let view = fixture.settings.view();
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
                .view()
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

        fixture.state.apply(|state| state.last_error = None);
        fixture.settings.view();
        assert_eq!(fixture.warning_code(), None);
    }
}
