use std::{
    sync::{Arc, Mutex, MutexGuard},
    time::Duration,
};

use whispernow_api::{
    ApiErrorDto, ClearApiKeyRequest, ErrorCategoryDto, LlmProvider, SetApiKeyRequest,
    SettingsError, SettingsView, SetupStateView,
};
use whispernow_core::{
    audio::{AudioDeviceChoice, AudioDeviceId, AudioInput},
    llm::{LlmProviderConfig, PromptKind},
};
use whispernow_domain::{ModelId, Settings, SetupCheckpoint, SetupStep, Shortcut, WindowGeometry};
use whispernow_platform::{ModelStore, SettingsNotice, SettingsStore};

use super::{
    dictation::{CleanupPlan, DictationSettings, RecordingSettings},
    lock,
    models::model_list_error,
    state::StateCell,
};

/// A caller-held settings transaction. The guard preserves the
/// model-transition -> settings -> state lock order.
pub(super) struct SettingsTransaction<'a> {
    coordinator: &'a SettingsCoordinator,
    _mutation: MutexGuard<'a, ()>,
}

impl SettingsTransaction<'_> {
    /// The saved settings. A file that could not be read is reported once here.
    pub(super) fn load(&self) -> Result<Settings, ApiErrorDto> {
        let loaded = self.coordinator.persistence.load().map_err(|error| {
            ApiErrorDto::new(ErrorCategoryDto::General, &error.code, &error.summary)
        });
        self.coordinator.report_notice();
        loaded
    }

    pub(super) fn save(&self, settings: &Settings) -> Result<(), ApiErrorDto> {
        self.coordinator
            .persistence
            .save(settings)
            .map_err(|error| {
                ApiErrorDto::new(ErrorCategoryDto::General, &error.code, &error.summary)
            })?;
        self.coordinator.follow_overlay(settings);
        Ok(())
    }
}

/// The recording shortcuts a settings save changes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PersistedShortcuts {
    shortcut: Shortcut,
    hold_shortcut_enabled: bool,
    toggle_shortcut: Option<Shortcut>,
}

impl PersistedShortcuts {
    pub(crate) fn of(settings: &Settings) -> Self {
        Self {
            shortcut: settings.hold_shortcut.clone(),
            hold_shortcut_enabled: settings.hold_shortcut_enabled,
            toggle_shortcut: settings.toggle_shortcut.clone(),
        }
    }

    pub(crate) fn apply_to(&self, settings: &mut Settings) {
        settings.hold_shortcut.clone_from(&self.shortcut);
        settings.hold_shortcut_enabled = self.hold_shortcut_enabled;
        settings.toggle_shortcut.clone_from(&self.toggle_shortcut);
    }
}

/// Owns settings persistence and serializes its writers.
pub(crate) struct SettingsCoordinator {
    persistence: Arc<SettingsStore>,
    state: Arc<StateCell>,
    mutation: Mutex<()>,
}

impl SettingsCoordinator {
    pub(super) fn new(persistence: Arc<SettingsStore>, state: Arc<StateCell>) -> Self {
        Self {
            persistence,
            state,
            mutation: Mutex::new(()),
        }
    }

    pub(super) fn transaction(&self) -> SettingsTransaction<'_> {
        SettingsTransaction {
            coordinator: self,
            _mutation: lock(&self.mutation),
        }
    }

    /// Where an LLM call reads its provider's API key when it is prepared, so settings
    /// snapshots and recording start never wait on the OS credential store.
    pub(super) fn api_keys(&self) -> Arc<SettingsStore> {
        Arc::clone(&self.persistence)
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
    pub(super) fn report_notice(&self) {
        let (code, summary) = match self.persistence.take_notice() {
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

    /// The overlay follows the saved style and position through the state snapshot. A save
    /// that leaves them unchanged publishes nothing.
    fn follow_overlay(&self, settings: &Settings) {
        let chosen = (settings.overlay_style, settings.overlay_position);
        let current = {
            let state = self.state.lock();
            (state.overlay_style, state.overlay_position)
        };
        if current != chosen {
            self.state.apply(|state| {
                (state.overlay_style, state.overlay_position) = chosen;
            });
        }
    }

    /// Publishes what happened to the settings file as it loaded, and warns when the saved
    /// settings break a rule, such as after a hand edit. They stay in effect, and an edit that
    /// leaves that rule alone still saves.
    pub(super) fn report_loaded(&self, saved: &Settings) {
        self.report_notice();
        if let Err(error) = saved.validate() {
            self.warn(
                "settings_break_a_rule",
                &format!("A saved setting needs attention: {error}"),
            );
        }
    }

    fn warn(&self, code: &str, summary: &str) {
        self.state
            .publish_error(ApiErrorDto::new(ErrorCategoryDto::General, code, summary));
    }

    /// Provider configurations in the snapshot carry no API key: the key is
    /// read when the LLM call is prepared, so starting a recording never waits
    /// on the OS credential store.
    pub(super) fn recording_settings(&self) -> Result<RecordingSettings, ApiErrorDto> {
        let settings = self.transaction().load()?;
        let cleanup = cleanup_plan(&settings);
        let (model_id, input_device) = (settings.model_id.clone(), settings.input_device.clone());
        Ok(recording(settings, model_id, input_device, cleanup))
    }

    /// The snapshot a first-run test dictation records with: the model and microphone setup
    /// has chosen so far, or the saved ones once setup has finished. It never runs AI cleanup,
    /// so the test never reaches a provider.
    pub(super) fn setup_test_settings(&self) -> Result<RecordingSettings, ApiErrorDto> {
        let settings = self.transaction().load()?;
        let (model_id, input_device) = if settings.first_run_complete {
            (settings.model_id.clone(), settings.input_device.clone())
        } else {
            let checkpoint = settings.setup.clone().unwrap_or_default();
            (checkpoint.model_id, checkpoint.input_device)
        };
        Ok(recording(settings, model_id, input_device, None))
    }

    /// The saved prompt `id` with the saved provider and LLM timeout, for a test run on
    /// sample text. The provider configuration carries no API key, like a recording's.
    pub(super) fn prompt_test_plan(
        &self,
        id: &str,
    ) -> Result<(CleanupPlan, Duration), ApiErrorDto> {
        let settings = self.transaction().load()?;
        let plan = prompt_plan(&settings, id).ok_or_else(|| {
            ApiErrorDto::new(
                ErrorCategoryDto::NotFound,
                "prompt_missing",
                "This prompt is no longer saved",
            )
        })?;
        Ok((
            plan,
            Duration::from_secs(u64::from(settings.llm_timeout_seconds)),
        ))
    }

    pub(crate) fn get(&self) -> Result<SettingsView, ApiErrorDto> {
        Ok(SettingsView::new(&self.transaction().load()?))
    }

    pub(crate) fn save_window_geometry(&self, geometry: WindowGeometry) -> Result<(), ApiErrorDto> {
        let transaction = self.transaction();
        let mut settings = transaction.load()?;
        if settings.window_geometry == Some(geometry) {
            return Ok(());
        }
        settings.window_geometry = Some(geometry);
        transaction.save(&settings)
    }

    /// Saves the settings the window edited, keeping what it does not edit: the providers
    /// with API keys, the model, setup's progress and the window geometry.
    /// The edit must not break a rule the saved settings kept, and a newly chosen microphone
    /// must be connected. Returns the settings before the edit with the saved view.
    pub(super) fn update(
        &self,
        mut edited: Settings,
        audio: &dyn AudioInput,
    ) -> Result<(Settings, SettingsView), ApiErrorDto> {
        let transaction = self.transaction();
        let before = transaction.load()?;
        edited.model_id.clone_from(&before.model_id);
        edited.first_run_complete = before.first_run_complete;
        edited.setup.clone_from(&before.setup);
        edited.window_geometry = before.window_geometry;
        edited.api_keys.clone_from(&before.api_keys);
        edited
            .validate_update(&before)
            .map_err(|error| settings_error(&error))?;
        if edited.input_device != before.input_device {
            require_connected_device(audio, edited.input_device.as_deref())?;
        }
        transaction.save(&edited)?;
        self.state.publish();
        Ok((before, SettingsView::new(&edited)))
    }

    pub(crate) fn set_api_key(
        &self,
        request: &SetApiKeyRequest,
    ) -> Result<SettingsView, ApiErrorDto> {
        let transaction = self.transaction();
        let mut settings = transaction.load()?;
        self.persistence
            .store_api_key(request.provider, &request.key)
            .map_err(|_| {
                ApiErrorDto::new(
                    ErrorCategoryDto::General,
                    "credential_store_unavailable",
                    "The OS credential store is unavailable, so the API key was not saved",
                )
            })?;
        settings.api_keys.insert(request.provider);
        transaction.save(&settings)?;
        self.state.publish();
        Ok(SettingsView::new(&settings))
    }

    pub(crate) fn clear_api_key(
        &self,
        request: ClearApiKeyRequest,
    ) -> Result<SettingsView, ApiErrorDto> {
        let transaction = self.transaction();
        let mut settings = transaction.load()?;
        self.persistence
            .delete_api_key(request.provider)
            .map_err(|_| {
                ApiErrorDto::new(
                    ErrorCategoryDto::General,
                    "credential_store_unavailable",
                    "The OS credential store is unavailable, so the API key was not removed",
                )
            })?;
        settings.api_keys.remove(&request.provider);
        transaction.save(&settings)?;
        self.state.publish();
        Ok(SettingsView::new(&settings))
    }

    pub(crate) fn get_setup_state(&self) -> Result<SetupStateView, ApiErrorDto> {
        let settings = self.transaction().load()?;
        Ok(SetupStateView {
            first_run_complete: settings.first_run_complete,
            checkpoint: settings.setup.unwrap_or_default(),
        })
    }

    pub(crate) fn save_setup_checkpoint(
        &self,
        checkpoint: SetupCheckpoint,
        models: &ModelStore,
    ) -> Result<SetupStateView, ApiErrorDto> {
        let transaction = self.transaction();
        if self.state.lock().setup_complete {
            return Err(ApiErrorDto::new(
                ErrorCategoryDto::General,
                "setup_already_complete",
                "Setup has already finished",
            ));
        }
        if !models
            .list()
            .map_err(model_list_error)?
            .iter()
            .any(|entry| entry.definition.id == checkpoint.model_id)
        {
            return Err(ApiErrorDto::new(
                ErrorCategoryDto::Model,
                "unknown_model",
                "The model is not supported",
            ));
        }
        let mut settings = transaction.load()?;
        settings.setup = Some(checkpoint.clone());
        transaction.save(&settings)?;
        self.state.publish();
        Ok(SetupStateView {
            first_run_complete: settings.first_run_complete,
            checkpoint,
        })
    }

    /// Starts first-run setup again from its first step. The checkpoint begins
    /// with the saved choices, and the saved settings stay in effect until setup
    /// finishes again.
    pub(crate) fn restart_setup(&self) -> Result<SetupStateView, ApiErrorDto> {
        {
            let state = self.state.lock();
            if !state.setup_complete {
                return Err(ApiErrorDto::new(
                    ErrorCategoryDto::General,
                    "setup_incomplete",
                    "Finish the remaining setup steps first",
                ));
            }
            state.phase.require_idle()?;
        }
        let transaction = self.transaction();
        let mut settings = transaction.load()?;
        let checkpoint = SetupCheckpoint {
            step: SetupStep::Welcome,
            model_id: settings.model_id.clone(),
            input_device: settings.input_device.clone(),
            hold_shortcut: settings.hold_shortcut.clone(),
            hold_shortcut_enabled: settings.hold_shortcut_enabled,
            toggle_shortcut: settings.toggle_shortcut.clone(),
        };
        settings.setup = Some(checkpoint.clone());
        settings.first_run_complete = false;
        transaction.save(&settings)?;
        self.state.apply(|state| state.setup_complete = false);
        Ok(SetupStateView {
            first_run_complete: false,
            checkpoint,
        })
    }
}

/// A settings rule an edit broke, worded for the control that made it.
pub(crate) fn settings_error(error: &SettingsError) -> ApiErrorDto {
    ApiErrorDto::new(
        ErrorCategoryDto::General,
        "invalid_settings",
        &error.to_string(),
    )
}

/// What a dictation records with: `settings`' vocabulary and LLM timeout, with the given
/// model, microphone and cleanup.
fn recording(
    settings: Settings,
    model_id: ModelId,
    input_device: Option<String>,
    cleanup: Option<CleanupPlan>,
) -> RecordingSettings {
    RecordingSettings {
        input_device,
        dictation: DictationSettings {
            model_id,
            vocabulary: settings.vocabulary_replacements,
            cleanup,
            llm_timeout: Duration::from_secs(u64::from(settings.llm_timeout_seconds)),
            history_limit: settings.history_limit,
            keep_audio: settings.keep_audio,
        },
    }
}

/// What capture opens for the saved input device: the system default when none is chosen.
pub(super) fn device_choice(selected: Option<&str>) -> AudioDeviceChoice {
    selected
        .and_then(|id| AudioDeviceId::new(id).ok())
        .map_or(AudioDeviceChoice::SystemDefault, AudioDeviceChoice::Device)
}

/// Refuses a chosen input device that is not connected; the system default always is.
pub(super) fn require_connected_device(
    audio: &dyn AudioInput,
    selected: Option<&str>,
) -> Result<(), ApiErrorDto> {
    let choice = device_choice(selected);
    if choice == AudioDeviceChoice::SystemDefault {
        return Ok(());
    }
    let devices = audio.list_input_devices().map_err(|_| {
        ApiErrorDto::new(
            ErrorCategoryDto::Audio,
            "audio_devices",
            "Audio input devices could not be listed",
        )
    })?;
    if devices.iter().any(|device| device.choice == choice) {
        Ok(())
    } else {
        Err(ApiErrorDto::new(
            ErrorCategoryDto::Audio,
            "audio_device_missing",
            "Choose an available microphone",
        ))
    }
}

fn cleanup_plan(settings: &Settings) -> Option<CleanupPlan> {
    prompt_plan(settings, settings.active_prompt_id.as_deref()?)
}

fn prompt_plan(settings: &Settings, id: &str) -> Option<CleanupPlan> {
    let prompt = settings
        .prompts
        .iter()
        .find(|prompt| prompt.id == id)?
        .clone();
    let config = provider_config(settings);
    let mode = if prompt.rewrites() {
        PromptKind::Rewrite
    } else {
        PromptKind::Cleanup
    };
    Some(CleanupPlan {
        prompt,
        key_saved: settings.api_keys.contains(&settings.llm_provider),
        config,
        mode,
        vocabulary: settings.vocabulary_replacements.clone(),
    })
}

/// The chosen provider, or `None` until it has a model and, for a custom provider, an
/// endpoint. Settings validation requires both while AI cleanup is on.
fn provider_config(settings: &Settings) -> Option<LlmProviderConfig> {
    let provider = settings.llm_provider;
    if provider == LlmProvider::LocalCleanup {
        // It runs in process: its model is a cleanup model's id, and an endpoint is ignored.
        return Some(LlmProviderConfig {
            provider,
            model: settings.cleanup_model_id().to_owned(),
            api_key: None,
            endpoint: None,
        });
    }
    let provider_settings = settings.provider_settings(provider);
    let model = provider.request_model(&provider_settings.model);
    let endpoint = provider_settings.api_base;
    if model.is_empty() || (provider == LlmProvider::Custom && endpoint.is_none()) {
        return None;
    }
    Some(LlmProviderConfig {
        provider,
        model: model.to_owned(),
        api_key: None,
        endpoint,
    })
}

#[cfg(test)]
mod tests {
    use std::{fs, sync::Arc};

    use whispernow_api::{
        ApiKey, ClearApiKeyRequest, LlmProvider, SetApiKeyRequest, ThemePreference,
    };
    use whispernow_platform::{AppPaths, MemoryCredentialStore, SettingsStore};

    use whispernow_domain::WindowGeometry;

    use super::{SettingsCoordinator, StateCell};
    use crate::services::{Events, state::State};

    struct Fixture {
        _root: tempfile::TempDir,
        paths: AppPaths,
        store: Arc<MemoryCredentialStore>,
        state: Arc<StateCell>,
        settings: SettingsCoordinator,
    }

    impl Fixture {
        fn start() -> Self {
            let root = tempfile::tempdir().expect("temporary directory");
            let paths =
                AppPaths::from_base_dirs(root.path().join("config"), root.path().join("data"));
            fs::create_dir_all(paths.config_dir()).expect("config directory");
            let store = Arc::new(MemoryCredentialStore::default());
            let persistence = Arc::new(SettingsStore::with_store(
                paths.clone(),
                Arc::clone(&store) as _,
            ));
            let state = Arc::new(StateCell::new(
                State::new(whispernow_domain::ModelId::default(), true),
                Events::channel().0,
            ));
            let settings = SettingsCoordinator::new(persistence, Arc::clone(&state));
            Self {
                _root: root,
                paths,
                store,
                state,
                settings,
            }
        }

        fn warning_code(&self) -> Option<String> {
            let state = self.state.lock();
            state.last_error.as_ref().map(|error| error.code.clone())
        }

        fn set_key(&self, secret: &str) -> Result<(), whispernow_api::ApiErrorDto> {
            self.settings
                .set_api_key(&SetApiKeyRequest {
                    provider: LlmProvider::OpenAi,
                    key: ApiKey::new(secret).expect("a key"),
                })
                .map(|_| ())
        }

        fn stored_key(&self) -> Option<String> {
            self.settings
                .api_keys()
                .api_key(LlmProvider::OpenAi)
                .expect("readable key")
                .map(|key| key.expose_secret().to_owned())
        }
    }

    #[test]
    fn the_local_cleanup_provider_names_the_chosen_cleanup_model() {
        use whispernow_domain::{DEFAULT_CLEANUP_MODEL_ID, MINICPM_CLEANUP_MODEL_ID, Settings};

        let mut settings = Settings {
            llm_provider: LlmProvider::LocalCleanup,
            ..Settings::default()
        };
        let model = |settings: &Settings| {
            super::provider_config(settings)
                .expect("the local cleanup provider always has a config")
                .model
        };
        assert_eq!(model(&settings), DEFAULT_CLEANUP_MODEL_ID);
        settings
            .provider_settings_mut(LlmProvider::LocalCleanup)
            .model = MINICPM_CLEANUP_MODEL_ID.to_owned();
        assert_eq!(model(&settings), MINICPM_CLEANUP_MODEL_ID);
    }

    #[test]
    fn a_long_model_name_still_configures_cleanup() {
        use whispernow_domain::Settings;

        let mut settings = Settings {
            llm_provider: LlmProvider::Ollama,
            ..Settings::default()
        };
        let long = "m".repeat(300);
        settings.provider_settings_mut(LlmProvider::Ollama).model = long.clone();
        assert_eq!(settings.validate(), Ok(()));
        let config = super::provider_config(&settings).expect("a valid name configures cleanup");
        assert_eq!(config.model, long);
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
    fn an_update_saves_the_edit_and_keeps_what_the_window_does_not_own() {
        let fixture = Fixture::start();
        fixture
            .settings
            .save_window_geometry(geometry())
            .expect("geometry save");
        fixture.set_key("kept-key").expect("store key");
        let saved = fixture.settings.get().expect("settings view").settings;
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
        assert_eq!(fixture.settings.get().expect("reload").settings, expected);
        assert_eq!(fixture.stored_key().as_deref(), Some("kept-key"));
    }

    #[test]
    fn saved_settings_that_break_a_rule_load_with_a_warning_and_other_edits_save() {
        let fixture = Fixture::start();
        fs::write(
            fixture.paths.settings_file(),
            r#"{"settings_version": 2, "llm_timeout_seconds": 1}"#,
        )
        .expect("hand-edited settings");

        let saved = fixture.settings.get().expect("settings load").settings;
        assert_eq!(saved.llm_timeout_seconds, 1);
        fixture.settings.report_loaded(&saved);
        assert_eq!(
            fixture.warning_code().as_deref(),
            Some("settings_break_a_rule")
        );

        let mut edited = saved;
        edited.theme = ThemePreference::Dark;
        fixture
            .settings
            .update(edited, &whispernow_platform::CpalAudioInput::new())
            .expect("an edit that leaves the broken rule alone saves");
        let reloaded = fixture.settings.get().expect("reload").settings;
        assert_eq!(
            (reloaded.theme, reloaded.llm_timeout_seconds),
            (ThemePreference::Dark, 1)
        );
    }

    #[test]
    fn an_update_that_breaks_a_rule_is_refused_and_nothing_is_saved() {
        let fixture = Fixture::start();
        let mut edited = fixture.settings.get().expect("settings view").settings;
        edited.llm_timeout_seconds = 1;

        let error = fixture
            .settings
            .update(edited, &whispernow_platform::CpalAudioInput::new())
            .expect_err("the time limit is out of range");

        assert_eq!(error.code, "invalid_settings");
        assert_eq!(
            error.summary,
            "Enter a time limit between 5 and 120 seconds"
        );
        assert!(!fixture.paths.settings_file().exists());
    }

    #[test]
    fn an_api_key_is_kept_only_in_the_store_and_can_be_cleared() {
        let fixture = Fixture::start();
        let secret = ["kept", "-in-", "store"].concat();
        fixture.set_key(&secret).expect("store key");

        let saved = fs::read_to_string(fixture.paths.settings_file()).expect("settings");
        assert!(!saved.contains(&secret));
        let view = fixture.settings.get().expect("settings view");
        assert!(view.settings.api_keys.contains(&LlmProvider::OpenAi));
        assert_eq!(fixture.stored_key(), Some(secret));

        fixture
            .settings
            .clear_api_key(ClearApiKeyRequest {
                provider: LlmProvider::OpenAi,
            })
            .expect("clear key");
        assert_eq!(fixture.stored_key(), None);
        assert!(
            !fixture
                .settings
                .get()
                .expect("settings view")
                .settings
                .api_keys
                .contains(&LlmProvider::OpenAi)
        );
    }

    #[test]
    fn an_unavailable_store_refuses_key_changes_and_keeps_settings() {
        let fixture = Fixture::start();
        fixture.set_key("existing").expect("store key");
        let before = fs::read(fixture.paths.settings_file()).expect("settings");
        fixture.store.set_available(false);

        let error = fixture.set_key("fresh").expect_err("store unavailable");
        assert_eq!(error.code, "credential_store_unavailable");
        let error = fixture
            .settings
            .clear_api_key(ClearApiKeyRequest {
                provider: LlmProvider::OpenAi,
            })
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
        let fixture = Fixture::start();
        let truncated = r#"{"settings_version": 2, "vocabulary_replacements": [["teh", "#;
        fs::write(fixture.paths.settings_file(), truncated).expect("truncated settings");

        fixture.settings.get().expect("defaults load");
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
        fixture.settings.get().expect("reload");
        assert_eq!(fixture.warning_code(), None);
    }
}
