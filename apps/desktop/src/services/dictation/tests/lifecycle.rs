use super::*;
use crate::{
    activation::ActivationGate,
    autostart::{AutostartError, AutostartRegistration},
    gateway::Gateway,
    services::{
        NativeServices, models::ModelCoordinator, output::OutputCoordinator,
        permissions::MicrophoneAccess,
    },
    shortcuts::{
        ShortcutBinding, ShortcutCoordinator, ShortcutRegistrar, ShortcutRegistrationError,
        recording_binding,
    },
    tray::TrayController,
};
use whispernow_platform::LocalCleanupService;

struct Registrar;

impl ShortcutRegistrar for Registrar {
    fn register(&self, _: &ShortcutBinding) -> Result<(), ShortcutRegistrationError> {
        Ok(())
    }

    fn unregister(&self, _: &ShortcutBinding) -> Result<(), ShortcutRegistrationError> {
        Ok(())
    }
}

struct Autostart;

impl AutostartRegistration for Autostart {
    fn is_enabled(&self) -> Result<bool, AutostartError> {
        Ok(false)
    }

    fn set_enabled(&self, _: bool) -> Result<(), AutostartError> {
        Ok(())
    }
}

fn native_services(fixture: Fixture) -> (NativeServices, tempfile::TempDir) {
    fixture.flush();
    let paths = AppPaths::from_base_dirs(
        fixture.root.path().join("config"),
        fixture.root.path().join("data"),
    );
    let store = Arc::new(ModelStore::new(paths.clone()));
    let local_cleanup = Arc::new(LocalCleanupService::new(
        fixture.root.path().join("llama"),
        Arc::new(|_| None),
        Arc::new(|| None),
    ));
    let services = NativeServices {
        settings: fixture.settings,
        history: fixture.history,
        audio: Arc::new(FakeAudio { audio: Vec::new() }),
        models: Arc::new(ModelCoordinator::new(
            store,
            Arc::clone(&fixture.state),
            Arc::clone(&local_cleanup),
        )),
        state: fixture.state,
        dictation: fixture.dictation,
        paths: paths.clone(),
        output: Arc::new(OutputCoordinator::native(None)),
        paste_helper_setup_dir: paths.paste_helper_setup_dir(),
        microphone_access: MicrophoneAccess::default(),
        local_cleanup,
    };
    (services, fixture.root)
}

#[test]
fn a_failed_tray_pause_does_not_leave_a_settings_warning() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let (services, _root) = native_services(Fixture::new());
    let shortcuts = Arc::new(ShortcutCoordinator::new(
        Arc::new(Registrar),
        recording_binding(&Settings::default()),
        Arc::new(|| true),
    ));
    let gateway = Gateway::new(
        Arc::new(services),
        shortcuts,
        Arc::new(Autostart),
        Arc::new(ActivationGate::default()),
        Arc::new(TrayController::default()),
    );
    gateway
        .services
        .state
        .apply(|state| state.phase = Phase::Updating);

    crate::toggle_pause(&gateway);

    let error = gateway
        .services_state()
        .error
        .expect("published pause failure");
    assert_eq!(error.code, "update_installing");
    for _ in 0..2 {
        let settings = runtime
            .block_on(gateway.settings_view())
            .expect("read settings");
        assert!(
            settings
                .warnings
                .iter()
                .all(|warning| warning != &error.summary)
        );
    }
}

#[test]
fn setup_refuses_active_downloads_and_cleanup_models_before_loading() {
    use crate::services::state::DownloadTask;
    use whispernow_domain::ipc::ActiveDownload;
    use whispernow_domain::{DEFAULT_CLEANUP_MODEL_ID, DownloadPhase, SetupCheckpoint, SetupStep};

    for expected in ["model_transition_busy", "model_not_speech"] {
        let (services, _root) = native_services(Fixture::new());
        services.state.lock().setup_complete = false;
        let mut checkpoint = SetupCheckpoint {
            step: SetupStep::TryIt,
            ..SetupCheckpoint::default()
        };
        if expected == "model_transition_busy" {
            services.state.lock().active_downloads.insert(
                checkpoint.model_id.clone(),
                DownloadTask {
                    progress: ActiveDownload {
                        model_id: checkpoint.model_id.clone(),
                        phase: DownloadPhase::Downloading,
                        downloaded_bytes: 0,
                        total_bytes: None,
                        bytes_per_second: None,
                    },
                    cancellation: CancellationToken::default(),
                },
            );
        } else {
            checkpoint.model_id = ModelId::new(DEFAULT_CLEANUP_MODEL_ID).expect("cleanup model id");
            let registry = ModelRegistry::builtin();
            let definition = registry
                .get(DEFAULT_CLEANUP_MODEL_ID)
                .expect("cleanup model");
            for artifact in definition.layout.required_artifacts() {
                let path = services
                    .paths
                    .models_dir()
                    .join(DEFAULT_CLEANUP_MODEL_ID)
                    .join(artifact);
                fs::create_dir_all(path.parent().expect("artifact parent"))
                    .expect("model directory");
                fs::write(path, b"fixture").expect("model artifact");
            }
        }
        let saved = services.settings.get().expect("saved settings").settings;
        let selected = services.state.lock().selected_model_id.clone();
        let loaded = services.state.lock().model.clone();

        let error = services
            .complete_setup(&checkpoint)
            .expect_err("setup refused");

        assert_eq!(error.code, expected);
        let state = services.state.snapshot();
        assert_eq!(services.state.lock().model.clone(), loaded);
        assert_eq!(state.selected_model_id, selected);
        assert!(!services.state.lock().setup_complete);
        assert_eq!(
            services
                .settings
                .get()
                .expect("unchanged settings")
                .settings,
            saved
        );
    }
}

#[test]
fn setup_loads_and_selects_the_model_then_saves_its_choices() {
    use whispernow_domain::{SetupCheckpoint, SetupStep};

    let (services, _root) = native_services(Fixture::new());
    services.state.lock().setup_complete = false;
    let checkpoint = SetupCheckpoint {
        step: SetupStep::TryIt,
        ..SetupCheckpoint::default()
    };

    let view = services
        .complete_setup(&checkpoint)
        .expect("complete setup");

    assert!(view.settings.first_run_complete);
    assert!(view.settings.setup.is_none());
    assert_eq!(view.settings.model_id, checkpoint.model_id);
    assert_eq!(
        view.settings.recording_shortcut,
        checkpoint.recording_shortcut
    );
    let state = services.state.lock();
    assert!(state.setup_complete);
    assert_eq!(state.selected_model_id, checkpoint.model_id);
    assert_eq!(state.model, LoadedModel::Ready(checkpoint.model_id));
}

#[test]
fn audio_device_listing_reports_the_system_failure() {
    struct UnavailableDevices;

    impl AudioInput for UnavailableDevices {
        fn list_input_devices(&self) -> Result<Vec<AudioDeviceDescriptor>, AudioInputError> {
            Err(AudioInputError::Backend {
                operation: "list devices",
            })
        }

        fn start(
            &self,
            _: AudioDeviceChoice,
            _: Arc<dyn AudioSink>,
        ) -> Result<Box<dyn AudioCapture>, AudioInputError> {
            panic!("device listing does not record audio")
        }
    }

    let (mut services, _root) = native_services(Fixture::new());
    services.audio = Arc::new(UnavailableDevices);
    let error = services
        .list_audio_devices()
        .expect_err("device listing failed");
    assert_eq!(error.category, ErrorCategory::Audio);
    assert_eq!(error.code, "audio_devices");
}

#[test]
fn invalid_setup_choices_leave_the_saved_and_selected_model_unchanged() {
    use whispernow_domain::{SetupCheckpoint, SetupStep, Shortcut};

    let (services, _root) = native_services(Fixture::new());
    services.state.lock().setup_complete = false;
    let before = services.settings.get().expect("saved settings").settings;
    let registry = ModelRegistry::builtin();
    let model = registry
        .speech_models()
        .find(|model| model.id != before.model_id)
        .expect("another speech model");
    for artifact in model.layout.required_artifacts() {
        let path = services
            .paths
            .models_dir()
            .join(model.id.as_str())
            .join(artifact);
        fs::create_dir_all(path.parent().expect("artifact parent")).expect("model directory");
        fs::write(path, b"fixture").expect("model artifact");
    }
    let checkpoint = SetupCheckpoint {
        step: SetupStep::TryIt,
        model_id: model.id.clone(),
        recording_shortcut: Shortcut::default_cancel(),
        ..SetupCheckpoint::default()
    };

    let error = services
        .complete_setup(&checkpoint)
        .expect_err("invalid setup choices");

    assert_eq!(error.code, "invalid_settings");
    assert_eq!(
        services
            .settings
            .get()
            .expect("saved model unchanged")
            .settings,
        before
    );
    assert_eq!(services.state.snapshot().selected_model_id, before.model_id);
    assert!(!services.state.lock().setup_complete);
}
