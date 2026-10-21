use super::clear_data::ReleaseOnDrop;
use super::*;
use crate::{
    activation::ActivationGate,
    autostart::AutostartRegistration,
    gateway::Gateway,
    services::{
        NativeServices,
        models::ModelCoordinator,
        output::OutputCoordinator,
        permissions::MicrophoneAccess,
        test_support::{install_artifacts, shut_down},
    },
    shortcuts::{
        ShortcutCoordinator, ShortcutRegistrationError, memory_registrar::MemoryRegistrar,
        recording_binding,
    },
    tray::TrayController,
};
use whispernow_core::audio::AudioInputError;
use whispernow_platform::LocalCleanupService;

#[derive(Default)]
struct Autostart {
    enabled: std::sync::atomic::AtomicBool,
}

impl AutostartRegistration for Autostart {
    fn is_enabled(&self) -> Option<bool> {
        Some(self.enabled.load(Ordering::SeqCst))
    }

    fn set_enabled(&self, enabled: bool) -> Result<(), AppError> {
        self.enabled.store(enabled, Ordering::SeqCst);
        Ok(())
    }
}

pub(super) fn native_services(fixture: Fixture) -> (NativeServices, tempfile::TempDir) {
    fixture.flush();
    native_services_before_preload(fixture)
}

pub(super) fn native_services_before_preload(
    fixture: Fixture,
) -> (NativeServices, tempfile::TempDir) {
    let paths = fixture.paths.clone();
    let store = Arc::new(ModelStore::new(paths.clone()));
    let local_cleanup = Arc::new(LocalCleanupService::new(
        fixture.root.path().join("llama"),
        Arc::clone(&store),
        |_| {},
    ));
    let services = NativeServices {
        settings: fixture.settings,
        history: fixture.history,
        audio: Arc::new(FakeAudioInput::recording(Vec::new())),
        models: Arc::new(ModelCoordinator::new(
            store,
            Arc::clone(&fixture.state),
            Arc::clone(&local_cleanup),
        )),
        state: fixture.state,
        dictation: fixture.dictation,
        paths: paths.clone(),
        output: Arc::new(OutputCoordinator::native(None)),
        microphone_access: MicrophoneAccess::default(),
        local_cleanup,
    };
    (services, fixture.root)
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "Keeping shortcut registration and geometry saves together exposes their ordering"
)]
fn concurrent_settings_and_geometry_saves_retain_both_and_register_the_final_settings() {
    use whispernow_domain::{Shortcut, ShortcutKey, ThemePreference, WindowGeometry};

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("runtime");
    let (services, _root) = native_services(Fixture::new());
    let saved = services.settings.get().settings;
    let registrar = Arc::new(MemoryRegistrar::default());
    let shortcuts = Arc::new(ShortcutCoordinator::new(
        registrar.clone(),
        recording_binding(&saved),
        Arc::new(|| true),
        None,
    ));
    let autostart = Arc::new(Autostart::default());
    let gateway = Arc::new(Gateway::new(
        Arc::new(services),
        shortcuts,
        autostart.clone(),
        Arc::new(ActivationGate::default()),
        TrayController::headless(true),
    ));
    let (entered, registering) = mpsc::channel();
    let (release, released) = mpsc::channel();
    *registrar.gate.lock().expect("registration gate") = Some((entered, released));
    let mut registration_release = ReleaseOnDrop(Some(release));
    let first = Settings {
        theme: ThemePreference::Dark,
        recording_shortcut: Shortcut {
            key: ShortcutKey::F1,
            ..Shortcut::default()
        },
        ..saved.clone()
    };
    let second = Settings {
        recording_shortcut: Shortcut {
            key: ShortcutKey::F2,
            ..Shortcut::default()
        },
        ..saved
    };
    let expected = second.clone();
    let saving = Arc::clone(&gateway);
    let first = runtime.spawn(async move { saving.update_settings(first).await });
    registering
        .recv_timeout(Duration::from_secs(10))
        .expect("first shortcut registration");
    let geometry = WindowGeometry {
        x: 12,
        y: 24,
        width: 900,
        height: 600,
    };
    gateway
        .services
        .settings
        .save_window_geometry(geometry)
        .expect("independent geometry save");
    let saving = Arc::clone(&gateway);
    let (started, starting) = mpsc::channel();
    let (finished, finishing) = mpsc::channel();
    let second = runtime.spawn(async move {
        started.send(()).expect("second-save observer");
        let result = saving.update_settings(second).await;
        finished.send(()).expect("second-save completion observer");
        result
    });
    starting
        .recv_timeout(Duration::from_secs(10))
        .expect("second save started");
    assert!(
        matches!(
            finishing.recv_timeout(Duration::from_millis(250)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ),
        "second save must wait for the first shortcut registration"
    );
    registration_release.release();
    runtime.block_on(async {
        tokio::time::timeout(Duration::from_secs(10), async {
            first.await.expect("first save task").expect("first save");
            second
                .await
                .expect("second save task")
                .expect("second save");
        })
        .await
        .expect("settings saves completed");
    });

    let saved = gateway.services.settings.get().settings;
    assert_eq!(saved.theme, expected.theme);
    assert_eq!(saved.window_geometry, Some(geometry));
    assert_eq!(saved.recording_shortcut, expected.recording_shortcut);
    assert_eq!(
        registrar.bindings(),
        [recording_binding(&saved)
            .expect("final recording shortcut")
            .binding]
    );
    runtime.block_on(async {
        gateway
            .set_start_on_login(true)
            .await
            .expect("enable autostart");
        assert!(
            gateway
                .settings_view()
                .await
                .expect("enabled view")
                .start_on_login
        );
        autostart.set_enabled(false).expect("external change");
        let view = gateway.settings_view().await.expect("changed view");
        assert!(!view.start_on_login);
        assert!(view.warnings.is_empty());
    });
    shut_down(&gateway.services);
}

#[test]
fn settings_views_after_a_change_keep_the_standing_shortcut_warning() {
    use whispernow_domain::{SetupCheckpoint, SetupStep};

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let (services, _root) = native_services(Fixture::new());
    let shortcuts = Arc::new(ShortcutCoordinator::new(
        Arc::new(MemoryRegistrar::default()),
        Err(ShortcutRegistrationError::ConflictOrUnsupported),
        Arc::new(|| true),
        None,
    ));
    let gateway = Gateway::new(
        Arc::new(services),
        shortcuts,
        Arc::new(Autostart::default()),
        Arc::new(ActivationGate::default()),
        TrayController::headless(true),
    );
    let standing = runtime
        .block_on(gateway.settings_view())
        .expect("settings view")
        .warnings;
    assert_eq!(
        standing,
        ["A configured recording shortcut is unavailable; choose another combination"]
    );

    runtime.block_on(async {
        let key = SetApiKeyRequest {
            provider: LlmProvider::OpenAi,
            key: ApiKey::new("saved-key").expect("key"),
        };
        let set = gateway.set_api_key(key).await.expect("set key");
        assert_eq!(set.warnings, standing);
        let cleared = gateway
            .clear_api_key(LlmProvider::OpenAi)
            .await
            .expect("clear key");
        assert_eq!(cleared.warnings, standing);
        let updated = gateway
            .update_settings(gateway.services.settings.load())
            .await
            .expect("update settings");
        assert_eq!(updated.settings.warnings, standing);
        let saved = gateway
            .save_setup_checkpoint(SetupCheckpoint::default())
            .await
            .expect("save checkpoint");
        assert_eq!(saved.warnings, standing);
        assert_eq!(saved.settings.setup, Some(SetupCheckpoint::default()));
        let checkpoint = SetupCheckpoint {
            step: SetupStep::TryIt,
            ..SetupCheckpoint::default()
        };
        gateway
            .services
            .complete_setup(&checkpoint)
            .expect("complete setup");
        let restarted = gateway.restart_setup().await.expect("restart setup");
        assert_eq!(restarted.warnings, standing);
        assert!(!restarted.settings.first_run_complete);
        assert_eq!(
            restarted.settings.setup.map(|setup| setup.step),
            Some(SetupStep::Welcome)
        );
    });
    shut_down(&gateway.services);
}

#[test]
fn setup_refuses_active_downloads_and_cleanup_models_before_loading() {
    use crate::services::state::DownloadTask;
    use whispernow_domain::ipc::ActiveDownload;
    use whispernow_domain::{DEFAULT_CLEANUP_MODEL_ID, DownloadPhase, SetupCheckpoint, SetupStep};

    for expected in ["model_transition_busy", "model_not_speech"] {
        let (services, _root) = native_services(Fixture::new());
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
            install_artifacts(
                &services.paths,
                registry
                    .get(DEFAULT_CLEANUP_MODEL_ID)
                    .expect("cleanup model"),
            );
        }
        let saved = services.settings.get().settings;
        let selected = services.state.lock().selected_model_id.clone();
        let loaded = services.state.lock().model.clone();

        let error = services
            .complete_setup(&checkpoint)
            .expect_err("setup refused");

        assert_eq!(error.code, expected);
        let state = services.state.snapshot();
        assert_eq!(services.state.lock().model.clone(), loaded);
        assert_eq!(state.selected_model_id, selected);
        assert!(!services.settings.load().first_run_complete);
        assert_eq!(services.settings.get().settings, saved);
    }
}

#[test]
fn setup_loads_and_selects_the_model_then_saves_its_choices() {
    use whispernow_domain::{SetupCheckpoint, SetupStep};

    let (services, _root) = native_services(Fixture::new());
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
    assert_eq!(state.selected_model_id, checkpoint.model_id);
    assert_eq!(state.model, LoadedModel::Ready(checkpoint.model_id.clone()));
    drop(state);
    assert_eq!(
        services
            .save_setup_checkpoint(checkpoint.clone())
            .expect_err("completed setup cannot save a checkpoint")
            .code,
        "setup_already_complete"
    );
    let restarted = services.settings.restart_setup().expect("restart setup");
    assert!(!restarted.settings.first_run_complete);
    assert_eq!(
        restarted.settings.setup.map(|setup| setup.step),
        Some(SetupStep::Welcome)
    );
    services
        .save_setup_checkpoint(checkpoint)
        .expect("restarted setup accepts checkpoints");
}

#[cfg(unix)]
#[test]
fn setup_checkpoints_do_not_scan_unrelated_installed_artifacts() {
    use std::os::unix::fs::PermissionsExt as _;
    use whispernow_domain::SetupCheckpoint;

    let (services, _root) = native_services(Fixture::new());
    let registry = ModelRegistry::builtin();
    let other = registry
        .speech_models()
        .find(|model| model.id != ModelId::default())
        .expect("another model");
    let inaccessible = services.paths.models_dir().join(other.id.as_str());
    fs::create_dir_all(&inaccessible).expect("unrelated installation");
    fs::set_permissions(&inaccessible, fs::Permissions::from_mode(0o000))
        .expect("inaccessible artifacts");
    let scan = services.models.model_store().list();
    let checkpoint = SetupCheckpoint::default();
    let saved = services.save_setup_checkpoint(checkpoint.clone());
    fs::set_permissions(&inaccessible, fs::Permissions::from_mode(0o700))
        .expect("restore artifacts access");

    assert!(
        scan.is_err(),
        "the unrelated installation cannot be scanned"
    );
    assert_eq!(
        saved.expect("supported checkpoint").settings.setup,
        Some(checkpoint.clone())
    );
    assert_eq!(services.settings.get().settings.setup, Some(checkpoint));
    let unknown = SetupCheckpoint {
        model_id: ModelId::new("unknown-model").expect("model id"),
        ..SetupCheckpoint::default()
    };
    assert_eq!(
        services
            .save_setup_checkpoint(unknown)
            .expect_err("unsupported choice")
            .code,
        "unknown_model"
    );
}

#[test]
fn audio_device_listing_reports_the_system_failure() {
    let (mut services, _root) = native_services(Fixture::new());
    services.audio = Arc::new(FakeAudioInput::listing(Err(AudioInputError::Backend {
        operation: "list devices",
    })));
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
    let before = services.settings.get().settings;
    let registry = ModelRegistry::builtin();
    let model = registry
        .speech_models()
        .find(|model| model.id != before.model_id)
        .expect("another speech model");
    install_artifacts(&services.paths, model);
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
    assert_eq!(services.settings.get().settings, before);
    assert_eq!(services.state.snapshot().selected_model_id, before.model_id);
    assert!(!services.settings.load().first_run_complete);
}
