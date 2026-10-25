use super::clear_data::ReleaseOnDrop;
use super::*;
use crate::{
    activation::ActivationGate,
    autostart::AutostartRegistration,
    gateway::Gateway,
    services::{
        Backend,
        models::ModelLibrary,
        output::Output,
        permissions::MicrophoneAccess,
        test_support::{install_artifacts, shut_down},
    },
    shortcuts::{Shortcuts, memory_registrar::MemoryRegistrar, recording_binding},
    tray::TrayController,
};
use whispernow_core::audio::AudioInputError;
use whispernow_domain::SettingChange;
use whispernow_platform::LocalCleanupService;

#[derive(Default)]
struct Autostart {
    enabled: std::sync::atomic::AtomicBool,
}

impl AutostartRegistration for Autostart {
    fn is_enabled(&self) -> Result<bool, AppError> {
        Ok(self.enabled.load(Ordering::SeqCst))
    }

    fn set_enabled(&self, enabled: bool) -> Result<(), AppError> {
        self.enabled.store(enabled, Ordering::SeqCst);
        Ok(())
    }
}

pub(super) fn test_backend(fixture: Fixture) -> (Backend, tempfile::TempDir) {
    fixture.flush();
    test_backend_before_preload(fixture)
}

pub(super) fn test_backend_before_preload(fixture: Fixture) -> (Backend, tempfile::TempDir) {
    let paths = fixture.paths.clone();
    let store = Arc::new(ModelStore::new(paths.clone()));
    let local_cleanup = Arc::new(LocalCleanupService::new(
        fixture.root.path().join("llama"),
        Arc::clone(&store),
        |_| {},
    ));
    let backend = Backend {
        settings: fixture.settings,
        history: fixture.history,
        audio: Arc::new(FakeAudioInput::recording(Vec::new())),
        models: ModelLibrary::new(
            store,
            Arc::clone(&fixture.state),
            Arc::clone(&local_cleanup),
        ),
        state: fixture.state,
        dictation: fixture.dictation,
        paths: paths.clone(),
        output: Arc::new(Output::native(None)),
        microphone_access: MicrophoneAccess::default(),
        local_cleanup,
    };
    (backend, fixture.root)
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "Keeping shortcut registration and geometry saves together exposes their ordering"
)]
fn concurrent_settings_and_geometry_saves_retain_both_and_register_the_final_settings() {
    use whispernow_domain::{Shortcut, ShortcutKey, WindowGeometry};

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("runtime");
    let (backend, _root) = test_backend(Fixture::new());
    let saved = backend.settings.view().settings;
    let registrar = Arc::new(MemoryRegistrar::default());
    let shortcuts = Arc::new(Shortcuts::new(
        registrar.clone(),
        recording_binding(&saved),
        Arc::new(|| true),
        None,
    ));
    let autostart = Arc::new(Autostart::default());
    let gateway = Arc::new(Gateway::new(
        Arc::new(backend),
        shortcuts,
        autostart.clone(),
        Arc::new(ActivationGate::default()),
        TrayController::headless(true),
    ));
    let (entered, registering) = mpsc::channel();
    let (release, released) = mpsc::channel();
    *registrar.gate.lock().expect("registration gate") = Some((entered, released));
    let mut registration_release = ReleaseOnDrop(Some(release));
    let recording = |key| SettingChange::RecordingShortcut {
        shortcut: Shortcut {
            key,
            ..Shortcut::default()
        },
        mode: saved.recording_mode,
    };
    let first = recording(ShortcutKey::F1);
    let second = recording(ShortcutKey::F2);
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
        .backend
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

    let saved = gateway.backend.settings.view().settings;
    assert_eq!(saved.window_geometry, Some(geometry));
    assert_eq!(saved.recording_shortcut.key, ShortcutKey::F2);
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
    shut_down(&gateway.backend);
}

#[test]
fn a_settings_save_keeps_registers_or_refuses_the_recording_shortcut() {
    use whispernow_domain::{Shortcut, ShortcutKey, ThemePreference};

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let (backend, _root) = test_backend(Fixture::new());
    let saved = backend.settings.view().settings;
    let registrar = Arc::new(MemoryRegistrar::default());
    let shortcuts = Arc::new(Shortcuts::new(
        registrar.clone(),
        recording_binding(&saved),
        Arc::new(|| true),
        None,
    ));
    let gateway = Gateway::new(
        Arc::new(backend),
        shortcuts,
        Arc::new(Autostart::default()),
        Arc::new(ActivationGate::default()),
        TrayController::headless(true),
    );
    gateway.register_persisted_shortcuts();
    let shortcut = |key| Shortcut {
        key,
        ..Shortcut::default()
    };
    let recording = |key| SettingChange::RecordingShortcut {
        shortcut: shortcut(key),
        mode: saved.recording_mode,
    };
    let held = |key| {
        let settings = Settings {
            recording_shortcut: shortcut(key),
            ..saved.clone()
        };
        recording_binding(&settings)
            .expect("a valid recording shortcut")
            .binding
    };
    runtime.block_on(async {
        let unchanged = gateway
            .update_settings(SettingChange::Theme(ThemePreference::Dark))
            .await
            .expect("theme save");
        assert!(unchanged.warnings.is_empty());
        assert_eq!(registrar.bindings(), [held(saved.recording_shortcut.key)]);

        let registered = gateway
            .update_settings(recording(ShortcutKey::F1))
            .await
            .expect("F1 save");
        assert!(registered.warnings.is_empty());
        assert_eq!(registrar.bindings(), [held(ShortcutKey::F1)]);

        *registrar.fail_register.lock().expect("fail register") = Some(held(ShortcutKey::F2).id.0);
        let refused = gateway
            .update_settings(recording(ShortcutKey::F2))
            .await
            .expect("the refusal is a warning");
        assert_eq!(refused.warnings.len(), 1);
        assert!(refused.warnings[0].contains("the previous shortcut is still in use"));
        assert_eq!(
            refused.settings.settings.recording_shortcut,
            shortcut(ShortcutKey::F1)
        );
        assert_eq!(registrar.bindings(), [held(ShortcutKey::F1)]);

        let settings_file = gateway.backend.paths.settings_file();
        std::fs::remove_file(&settings_file).expect("remove settings");
        std::fs::create_dir(&settings_file).expect("block settings replacement");
        let error = gateway
            .update_settings(recording(ShortcutKey::F3))
            .await
            .expect_err("settings write refused");
        assert_eq!(error.code, "settings_write");
        assert_eq!(
            registrar.bindings(),
            [held(ShortcutKey::F1)],
            "the saved shortcut is registered again"
        );
    });
    shut_down(&gateway.backend);
}

#[test]
fn a_cancel_shortcut_change_swaps_the_cancel_shortcut_and_other_changes_leave_it() {
    use whispernow_domain::{Shortcut, ShortcutKey, ThemePreference};

    use crate::shortcuts::{binding, cancel_binding};

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let (backend, _root) = test_backend(Fixture::new());
    let saved = backend.settings.view().settings;
    let registrar = Arc::new(MemoryRegistrar::default());
    let shortcuts = Arc::new(Shortcuts::new(
        registrar.clone(),
        recording_binding(&saved),
        Arc::new(|| true),
        None,
    ));
    shortcuts.set_cancel_binding(cancel_binding(&saved));
    shortcuts.observe_dictation(true);
    shortcuts.sync_cancel_shortcut();
    let gateway = Gateway::new(
        Arc::new(backend),
        shortcuts,
        Arc::new(Autostart::default()),
        Arc::new(ActivationGate::default()),
        TrayController::headless(true),
    );
    let escape = binding(&Shortcut::default_cancel());
    assert!(registrar.holds(&escape));

    runtime
        .block_on(gateway.update_settings(SettingChange::Theme(ThemePreference::Dark)))
        .expect("theme save");
    assert!(registrar.holds(&escape));

    let f12 = Shortcut {
        modifiers: [].into(),
        key: ShortcutKey::F12,
    };
    runtime
        .block_on(gateway.update_settings(SettingChange::CancelShortcut(Some(f12.clone()))))
        .expect("cancel shortcut save");
    assert!(registrar.holds(&binding(&f12)));
    assert!(!registrar.holds(&escape));
    shut_down(&gateway.backend);
}

#[test]
fn a_history_limit_change_prunes_history_and_other_changes_do_not() {
    use whispernow_domain::{
        EntryStatus, HistoryLimit, HistoryRetention, NewHistoryEntry, ThemePreference, UtcMillis,
    };

    let (backend, _root) = test_backend(Fixture::new());
    backend
        .update_settings(&SettingChange::HistoryLimit(HistoryLimit::Unlimited))
        .expect("unlimited history");
    for _ in 0..21 {
        let entry = NewHistoryEntry {
            created_at: UtcMillis::now(),
            duration_millis: 1_000,
            raw_text: Some("kept".to_owned()),
            cleaned_text: None,
            status: EntryStatus::Ok,
            error: None,
            asr_model_id: ModelId::default(),
        };
        backend
            .history
            .insert(
                entry,
                None,
                HistoryLimit::Unlimited,
                HistoryRetention::Forever,
            )
            .expect("insert");
    }
    let count = || {
        backend
            .history_page(&HistoryPageRequest::first_page(HistoryFilter::default()))
            .expect("history")
            .entries
            .len()
    };
    backend
        .update_settings(&SettingChange::Theme(ThemePreference::Dark))
        .expect("theme save");
    assert_eq!(count(), 21);
    let saved = backend
        .update_settings(&SettingChange::HistoryLimit(HistoryLimit::Last20))
        .expect("limit save");
    assert!(saved.warnings.is_empty());
    assert_eq!(count(), 20);
    shut_down(&backend);
}

#[test]
fn only_cleanup_changes_move_the_local_cleanup_model() {
    use whispernow_domain::{DEFAULT_CLEANUP_MODEL_ID, LlmProvider, ThemePreference};

    use crate::services::moves_cleanup;

    for change in [
        SettingChange::ActiveProvider(LlmProvider::LocalCleanup),
        SettingChange::CleanupEnabled(true),
        SettingChange::CleanupUsesGpu(false),
        SettingChange::CleanupModel(ModelId::new(DEFAULT_CLEANUP_MODEL_ID).expect("model id")),
    ] {
        assert!(moves_cleanup(&change), "{change:?}");
    }
    for change in [
        SettingChange::Theme(ThemePreference::Dark),
        SettingChange::ChooseProviderModel {
            provider: LlmProvider::OpenAi,
            model: "gpt-test".to_owned(),
        },
    ] {
        assert!(!moves_cleanup(&change), "{change:?}");
    }
}

#[test]
fn saved_shortcuts_that_break_a_rule_leave_the_settings_warning_published() {
    let fixture = Fixture::new();
    let invalid = Settings {
        cancel_shortcut: Some(Settings::default().recording_shortcut),
        ..Settings::default()
    };
    fixture.settings.report_loaded(&invalid, None);
    let shortcuts = Shortcuts::new(
        Arc::new(MemoryRegistrar::default()),
        recording_binding(&invalid),
        Arc::new(|| true),
        None,
    );
    assert_eq!(
        shortcuts.failure(),
        None,
        "startup publishes a shortcut error only for a registration failure"
    );
    let published = fixture.state.snapshot().error.expect("settings warning");
    assert!(
        published
            .summary
            .starts_with("A saved setting needs attention"),
        "{published:?}"
    );
    fixture.flush();
}

#[test]
fn the_persisted_shortcut_registration_publishes_its_failure_when_it_ends() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let (backend, _root) = test_backend(Fixture::new());
    let saved = backend.settings.view().settings;
    let recording = recording_binding(&saved).expect("the default recording shortcut");
    let registrar = Arc::new(MemoryRegistrar::default());
    *registrar.fail_register.lock().expect("fail register") = Some(recording.binding.id.0);
    let gateway = Gateway::new(
        Arc::new(backend),
        Arc::new(Shortcuts::new(
            registrar,
            Some(recording),
            Arc::new(|| true),
            None,
        )),
        Arc::new(Autostart::default()),
        Arc::new(ActivationGate::default()),
        TrayController::headless(true),
    );

    let pending = runtime
        .block_on(gateway.settings_view())
        .expect("settings view while the shortcut registers");
    assert!(pending.warnings.is_empty(), "{:?}", pending.warnings);
    assert!(gateway.backend.state.snapshot().error.is_none());

    gateway.register_persisted_shortcuts();
    let failed = runtime
        .block_on(gateway.settings_view())
        .expect("settings view after the registration failed");
    assert!(
        failed
            .warnings
            .iter()
            .any(|warning| warning.starts_with("A configured recording shortcut is unavailable")),
        "{:?}",
        failed.warnings
    );
    let published = gateway
        .backend
        .state
        .snapshot()
        .error
        .expect("the registration failure is published");
    assert_eq!(published.code, "shortcut_registration_failed");
    shut_down(&gateway.backend);
}

#[test]
fn settings_views_after_a_change_keep_the_standing_shortcut_warning() {
    use whispernow_domain::{SetupCheckpoint, SetupStep};

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let (backend, _root) = test_backend(Fixture::new());
    let shortcuts = Arc::new(Shortcuts::new(
        Arc::new(MemoryRegistrar::default()),
        None,
        Arc::new(|| true),
        None,
    ));
    let gateway = Gateway::new(
        Arc::new(backend),
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
            .update_settings(SettingChange::StartMinimized(true))
            .await
            .expect("update settings");
        assert_eq!(updated.settings.warnings, standing);
        let saved = gateway
            .save_setup_checkpoint(SetupCheckpoint::default())
            .await
            .expect("save checkpoint");
        assert_eq!(saved.warnings, standing);
        assert_eq!(
            saved.settings.setup.checkpoint(),
            Some(&SetupCheckpoint::default())
        );
        let checkpoint = SetupCheckpoint {
            step: SetupStep::TryIt,
            ..SetupCheckpoint::default()
        };
        gateway
            .backend
            .complete_setup(&checkpoint)
            .expect("complete setup");
        let restarted = gateway.restart_setup().await.expect("restart setup");
        assert_eq!(restarted.warnings, standing);
        assert!(!restarted.settings.setup.is_complete());
        assert_eq!(
            restarted
                .settings
                .setup
                .checkpoint()
                .map(|setup| setup.step),
            Some(SetupStep::Welcome)
        );
    });
    shut_down(&gateway.backend);
}

#[test]
fn a_settings_view_reads_the_autostart_registration_off_the_runtime_thread() {
    use std::thread::{self, ThreadId};

    #[derive(Default)]
    struct ReadingThread(Mutex<Option<ThreadId>>);

    impl AutostartRegistration for ReadingThread {
        fn is_enabled(&self) -> Result<bool, AppError> {
            *self.0.lock().expect("reading thread") = Some(thread::current().id());
            Ok(false)
        }

        fn set_enabled(&self, _enabled: bool) -> Result<(), AppError> {
            Ok(())
        }
    }

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let (backend, _root) = test_backend(Fixture::new());
    let autostart = Arc::new(ReadingThread::default());
    let gateway = Gateway::new(
        Arc::new(backend),
        Arc::new(Shortcuts::new(
            Arc::new(MemoryRegistrar::default()),
            None,
            Arc::new(|| true),
            None,
        )),
        autostart.clone(),
        Arc::new(ActivationGate::default()),
        TrayController::headless(true),
    );

    runtime
        .block_on(gateway.settings_view())
        .expect("settings view");

    let read_on = autostart.0.lock().expect("reading thread").expect("read");
    assert_ne!(read_on, thread::current().id());
    shut_down(&gateway.backend);
}

#[test]
fn setup_refuses_active_downloads_and_cleanup_models_before_loading() {
    use crate::services::state::DownloadTask;
    use whispernow_domain::ipc::ActiveDownload;
    use whispernow_domain::{DEFAULT_CLEANUP_MODEL_ID, DownloadPhase, SetupCheckpoint, SetupStep};

    for expected in ["model_transition_busy", "model_not_speech"] {
        let (backend, _root) = test_backend(Fixture::new());
        let mut checkpoint = SetupCheckpoint {
            step: SetupStep::TryIt,
            ..SetupCheckpoint::default()
        };
        if expected == "model_transition_busy" {
            backend.state.apply(|state| {
                state.active_downloads.insert(
                    checkpoint.model_id.clone(),
                    DownloadTask {
                        progress: ActiveDownload {
                            model_id: checkpoint.model_id.clone(),
                            phase: DownloadPhase::Downloading,
                            downloaded_bytes: 0,
                            total_bytes: 1_024,
                            bytes_per_second: None,
                        },
                        cancellation: CancellationToken::default(),
                    },
                );
            });
        } else {
            checkpoint.model_id = ModelId::new(DEFAULT_CLEANUP_MODEL_ID).expect("cleanup model id");
            let registry = ModelRegistry::builtin();
            install_artifacts(
                &backend.paths,
                registry
                    .get(DEFAULT_CLEANUP_MODEL_ID)
                    .expect("cleanup model"),
            );
        }
        let saved = backend.settings.view().settings;
        let selected = backend.state.lock().selected_model_id.clone();
        let loaded = backend.state.lock().model.clone();

        let error = backend
            .complete_setup(&checkpoint)
            .expect_err("setup refused");

        assert_eq!(error.code, expected);
        let state = backend.state.snapshot();
        assert_eq!(backend.state.lock().model.clone(), loaded);
        assert_eq!(state.selected_model_id, selected);
        assert!(!backend.settings.snapshot().setup.is_complete());
        assert_eq!(backend.settings.view().settings, saved);
    }
}

#[test]
fn setup_loads_and_selects_the_model_then_saves_its_choices() {
    use whispernow_domain::{SetupCheckpoint, SetupStep};

    let (backend, _root) = test_backend(Fixture::new());
    let checkpoint = SetupCheckpoint {
        step: SetupStep::TryIt,
        ..SetupCheckpoint::default()
    };

    let view = backend.complete_setup(&checkpoint).expect("complete setup");

    assert!(view.settings.setup.is_complete());
    assert_eq!(view.settings.model_id, checkpoint.model_id);
    assert_eq!(
        view.settings.recording_shortcut,
        checkpoint.recording_shortcut
    );
    let state = backend.state.lock();
    assert_eq!(state.selected_model_id, checkpoint.model_id);
    assert_eq!(state.model, LoadedModel::Ready(checkpoint.model_id.clone()));
    drop(state);
    assert_eq!(
        backend
            .save_setup_checkpoint(checkpoint.clone())
            .expect_err("completed setup cannot save a checkpoint")
            .code,
        "setup_already_complete"
    );
    let restarted = backend.settings.restart_setup().expect("restart setup");
    assert!(!restarted.settings.setup.is_complete());
    assert_eq!(
        restarted
            .settings
            .setup
            .checkpoint()
            .map(|setup| setup.step),
        Some(SetupStep::Welcome)
    );
    backend
        .save_setup_checkpoint(checkpoint)
        .expect("restarted setup accepts checkpoints");
}

#[test]
fn a_saved_setup_checkpoint_selects_its_model_and_keeps_the_saved_one() {
    use whispernow_domain::{SetupCheckpoint, SetupStep};

    let (backend, _root) = test_backend(Fixture::new());
    let saved = backend.settings.view().settings.model_id;
    let registry = ModelRegistry::builtin();
    let chosen = registry
        .speech_models()
        .find(|model| model.id != saved)
        .expect("another speech model");
    let checkpoint = SetupCheckpoint {
        step: SetupStep::Shortcuts,
        model_id: chosen.id.clone(),
        ..SetupCheckpoint::default()
    };

    backend
        .save_setup_checkpoint(checkpoint)
        .expect("save checkpoint");

    assert_eq!(backend.state.snapshot().selected_model_id, chosen.id);
    assert_eq!(backend.settings.view().settings.model_id, saved);
}

#[cfg(unix)]
#[test]
fn setup_checkpoints_do_not_scan_unrelated_installed_artifacts() {
    use std::os::unix::fs::PermissionsExt as _;
    use whispernow_domain::SetupCheckpoint;

    let (backend, _root) = test_backend(Fixture::new());
    let registry = ModelRegistry::builtin();
    let other = registry
        .speech_models()
        .find(|model| model.id != ModelId::default())
        .expect("another model");
    let inaccessible = backend.paths.models_dir().join(other.id.as_str());
    fs::create_dir_all(&inaccessible).expect("unrelated installation");
    fs::set_permissions(&inaccessible, fs::Permissions::from_mode(0o000))
        .expect("inaccessible artifacts");
    let scan = backend.models.model_store().list();
    let checkpoint = SetupCheckpoint::default();
    let saved = backend.save_setup_checkpoint(checkpoint.clone());
    fs::set_permissions(&inaccessible, fs::Permissions::from_mode(0o700))
        .expect("restore artifacts access");

    assert!(
        scan.is_err(),
        "the unrelated installation cannot be scanned"
    );
    assert_eq!(
        saved
            .expect("supported checkpoint")
            .settings
            .setup
            .checkpoint(),
        Some(&checkpoint)
    );
    assert_eq!(
        backend.settings.view().settings.setup.checkpoint(),
        Some(&checkpoint)
    );
    let unknown = SetupCheckpoint {
        model_id: ModelId::new("unknown-model").expect("model id"),
        ..SetupCheckpoint::default()
    };
    assert_eq!(
        backend
            .save_setup_checkpoint(unknown)
            .expect_err("unsupported choice")
            .code,
        "unknown_model"
    );
}

#[test]
fn audio_device_listing_reports_the_system_failure() {
    let (mut backend, _root) = test_backend(Fixture::new());
    backend.audio = Arc::new(FakeAudioInput::listing(Err(AudioInputError::Backend {
        operation: "list devices",
    })));
    let error = backend
        .list_audio_devices()
        .expect_err("device listing failed");
    assert_eq!(error.category, ErrorCategory::Audio);
    assert_eq!(error.code, "audio_devices");
}

#[test]
fn invalid_setup_choices_leave_the_saved_and_selected_model_unchanged() {
    use whispernow_domain::{SetupCheckpoint, SetupStep, Shortcut};

    let (backend, _root) = test_backend(Fixture::new());
    let before = backend.settings.view().settings;
    let registry = ModelRegistry::builtin();
    let model = registry
        .speech_models()
        .find(|model| model.id != before.model_id)
        .expect("another speech model");
    install_artifacts(&backend.paths, model);
    let checkpoint = SetupCheckpoint {
        step: SetupStep::TryIt,
        model_id: model.id.clone(),
        recording_shortcut: Shortcut::default_cancel(),
        ..SetupCheckpoint::default()
    };

    let error = backend
        .complete_setup(&checkpoint)
        .expect_err("invalid setup choices");

    assert_eq!(error.code, "invalid_settings");
    assert_eq!(backend.settings.view().settings, before);
    assert_eq!(backend.state.snapshot().selected_model_id, before.model_id);
    assert!(!backend.settings.snapshot().setup.is_complete());
}
