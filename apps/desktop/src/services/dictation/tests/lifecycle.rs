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

#[test]
fn a_failed_tray_pause_does_not_leave_a_settings_warning() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let fixture = Fixture::new();
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
    let services = Arc::new(NativeServices {
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
    });
    let shortcuts = Arc::new(ShortcutCoordinator::new(
        Arc::new(Registrar),
        recording_binding(&Settings::default()),
        Arc::new(|| true),
    ));
    let gateway = Gateway::new(
        services,
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
