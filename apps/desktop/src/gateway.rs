pub(crate) mod commands;
mod events;
pub(crate) mod lifecycle;
mod permissions;
mod settings;

use std::sync::{Arc, Mutex, PoisonError, RwLock, atomic::AtomicBool};

use whispernow_api::{ApiErrorDto, AppStateDto, ErrorCategoryDto, EventPayload};
use whispernow_core::CancellationToken;

use crate::{
    activation::ActivationGate, autostart::AutostartRegistration, services::NativeServices,
    shortcuts::ShortcutCoordinator, tray::TrayController,
};

pub(crate) use events::{emit, forward_events};
pub(crate) use settings::{autostart_error, shortcut_error};

pub(crate) struct Gateway {
    pub(crate) services: Arc<NativeServices>,
    /// Cancelled when shutdown begins; running commands see it and new ones are refused.
    shutdown: CancellationToken,
    /// Held for reading by every running command, so shutdown, which takes it for writing,
    /// releases the services only after they finished.
    commands: Arc<RwLock<()>>,
    pub(crate) shortcuts: Arc<ShortcutCoordinator>,
    autostart: Arc<dyn AutostartRegistration>,
    /// Serializes persisting settings with applying their native side effects,
    /// so the settings saved last are the settings registered last, and a
    /// shortcut capture or clear-data never interleaves with that application.
    /// It is held across awaits only; native calls run on blocking workers.
    native_settings: tokio::sync::Mutex<()>,
    activation: Arc<ActivationGate>,
    tray: Arc<TrayController>,
    shutdown_begun: AtomicBool,
    shutdown_claimed: AtomicBool,
    shutdown_complete: AtomicBool,
    /// The latest pause, resume or shortcut failure, shown with the settings.
    lifecycle_warning: Mutex<Option<String>>,
}

impl Gateway {
    pub(crate) fn new(
        services: Arc<NativeServices>,
        shortcuts: Arc<ShortcutCoordinator>,
        autostart: Arc<dyn AutostartRegistration>,
        activation: Arc<ActivationGate>,
        tray: Arc<TrayController>,
    ) -> Self {
        Self {
            services,
            shutdown: CancellationToken::default(),
            commands: Arc::new(RwLock::new(())),
            shortcuts,
            autostart,
            native_settings: tokio::sync::Mutex::new(()),
            activation,
            tray,
            shutdown_begun: AtomicBool::new(false),
            shutdown_claimed: AtomicBool::new(false),
            shutdown_complete: AtomicBool::new(false),
            lifecycle_warning: Mutex::new(None),
        }
    }

    pub(crate) fn services_state(&self) -> AppStateDto {
        self.services.state.snapshot()
    }

    /// Publishes the current snapshot so late subscribers such as the tray converge.
    pub(crate) fn publish_state(&self) {
        self.services.state.publish();
    }

    pub(crate) fn request_navigation(&self, section: whispernow_api::MainWindowSectionDto) {
        self.services
            .state
            .events()
            .publish(EventPayload::NavigationRequested { section });
    }

    /// Runs a command's work on a blocking worker, away from the async runtime.
    pub(crate) async fn run_command<R: Send + 'static>(
        &self,
        operation: &'static str,
        failure: Failure,
        work: impl FnOnce(Arc<NativeServices>, CancellationToken) -> Result<R, ApiErrorDto>
        + Send
        + 'static,
    ) -> Result<R, ApiErrorDto> {
        let services = Arc::clone(&self.services);
        let commands = Arc::clone(&self.commands);
        let shutdown = self.shutdown.clone();
        tracing::info!(operation, "desktop command started");
        let result = run_blocking(move || {
            let _running = commands.read().unwrap_or_else(PoisonError::into_inner);
            if shutdown.is_cancelled() {
                return Err(ApiErrorDto::new(
                    ErrorCategoryDto::General,
                    "application_shutting_down",
                    "The application is closing",
                ));
            }
            work(services, shutdown)
        })
        .await
        .and_then(|result| result);
        match &result {
            Ok(_) => tracing::info!(operation, "desktop command completed"),
            Err(error) => {
                if matches!(failure, Failure::Published) {
                    self.publish_error(error);
                }
                tracing::warn!(operation, code = error.code, "desktop command failed");
            }
        }
        result
    }

    fn publish_error(&self, error: &ApiErrorDto) {
        self.services.state.publish_error(error.clone());
    }
}

/// Whether a failed command becomes the application's current problem (`last_error`), which
/// Home and the tray keep showing until it is resolved.
#[derive(Clone, Copy)]
pub(crate) enum Failure {
    Published,
    /// The control that asked shows the failure in place: a probe, a test, or one setting's
    /// save.
    Local,
}

async fn run_blocking<T, F>(work: F) -> Result<T, ApiErrorDto>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    tokio::task::spawn_blocking(work).await.map_err(|_| {
        ApiErrorDto::new(
            ErrorCategoryDto::General,
            "worker_stopped",
            "The application could not complete this operation",
        )
    })
}
