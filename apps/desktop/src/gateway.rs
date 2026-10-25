pub(crate) mod commands;
mod events;
pub(crate) mod lifecycle;
mod permissions;
mod settings;
mod shutdown;

use std::sync::{Arc, RwLock};
use whispernow_domain::{AppError, ErrorCategory};

use crate::{
    activation::ActivationGate,
    autostart::AutostartRegistration,
    services::{Backend, shutting_down},
    shortcuts::Shortcuts,
    tray::TrayController,
};
use shutdown::Shutdown;

pub(crate) use events::forward_events;
pub(crate) use settings::shortcut_error;

pub(crate) struct Gateway {
    pub(crate) backend: Arc<Backend>,
    shutdown: Arc<Shutdown>,
    /// Held for reading by every running command, so shutdown, which takes it for writing,
    /// releases the backend only after they finished.
    commands: Arc<RwLock<()>>,
    pub(crate) shortcuts: Arc<Shortcuts>,
    autostart: Arc<dyn AutostartRegistration>,
    /// Serializes persisting settings with applying their native side effects, so the settings
    /// saved last are the settings registered last, and a shortcut capture or clear-data never
    /// interleaves with that application. Held across awaits only; native calls run on blocking
    /// workers.
    native_settings: tokio::sync::Mutex<()>,
    activation: Arc<ActivationGate>,
    tray: TrayController,
}

impl Gateway {
    pub(crate) fn new(
        backend: Arc<Backend>,
        shortcuts: Arc<Shortcuts>,
        autostart: Arc<dyn AutostartRegistration>,
        activation: Arc<ActivationGate>,
        tray: TrayController,
    ) -> Self {
        Self {
            backend,
            shutdown: Arc::new(Shutdown::new()),
            commands: Arc::new(RwLock::new(())),
            shortcuts,
            autostart,
            native_settings: tokio::sync::Mutex::new(()),
            activation,
            tray,
        }
    }

    pub(crate) async fn run_command<R: Send + 'static>(
        &self,
        operation: &'static str,
        failure: Failure,
        work: impl FnOnce(Arc<Backend>) -> Result<R, AppError> + Send + 'static,
    ) -> Result<R, AppError> {
        let backend = Arc::clone(&self.backend);
        let commands = Arc::clone(&self.commands);
        let shutdown = Arc::clone(&self.shutdown);
        tracing::info!(operation, "desktop command started");
        let result = run_blocking(move || {
            let _running = commands.read().expect("command barrier lock poisoned");
            if shutdown.has_begun() {
                return Err(shutting_down());
            }
            work(backend)
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

    pub(crate) fn publish_error(&self, error: &AppError) {
        self.backend.state.publish_error(error.clone());
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

async fn run_blocking<T, F>(work: F) -> Result<T, AppError>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    tokio::task::spawn_blocking(work).await.map_err(|error| {
        tracing::error!(%error, "a blocking task did not finish");
        AppError::new(
            ErrorCategory::General,
            "operation_aborted",
            "The application could not complete this operation",
        )
    })
}
