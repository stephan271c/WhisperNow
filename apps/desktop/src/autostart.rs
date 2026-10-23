use tauri_plugin_autostart::ManagerExt as _;
use whispernow_domain::{AppError, ErrorCategory};

pub(crate) trait AutostartRegistration: Send + Sync {
    fn is_enabled(&self) -> Result<bool, AppError>;
    fn set_enabled(&self, enabled: bool) -> Result<(), AppError>;
}

#[derive(Clone)]
pub(crate) struct TauriAutostart(tauri::AppHandle);

impl TauriAutostart {
    pub(crate) fn new(app: tauri::AppHandle) -> Self {
        Self(app)
    }
}

impl AutostartRegistration for TauriAutostart {
    fn is_enabled(&self) -> Result<bool, AppError> {
        self.0.autolaunch().is_enabled().map_err(|error| {
            tracing::warn!(%error, "the start-on-login registration could not be inspected");
            AppError::new(
                ErrorCategory::Settings,
                "autostart_inspect_failed",
                "The OS startup registration could not be inspected",
            )
        })
    }

    fn set_enabled(&self, enabled: bool) -> Result<(), AppError> {
        let manager = self.0.autolaunch();
        let result = if enabled {
            manager.enable()
        } else {
            manager.disable()
        };
        result.map_err(|error| {
            tracing::warn!(%error, "the start-on-login registration could not be updated");
            AppError::new(
                ErrorCategory::Settings,
                "autostart_update_failed",
                "The start-on-login registration could not be updated",
            )
        })
    }
}
