use tauri_plugin_autostart::ManagerExt as _;
use whispernow_domain::{AppError, ErrorCategory};

pub(crate) trait AutostartRegistration: Send + Sync {
    /// `None` when the OS registration cannot be inspected.
    fn is_enabled(&self) -> Option<bool>;
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
    fn is_enabled(&self) -> Option<bool> {
        self.0.autolaunch().is_enabled().ok()
    }

    fn set_enabled(&self, enabled: bool) -> Result<(), AppError> {
        let manager = self.0.autolaunch();
        let result = if enabled {
            manager.enable()
        } else {
            manager.disable()
        };
        result.map_err(|_| {
            AppError::new(
                ErrorCategory::Settings,
                "autostart_update_failed",
                "The start-on-login registration could not be updated",
            )
        })
    }
}
