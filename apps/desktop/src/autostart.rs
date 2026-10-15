use tauri_plugin_autostart::ManagerExt as _;

pub(crate) trait AutostartRegistration: Send + Sync {
    fn is_enabled(&self) -> Result<bool, AutostartError>;
    fn set_enabled(&self, enabled: bool) -> Result<(), AutostartError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AutostartError {
    Query,
    Update,
}

#[derive(Clone)]
pub(crate) struct TauriAutostart(tauri::AppHandle);

impl TauriAutostart {
    pub(crate) fn new(app: tauri::AppHandle) -> Self {
        Self(app)
    }
}

impl AutostartRegistration for TauriAutostart {
    fn is_enabled(&self) -> Result<bool, AutostartError> {
        self.0
            .autolaunch()
            .is_enabled()
            .map_err(|_| AutostartError::Query)
    }

    fn set_enabled(&self, enabled: bool) -> Result<(), AutostartError> {
        let manager = self.0.autolaunch();
        let result = if enabled {
            manager.enable()
        } else {
            manager.disable()
        };
        result.map_err(|_| AutostartError::Update)
    }
}
