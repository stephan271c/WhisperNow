use tauri_plugin_autostart::ManagerExt as _;

pub(crate) trait AutostartRegistration: Send + Sync {
    fn is_enabled(&self) -> Result<bool, AutostartError>;
    fn set_enabled(&self, enabled: bool) -> Result<(), AutostartError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AutostartError {
    Query,
    Update,
    Verification,
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

/// Applies a persisted setting to the OS registration and verifies the resulting state.
pub(crate) fn reconcile(
    registration: &dyn AutostartRegistration,
    desired: bool,
) -> Result<(), AutostartError> {
    if registration.is_enabled()? != desired {
        registration.set_enabled(desired)?;
    }
    if registration.is_enabled()? == desired {
        Ok(())
    } else {
        Err(AutostartError::Verification)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    use super::*;

    #[derive(Default)]
    struct MemoryRegistration {
        enabled: Mutex<bool>,
        fail_update: Mutex<bool>,
        query_count: Mutex<u32>,
        fail_query_at: Mutex<Option<u32>>,
    }

    impl AutostartRegistration for MemoryRegistration {
        fn is_enabled(&self) -> Result<bool, AutostartError> {
            let mut query_count = self.query_count.lock().expect("query count");
            *query_count += 1;
            if *self.fail_query_at.lock().expect("query failure") == Some(*query_count) {
                return Err(AutostartError::Query);
            }
            Ok(*self.enabled.lock().expect("enabled"))
        }

        fn set_enabled(&self, enabled: bool) -> Result<(), AutostartError> {
            if *self.fail_update.lock().expect("failure") {
                return Err(AutostartError::Update);
            }
            *self.enabled.lock().expect("enabled") = enabled;
            Ok(())
        }
    }

    #[test]
    fn reconcile_repairs_drift_after_restart() {
        let native = MemoryRegistration::default();
        reconcile(&native, true).expect("enable");
        assert!(*native.enabled.lock().expect("enabled"));
        reconcile(&native, false).expect("disable");
        assert!(!*native.enabled.lock().expect("enabled"));
    }

    #[test]
    fn failed_query_leaves_the_registration_unchanged() {
        let native = MemoryRegistration::default();
        *native.fail_query_at.lock().expect("query failure") = Some(1);

        assert_eq!(reconcile(&native, true), Err(AutostartError::Query));
        assert!(!*native.enabled.lock().expect("enabled"));
    }

    #[test]
    fn failed_update_is_reported() {
        let native = MemoryRegistration::default();
        *native.fail_update.lock().expect("failure") = true;

        assert_eq!(reconcile(&native, true), Err(AutostartError::Update));
        assert!(!*native.enabled.lock().expect("enabled"));
    }

    #[derive(Default)]
    struct RefusingRegistration(AtomicUsize);

    impl AutostartRegistration for RefusingRegistration {
        fn is_enabled(&self) -> Result<bool, AutostartError> {
            Ok(false)
        }

        fn set_enabled(&self, _enabled: bool) -> Result<(), AutostartError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    #[test]
    fn registration_that_ignores_the_update_fails_verification() {
        let native = RefusingRegistration::default();
        assert_eq!(reconcile(&native, true), Err(AutostartError::Verification));
        assert_eq!(native.0.load(Ordering::Relaxed), 1);
    }
}
