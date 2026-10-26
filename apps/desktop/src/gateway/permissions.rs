use whispernow_domain::AppError;
use whispernow_domain::ipc::{PermissionKind, PermissionStatus, PermissionsResponse};

use std::sync::Arc;

use super::{Failure, Gateway, run_blocking, settings::shortcut_error};
use crate::shortcuts::ShortcutRegistrationError;

impl Gateway {
    pub(crate) async fn permissions(&self) -> Result<PermissionsResponse, AppError> {
        let global_shortcuts = self.global_shortcuts_consent();
        self.run_command("get_permissions", Failure::Local, move |backend| {
            Ok(backend.permissions(global_shortcuts))
        })
        .await
    }

    /// Asks for one permission and returns the cards with its answer. A refusal
    /// is an answer, shown on its card; only a failure to ask is an error.
    pub(crate) async fn request_permission(
        &self,
        request: PermissionKind,
    ) -> Result<PermissionsResponse, AppError> {
        if request == PermissionKind::GlobalShortcuts {
            let shortcuts = Arc::clone(&self.shortcuts);
            self.backend.state.lock().phase.require_idle()?;
            // Registering the saved shortcuts again opens the compositor's consent
            // dialog. Serialized with settings saves like a shortcut capture.
            let _native_settings = self.native_settings.lock().await;
            if let Err(error) = run_blocking(move || shortcuts.sync_lifecycle()).await?
                && !matches!(
                    error,
                    ShortcutRegistrationError::PermissionDenied
                        | ShortcutRegistrationError::PortalUnavailable
                )
            {
                return Err(shortcut_error(error));
            }
        } else {
            self.run_command("request_permission", Failure::Local, move |backend| {
                backend.request_permission(request)
            })
            .await?;
        }
        self.permissions().await
    }

    /// `true` once the helper is installed.
    pub(crate) async fn set_up_paste_helper(&self) -> Result<bool, AppError> {
        self.run_command("set_up_paste_helper", Failure::Local, |backend| {
            backend.set_up_paste_helper()
        })
        .await
    }

    fn global_shortcuts_consent(&self) -> PermissionStatus {
        shortcut_portal_consent(
            self.shortcuts.registration_active(),
            self.shortcuts.failure(),
        )
    }
}

/// The shortcut portal's consent, as the registration shows it. A refusal or a
/// missing portal is an answer; any other failure is a conflict the Shortcuts
/// step resolves, not a consent question.
const fn shortcut_portal_consent(
    registered: bool,
    diagnostic: Option<ShortcutRegistrationError>,
) -> PermissionStatus {
    if registered {
        return PermissionStatus::Granted;
    }
    match diagnostic {
        Some(ShortcutRegistrationError::PermissionDenied) => PermissionStatus::Denied,
        Some(ShortcutRegistrationError::PortalUnavailable) => PermissionStatus::Unavailable,
        Some(
            ShortcutRegistrationError::ConflictOrUnsupported
            | ShortcutRegistrationError::SessionClosed,
        )
        | None => PermissionStatus::NotDetermined,
    }
}

#[cfg(test)]
mod tests {
    use whispernow_domain::ipc::PermissionStatus as Status;

    use super::shortcut_portal_consent;
    use crate::shortcuts::ShortcutRegistrationError as Error;

    #[test]
    fn the_shortcut_registration_answers_the_portal_consent_question() {
        assert_eq!(
            shortcut_portal_consent(true, Some(Error::PermissionDenied)),
            Status::Granted,
            "active shortcuts were granted, whatever an earlier attempt said"
        );
        assert_eq!(
            shortcut_portal_consent(false, Some(Error::PermissionDenied)),
            Status::Denied
        );
        assert_eq!(
            shortcut_portal_consent(false, Some(Error::ConflictOrUnsupported)),
            Status::NotDetermined
        );
    }
}
