//! First-run permission commands: the permission cards, asking for one, and
//! opening the system page that grants one. Probes and requests run on
//! background workers, never on the async runtime.

use whispernow_api::{
    ApiErrorDto, PermissionKindDto, PermissionRequest, PermissionStatusDto, PermissionsResponse,
};

use std::sync::Arc;

use super::{Failure, Gateway, run_blocking, settings::shortcut_error};
use crate::shortcuts::ShortcutRegistrationError;

impl Gateway {
    /// The permission cards this platform needs, with what the system reports now.
    pub(crate) async fn permissions(&self) -> Result<PermissionsResponse, ApiErrorDto> {
        let global_shortcuts = self.global_shortcuts_consent();
        self.run_command("get_permissions", Failure::Local, move |services, _| {
            Ok(services.permissions(global_shortcuts))
        })
        .await
    }

    /// Asks for one permission and returns the cards with its answer. A refusal
    /// is an answer, shown on its card; only a failure to ask is an error.
    pub(crate) async fn request_permission(
        &self,
        request: PermissionRequest,
    ) -> Result<PermissionsResponse, ApiErrorDto> {
        if request.kind == PermissionKindDto::GlobalShortcuts {
            let shortcuts = Arc::clone(&self.shortcuts);
            self.services.state.lock().phase.require_idle()?;
            // Registering the saved shortcuts again opens the compositor's consent
            // dialog. Serialized with settings saves like a shortcut capture.
            let _native_settings = self.native_settings.lock().await;
            // A refusal or a missing portal is an answer, shown on the card.
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
            self.run_command("request_permission", Failure::Local, move |services, _| {
                services.request_permission(request.kind)
            })
            .await?;
        }
        self.permissions().await
    }

    fn global_shortcuts_consent(&self) -> PermissionStatusDto {
        shortcut_portal_consent(
            self.shortcuts.registration_active(),
            self.shortcuts.registration_diagnostic(),
        )
    }
}

/// The shortcut portal's consent, as the registration shows it. A refusal or a
/// missing portal is an answer; any other failure is a conflict the Shortcuts
/// step resolves, not a consent question.
const fn shortcut_portal_consent(
    registered: bool,
    diagnostic: Option<ShortcutRegistrationError>,
) -> PermissionStatusDto {
    if registered {
        return PermissionStatusDto::Granted;
    }
    match diagnostic {
        Some(ShortcutRegistrationError::PermissionDenied) => PermissionStatusDto::Denied,
        Some(ShortcutRegistrationError::PortalUnavailable) => PermissionStatusDto::Unavailable,
        Some(
            ShortcutRegistrationError::ConflictOrUnsupported
            | ShortcutRegistrationError::SessionClosed,
        )
        | None => PermissionStatusDto::NotDetermined,
    }
}

#[cfg(test)]
mod tests {
    use whispernow_api::PermissionStatusDto as Status;

    use super::shortcut_portal_consent;
    use crate::shortcuts::ShortcutRegistrationError as Error;

    #[test]
    fn the_shortcut_registration_answers_the_portal_consent_question() {
        assert_eq!(
            shortcut_portal_consent(true, Some(Error::PermissionDenied)),
            Status::Granted,
            "active shortcuts were granted, whatever an earlier attempt said"
        );
        for (diagnostic, expected) in [
            (Some(Error::PermissionDenied), Status::Denied),
            (Some(Error::PortalUnavailable), Status::Unavailable),
            (Some(Error::SessionClosed), Status::NotDetermined),
            (Some(Error::ConflictOrUnsupported), Status::NotDetermined),
            (None, Status::NotDetermined),
        ] {
            assert_eq!(
                shortcut_portal_consent(false, diagnostic),
                expected,
                "{diagnostic:?}"
            );
        }
    }
}
