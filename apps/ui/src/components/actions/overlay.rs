use std::future::Future;
use whispernow_domain::AppError;

use whispernow_domain::ipc::{MainWindowSection, PermissionKind, Phase};

use crate::UiContext;
use crate::client;
use crate::operations::UiOperation;

use dioxus::prelude::*;

impl UiContext {
    /// Runs an overlay control; the overlay shows a failure in its own status line.
    fn overlay_command(
        self,
        operation: UiOperation,
        call: impl Future<Output = Result<(), AppError>> + 'static,
    ) {
        self.run(operation, call, move |actions, answer| {
            let Err(error) = answer else {
                return;
            };
            let mut state = actions.state;
            let mut state = state.write();
            if operation == UiOperation::OverlayAction
                || state
                    .snapshot
                    .as_ref()
                    .is_some_and(|snapshot| matches!(snapshot.phase, Phase::Recording { .. }))
            {
                state.overlay.action_error = Some(error.summary);
            }
        });
    }

    /// Opens the system page for a permission and then dismisses the failure.
    pub(crate) fn overlay_permission_settings(self, kind: PermissionKind) {
        self.overlay_command(UiOperation::OverlayAction, async move {
            client::invoke::<_, ()>("open_permission_settings", kind).await?;
            client::call("dismiss_recording_overlay").await
        });
    }

    pub(crate) fn stop_recording(self) {
        self.overlay_command(UiOperation::StopRecording, client::call("stop_recording"));
    }

    pub(crate) fn cancel_recording(self) {
        self.overlay_command(
            UiOperation::CancelRecording,
            client::call("cancel_recording"),
        );
    }

    pub(crate) fn dismiss_overlay(self) {
        self.overlay_command(
            UiOperation::OverlayAction,
            client::call("dismiss_recording_overlay"),
        );
    }

    /// Runs a failure's action; the desktop then dismisses the failure.
    pub(crate) fn overlay_action(self, section: MainWindowSection) {
        self.overlay_command(
            UiOperation::OverlayAction,
            client::invoke("open_main_window", section),
        );
    }
}
