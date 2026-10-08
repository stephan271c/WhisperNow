use std::future::Future;

use whispernow_api::{MainWindowSectionDto, OpenMainWindowRequest};

use super::UiActions;
use crate::UiOperation;
use crate::client::{self, ClientError};
use crate::overlay::OverlayAction;
use dioxus::prelude::*;

impl UiActions {
    /// Runs an overlay control; the overlay shows a failure in its own status line.
    fn overlay_command(
        self,
        operation: UiOperation,
        call: impl Future<Output = Result<(), ClientError>> + 'static,
    ) {
        self.run(operation, call, move |actions, answer| {
            let Err(error) = answer else {
                return;
            };
            let mut state = actions.state();
            let mut state = state.write();
            if operation == UiOperation::OverlayAction {
                state.overlay.report_action_failure(error.summary());
            } else {
                state
                    .overlay
                    .report_recording_command_failure(error.summary());
            }
        });
    }

    pub(crate) fn stop_recording(self) {
        self.overlay_command(UiOperation::StopRecording, client::stop_recording());
    }

    pub(crate) fn cancel_recording(self) {
        self.overlay_command(UiOperation::CancelRecording, client::cancel_recording());
    }

    pub(crate) fn dismiss_overlay(self) {
        self.overlay_command(
            UiOperation::OverlayAction,
            client::dismiss_recording_overlay(),
        );
    }

    /// Runs a failure's action; the desktop then dismisses the failure.
    pub(crate) fn overlay_action(self, action: OverlayAction) {
        let section = match action {
            OverlayAction::CopyTranscript => {
                self.overlay_command(UiOperation::OverlayAction, client::copy_last_transcript());
                return;
            }
            OverlayAction::OpenHistory => MainWindowSectionDto::History,
            OverlayAction::OpenSettings => MainWindowSectionDto::Settings,
        };
        self.overlay_command(UiOperation::OverlayAction, async move {
            client::open_main_window(&OpenMainWindowRequest { section }).await
        });
    }
}
