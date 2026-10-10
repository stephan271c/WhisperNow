use std::future::Future;
use whispernow_domain::AppError;

use whispernow_domain::ipc::{MainWindowSection, Phase};

use crate::UiContext;
use crate::UiOperation;
use crate::client;

use crate::overlay::OverlayAction;
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
    pub(crate) fn overlay_action(self, action: OverlayAction) {
        let section = match action.route {
            crate::Route::History => MainWindowSection::History,
            crate::Route::Settings(crate::SettingsRoute::General) => MainWindowSection::Settings,
            crate::Route::Settings(crate::SettingsRoute::Audio) => MainWindowSection::Audio,
            crate::Route::Settings(crate::SettingsRoute::SpeechModels) => {
                MainWindowSection::SpeechModels
            }
            crate::Route::Settings(crate::SettingsRoute::AiCleanup) => MainWindowSection::AiCleanup,
            _ => unreachable!("error fix must name an available main window section"),
        };
        self.overlay_command(UiOperation::OverlayAction, async move {
            client::invoke("open_main_window", Some(&section)).await
        });
    }
}
