use std::future::Future;
use whispernow_domain::AppError;

use whispernow_domain::ipc::UpdateState;

use crate::UiContext;
use crate::client;
use crate::operations::UiOperation;

use dioxus::prelude::*;

impl UiContext {
    fn update_command(
        self,
        operation: UiOperation,
        call: impl Future<Output = Result<UpdateState, AppError>> + 'static,
    ) {
        self.run(operation, call, |mut actions, answer| match answer {
            Ok(update) => actions.state.write().apply_update_state(update),
            Err(error) => actions
                .state
                .write()
                .report_update_command_failure(error.summary),
        });
    }

    pub(crate) fn check_for_updates(self) {
        self.update_command(
            UiOperation::CheckForUpdates,
            client::call("check_for_updates"),
        );
    }

    pub(crate) fn dismiss_update(self) {
        self.update_command(UiOperation::DismissUpdate, client::call("dismiss_update"));
    }

    pub(crate) fn open_update_release_page(self) {
        self.update_command(
            UiOperation::OpenUpdateReleasePage,
            client::call("open_update_release_page"),
        );
    }

    pub(crate) fn restart_to_update(self) {
        self.update_command(
            UiOperation::UpdateInstallRestart,
            client::call("restart_to_update"),
        );
    }
}
