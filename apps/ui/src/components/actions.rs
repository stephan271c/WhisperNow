//! One path from a control to the desktop: mark busy, call the client, apply the answer. A
//! failed call is explained by a notice in the scope of the control that made it, unless the
//! control shows the failure itself.

mod history;
mod models;
mod overlay;
mod settings;
mod setup;
mod updates;

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) use setup::PERMISSION_POLLS;

use std::future::Future;
use whispernow_domain::AppError;

use dioxus::core::spawn_forever;
use dioxus::prelude::*;
use whispernow_domain::ClearDataCategory;
use whispernow_domain::ipc::{EventPayload, PermissionKind, Phase};

use super::ui::Tone;
use crate::client;

use crate::{BlockedReason, NoticeScope, Route, UiContext, UiOperation};

impl UiContext {
    pub(crate) fn blocked(self, operation: UiOperation) -> Option<BlockedReason> {
        if self.state.read().snapshot.is_none() {
            Some(BlockedReason::Connecting)
        } else {
            self.management
                .read()
                .is_busy(operation)
                .then_some(BlockedReason::CommandPending)
        }
    }

    pub(crate) fn phase_blocked(self) -> Option<BlockedReason> {
        match self
            .state
            .read()
            .snapshot
            .as_ref()
            .map(|snapshot| &snapshot.phase)
        {
            None => Some(BlockedReason::Connecting),
            Some(phase) if phase.is_busy() => Some(BlockedReason::Dictating),
            Some(Phase::Paused) => Some(BlockedReason::Paused),
            Some(Phase::Updating) => Some(BlockedReason::UpdateInstalling),
            Some(_) => None,
        }
    }

    pub(crate) fn download(
        self,
        model_id: &whispernow_domain::ModelId,
    ) -> Option<whispernow_domain::ipc::ActiveDownload> {
        self.state
            .read()
            .snapshot
            .as_ref()?
            .active_downloads
            .iter()
            .find(|download| download.model_id == *model_id)
            .cloned()
    }

    /// Whether a request for permission `kind` is waiting for the system's answer.
    pub(crate) fn permission_requested(self, kind: PermissionKind) -> bool {
        self.management.read().requested_permission == Some(kind)
    }

    // Actions read state with `peek`: one called from an effect must not subscribe it.

    /// Runs `call` with `operation` busy until it is answered, then hands the answer to
    /// `finish`. The call outlives the control that made it.
    fn run<R: 'static>(
        mut self,
        operation: UiOperation,
        call: impl Future<Output = Result<R, AppError>> + 'static,
        finish: impl FnOnce(Self, Result<R, AppError>) + 'static,
    ) {
        self.management.write().begin_busy(operation);
        spawn_forever(async move {
            let answer = call.await;
            self.management.write().end_busy(operation);
            finish(self, answer);
        });
    }

    fn report(mut self, operation: UiOperation, error: &AppError) {
        self.management.write().notify(
            Tone::Error,
            operation.notice_scope(),
            error.summary.clone(),
        );
    }

    /// Reads everything this window shows: the state snapshot, settings, setup, models and
    /// history. Used at startup and after clearing data.
    pub(crate) fn refresh_all(mut self) {
        spawn_forever(async move {
            match client::call("get_state").await {
                Ok(snapshot) => self.state.write().hydrate(snapshot),
                Err(error) => self.state.write().report_client_error(error),
            }
        });
        spawn_forever(async move {
            match client::call("get_settings").await {
                Ok(settings) => self.management.write().install_settings(settings),
                Err(error) => self.load_failed("Settings", &error),
            }
        });
        spawn_forever(async move {
            match client::call("get_setup_state").await {
                Ok(setup) => {
                    self.management.write().setup = Some(setup);
                    if self.management.peek().may_list_devices() {
                        self.list_devices();
                    }
                }
                Err(error) => self.load_failed("Setup state", &error),
            }
        });
        self.reload_models();
        self.refresh_history();
    }

    fn load_failed(mut self, what: &str, error: &AppError) {
        self.management.write().notify(
            Tone::Error,
            NoticeScope::Global,
            format!("{what} could not be loaded: {}", error.summary.clone()),
        );
    }

    #[cfg_attr(
        not(any(test, target_arch = "wasm32")),
        expect(dead_code, reason = "called by the browser's listener and by tests")
    )]
    pub(crate) fn apply_event(mut self, event: EventPayload) {
        match &event {
            EventPayload::StateChanged { state } => {
                let download_ended = self.state.peek().snapshot.as_ref().is_some_and(|old| {
                    old.active_downloads.iter().any(|download| {
                        !state
                            .active_downloads
                            .iter()
                            .any(|active| active.model_id == download.model_id)
                    })
                });
                self.management.write().observe_setup_test(state);
                if download_ended {
                    self.reload_models();
                }
            }
            EventPayload::HistoryChanged => self.refresh_history(),
            EventPayload::NavigationRequested { section } => {
                self.navigator.go(Route::from_section(*section));
            }
            _ => {}
        }
        self.state.write().apply_event(event);
    }

    pub(crate) fn copy_text(mut self, text: String, scope: NoticeScope) {
        spawn_forever(async move {
            let copied = client::write_clipboard(&text).await.is_ok();
            self.management.write().finish_copy(scope, copied);
        });
    }

    pub(crate) fn test_microphone(self) {
        let operation = UiOperation::TestMicrophone;
        {
            let mut management = self.management;
            let mut management = management.write();
            management.microphone_test = None;
            management.clear_notice_scope(NoticeScope::Home);
        }
        self.run(
            operation,
            client::call("test_microphone"),
            move |mut actions, answer| match answer {
                Ok(result) => actions.management.write().finish_microphone_test(result),
                Err(error) => actions.report(operation, &error),
            },
        );
    }

    pub(crate) fn review_data(mut self) {
        let operation = UiOperation::ListDataLocations;
        self.management.write().clear_result = None;
        self.run(
            operation,
            client::call("data_locations"),
            move |mut actions, answer| match answer {
                Ok(locations) => actions.management.write().data_locations = Some(locations),
                Err(error) => actions.report(operation, &error),
            },
        );
    }

    /// Removes every kind of application data, then reads everything again.
    pub(crate) fn clear_data(self) {
        let operation = UiOperation::ClearData;
        let request = ClearDataCategory::ALL.to_vec();
        self.run(
            operation,
            async move { client::invoke("clear_data", Some(&request)).await },
            move |mut actions, answer| {
                match answer {
                    Ok(result) => actions.management.write().apply_clear_result(result),
                    Err(error) => actions.report(operation, &error),
                }
                actions.refresh_all();
            },
        );
    }
}

#[component]
pub(crate) fn DisabledReason(id: String, reason: Option<BlockedReason>) -> Element {
    let Some(reason) = reason else {
        return rsx! {};
    };
    rsx! { p { id, class: "disabled-reason", "{reason.message()}" } }
}
