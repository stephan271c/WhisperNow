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
use whispernow_domain::ipc::{EventPayload, PermissionKind};

use super::ui::Tone;
use crate::client;

use crate::{BlockedReason, ManagementState, NoticeScope, Route, UiContext, UiOperation, UiState};

/// The window's command facade; cheap to copy into event handlers.
#[derive(Clone, Copy, PartialEq)]
pub(crate) struct UiActions {
    context: UiContext,
}

pub(crate) fn use_actions() -> UiActions {
    UiActions::new(use_context::<UiContext>())
}

impl UiActions {
    pub(crate) const fn new(context: UiContext) -> Self {
        Self { context }
    }

    fn management(self) -> Signal<ManagementState> {
        self.context.management
    }

    fn state(self) -> Signal<UiState> {
        self.context.state
    }

    pub(crate) fn blocked(self, operation: UiOperation) -> Option<BlockedReason> {
        crate::blocked(
            operation,
            self.context.state.read().snapshot.as_ref(),
            &self.context.management.read(),
        )
    }

    /// Whether a request for permission `kind` is waiting for the system's answer.
    pub(crate) fn permission_requested(self, kind: PermissionKind) -> bool {
        self.context.management.read().requested_permission == Some(kind)
    }

    // Actions read state with `peek`: one called from an effect must not subscribe it.

    /// Runs `call` with `operation` busy until it is answered, then hands the answer to
    /// `finish`. The call outlives the control that made it.
    fn run<R: 'static>(
        self,
        operation: UiOperation,
        call: impl Future<Output = Result<R, AppError>> + 'static,
        finish: impl FnOnce(Self, Result<R, AppError>) + 'static,
    ) {
        self.management().write().begin_busy(operation);
        spawn_forever(async move {
            let answer = call.await;
            self.management().write().end_busy(operation);
            finish(self, answer);
        });
    }

    fn report(self, operation: UiOperation, error: &AppError) {
        self.management().write().notify(
            Tone::Error,
            operation.notice_scope(),
            error.summary.clone(),
        );
    }

    /// Reads everything this window shows: the state snapshot, settings, setup, models and
    /// history. Used at startup and after clearing data.
    pub(crate) fn refresh_all(self) {
        spawn_forever(async move {
            match client::get_state().await {
                // An event that arrived first is newer and wins.
                Ok(snapshot) => {
                    if self.state().peek().snapshot.is_none() {
                        self.management().write().install_snapshot(&snapshot);
                    }
                    self.state().write().hydrate(snapshot);
                }
                Err(error) => self.state().write().report_client_error(error),
            }
        });
        spawn_forever(async move {
            match client::get_settings().await {
                Ok(settings) => self.management().write().install_settings(settings),
                Err(error) => self.load_failed("Settings", &error),
            }
        });
        spawn_forever(async move {
            match client::get_setup_state().await {
                Ok(setup) => {
                    self.management().write().install_setup(setup);
                    if self.management().peek().may_list_devices() {
                        self.list_devices();
                    }
                }
                Err(error) => self.load_failed("Setup state", &error),
            }
        });
        self.reload_models();
        self.refresh_history();
    }

    fn load_failed(self, what: &str, error: &AppError) {
        self.management().write().notify(
            Tone::Error,
            NoticeScope::Global,
            format!("{what} could not be loaded: {}", error.summary.clone()),
        );
    }

    /// A failed history read is retried once the workflow returns to idle, so a change
    /// published while the read could not be served still reaches the history window.
    #[cfg_attr(
        not(any(test, target_arch = "wasm32")),
        expect(dead_code, reason = "called by the browser's listener and by tests")
    )]
    pub(crate) fn apply_event(self, event: EventPayload) {
        let was_idle = self
            .state()
            .peek()
            .snapshot
            .as_ref()
            .is_none_or(|state| !state.phase.is_busy());
        if self.management().write().apply_event(&event) {
            self.reload_models();
        }
        match &event {
            EventPayload::StateChanged { state }
                if !was_idle
                    && !state.phase.is_busy()
                    && self.management().peek().history_refresh_failed() =>
            {
                self.refresh_history();
            }
            EventPayload::HistoryChanged => self.refresh_history(),
            EventPayload::NavigationRequested { section } => {
                self.context.navigator.go(Route::from_section(*section));
            }
            _ => {}
        }
        self.state().write().apply_event(event);
    }

    pub(crate) fn copy_text(self, text: String, scope: NoticeScope) {
        spawn_forever(async move {
            let copied = client::write_clipboard(&text).await.is_ok();
            self.management().write().finish_copy(scope, copied);
        });
    }

    pub(crate) fn test_microphone(self) {
        let operation = UiOperation::TestMicrophone;
        {
            let mut management = self.management();
            let mut management = management.write();
            management.microphone_test = None;
            management.clear_notice_scope(NoticeScope::Home);
        }
        self.run(
            operation,
            client::test_microphone(),
            move |actions, answer| match answer {
                Ok(result) => actions.management().write().finish_microphone_test(result),
                Err(error) => actions.report(operation, &error),
            },
        );
    }

    pub(crate) fn review_data(self) {
        let operation = UiOperation::ListDataLocations;
        self.management().write().clear_result = None;
        self.run(
            operation,
            client::data_locations(),
            move |actions, answer| match answer {
                Ok(locations) => actions.management().write().data_locations = Some(locations),
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
            async move { client::clear_data(&request).await },
            move |actions, answer| {
                match answer {
                    Ok(result) => actions.management().write().apply_clear_result(result),
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
