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
use whispernow_domain::ipc::EventPayload;

use crate::client;

use crate::UiContext;
use crate::management::replace_task;
use crate::notice::{NoticeScope, Tone};
use crate::operations::UiOperation;
use crate::route::Route;

const METER_MS: u32 = 20_000;

impl UiContext {
    pub(crate) fn active_download(
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

    // Actions read state with `peek`: one called from an effect must not subscribe it.

    /// Runs `call` with `operation` busy until it is answered, then hands the answer to
    /// `finish`. The call outlives the control that made it.
    fn run<R: 'static>(
        mut self,
        operation: UiOperation,
        call: impl Future<Output = Result<R, AppError>> + 'static,
        finish: impl FnOnce(Self, Result<R, AppError>) + 'static,
    ) {
        if !self.management.write().begin_busy(operation) {
            return;
        }
        spawn_forever(async move {
            let answer = call.await;
            self.management.write().end_busy(operation);
            finish(self, answer);
        });
    }

    fn run_reporting(
        self,
        operation: UiOperation,
        call: impl Future<Output = Result<(), AppError>> + 'static,
        scope: NoticeScope,
    ) {
        self.run(operation, call, move |actions, answer| {
            if let Err(error) = answer {
                actions.report(scope, &error);
            }
        });
    }

    fn run_unless_busy<R: 'static>(
        self,
        operation: UiOperation,
        call: impl Future<Output = Result<R, AppError>> + 'static,
        finish: impl FnOnce(Self, Result<R, AppError>) + 'static,
    ) {
        if !self.management.peek().is_busy(operation) {
            self.run(operation, call, finish);
        }
    }

    fn report(mut self, scope: NoticeScope, error: &AppError) {
        self.management
            .write()
            .notify(Tone::Error, scope, error.summary.clone());
    }

    pub(crate) async fn hydrate_state(mut self) {
        match client::call("get_state").await {
            Ok(snapshot) => self.state.write().hydrate(snapshot),
            Err(_) => self.state.write().report_state_read_failure(),
        }
    }

    /// Reads everything this window shows: the state snapshot, settings and the setup they
    /// hold, models and history. Used at startup and after clearing data.
    pub(crate) fn refresh_all(self) {
        spawn_forever(async move {
            self.hydrate_state().await;
        });
        self.run(
            UiOperation::LoadSettings,
            client::call("get_settings"),
            |mut actions, answer| match answer {
                Ok(settings) => {
                    actions.management.write().install_setup(settings);
                    if actions.management.peek().may_list_devices() {
                        actions.list_devices();
                    }
                }
                Err(error) => actions.load_failed("Settings", &error),
            },
        );
        self.reload_models();
        self.refresh_history();
    }

    fn load_failed(mut self, what: &str, error: &AppError) {
        self.management.write().notify(
            Tone::Error,
            NoticeScope::Global,
            format!("{what} could not be loaded: {}", error.summary),
        );
    }

    pub(crate) fn apply_event(mut self, event: EventPayload) {
        match event {
            EventPayload::StateChanged { state } => {
                // A write re-renders everything that reads `management`, so take one only
                // when the snapshot can end a test.
                if self.management.peek().setup_test_running() {
                    self.management.write().observe_setup_test(&state);
                }
                self.state.write().apply_state(state);
            }
            EventPayload::UpdateChanged { state } => self.state.write().apply_update_state(state),
            EventPayload::HistoryChanged => self.refresh_history(),
            EventPayload::ModelsChanged => self.reload_models(),
            EventPayload::NavigationRequested { section } => {
                self.navigator.go(Route::from(section));
            }
        }
    }

    pub(crate) fn test_microphone(self) {
        {
            let mut management = self.management;
            let mut management = management.write();
            management.microphone_test = None;
            replace_task(&mut management.microphone_test_task, None);
            management.clear_notice_scope(NoticeScope::Home);
        }
        self.run(
            UiOperation::TestMicrophone,
            client::call("test_microphone"),
            move |mut actions, answer| match answer {
                Ok(result) => {
                    actions.management.write().finish_microphone_test(result);
                    actions.hide_meter_later();
                }
                Err(error) => actions.report(NoticeScope::Home, &error),
            },
        );
    }

    /// Hides the level meter a moment after the test that showed it; a newer test
    /// cancels the hide.
    fn hide_meter_later(self) {
        let mut management = self.management;
        let hide_task = spawn_forever(async move {
            client::delay(METER_MS).await;
            let mut management = management.write();
            management.microphone_test = None;
            management.microphone_test_task = None;
        });
        replace_task(
            &mut management.write().microphone_test_task,
            Some(hide_task),
        );
    }

    pub(crate) fn review_data(mut self) {
        self.management.write().clear_result = None;
        self.run(
            UiOperation::ListDataLocations,
            client::call("data_locations"),
            move |mut actions, answer| match answer {
                Ok(locations) => actions.management.write().data_locations = Some(locations),
                Err(error) => actions.report(NoticeScope::Data, &error),
            },
        );
    }

    pub(crate) fn clear_data(self) {
        self.run(
            UiOperation::ClearData,
            client::call("clear_data"),
            move |mut actions, answer| {
                match answer {
                    Ok(result) => actions.management.write().apply_clear_result(result),
                    Err(error) => actions.report(NoticeScope::Data, &error),
                }
                actions.refresh_all();
            },
        );
    }
}
