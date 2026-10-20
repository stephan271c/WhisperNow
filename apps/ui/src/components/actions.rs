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
use whispernow_domain::ipc::EventPayload;

use super::ui::Tone;
use crate::client;

use crate::{Copied, CopyTarget, NoticeScope, Route, UiContext, UiOperation};

/// How long the level meter stays up after a microphone test.
const METER_MS: u32 = 20_000;
const COPIED_MS: u32 = 2_000;

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

    fn report(mut self, scope: NoticeScope, error: &AppError) {
        self.management
            .write()
            .notify(Tone::Error, scope, error.summary.clone());
    }

    /// Reads everything this window shows: the state snapshot, settings, setup, models and
    /// history. Used at startup and after clearing data.
    pub(crate) fn refresh_all(mut self) {
        spawn_forever(async move {
            self.hydrate_state().await;
        });
        self.run(
            UiOperation::LoadSettings,
            client::call("get_settings"),
            |mut actions, answer| match answer {
                Ok(settings) => actions.management.write().install_settings(settings),
                Err(error) => actions.load_failed("Settings", &error),
            },
        );
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

    /// Copies `text` and shows the outcome on the Copy button for `target` for a moment; a
    /// newer copy replaces it.
    pub(crate) fn copy_text(mut self, text: String, target: CopyTarget) {
        let copy_task = spawn_forever(async move {
            let succeeded = client::invoke::<_, ()>("copy_text", Some(&text))
                .await
                .is_ok();
            self.management.write().copied = Some(Copied { target, succeeded });
            client::delay(COPIED_MS).await;
            let mut management = self.management;
            let mut management = management.write();
            management.copied = None;
            management.copy_task = None;
        });
        let mut management = self.management;
        let mut management = management.write();
        management.copied = None;
        if let Some(previous) = management.copy_task.replace(copy_task) {
            previous.cancel();
        }
    }

    pub(crate) fn test_microphone(self) {
        let operation = UiOperation::TestMicrophone;
        {
            let mut management = self.management;
            let mut management = management.write();
            management.microphone_test = None;
            if let Some(previous) = management.microphone_test_task.take() {
                previous.cancel();
            }
            management.clear_notice_scope(NoticeScope::Home);
        }
        self.run(
            operation,
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
        if let Some(previous) = management.write().microphone_test_task.replace(hide_task) {
            previous.cancel();
        }
    }

    pub(crate) fn review_data(mut self) {
        let operation = UiOperation::ListDataLocations;
        self.management.write().clear_result = None;
        self.run(
            operation,
            client::call("data_locations"),
            move |mut actions, answer| match answer {
                Ok(locations) => actions.management.write().data_locations = Some(locations),
                Err(error) => actions.report(NoticeScope::Data, &error),
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
                    Err(error) => actions.report(NoticeScope::Data, &error),
                }
                actions.refresh_all();
            },
        );
    }
}
