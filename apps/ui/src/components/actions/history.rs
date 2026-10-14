use std::future::Future;
use whispernow_domain::AppError;

use dioxus::core::spawn_forever;
use dioxus::prelude::*;
use whispernow_domain::ipc::ErrorCategory;
use whispernow_domain::{EntryId, HistoryPageRequest};

use crate::client;
use crate::{NoticeScope, UiContext};

use crate::{HistoryCopied, UiOperation};

const COPIED_MS: u32 = 2_000;

impl UiContext {
    pub(crate) fn refresh_history(mut self) {
        let (request_id, request) = self.management.write().history_request();
        self.read_history(request_id, request);
    }

    pub(crate) fn load_older_history(mut self) {
        let (request_id, request) = self.management.write().older_history_request();
        self.read_history(request_id, request);
    }

    fn read_history(self, request_id: u64, request: HistoryPageRequest) {
        spawn_forever(async move {
            let answer = client::invoke("get_history", Some(&request)).await;
            let mut management = self.management;
            match answer {
                Ok(page) => management.write().finish_history_refresh(request_id, page),
                Err(error) => management.write().fail_history_refresh(request_id, &error),
            }
        });
    }

    /// Searches for the typed search text, unless it is already the search shown.
    pub(crate) fn search_history(mut self) {
        if self.management.write().apply_history_search() {
            self.refresh_history();
        }
    }

    pub(crate) fn show_failed_history_only(mut self, failed_only: bool) {
        if self
            .management
            .write()
            .show_failed_history_only(failed_only)
        {
            self.refresh_history();
        }
    }

    pub(crate) fn clear_history_filter(mut self) {
        if self.management.write().clear_history_filter() {
            self.refresh_history();
        }
    }

    /// Copies one entry's text and shows the outcome on its Copy button for a moment; a newer
    /// copy replaces it.
    pub(crate) fn copy_history_entry(mut self, entry: EntryId) {
        let Some(text) = self.management.peek().history_copy(entry) else {
            return;
        };
        let copy_task = spawn_forever(async move {
            let succeeded = client::write_clipboard(&text).await.is_ok();
            self.management.write().history_copied = Some(HistoryCopied { entry, succeeded });
            client::delay(COPIED_MS).await;
            let mut management = self.management;
            let mut management = management.write();
            management.history_copied = None;
            management.history_copy_task = None;
        });
        let mut management = self.management;
        let mut management = management.write();
        management.history_copied = None;
        if let Some(previous) = management.history_copy_task.replace(copy_task) {
            previous.cancel();
        }
    }

    pub(crate) fn delete_history_entry(self, id: EntryId) {
        self.change_history(async move { client::invoke("delete_history_entry", Some(&id)).await });
    }

    pub(crate) fn clear_history(self) {
        self.change_history(client::call("clear_history"));
    }

    /// Changes history, then reads it again whatever the answer: last write wins, and a row
    /// another surface already removed is the refresh's result rather than an error.
    fn change_history(self, call: impl Future<Output = Result<(), AppError>> + 'static) {
        let operation = UiOperation::ChangeHistory;
        self.run(operation, call, move |actions, answer| {
            actions.refresh_history();
            if let Err(error) = answer
                && error.category != ErrorCategory::NotFound
            {
                actions.report(NoticeScope::History, &error);
            }
        });
    }

    pub(crate) fn open_recordings_folder(self) {
        let operation = UiOperation::OpenRecordingsFolder;
        self.run(
            operation,
            client::call::<()>("open_recordings_folder"),
            move |actions, answer| {
                if let Err(error) = answer {
                    actions.report(NoticeScope::History, &error);
                }
            },
        );
    }
}
