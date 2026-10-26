use std::future::Future;
use whispernow_domain::{AppError, ErrorCategory};

use dioxus::core::spawn_forever;
use dioxus::prelude::*;
use whispernow_domain::{EntryId, HistoryPageRequest};

use crate::UiContext;
use crate::client;
use crate::management::replace_task;
use crate::notice::NoticeScope;
use crate::operations::UiOperation;

impl UiContext {
    pub(crate) fn refresh_history(self) {
        let request = self.management.peek().history_request();
        self.read_history(request);
    }

    pub(crate) fn load_older_history(mut self) {
        let request = self.management.write().older_history_request();
        self.read_history(request);
    }

    /// Reads with the same filter can finish out of order after Load more or deletion, so a
    /// new read cancels the one in flight.
    fn read_history(self, request: HistoryPageRequest) {
        let (mut management, mut background) = (self.management, self.background);
        let read = spawn_forever(async move {
            let answer = client::invoke("get_history", request).await;
            background.write().history_read = None;
            match answer {
                Ok(page) if management.peek().shows_history(&page) => {}
                Ok(page) => management.write().finish_history_refresh(page),
                Err(error) => management.write().fail_history_refresh(&error),
            }
        });
        replace_task(&mut background.write().history_read, Some(read));
    }

    pub(crate) fn search_history(mut self, text: &str) {
        if self.management.write().apply_history_search(text) {
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

    pub(crate) fn delete_history_entry(self, id: EntryId) {
        self.change_history(client::invoke("delete_history_entry", id));
    }

    pub(crate) fn clear_history(self) {
        self.change_history(client::call("clear_history"));
    }

    /// The desktop announces every change and failure with `HistoryChanged`, which reads the
    /// window again, so a row another surface already removed is not an error.
    fn change_history(self, call: impl Future<Output = Result<(), AppError>> + 'static) {
        self.run(UiOperation::ChangeHistory, call, move |actions, answer| {
            if let Err(error) = answer
                && error.category != ErrorCategory::NotFound
            {
                actions.report(NoticeScope::History, &error);
            }
        });
    }

    pub(crate) fn open_recordings_folder(self) {
        self.run_reporting(
            UiOperation::OpenRecordingsFolder,
            client::call("open_recordings_folder"),
            NoticeScope::History,
        );
    }
}
