use whispernow_domain::AppError;
use whispernow_domain::ipc::ErrorCategory;
use whispernow_domain::{HistoryFilter, HistoryPage, HistoryPageRequest};

use super::ManagementState;
use crate::NoticeScope;
use crate::components::ui::Tone;

impl ManagementState {
    pub(crate) fn history_request(&mut self) -> (u64, HistoryPageRequest) {
        // Reads with the same filter can finish out of order after Load more or deletion.
        self.history_request_id += 1;
        let request = HistoryPageRequest {
            limit: self
                .history_window_pages()
                .saturating_mul(HistoryPageRequest::DEFAULT_PAGE_SIZE),
            ..HistoryPageRequest::first_page(self.history_filter.clone())
        };
        (self.history_request_id, request)
    }

    /// Grows the window by one page and returns the read for it.
    pub(crate) fn older_history_request(&mut self) -> (u64, HistoryPageRequest) {
        self.history_pages = self.history_window_pages().saturating_add(1);
        self.history_request()
    }

    #[must_use]
    pub const fn history_loaded(&self) -> bool {
        self.history_rows_filter.is_some()
    }

    #[must_use]
    pub fn history_window_pages(&self) -> u32 {
        self.history_pages.max(1)
    }

    pub fn finish_history_refresh(&mut self, request_id: u64, page: HistoryPage) {
        if request_id != self.history_request_id {
            return;
        }
        self.history_unavailable = None;
        self.history_search_error = None;
        self.history_rows_filter = Some(self.history_filter.clone());
        self.history = page.entries;
        self.history_has_more = page.has_more;
    }

    pub fn fail_history_refresh(&mut self, request_id: u64, error: &AppError) {
        if request_id != self.history_request_id {
            return;
        }
        let summary = error.summary.clone();
        let unreadable = error.category == ErrorCategory::HistoryUnavailable;
        if !self.history_loaded() || unreadable {
            // Degraded mode: show the reason instead of rows that can no longer be confirmed.
            self.history.clear();
            self.history_rows_filter = None;
            self.history_has_more = false;
            self.history_unavailable = Some(summary);
        } else if self.history_rows_filter.as_ref() != Some(&self.history_filter) {
            // The rows answer an earlier search; showing them under this one would mislead.
            self.history.clear();
            self.history_has_more = false;
            self.history_search_error = Some(summary);
        } else {
            self.notify(Tone::Error, NoticeScope::History, summary);
        }
    }

    /// Applies the typed search text; returns whether the window must be read again from its
    /// first page.
    pub fn apply_history_search(&mut self) -> bool {
        if self.history_search == self.history_filter.query {
            return false;
        }
        self.history_filter.query.clone_from(&self.history_search);
        self.history_pages = 1;
        true
    }

    /// Shows only failed entries, or all of them; returns whether the filter changed.
    pub fn show_failed_history_only(&mut self, failed_only: bool) -> bool {
        if self.history_filter.failed_only == failed_only {
            return false;
        }
        self.history_filter.failed_only = failed_only;
        self.history_pages = 1;
        true
    }

    /// Removes the search text and the failed filter; returns whether anything changed.
    pub fn clear_history_filter(&mut self) -> bool {
        self.history_search.clear();
        if self.history_filter == HistoryFilter::default() {
            return false;
        }
        self.history_filter = HistoryFilter::default();
        self.history_pages = 1;
        true
    }
}
