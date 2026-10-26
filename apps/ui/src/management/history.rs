use whispernow_domain::{AppError, ErrorCategory};
use whispernow_domain::{HistoryEntry, HistoryFilter, HistoryPage, HistoryPageRequest};

use super::ManagementState;
use crate::notice::{NoticeScope, Tone};

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) enum HistoryRows {
    #[default]
    Loading,
    /// The store cannot be read; the reason replaces the page.
    Unavailable(String),
    /// The current search could not be read; the reason replaces the list.
    SearchFailed(String),
    Loaded {
        filter: HistoryFilter,
        entries: Vec<HistoryEntry>,
        has_more: bool,
    },
}

impl HistoryRows {
    #[must_use]
    pub(crate) fn entries(&self) -> &[HistoryEntry] {
        match self {
            Self::Loaded { entries, .. } => entries,
            _ => &[],
        }
    }

    #[must_use]
    pub(crate) const fn has_more(&self) -> bool {
        matches!(self, Self::Loaded { has_more: true, .. })
    }
}

impl ManagementState {
    #[must_use]
    pub(crate) fn history_request(&self) -> HistoryPageRequest {
        HistoryPageRequest {
            limit: self
                .history_window_pages()
                .saturating_mul(HistoryPageRequest::DEFAULT_PAGE_SIZE),
            ..HistoryPageRequest::first_page(self.history_filter.clone())
        }
    }

    pub(crate) fn older_history_request(&mut self) -> HistoryPageRequest {
        self.history_extra_pages = self.history_extra_pages.saturating_add(1);
        self.history_request()
    }

    #[must_use]
    pub(crate) const fn history_window_pages(&self) -> u32 {
        self.history_extra_pages.saturating_add(1)
    }

    /// Whether the rows shown are already `page`'s, read for the current filter.
    #[must_use]
    pub(crate) fn shows_history(&self, page: &HistoryPage) -> bool {
        matches!(
            &self.history,
            HistoryRows::Loaded { filter, entries, has_more }
                if *filter == self.history_filter
                    && *entries == page.entries
                    && *has_more == page.has_more
        )
    }

    pub(crate) fn finish_history_refresh(&mut self, page: HistoryPage) {
        self.history = HistoryRows::Loaded {
            filter: self.history_filter.clone(),
            entries: page.entries,
            has_more: page.has_more,
        };
    }

    pub(crate) fn fail_history_refresh(&mut self, error: &AppError) {
        let summary = error.summary.clone();
        self.history = match &self.history {
            // Rows that can no longer be confirmed give way to the reason.
            _ if error.category == ErrorCategory::HistoryUnavailable => {
                HistoryRows::Unavailable(summary)
            }
            HistoryRows::Loading | HistoryRows::Unavailable(_) => HistoryRows::Unavailable(summary),
            HistoryRows::Loaded { filter, .. } if *filter == self.history_filter => {
                self.notify(Tone::Error, NoticeScope::History, summary);
                return;
            }
            // Rows read for another search would mislead under this one.
            HistoryRows::Loaded { .. } | HistoryRows::SearchFailed(_) => {
                HistoryRows::SearchFailed(summary)
            }
        };
    }

    /// Applies the typed search text; returns whether the window must be read again from its
    /// first page.
    pub(crate) fn apply_history_search(&mut self, text: &str) -> bool {
        if text == self.history_filter.query {
            return false;
        }
        text.clone_into(&mut self.history_filter.query);
        self.history_extra_pages = 0;
        true
    }

    /// Shows only failed entries, or all of them; returns whether the filter changed.
    pub(crate) fn show_failed_history_only(&mut self, failed_only: bool) -> bool {
        if self.history_filter.failed_only == failed_only {
            return false;
        }
        self.history_filter.failed_only = failed_only;
        self.history_extra_pages = 0;
        true
    }

    /// Removes the search text and the failed filter; returns whether anything changed.
    pub(crate) fn clear_history_filter(&mut self) -> bool {
        if self.history_filter == HistoryFilter::default() {
            return false;
        }
        self.history_filter = HistoryFilter::default();
        self.history_extra_pages = 0;
        true
    }
}
