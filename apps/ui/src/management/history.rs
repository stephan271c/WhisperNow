use whispernow_domain::{AppError, ErrorCategory};
use whispernow_domain::{HistoryEntry, HistoryFilter, HistoryPage, HistoryPageRequest};

use super::ManagementState;
use crate::NoticeScope;
use crate::components::ui::Tone;

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) enum HistoryRows {
    #[default]
    Loading,
    /// The store cannot be read; the reason replaces the page.
    Unavailable(String),
    /// The current search could not be read; the reason replaces the list.
    SearchFailed(String),
    Loaded {
        /// The search and failed filter the rows were read with.
        filter: HistoryFilter,
        entries: Vec<HistoryEntry>,
        has_more: bool,
    },
}

impl HistoryRows {
    #[must_use]
    pub fn entries(&self) -> &[HistoryEntry] {
        match self {
            Self::Loaded { entries, .. } => entries,
            _ => &[],
        }
    }

    #[must_use]
    pub const fn has_more(&self) -> bool {
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

    /// Grows the window by one page and returns the read for it.
    pub(crate) fn older_history_request(&mut self) -> HistoryPageRequest {
        self.history_extra_pages = self.history_extra_pages.saturating_add(1);
        self.history_request()
    }

    #[must_use]
    pub const fn history_window_pages(&self) -> u32 {
        self.history_extra_pages.saturating_add(1)
    }

    pub fn finish_history_refresh(&mut self, page: HistoryPage) {
        self.history = HistoryRows::Loaded {
            filter: self.history_filter.clone(),
            entries: page.entries,
            has_more: page.has_more,
        };
    }

    pub fn fail_history_refresh(&mut self, error: &AppError) {
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
    pub fn apply_history_search(&mut self) -> bool {
        if self.history_search == self.history_filter.query {
            return false;
        }
        self.history_filter.query.clone_from(&self.history_search);
        self.history_extra_pages = 0;
        true
    }

    /// Shows only failed entries, or all of them; returns whether the filter changed.
    pub fn show_failed_history_only(&mut self, failed_only: bool) -> bool {
        if self.history_filter.failed_only == failed_only {
            return false;
        }
        self.history_filter.failed_only = failed_only;
        self.history_extra_pages = 0;
        true
    }

    /// Removes the search text and the failed filter; returns whether anything changed.
    pub fn clear_history_filter(&mut self) -> bool {
        self.history_search.clear();
        if self.history_filter == HistoryFilter::default() {
            return false;
        }
        self.history_filter = HistoryFilter::default();
        self.history_extra_pages = 0;
        true
    }
}
