mod audio;
mod format;
mod row;

use dioxus::core::Task;
use dioxus::prelude::*;
use whispernow_domain::{HistoryFilter, RecordingMode};

pub(crate) use format::LocalClock;
use format::{group_by_day, retention_summary};
use row::HistoryRow;

use super::ui::{
    Button, ButtonSize, ConfirmButton, EmptyState, Icon, IconName, ShortcutLabel, TextInput,
};
use crate::UiContext;
use crate::blocked_reason::DisabledReason;
use crate::management::{HistoryRows, replace_task};
use crate::operations::UiOperation;
use crate::route::{Route, SettingsRoute};

// Page-level control ids; as expressions they render as attributes tests can locate.
const SEARCH_ID: &str = "history-search";
const SEARCH_ERROR_ID: &str = "history-search-error";
const FAILED_FILTER_ID: &str = "history-failed-filter";
const CLEAR_FILTER_ID: &str = "history-clear-filter";
const RELOAD_ID: &str = "history-reload";
const LOAD_MORE_ID: &str = "history-load-more";
const CLEAR_ID: &str = "history-clear";
const OPEN_FOLDER_ID: &str = "history-open-folder";
const RETENTION_ID: &str = "history-retention-settings";
const CHANGE_REASON_ID: &str = "history-change-reason";

const SEARCH_PAUSE_MS: u32 = 300;

#[component]
pub(super) fn HistoryScreen() -> Element {
    let context = use_context::<UiContext>();
    let navigator = context.navigator;
    let management = context.management;
    let clock = use_clock();
    let mut typed = use_signal(|| management.peek().history_filter.query.clone());
    let blocked = |operation| context.blocked(operation);
    let change_blocked = blocked(UiOperation::ChangeHistory);
    let folder_blocked = blocked(UiOperation::OpenRecordingsFolder).is_some();
    let shown = management.read();
    let settings = shown.shown_settings();
    let rows = &shown.history;
    let entries = rows.entries();
    let search_error = match rows {
        HistoryRows::SearchFailed(problem) => Some(problem.as_str()),
        _ => None,
    };
    let failed_only = shown.history_filter.failed_only;
    let filtered = shown.history_filter != HistoryFilter::default();
    let groups = group_by_day(clock, entries);
    rsx! { div { class: "history-page",
        header { class: "history-header",
            p { class: "history-retention",
                if let Some(settings) = settings {
                    span {
                        "{retention_summary(settings.history_limit, settings.history_retention, settings.keep_audio)}. "
                    }
                }
                button { r#type: "button", class: "link-button", id: RETENTION_ID,
                    onclick: move |_| navigator.go(Route::Settings(SettingsRoute::PrivacyData)),
                    "Change in Privacy & Data"
                }
            }
            div { class: "history-header-actions",
                Button { id: OPEN_FOLDER_ID, size: ButtonSize::Sm,
                    disabled: folder_blocked,
                    onclick: move |_| context.open_recordings_folder(),
                    Icon { name: IconName::Folder }
                    "Open recordings folder"
                }
                if !entries.is_empty() {
                    ConfirmButton { id: CLEAR_ID, size: ButtonSize::Sm,
                        label: "Clear history", confirm_label: "Delete every transcript",
                        disabled: change_blocked.is_some(),
                        aria_describedby: change_blocked.as_ref().map(|_| CHANGE_REASON_ID.to_owned()),
                        onconfirm: move |()| context.clear_history(),
                    }
                }
            }
        }
        DisabledReason { id: CHANGE_REASON_ID, reason: change_blocked.clone().filter(|_| !entries.is_empty()) }
        if let HistoryRows::Unavailable(summary) = rows {
            div { class: "history-unavailable", role: "alert",
                p { "{summary}" }
                Button { id: RELOAD_ID, onclick: move |_| context.refresh_history(),
                    "Try again"
                }
            }
        } else {
            div { class: "history-filters", role: "search",
                SearchBox { typed, failed: search_error.is_some() }
                button { r#type: "button", id: FAILED_FILTER_ID,
                    class: if failed_only { "filter-chip selected" } else { "filter-chip" },
                    aria_pressed: if failed_only { "true" } else { "false" },
                    onclick: move |_| context.show_failed_history_only(!failed_only),
                    Icon { name: IconName::AlertCircle }
                    "Failed only"
                }
            }
            if matches!(rows, HistoryRows::Loading) {
                p { class: "history-status", role: "status", "Loading transcripts…" }
            } else if let Some(problem) = search_error {
                p { class: "history-status", id: SEARCH_ERROR_ID, role: "alert", "{problem}" }
            } else if entries.is_empty() && filtered {
                EmptyState { icon: IconName::History, title: "No matching transcripts",
                    if shown.history_filter.search_text().is_some() {
                        p { "Try other words, or show every transcript." }
                    }
                    Button { id: CLEAR_FILTER_ID, size: ButtonSize::Sm,
                        onclick: move |_| {
                            typed.set(String::new());
                            context.clear_history_filter();
                        },
                        "Show all transcripts"
                    }
                }
            } else if entries.is_empty() {
                EmptyState { icon: IconName::History, title: "No transcripts yet",
                    if let Some(settings) = settings {
                        p {
                            match settings.recording_mode {
                                RecordingMode::Hold => "Hold ",
                                RecordingMode::Toggle => "Press ",
                            }
                            ShortcutLabel { shortcut: settings.recording_shortcut.clone() }
                            " to dictate."
                        }
                    }
                }
            }
            for (day, members) in groups {
                section { class: "history-day", key: "{day}", aria_labelledby: "history-day-{day}",
                    h2 { class: "history-day-title", id: "history-day-{day}", "{clock.day_label(day)}" }
                    ul { class: "history-list",
                        for entry in members {
                            HistoryRow { key: "{entry.id.0}",
                                delete_blocked: change_blocked.is_some(),
                                entry: entry.clone(),
                                clock,
                            }
                        }
                    }
                }
            }
            if rows.has_more() {
                div { class: "history-more",
                    Button { id: LOAD_MORE_ID, onclick: move |_| context.load_older_history(),
                        "Load more"
                    }
                }
            }
        }
    } }
}

/// The search field. Typing re-renders only this field; the filter follows once typing pauses.
#[component]
fn SearchBox(mut typed: Signal<String>, failed: bool) -> Element {
    let context = use_context::<UiContext>();
    // The search waiting for typing to pause; a key press restarts it.
    let mut search_task = use_signal(|| None::<Task>);
    rsx! {
        label { class: "visually-hidden", r#for: SEARCH_ID, "Search transcripts" }
        TextInput { id: SEARCH_ID, r#type: "search", value: "{typed}",
            placeholder: "Search transcripts",
            maxlength: "{HistoryFilter::MAX_QUERY_BYTES}",
            aria_describedby: failed.then_some(SEARCH_ERROR_ID),
            aria_invalid: failed.then_some("true"),
            oninput: move |event: FormEvent| {
                typed.set(event.value());
                let waiting = spawn(async move {
                    crate::client::delay(SEARCH_PAUSE_MS).await;
                    search_task.set(None);
                    context.search_history(&typed.peek());
                });
                replace_task(&mut search_task.write(), Some(waiting));
            },
        }
    }
}

/// The system clock, read again each minute so relative times stay current; tests may provide
/// a fixed one. Between ticks it stays equal, so rows skip re-rendering.
fn use_clock() -> LocalClock {
    #[cfg(test)]
    let provided = try_use_context::<LocalClock>();
    let mut system = use_signal(LocalClock::system);
    use_future(move || async move {
        loop {
            crate::client::delay(60_000).await;
            system.set(LocalClock::system());
        }
    });
    #[cfg(test)]
    if let Some(provided) = provided {
        return provided;
    }
    *system.read()
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
