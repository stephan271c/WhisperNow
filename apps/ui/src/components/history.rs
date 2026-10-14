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
use crate::{Route, SettingsRoute, UiContext, UiOperation};

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

const SEARCH_PAUSE_MS: u32 = 300;

#[component]
pub(super) fn HistoryScreen() -> Element {
    let context = use_context::<UiContext>();
    let navigator = context.navigator;
    let mut management = context.management;
    let clock = use_clock();
    // The search waiting for typing to pause; a key press restarts it.
    let mut search_task = use_signal(|| None::<Task>);
    let copied = management.read().history_copied;
    let blocked = |operation| context.blocked(operation);
    let settings = management.read().shown_settings().cloned();
    let change_blocked = blocked(UiOperation::ChangeHistory);
    let unavailable = management.read().history_unavailable.clone();
    let loading = !management.read().history_loaded();
    let entries = management.read().history.clone();
    let has_more = management.read().history_has_more;
    let filter = management.read().history_filter.clone();
    let search_text = management.read().history_search.clone();
    let search_error = management.read().history_search_error.clone();
    let filtered = filter != HistoryFilter::default();
    let groups = group_by_day(clock, &entries);
    let folder_blocked = blocked(UiOperation::OpenRecordingsFolder).is_some();
    rsx! { div { class: "history-page",
        header { class: "history-header",
            p { class: "history-retention",
                if let Some(settings) = &settings {
                    span { "{retention_summary(settings.history_limit, settings.keep_audio)}. " }
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
                if unavailable.is_none() && !entries.is_empty() {
                    ConfirmButton { id: CLEAR_ID, size: ButtonSize::Sm,
                        label: "Clear history", confirm_label: "Delete every transcript",
                        disabled: change_blocked.is_some(),
                        onconfirm: move |()| context.clear_history(),
                    }
                }
            }
        }
        if let Some(reason) = change_blocked.as_ref().filter(|_| !entries.is_empty()) {
            p { class: "disabled-reason", id: "history-change-reason", "{reason.message()}" }
        }
        if let Some(summary) = &unavailable {
            div { class: "history-unavailable", role: "alert",
                p { "{summary}" }
                Button { id: RELOAD_ID, onclick: move |_| context.refresh_history(),
                    "Try again"
                }
            }
        } else {
            div { class: "history-filters", role: "search",
                label { class: "visually-hidden", r#for: SEARCH_ID, "Search transcripts" }
                TextInput { id: SEARCH_ID, r#type: "search", value: "{search_text}",
                    placeholder: "Search transcripts",
                    maxlength: "{HistoryFilter::MAX_QUERY_BYTES}",
                    aria_describedby: search_error.as_ref().map(|_| SEARCH_ERROR_ID),
                    aria_invalid: search_error.as_ref().map(|_| "true"),
                    oninput: move |event: FormEvent| {
                        management.write().history_search = event.value();
                        if let Some(waiting) = search_task.take() {
                            waiting.cancel();
                        }
                        search_task.set(Some(spawn(async move {
                            crate::client::delay(SEARCH_PAUSE_MS).await;
                            search_task.set(None);
                            context.search_history();
                        })));
                    },
                }
                button { r#type: "button", id: FAILED_FILTER_ID,
                    class: if filter.failed_only { "filter-chip selected" } else { "filter-chip" },
                    aria_pressed: if filter.failed_only { "true" } else { "false" },
                    onclick: move |_| context.show_failed_history_only(!filter.failed_only),
                    Icon { name: IconName::AlertCircle }
                    "Failed only"
                }
            }
            if loading {
                p { class: "history-status", role: "status", "Loading transcripts…" }
            } else if let Some(problem) = &search_error {
                p { class: "history-status", id: SEARCH_ERROR_ID, role: "alert", "{problem}" }
            } else if entries.is_empty() && filtered {
                EmptyState { icon: IconName::History, title: "No matching transcripts",
                    p { "Try other words, or show every transcript." }
                    Button { id: CLEAR_FILTER_ID, size: ButtonSize::Sm,
                        onclick: move |_| context.clear_history_filter(),
                        "Show all transcripts"
                    }
                }
            } else if entries.is_empty() {
                EmptyState { icon: IconName::History, title: "No transcripts yet",
                    if let Some(settings) = settings.as_ref() {
                        p {
                            match settings.recording_mode {
                                RecordingMode::Hold => "Hold ",
                                RecordingMode::Toggle => "Press ",
                            }
                            ShortcutLabel { shortcut: settings.recording_shortcut.clone() }
                            " to dictate. Your words will be waiting here."
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
                                copied: copied.filter(|copied| copied.entry == entry.id),
                                delete_blocked: change_blocked.is_some(),
                                entry,
                                clock,
                            }
                        }
                    }
                }
            }
            if has_more && !loading {
                div { class: "history-more",
                    Button { id: LOAD_MORE_ID, onclick: move |_| context.load_older_history(),
                        "Load more"
                    }
                }
            }
        }
    } }
}

/// The page's clock: a provided one in tests, otherwise the system's, read again each minute
/// so relative times stay current.
fn use_clock() -> LocalClock {
    let provided = try_use_context::<LocalClock>();
    use_minute_tick();
    provided.unwrap_or_else(LocalClock::system)
}

fn use_minute_tick() {
    let mut minute = use_signal(|| 0_u64);
    use_future(move || async move {
        loop {
            crate::client::delay(60_000).await;
            minute += 1;
        }
    });
    let _ = minute();
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
