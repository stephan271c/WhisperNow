use crate::UiContext;
use dioxus::prelude::*;
use whispernow_domain::{EntryStatus, HistoryEntry};

use super::audio::AudioPlayer;
use super::format::{LocalClock, duration_label, machine_time};
use crate::components::copy_button::CopyButton;
use crate::components::ui::{ButtonSize, ConfirmButton, Icon, IconName};

struct Badge {
    label: &'static str,
    class: &'static str,
}

/// The marker a row shows; a dictation that was pasted needs none.
const fn status_badge(status: EntryStatus) -> Option<Badge> {
    let (label, class) = match status {
        EntryStatus::Ok => return None,
        EntryStatus::Failed => ("Failed", "history-status-badge failed"),
        EntryStatus::Cancelled => ("Cancelled", "history-status-badge cancelled"),
        EntryStatus::NotDelivered => ("Not pasted", "history-status-badge not-pasted"),
    };
    Some(Badge { label, class })
}

#[component]
pub(super) fn HistoryRow(entry: HistoryEntry, delete_blocked: bool, clock: LocalClock) -> Element {
    let context = use_context::<UiContext>();
    let id = entry.id;
    let rid = id.0;
    let text = entry.text().map(str::to_owned);
    let original = entry.cleaned_text.as_ref().and(entry.raw_text.clone());
    let when = clock.relative_time(entry.created_at);
    let target = format!("the transcript from {when}");
    let duration = entry.duration_millis.map(duration_label);
    let badge = status_badge(entry.status);
    rsx! { li { class: "history-row", id: "history-entry-{rid}",
        div { class: "history-row-main",
            div { class: "history-row-summary",
                span { class: "history-row-meta",
                    time { datetime: machine_time(entry.created_at),
                        title: clock.full_time(entry.created_at),
                        "{when}"
                    }
                    if let Some(duration) = &duration {
                        span { class: "history-row-duration", "{duration}" }
                    }
                    if let Some(Badge { label, class }) = badge {
                        span { class, "{label}" }
                    }
                    if original.is_some() {
                        span { class: "history-row-cleaned", id: "history-cleaned-{rid}",
                            Icon { name: IconName::Sparkles }
                            "Cleaned"
                        }
                    }
                }
                if let (Some(text), Some(original)) = (&text, &original) {
                    details { class: "history-row-text", id: "history-text-{rid}",
                        summary { class: "history-row-preview", "{text}" }
                        div { class: "history-row-original", id: "history-original-{rid}",
                            p { class: "history-row-original-label", "Original transcript" }
                            p { class: "transcript-text", "{original}" }
                        }
                    }
                } else if let Some(text) = &text {
                    p { class: "history-row-preview", "{text}" }
                } else {
                    p { class: "history-row-preview empty", "No transcript" }
                }
            }
            div { class: "history-row-actions",
                CopyButton { id: "history-copy-{rid}",
                    text, label: "Copy", names: target.clone(), size: ButtonSize::Sm, icon: true,
                }
                ConfirmButton { id: "history-delete-{rid}", size: ButtonSize::Sm,
                    label: "Delete", confirm_label: "Delete this dictation",
                    target: target.clone(),
                    disabled: delete_blocked,
                    aria_describedby: delete_blocked.then(|| super::CHANGE_REASON_ID.to_owned()),
                    onconfirm: move |()| context.delete_history_entry(id),
                }
            }
        }
        if let Some(error) = &entry.error {
            div { class: "history-problem", id: "history-problem-{rid}",
                Icon { name: IconName::AlertCircle }
                p { "{error}" }
            }
        }
        if entry.audio_kept {
            AudioPlayer { id }
        }
    } }
}
