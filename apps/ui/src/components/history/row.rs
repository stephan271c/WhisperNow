use crate::UiContext;
use dioxus::prelude::*;
use whispernow_domain::{EntryStatus, HistoryEntry};

use super::audio::AudioPlayer;
use super::format::{LocalClock, duration_label, machine_time};
use crate::HistoryCopied;
use crate::components::ui::{Button, ButtonSize, ButtonVariant, ConfirmButton, Icon, IconName};

/// The marker a row shows; a dictation that was pasted needs none.
pub(super) const fn status_badge(status: EntryStatus) -> Option<(&'static str, &'static str)> {
    match status {
        EntryStatus::Ok => None,
        EntryStatus::Failed => Some(("Failed", "history-status-badge failed")),
        EntryStatus::Cancelled => Some(("Cancelled", "history-status-badge cancelled")),
        EntryStatus::NotDelivered => Some(("Not pasted", "history-status-badge not-pasted")),
    }
}

#[component]
pub(super) fn HistoryRow(
    entry: HistoryEntry,
    copied: Option<HistoryCopied>,
    delete_blocked: bool,
    clock: LocalClock,
) -> Element {
    let context = use_context::<UiContext>();
    let id = entry.id;
    let rid = id.0;
    let text = entry.text().map(str::to_owned);
    let original = entry.cleaned_text.as_ref().and(entry.raw_text.clone());
    let when = clock.relative_time(entry.created_at);
    let target = format!("the transcript from {when}");
    let duration = entry.duration_millis.map(duration_label);
    let badge = status_badge(entry.status);
    let copy_label = match copied {
        Some(copied) if copied.succeeded => "Copied",
        Some(_) => "Copy failed",
        None => "Copy",
    };
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
                    if let Some((label, class)) = badge {
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
                Button { id: "history-copy-{rid}", size: ButtonSize::Sm,
                    variant: ButtonVariant::Secondary,
                    aria_label: "{copy_label} {target}",
                    disabled: text.is_none(),
                    onclick: move |_| context.copy_history_entry(id),
                    Icon { name: if copy_label == "Copied" { IconName::CheckCircle } else { IconName::Copy } }
                    "{copy_label}"
                }
                ConfirmButton { id: "history-delete-{rid}", size: ButtonSize::Sm,
                    label: "Delete", confirm_label: "Delete this dictation",
                    target: target.clone(),
                    disabled: delete_blocked,
                    onconfirm: move |()| context.delete_history_entry(id),
                }
            }
        }
        span { class: "visually-hidden", role: "status",
            if copy_label != "Copy" { "{copy_label}" }
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
