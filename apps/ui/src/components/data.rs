use dioxus::prelude::*;

use super::ui::{Banner, ConfirmButton, Section, Tone};
use super::{NoticeRegion, labels};
use crate::blocked_reason::{BlockedButton, DisabledReason};
use crate::{NoticeScope, UiContext, UiOperation};

#[component]
pub(super) fn ClearDataCard() -> Element {
    let context = use_context::<UiContext>();
    let locations = context.management.read().data_locations.clone();
    let result = context.management.read().clear_result.clone();
    let review_blocked = context.blocked(UiOperation::ListDataLocations);
    let clear_blocked = context
        .blocked(UiOperation::ClearData)
        .or(context.settings_blocked());
    rsx! {
        Section { id: "clear-data", title: "Application data",
            description: "Settings, history, downloaded models and logs.",
            div { class: "section-body",
                BlockedButton {
                    id: "preview-clear-data",
                    blocked: review_blocked,
                    onclick: move |_| context.review_data(),
                    "Review data to remove"
                }
                NoticeRegion { scope: NoticeScope::Data }
                if let Some(locations) = locations {
                    div { class: "danger-zone",
                        if locations.is_empty() {
                            p { "There is no application data to remove." }
                        } else {
                            h3 { "These items will be removed" }
                            ul { class: "clear-data-list",
                                for location in locations.iter() {
                                    li {
                                        strong { "{labels::clear_category(location.category)}: " }
                                        code { "{location.path.display()}" }
                                    }
                                }
                            }
                            ConfirmButton {
                                id: "execute-clear-data",
                                label: "Remove these items",
                                confirm_label: "Remove permanently",
                                disabled: clear_blocked.is_some(),
                                aria_describedby: clear_blocked.as_ref().map(|_| "execute-disabled-reason".to_owned()),
                                onconfirm: move |()| context.clear_data(),
                            }
                            DisabledReason { id: "execute-disabled-reason", reason: clear_blocked }
                        }
                    }
                }
                if let Some(result) = result {
                    h3 { "Removal result" }
                    ul { class: "clear-data-list",
                        for item in result.items.iter() {
                            li {
                                "{labels::clear_category(item.location.category)}: "
                                if item.deleted { "Removed" } else { "Could not be removed" }
                                " — {item.location.path.display()}"
                            }
                        }
                    }
                    if result.restart_required {
                        Banner { tone: Tone::Warning, p { "Restart WhisperNow before dictating again." } }
                    }
                }
            }
        }
    }
}
