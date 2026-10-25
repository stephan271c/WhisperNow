use dioxus::prelude::*;

use super::{Icon, IconName};

#[component]
pub(crate) fn Section(
    id: String,
    title: String,
    description: Option<String>,
    children: Element,
) -> Element {
    rsx! {
        section { class: "section", aria_labelledby: "{id}-title",
            header { class: "section-header",
                h2 { id: "{id}-title", class: "section-title", "{title}" }
                if let Some(description) = description {
                    p { class: "section-description", "{description}" }
                }
            }
            div { class: "card section-card", {children} }
        }
    }
}

#[component]
pub(crate) fn EmptyState(icon: IconName, title: String, children: Element) -> Element {
    rsx! {
        div { class: "empty-state",
            span { class: "empty-state-icon", Icon { name: icon } }
            p { class: "empty-state-title", "{title}" }
            div { class: "empty-state-body", {children} }
        }
    }
}
