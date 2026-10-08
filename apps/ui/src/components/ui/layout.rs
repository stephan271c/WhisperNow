use dioxus::prelude::*;

use super::{Icon, IconName};

#[component]
pub(crate) fn Card(
    #[props(extends = GlobalAttributes)] attributes: Vec<Attribute>,
    children: Element,
) -> Element {
    rsx! { div { class: "card", ..attributes, {children} } }
}

/// A titled group of settings rows on one card; the heading names the region.
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

/// Placeholder for a list or panel with nothing to show yet; `children` explains what to do.
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
