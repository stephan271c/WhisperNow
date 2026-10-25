use dioxus::core::Task;
use dioxus::prelude::*;

use super::ui::{Button, ButtonSize, Icon, IconName};
use crate::client;
use crate::management::replace_task;

const COPIED_MS: u32 = 2_000;

#[derive(Clone, Copy, Eq, PartialEq)]
enum CopyOutcome {
    Copied,
    Failed,
}

/// Copies `text` and shows the outcome in its label for a moment; without `text` it is
/// disabled. In a repeated row, `names` says what it copies; it follows the label in the
/// accessible name.
#[component]
pub(super) fn CopyButton(
    id: String,
    text: Option<String>,
    label: String,
    names: Option<String>,
    #[props(default)] size: ButtonSize,
    #[props(default)] icon: bool,
) -> Element {
    let mut outcome = use_signal(|| None::<CopyOutcome>);
    // The latest copy, which clears its outcome after a moment; a newer copy replaces it.
    let mut copy_task = use_signal(|| None::<Task>);
    let shown_outcome = outcome();
    let shown = match shown_outcome {
        Some(CopyOutcome::Copied) => "Copied",
        Some(CopyOutcome::Failed) => "Copy failed",
        None => label.as_str(),
    };
    rsx! {
        Button { id: "{id}", size, disabled: text.is_none(),
            aria_label: names.as_ref().map(|names| format!("{shown} {names}")),
            onclick: move |_| {
                if let Some(text) = text.clone() {
                    outcome.set(None);
                    let copying = spawn(async move {
                        let copied = client::invoke::<_, ()>("copy_text", text).await.is_ok();
                        outcome.set(Some(if copied {
                            CopyOutcome::Copied
                        } else {
                            CopyOutcome::Failed
                        }));
                        client::delay(COPIED_MS).await;
                        outcome.set(None);
                        copy_task.set(None);
                    });
                    replace_task(&mut copy_task.write(), Some(copying));
                }
            },
            if icon {
                Icon { name: if shown_outcome == Some(CopyOutcome::Copied) { IconName::CheckCircle } else { IconName::Copy } }
            }
            "{shown}"
        }
        span { class: "visually-hidden", role: "status",
            if shown_outcome.is_some() { "{shown}" }
        }
    }
}
