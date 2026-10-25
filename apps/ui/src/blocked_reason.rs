use dioxus::prelude::*;

use crate::UiContext;
use crate::components::ui::{Button, ButtonVariant};
use crate::operations::UiOperation;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum BlockedReason {
    Connecting,
    Phase(String),
    CommandPending,
}

impl BlockedReason {
    #[must_use]
    pub(crate) fn message(&self) -> &str {
        match self {
            Self::Connecting => "WhisperNow is still connecting",
            Self::Phase(summary) => summary,
            Self::CommandPending => "Wait for the current change to finish",
        }
    }
}

impl UiContext {
    pub(crate) fn blocked(self, operation: UiOperation) -> Option<BlockedReason> {
        if self.state.read().snapshot.is_none() {
            Some(BlockedReason::Connecting)
        } else {
            self.management
                .read()
                .is_busy(operation)
                .then_some(BlockedReason::CommandPending)
        }
    }

    pub(crate) fn phase_blocked(self) -> Option<BlockedReason> {
        match self.state.read().snapshot.as_ref() {
            None => Some(BlockedReason::Connecting),
            Some(snapshot) => snapshot
                .phase
                .require_idle()
                .err()
                .map(|error| BlockedReason::Phase(error.summary)),
        }
    }

    pub(crate) fn model_download_blocked(self) -> Option<BlockedReason> {
        self.blocked(UiOperation::DownloadModel)
    }

    pub(crate) fn model_delete_blocked(self) -> Option<BlockedReason> {
        self.blocked(UiOperation::ModelDelete)
            .or(self.phase_blocked())
    }

    pub(crate) fn settings_pending(self) -> Option<BlockedReason> {
        self.management
            .read()
            .settings_pending()
            .then_some(BlockedReason::CommandPending)
    }

    pub(crate) fn settings_blocked(self) -> Option<BlockedReason> {
        self.phase_blocked().or(self.settings_pending())
    }
}

#[component]
pub(crate) fn DisabledReason(id: String, reason: Option<BlockedReason>) -> Element {
    let Some(reason) = reason else {
        return rsx! {};
    };
    // A save in flight ends within moments; showing it would shift the page twice.
    let class = match reason {
        BlockedReason::CommandPending => "disabled-reason visually-hidden",
        BlockedReason::Connecting | BlockedReason::Phase(_) => "disabled-reason",
    };
    rsx! { p { id, class, "{reason.message()}" } }
}

/// A button that `blocked` disables, followed by the reason, which describes it as
/// `{id}-disabled-reason`. `disabled` disables it without a reason.
#[component]
pub(crate) fn BlockedButton(
    id: String,
    blocked: Option<BlockedReason>,
    #[props(default)] disabled: bool,
    #[props(default)] variant: ButtonVariant,
    onclick: EventHandler<MouseEvent>,
    children: Element,
) -> Element {
    let reason_id = format!("{id}-disabled-reason");
    rsx! {
        Button {
            id,
            variant,
            disabled: disabled || blocked.is_some(),
            aria_describedby: blocked.is_some().then(|| reason_id.clone()),
            onclick: move |event| onclick.call(event),
            {children}
        }
        DisabledReason { id: reason_id, reason: blocked }
    }
}
