use dioxus::prelude::*;
use whispernow_domain::ipc::{ManualUpdatePackage, UpdateError, UpdatePhase};

use super::ui::{Banner, Button, ButtonVariant, ProgressBar, Tone};
use crate::blocked_reason::{BlockedButton, DisabledReason};
use crate::{UiContext, UiOperation};

#[component]
pub(super) fn AboutPage() -> Element {
    let context = use_context::<UiContext>();
    let check_blocked = context.blocked(UiOperation::CheckForUpdates);
    let (checking, updates_disabled) = {
        let state = context.state.read();
        let phase = state
            .update_snapshot
            .as_ref()
            .map(|snapshot| &snapshot.phase);
        (
            matches!(phase, Some(UpdatePhase::Checking)),
            matches!(phase, Some(UpdatePhase::Disabled)),
        )
    };
    rsx! { div { class: "settings-stack",
        div { class: "card", aria_labelledby: "about-title",
            h2 { id: "about-title", "WhisperNow" }
            p { id: "about-version", "Version {env!(\"CARGO_PKG_VERSION\")}" }
            UpdateStatus {}
            BlockedButton {
                id: "check-for-updates",
                blocked: check_blocked,
                disabled: checking || updates_disabled,
                onclick: move |_| context.check_for_updates(),
                if checking { "Checking…" } else { "Check for updates" }
            }
            if updates_disabled {
                p { class: "status-detail", "Updates are not available in this build." }
            }
        }
    } }
}

#[component]
pub(super) fn UpdateStatus() -> Element {
    let context = use_context::<UiContext>();
    let state = context.state;
    let restart_blocked = context.blocked(UiOperation::UpdateInstallRestart);
    let dismiss_blocked = context.blocked(UiOperation::DismissUpdate);
    let open_page_blocked = context.blocked(UiOperation::OpenUpdateReleasePage);
    let snapshot = state.read().update_snapshot.clone();
    let command_error = state.read().update_command_error.clone();
    if let Some(message) = command_error {
        return rsx! { UpdateFailure { title: "Update check failed", message, retryable: true } };
    }
    let Some(snapshot) = snapshot else {
        return rsx! {};
    };

    match snapshot.phase {
        UpdatePhase::Disabled | UpdatePhase::Idle => rsx! {},
        UpdatePhase::Checking => {
            rsx! { section { class: "update-banner", role: "status", aria_live: "polite",
                h3 { "Checking for updates…" }
            } }
        }
        UpdatePhase::Downloading {
            version, progress, ..
        } => {
            rsx! { section { class: "update-banner", role: "status", aria_live: "polite",
                h3 { "Downloading update to v{version}" }
                ProgressBar { label: "Update download progress", downloaded: progress.downloaded_bytes, total: progress.total_bytes }
            } }
        }
        UpdatePhase::ReadyToRestart { version, error, .. } if snapshot.notification_dismissed => {
            rsx! {
                section { class: "update-banner update-secondary", role: "status", aria_live: "polite",
                    div {
                        strong { "Update v{version} is ready" }
                        if let Some(error) = error {
                            p { role: "alert", "{update_error_message(error)}" }
                        }
                        if snapshot.restart_deferred {
                            p { "The restart waits until the current dictation finishes." }
                        }
                    }
                    BlockedButton { id: "restart-to-update", blocked: restart_blocked,
                        onclick: move |_| context.restart_to_update(), "Restart to update" }
                }
            }
        }
        UpdatePhase::ReadyToRestart {
            version,
            release_notes,
            error,
        } => rsx! {
            section { class: "update-banner update-ready", role: "status", aria_live: "polite",
                div {
                    h3 { "Restart to update to v{version}" }
                    if let Some(error) = error {
                        p { role: "alert", "{update_error_message(error)}" }
                    }
                    if snapshot.restart_deferred {
                        p { "The restart waits until the current dictation finishes." }
                    } else {
                        p { "The update is downloaded and verified. WhisperNow restarts only when you choose." }
                    }
                    if let Some(notes) = release_notes.as_deref() {
                        h4 { "What’s new" }
                        p { class: "release-notes", "{notes}" }
                    }
                }
                div { class: "update-actions",
                    Button { id: "restart-to-update", variant: ButtonVariant::Primary,
                        disabled: restart_blocked.is_some(),
                        aria_describedby: restart_blocked.as_ref().map(|_| "restart-to-update-disabled-reason"),
                        onclick: move |_| context.restart_to_update(), "Restart to update" }
                    Button { variant: ButtonVariant::Ghost, disabled: dismiss_blocked.is_some(),
                        onclick: move |_| context.dismiss_update(), "Dismiss" }
                }
                DisabledReason { id: "restart-to-update-disabled-reason", reason: restart_blocked }
            }
        },
        UpdatePhase::ManualInstallAvailable { version, error, .. }
            if snapshot.notification_dismissed =>
        {
            rsx! {
                section { class: "update-banner update-secondary", role: "status", aria_live: "polite",
                    div {
                        strong { "WhisperNow v{version} is available" }
                        if let Some(error) = error {
                            p { role: "alert", "{update_error_message(error)}" }
                        }
                    }
                    Button { disabled: open_page_blocked.is_some(),
                        onclick: move |_| context.open_update_release_page(), "Open release page" }
                }
            }
        }
        UpdatePhase::ManualInstallAvailable {
            version,
            release_notes,
            package,
            error,
        } => {
            let package = manual_package_label(package);
            let opener_error = error.map(update_error_message);
            rsx! { section { class: "update-banner update-ready", role: "status", aria_live: "polite",
                div {
                    h3 { "WhisperNow v{version} is available" }
                    p { "Download {package} from the release page and install it with your system package manager." }
                    if let Some(message) = opener_error {
                        p { role: "alert", "{message}" }
                    }
                    if let Some(notes) = release_notes.as_deref() {
                        h4 { "What’s new" }
                        p { class: "release-notes", "{notes}" }
                    }
                }
                div { class: "update-actions",
                    Button { variant: ButtonVariant::Primary, disabled: open_page_blocked.is_some(),
                        onclick: move |_| context.open_update_release_page(), "Open release page" }
                    Button { variant: ButtonVariant::Ghost, disabled: dismiss_blocked.is_some(),
                        onclick: move |_| context.dismiss_update(), "Dismiss" }
                }
            } }
        }
        UpdatePhase::Installing { version, .. } => {
            rsx! { section { class: "update-banner", role: "status", aria_live: "polite",
                h3 { "Installing update v{version}…" }
                p { "WhisperNow restarts when the installation finishes." }
            } }
        }
        UpdatePhase::UpToDate => {
            rsx! { Banner { tone: Tone::Success, title: "WhisperNow is up to date",
                p { "No newer stable release is available." }
            } }
        }
        UpdatePhase::Error { error } => rsx! {
            UpdateFailure {
                title: "Update needs attention",
                message: update_error_message(error),
                retryable: error.is_recoverable(),
            }
        },
    }
}

#[component]
fn UpdateFailure(title: &'static str, message: String, retryable: bool) -> Element {
    let context = use_context::<UiContext>();
    let check_blocked = context.blocked(UiOperation::CheckForUpdates);
    rsx! { section { class: "update-banner update-error", role: "alert",
        h3 { "{title}" }
        p { "{message}" }
        if retryable {
            BlockedButton { id: "update-retry", blocked: check_blocked,
                onclick: move |_| context.check_for_updates(), "Try again" }
        }
    } }
}

const fn manual_package_label(package: ManualUpdatePackage) -> &'static str {
    match package {
        ManualUpdatePackage::Deb => "the .deb package",
        ManualUpdatePackage::Rpm => "the .rpm package",
    }
}

const fn update_error_message(code: UpdateError) -> &'static str {
    match code {
        UpdateError::Offline => "You appear to be offline. Reconnect and try again.",
        UpdateError::ServerUnavailable => "The update service is unavailable. Try again later.",
        UpdateError::MalformedMetadata => "The release information could not be verified.",
        UpdateError::UnsupportedTarget => {
            "This installation type is not supported by automatic updates."
        }
        UpdateError::InvalidSignature => {
            "The downloaded update could not be verified and will not be installed."
        }
        UpdateError::DownloadInterrupted => {
            "The update download was interrupted. You can try again."
        }
        UpdateError::InstallFailed => {
            "The verified update could not be installed. You can retry it."
        }
        UpdateError::InsufficientPrivileges => {
            "The system did not grant permission to install this update."
        }
        UpdateError::ReadOnlyFilesystem => {
            "This installation is on a read-only filesystem and cannot be updated in place."
        }
        UpdateError::ReleasePageUnavailable => {
            "Your web browser could not be opened. Download the update from the WhisperNow releases page on GitHub."
        }
        UpdateError::Internal => "WhisperNow could not complete the update. Try again.",
    }
}
