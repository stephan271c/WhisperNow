use dioxus::prelude::*;
use whispernow_domain::ipc::{ManualUpdatePackage, UpdateError, UpdatePhase};

use super::NoticeRegion;
use super::actions::DisabledReason;
use super::labels::format_bytes;
use super::ui::{Banner, Button, ButtonVariant, ProgressBar, Tone};
use crate::{NoticeScope, UiContext, UiOperation};

#[component]
pub(super) fn AboutPage() -> Element {
    let context = use_context::<UiContext>();
    let actions = context;
    let check_blocked = actions.blocked(UiOperation::CheckForUpdates);
    let phase = context
        .state
        .read()
        .update_snapshot
        .as_ref()
        .map(|snapshot| snapshot.phase);
    let checking = phase == Some(UpdatePhase::Checking);
    let updates_disabled = phase == Some(UpdatePhase::Disabled);
    rsx! { div { class: "settings-stack",
        div { class: "card", aria_labelledby: "about-title",
            h2 { id: "about-title", "WhisperNow" }
            p { id: "about-version", "Version {env!(\"CARGO_PKG_VERSION\")}" }
            UpdateStatus {}
            NoticeRegion { scope: NoticeScope::Updates }
            Button {
                id: "check-for-updates",
                disabled: checking || updates_disabled || check_blocked.is_some(),
                aria_describedby: check_blocked.map(|_| "check-disabled-reason"),
                onclick: move |_| actions.check_for_updates(),
                if checking { "Checking…" } else { "Check for updates" }
            }
            DisabledReason { id: "check-disabled-reason", reason: check_blocked }
            if updates_disabled {
                p { class: "status-detail", "Updates are not available in this build." }
            }
        }
    } }
}

#[component]
pub(super) fn UpdateStatus() -> Element {
    let context = use_context::<UiContext>();
    let actions = context;
    let state = context.state;
    let restart_blocked = actions.blocked(UiOperation::UpdateInstallRestart);
    let check_blocked = actions.blocked(UiOperation::CheckForUpdates);
    let dismiss_blocked = actions.blocked(UiOperation::DismissUpdate);
    let open_page_blocked = actions.blocked(UiOperation::OpenUpdateReleasePage);
    let snapshot = state.read().update_snapshot.clone();
    let command_error = state.read().update_command_error.clone();
    if let Some(message) = command_error {
        return rsx! { section { class: "update-banner update-error", role: "alert",
            h3 { "Update check failed" }
            p { "{message}" }
            Button { disabled: check_blocked.is_some(),
                onclick: move |_| actions.check_for_updates(), "Try again" }
            DisabledReason { id: "update-retry-disabled-reason", reason: check_blocked }
        } };
    }
    let Some(snapshot) = snapshot else {
        return rsx! {};
    };

    let version = snapshot
        .available_version
        .as_deref()
        .map_or_else(|| "the latest version".to_owned(), display_version);
    match snapshot.phase {
        UpdatePhase::Disabled | UpdatePhase::Idle => rsx! {},
        UpdatePhase::Checking => {
            rsx! { section { class: "update-banner", role: "status", aria_live: "polite",
                h3 { "Checking for updates…" }
            } }
        }
        UpdatePhase::Downloading => {
            let progress = snapshot.progress;
            let downloaded = progress.map_or(0, |progress| progress.downloaded_bytes);
            let total = progress.and_then(|progress| progress.total_bytes);
            let max = total.unwrap_or(downloaded.max(1));
            let progress_label = total.map_or_else(
                || format!("{} downloaded", format_bytes(downloaded)),
                |total| {
                    format!(
                        "{} of {} downloaded",
                        format_bytes(downloaded),
                        format_bytes(total)
                    )
                },
            );
            rsx! { section { class: "update-banner", role: "status", aria_live: "polite",
                h3 { "Downloading update to {version}" }
                ProgressBar { label: "Update download progress", value: downloaded, max, value_text: progress_label.clone() }
                p { "{progress_label}" }
            } }
        }
        UpdatePhase::ReadyToRestart if snapshot.notification_dismissed => rsx! {
            section { class: "update-banner update-secondary", role: "status", aria_live: "polite",
                div {
                    strong { "Update {version} is ready" }
                    if snapshot.restart_deferred {
                        p { "The restart waits until the current dictation finishes." }
                    }
                }
                Button { disabled: restart_blocked.is_some(),
                    onclick: move |_| actions.restart_to_update(), "Restart to update" }
                DisabledReason { id: "restart-disabled-reason", reason: restart_blocked }
            }
        },
        UpdatePhase::ReadyToRestart => rsx! {
            section { class: "update-banner update-ready", role: "status", aria_live: "polite",
                div {
                    h3 { "Restart to update to {version}" }
                    if snapshot.restart_deferred {
                        p { "The restart waits until the current dictation finishes." }
                    } else {
                        p { "The update is downloaded and verified. WhisperNow restarts only when you choose." }
                    }
                    if let Some(notes) = snapshot.release_notes.as_deref() {
                        h4 { "What’s new" }
                        p { class: "release-notes", "{notes}" }
                    }
                }
                div { class: "update-actions",
                    Button { variant: ButtonVariant::Primary, disabled: restart_blocked.is_some(),
                        onclick: move |_| actions.restart_to_update(), "Restart to update" }
                    Button { variant: ButtonVariant::Ghost, disabled: dismiss_blocked.is_some(),
                        onclick: move |_| actions.dismiss_update(), "Dismiss" }
                }
                DisabledReason { id: "restart-disabled-reason", reason: restart_blocked }
            }
        },
        UpdatePhase::ManualInstallAvailable if snapshot.notification_dismissed => rsx! {
            section { class: "update-banner update-secondary", role: "status", aria_live: "polite",
                div { strong { "WhisperNow {version} is available" } }
                Button { disabled: open_page_blocked.is_some(),
                    onclick: move |_| actions.open_update_release_page(), "Open release page" }
            }
        },
        UpdatePhase::ManualInstallAvailable => {
            let package = manual_package_label(snapshot.manual_package);
            let opener_error = snapshot.error.map(update_error_message);
            rsx! { section { class: "update-banner update-ready", role: "status", aria_live: "polite",
                div {
                    h3 { "WhisperNow {version} is available" }
                    p { "Download {package} from the release page and install it with your system package manager." }
                    if let Some(message) = opener_error {
                        p { role: "alert", "{message}" }
                    }
                    if let Some(notes) = snapshot.release_notes.as_deref() {
                        h4 { "What’s new" }
                        p { class: "release-notes", "{notes}" }
                    }
                }
                div { class: "update-actions",
                    Button { variant: ButtonVariant::Primary, disabled: open_page_blocked.is_some(),
                        onclick: move |_| actions.open_update_release_page(), "Open release page" }
                    Button { variant: ButtonVariant::Ghost, disabled: dismiss_blocked.is_some(),
                        onclick: move |_| actions.dismiss_update(), "Dismiss" }
                }
            } }
        }
        UpdatePhase::Installing => {
            rsx! { section { class: "update-banner", role: "status", aria_live: "polite",
                h3 { "Installing update {version}…" }
                p { "WhisperNow restarts when the installation finishes." }
            } }
        }
        UpdatePhase::UpToDate => {
            rsx! { Banner { tone: Tone::Success, title: "WhisperNow is up to date",
                p { "No newer stable release is available." }
            } }
        }
        UpdatePhase::Error => {
            let error = snapshot.error;
            let (message, recoverable) = error
                .map_or(("The update could not be completed.", true), |error| {
                    (update_error_message(error), error.is_recoverable())
                });
            rsx! { section { class: "update-banner update-error", role: "alert",
                h3 { "Update needs attention" }
                p { "{message}" }
                if recoverable {
                    Button { disabled: check_blocked.is_some(),
                        onclick: move |_| actions.check_for_updates(), "Try again" }
                    DisabledReason { id: "update-retry-disabled-reason", reason: check_blocked }
                }
            } }
        }
    }
}

const fn manual_package_label(package: Option<ManualUpdatePackage>) -> &'static str {
    match package {
        Some(ManualUpdatePackage::Deb) => "the .deb package",
        Some(ManualUpdatePackage::Rpm) => "the .rpm package",
        None => "the package for your system",
    }
}

fn display_version(version: &str) -> String {
    if version.starts_with('v') {
        version.to_owned()
    } else {
        format!("v{version}")
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
