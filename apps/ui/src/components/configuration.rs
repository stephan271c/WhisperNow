use dioxus::prelude::*;
use whispernow_domain::ipc::{AudioDevice, PermissionKind, PermissionStatus};
use whispernow_domain::{
    CleanupTarget, HistoryLimit, HistoryRetention, RecordingMode, RemoteProvider, Settings,
    Shortcut, ThemePreference,
};

use super::microphone_test::MicrophoneTest;
use super::ui::{Button, Section, Segmented, Select, SettingRow, Toggle};
use super::{PasteHelperNotice, ShortcutCapture};
use crate::blocked_reason::DisabledReason;
use crate::{Route, SettingKey, UiContext, UiOperation};

const THEMES: [(ThemePreference, &str); 3] = [
    (ThemePreference::System, "Match system"),
    (ThemePreference::Light, "Light"),
    (ThemePreference::Dark, "Dark"),
];

pub(super) const RECORDING_MODES: [(RecordingMode, &str, &str); 2] = [
    (RecordingMode::Hold, "hold", "Hold"),
    (RecordingMode::Toggle, "toggle", "Toggle"),
];

/// Input device choices for a select: the system default (`""`), then each device. Give the select
/// the saved device as its placeholder, so one that is no longer connected still shows.
pub(super) fn device_options(devices: &[AudioDevice]) -> Vec<(String, String)> {
    std::iter::once((String::new(), "System default".to_owned()))
        .chain(
            devices
                .iter()
                .map(|device| (device.id.clone(), device.display_name.clone())),
        )
        .collect()
}

#[component]
pub(super) fn AutosaveRow(
    id: String,
    title: String,
    description: Option<String>,
    setting: SettingKey,
    children: Element,
) -> Element {
    let context = use_context::<UiContext>();
    let management = context.management;
    let error = management.read().setting_error(setting);
    rsx! {
        SettingRow { id, title, description, error, {children} }
    }
}

#[component]
pub(super) fn SettingsLoading() -> Element {
    rsx! { p { role: "status", "Loading settings…" } }
}

#[component]
fn MicrophonePermissionNotice(status: Option<PermissionStatus>) -> Element {
    let (label, guidance) = match status {
        Some(PermissionStatus::Granted) => ("Microphone access is available", None),
        Some(PermissionStatus::Denied) => (
            "Microphone access is denied",
            Some("Allow microphone access in system privacy settings, then reopen or retry setup."),
        ),
        _ => (
            "Microphone permission is unknown",
            Some(
                "Your system will request access when capture starts. If it fails, enable microphone access in system privacy settings.",
            ),
        ),
    };
    rsx! { div { class: "permission-status", role: "status",
        strong { "{label}" }
        if let Some(guidance) = guidance {
            p { "{guidance}" }
        }
    } }
}

#[component]
pub(super) fn GeneralPage() -> Element {
    let context = use_context::<UiContext>();
    let settings_blocked = context.settings_blocked();
    let disabled = settings_blocked.is_some();
    let management = context.management;
    let navigator = context.navigator;
    let Some(settings) = management.read().shown_settings().cloned() else {
        return rsx! { SettingsLoading {} };
    };
    let start_on_login = management
        .read()
        .settings
        .as_ref()
        .is_some_and(|view| view.start_on_login);
    let restart_blocked = context
        .blocked(UiOperation::RestartSetup)
        .or(context.settings_blocked());
    use_effect(move || context.load_permissions());
    let paste_helper_setup = management
        .read()
        .permissions
        .as_ref()
        .and_then(|permissions| permissions.paste_helper_setup.clone());
    rsx! { div { class: "settings-stack",
        DisabledReason { id: "settings-disabled-reason", reason: settings_blocked }
        Section { id: "shortcuts", title: "Shortcuts",
            AutosaveRow { id: "recording-mode", title: "Recording shortcut",
                description: match settings.recording_mode {
                    RecordingMode::Hold => "Hold while you speak.",
                    RecordingMode::Toggle => "Press to start, press again to stop.",
                },
                setting: SettingKey::RecordingShortcut,
                div { class: "shortcut-control",
                    Segmented {
                        disabled,
                        value: settings.recording_mode,
                        options: RECORDING_MODES.to_vec(),
                        onchange: move |mode: RecordingMode| {
                            context.change_setting(SettingKey::RecordingShortcut, move |settings| settings.recording_mode = mode);
                        },
                    }
                    ShortcutCapture { id: "recording-shortcut", label: "recording shortcut",
                        disabled,
                        shortcut: settings.recording_shortcut.clone(),
                        oncapture: move |shortcut: Shortcut| { context.change_setting(SettingKey::RecordingShortcut, move |settings| settings.recording_shortcut = shortcut.clone()); } }
                }
            }
            AutosaveRow { id: "cancel-dictation", title: "Cancel dictation",
                setting: SettingKey::CancelShortcut,
                div { class: "shortcut-control",
                    Toggle { checked: settings.cancel_shortcut.is_some(),
                        disabled,
                        onchange: move |enabled: bool| {
                            context.change_setting(SettingKey::CancelShortcut, move |settings| {
                                settings.cancel_shortcut = enabled.then(Shortcut::default_cancel);
                            });
                        }
                    }
                    if let Some(cancel) = settings.cancel_shortcut.clone() {
                        ShortcutCapture { id: "cancel-shortcut", label: "cancel-dictation shortcut",
                            disabled,
                            shortcut: cancel,
                            lone_keys: true,
                            oncapture: move |shortcut: Shortcut| { context.change_setting(SettingKey::CancelShortcut, move |settings| settings.cancel_shortcut = Some(shortcut.clone())); } }
                    }
                }
            }
        }
        Section { id: "appearance", title: "Appearance",
            AutosaveRow { id: "theme", title: "Theme", setting: SettingKey::Theme,
                Select { value: settings.theme,
                    options: THEMES.map(|(theme, label)| (theme, label.to_owned())).to_vec(),
                    disabled,
                    onchange: move |theme: ThemePreference| {
                        context.change_setting(SettingKey::Theme, move |settings| settings.theme = theme);
                    },
                }
            }
        }
        Section { id: "startup", title: "Startup",
            AutosaveRow { id: "start-minimized", title: "Start minimized",
                setting: SettingKey::StartMinimized,
                Toggle { checked: settings.start_minimized,
                    disabled,
                    onchange: move |value| { context.change_setting(SettingKey::StartMinimized, move |settings| settings.start_minimized = value); } }
            }
            AutosaveRow { id: "start-on-login", title: "Start on login",
                setting: SettingKey::StartOnLogin,
                Toggle { checked: start_on_login,
                    disabled,
                    onchange: move |value| context.set_start_on_login(value) }
            }
        }
        if let Some(setup) = paste_helper_setup {
            Section { id: "pasting", title: "Pasting",
                PasteHelperNotice { setup }
            }
        }
        if management.read().first_run_complete() {
            Section { id: "setup", title: "Setup",
                SettingRow { id: "run-setup-again", title: "Run setup again",
                    Button {
                        id: "run-setup-again",
                        disabled: restart_blocked.is_some(),
                        aria_describedby: restart_blocked.as_ref().map(|_| "run-setup-again-disabled-reason"),
                        onclick: move |_| {
                            context.restart_setup();
                            navigator.go(Route::Setup);
                        },
                        "Run setup again"
                    }
                }
                DisabledReason { id: "run-setup-again-disabled-reason", reason: restart_blocked }
            }
        }
    } }
}

#[component]
pub(super) fn AudioPage() -> Element {
    let context = use_context::<UiContext>();
    let settings_blocked = context.settings_blocked();
    let disabled = settings_blocked.is_some();
    let management = context.management;
    let Some(settings) = management.read().shown_settings().cloned() else {
        return rsx! { SettingsLoading {} };
    };
    let devices = management.read().devices.clone();
    let saved_device = settings.input_device.clone().unwrap_or_default();
    let permission = management
        .read()
        .permission_status(PermissionKind::Microphone);
    rsx! { div { class: "settings-stack",
        DisabledReason { id: "settings-disabled-reason", reason: settings_blocked }
        Section { id: "microphone", title: "Microphone",
            div { class: "section-note",
                MicrophonePermissionNotice { status: permission }
            }
            AutosaveRow { id: "input-device", title: "Input device",
                setting: SettingKey::InputDevice,
                Select { value: saved_device.clone(),
                    placeholder: saved_device,
                    options: device_options(&devices),
                    disabled,
                    onchange: move |value: String| {
                        let device = (!value.is_empty()).then_some(value);
                        context.change_setting(SettingKey::InputDevice, move |settings| settings.input_device.clone_from(&device));
                    },
                }
            }
            SettingRow { id: "audio-test-microphone", title: "Test microphone",
                MicrophoneTest { id_prefix: "audio-test" }
            }
        }
    } }
}

#[component]
pub(super) fn PrivacyPage() -> Element {
    let context = use_context::<UiContext>();
    let settings_blocked = context.settings_blocked();
    let disabled = settings_blocked.is_some();
    let management = context.management;
    let Some(settings) = management.read().shown_settings().cloned() else {
        return rsx! { SettingsLoading {} };
    };
    let summary = cleanup_destination(&settings).map_or_else(
        || "Your recordings and transcripts stay on this computer.".to_owned(),
        |destination| {
            format!("Transcripts are sent to {destination} for AI cleanup. Recordings stay on this computer.")
        },
    );
    rsx! { div { class: "settings-stack",
        DisabledReason { id: "settings-disabled-reason", reason: settings_blocked }
        p { id: "privacy-summary", "{summary}" }
        Section { id: "transcript-history", title: "History",
            AutosaveRow { id: "history-limit", title: "Keep", setting: SettingKey::HistoryLimit,
                Select { value: settings.history_limit,
                    options: HistoryLimit::ALL.map(|limit| (limit, limit.label().to_owned())).to_vec(),
                    disabled,
                    onchange: move |limit: HistoryLimit| {
                        context.change_setting(SettingKey::HistoryLimit, move |settings| settings.history_limit = limit);
                    },
                }
            }
            AutosaveRow { id: "history-retention", title: "For",
                description: "Older entries are deleted.",
                setting: SettingKey::HistoryRetention,
                Select { value: settings.history_retention,
                    options: HistoryRetention::ALL.map(|retention| (retention, retention.label().to_owned())).to_vec(),
                    disabled,
                    onchange: move |retention: HistoryRetention| {
                        context.change_setting(SettingKey::HistoryRetention, move |settings| settings.history_retention = retention);
                    },
                }
            }
            AutosaveRow { id: "history-keep-audio", title: "Keep audio",
                description: "About 2 MB per minute.",
                setting: SettingKey::KeepAudio,
                Toggle { checked: settings.keep_audio,
                    disabled,
                    onchange: move |value| { context.change_setting(SettingKey::KeepAudio, move |settings| settings.keep_audio = value); } }
            }
        }
    } }
}

/// Where AI cleanup sends a transcript, or `None` when it stays on this computer.
pub(crate) fn cleanup_destination(settings: &Settings) -> Option<String> {
    let CleanupTarget::Remote(target) = settings.active_cleanup_target()? else {
        return None;
    };
    let provider = target.provider.provider();
    let endpoint = match &target.provider {
        RemoteProvider::Ollama(None) => return None,
        RemoteProvider::OpenAi(endpoint)
        | RemoteProvider::Anthropic(endpoint)
        | RemoteProvider::OpenRouter(endpoint)
        | RemoteProvider::Gemini(endpoint)
        | RemoteProvider::Ollama(endpoint) => endpoint.as_ref(),
        RemoteProvider::Custom(endpoint) => Some(endpoint),
    };
    match endpoint {
        Some(endpoint) if endpoint.is_on_this_computer() => None,
        Some(endpoint) => Some(endpoint.origin()),
        None => Some(provider.label().to_owned()),
    }
}
