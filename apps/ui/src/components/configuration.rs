use dioxus::prelude::*;
use whispernow_domain::ipc::{PermissionKind, PermissionStatus};
use whispernow_domain::{
    HistoryLimit, LlmProvider, RecordingMode, Settings, Shortcut, ThemePreference,
};

use super::actions::DisabledReason;
use super::home::LevelMeter;
use super::ui::{Button, Section, Segmented, Select, SettingRow, Toggle};
use super::{PasteHelperNotice, ShortcutCapture};
use crate::{NoticeScope, Route, SaveStatus, SettingKey, UiContext, UiOperation};

/// Theme choices as (setting, select value, label).
const THEMES: [(ThemePreference, &str, &str); 3] = [
    (ThemePreference::System, "system", "Match system"),
    (ThemePreference::Light, "light", "Light"),
    (ThemePreference::Dark, "dark", "Dark"),
];

pub(super) const RECORDING_MODES: [(RecordingMode, &str, &str); 2] = [
    (RecordingMode::Hold, "hold", "Hold"),
    (RecordingMode::Toggle, "toggle", "Toggle"),
];

/// The select value for a history limit: its entry count, or `"unlimited"`.
fn history_limit_value(limit: HistoryLimit) -> String {
    limit
        .max_entries()
        .map_or_else(|| "unlimited".to_owned(), |entries| entries.to_string())
}

pub(super) fn choice_value<T: PartialEq>(
    choices: &[(T, &'static str, &str)],
    current: &T,
) -> &'static str {
    choices
        .iter()
        .find(|(value, _, _)| value == current)
        .map_or("", |(_, value, _)| value)
}

pub(super) fn chosen<T: Copy>(choices: &[(T, &str, &str)], value: &str) -> Option<T> {
    choices
        .iter()
        .find(|(_, candidate, _)| *candidate == value)
        .map(|(setting, _, _)| *setting)
}

/// The live status text and the error a control shows for `status`.
pub(super) fn status_parts(status: SaveStatus) -> (String, Option<String>) {
    match status {
        SaveStatus::Idle => (String::new(), None),
        SaveStatus::Saving => ("Saving…".to_owned(), None),
        SaveStatus::Saved => ("Saved".to_owned(), None),
        SaveStatus::Failed(message) => (String::new(), Some(message)),
    }
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
    let (status, error) = status_parts(management.read().setting_status(setting));
    rsx! {
        SettingRow { id, title, description, error, status, {children} }
    }
}

#[component]
pub(super) fn SettingsLoading() -> Element {
    rsx! { p { role: "status", "Loading settings…" } }
}

#[component]
fn MicrophonePermissionNotice(status: Option<PermissionStatus>) -> Element {
    let (label, guidance) = match status {
        Some(PermissionStatus::Granted) => (
            "Microphone access is available",
            "Starting a recording remains the final permission check.",
        ),
        Some(PermissionStatus::Denied) => (
            "Microphone access is denied",
            "Allow microphone access in system privacy settings, then reopen or retry setup.",
        ),
        _ => (
            "Microphone permission is unknown",
            "Your system will request access when capture starts. If it fails, enable microphone access in system privacy settings.",
        ),
    };
    rsx! { div { class: "permission-status", role: "status",
        strong { "{label}" }
        p { "{guidance}" }
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
                        value: choice_value(&RECORDING_MODES, &settings.recording_mode),
                        options: RECORDING_MODES.iter().map(|(_, value, label)| (*value, *label)).collect::<Vec<_>>(),
                        onchange: move |value: &'static str| if let Some(mode) = chosen(&RECORDING_MODES, value) {
                            context.change_setting(SettingKey::RecordingShortcut, move |settings| settings.recording_mode = mode);
                        },
                    }
                    ShortcutCapture { id: "recording-shortcut", label: "recording shortcut",
                        disabled,
                        shortcut: Some(settings.recording_shortcut.clone()),
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
                    if settings.cancel_shortcut.is_some() {
                        ShortcutCapture { id: "cancel-shortcut", label: "cancel-dictation shortcut",
                            disabled,
                            shortcut: settings.cancel_shortcut.clone(),
                            lone_keys: true,
                            oncapture: move |shortcut: Shortcut| { context.change_setting(SettingKey::CancelShortcut, move |settings| settings.cancel_shortcut = Some(shortcut.clone())); } }
                    }
                }
            }
        }
        Section { id: "appearance", title: "Appearance",
            AutosaveRow { id: "theme", title: "Theme", setting: SettingKey::Theme,
                Select { value: choice_value(&THEMES, &settings.theme),
                    disabled,
                    onchange: move |event: FormEvent| if let Some(theme) = chosen(&THEMES, &event.value()) {
                        context.change_setting(SettingKey::Theme, move |settings| settings.theme = theme);
                    },
                    for (_, value, label) in THEMES { option { value, "{label}" } }
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
                Toggle { checked: settings.auto_start_on_login,
                    disabled,
                    onchange: move |value| { context.change_setting(SettingKey::StartOnLogin, move |settings| settings.auto_start_on_login = value); } }
            }
        }
        if let Some(setup) = paste_helper_setup {
            Section { id: "pasting", title: "Pasting",
                PasteHelperNotice { setup, scope: NoticeScope::Settings }
            }
        }
        if management.read().setup.as_ref().is_some_and(|setup| setup.first_run_complete) {
            Section { id: "setup", title: "Setup",
                SettingRow { id: "run-setup-again", title: "Run setup again",
                    description: "Your settings are kept.",
                    Button {
                        id: "run-setup-again",
                        disabled: restart_blocked.is_some(),
                        aria_describedby: restart_blocked.map(|_| "run-setup-again-disabled-reason"),
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
    let permission = management
        .read()
        .permission_status(PermissionKind::Microphone);
    let test = management.read().microphone_test.clone();
    let test_blocked = context
        .blocked(UiOperation::TestMicrophone)
        .or(context.phase_blocked());
    let testing = management.read().is_busy(UiOperation::TestMicrophone);
    rsx! { div { class: "settings-stack",
        DisabledReason { id: "settings-disabled-reason", reason: settings_blocked }
        Section { id: "microphone", title: "Microphone",
            div { class: "section-note",
                MicrophonePermissionNotice { status: permission }
            }
            AutosaveRow { id: "input-device", title: "Input device",
                setting: SettingKey::InputDevice,
                Select { value: settings.input_device.clone().unwrap_or_default(),
                    disabled,
                    onchange: move |event: FormEvent| {
                        let value = event.value();
                        let device = (!value.is_empty()).then_some(value);
                        context.change_setting(SettingKey::InputDevice, move |settings| settings.input_device.clone_from(&device));
                    },
                    option { value: "", "System default" }
                    for device in devices { option { value: "{device.id}", "{device.display_name}" } }
                }
            }
            SettingRow { id: "audio-test-microphone", title: "Test microphone",
                description: "Listens for three seconds. Nothing is kept.",
                Button {
                    disabled: test_blocked.is_some(),
                    aria_describedby: (test_blocked.is_some() && !testing).then_some("audio-test-disabled-reason"),
                    onclick: move |_| context.test_microphone(),
                    if testing { "Listening…" } else { "Test microphone" }
                }
            }
            if testing {
                p { class: "section-note status-detail", role: "status", "Say a few words." }
            } else {
                DisabledReason { id: "audio-test-disabled-reason", reason: test_blocked }
            }
            if let Some(test) = test {
                div { class: "section-note", LevelMeter { test } }
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
                Select { value: history_limit_value(settings.history_limit),
                    disabled,
                    onchange: move |event: FormEvent| {
                        if let Some(limit) = HistoryLimit::ALL
                            .into_iter()
                            .find(|limit| history_limit_value(*limit) == event.value())
                        {
                            context.change_setting(SettingKey::HistoryLimit, move |settings| settings.history_limit = limit);
                        }
                    },
                    for limit in HistoryLimit::ALL {
                        option { value: history_limit_value(limit), "{limit.label()}" }
                    }
                }
            }
            AutosaveRow { id: "history-keep-audio", title: "Keep audio",
                description: "About 2 MB per minute. Failed and cancelled dictations always keep it.",
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
    if !settings.cleanup_enabled {
        return None;
    }
    let provider = settings.llm_provider;
    match settings.provider_settings(provider).api_base.as_ref() {
        Some(endpoint) if endpoint.is_on_this_computer() => None,
        Some(endpoint) => Some(endpoint.origin()),
        None => match provider {
            LlmProvider::LocalCleanup | LlmProvider::Ollama => None,
            LlmProvider::Custom => Some("a custom server".to_owned()),
            _ => Some(provider.label().to_owned()),
        },
    }
}
