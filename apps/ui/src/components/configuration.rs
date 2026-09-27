//! General & Shortcuts, Audio, and Privacy & Data, and the row every self-saving control uses.

use dioxus::prelude::*;
use whispernow_api::{
    HistoryLimit, LlmProvider, OverlayPosition, OverlayStyle, PermissionKindDto, Settings,
    Shortcut, ThemePreference,
};

use super::actions::{DisabledReason, use_actions};
use super::home::LevelMeter;
use super::ui::{Button, Section, Select, SettingRow, Toggle};
use super::{MicrophonePermissionNotice, PasteHelperNotice, ShortcutCapture, use_navigator};
use crate::{BlockedReason, NoticeScope, Route, SaveStatus, SettingKey, UiContext, UiOperation};

/// Theme choices as (setting, select value, label).
const THEMES: [(ThemePreference, &str, &str); 3] = [
    (ThemePreference::System, "system", "Match system"),
    (ThemePreference::Light, "light", "Light"),
    (ThemePreference::Dark, "dark", "Dark"),
];

/// Recording overlay styles as (setting, select value, label).
const OVERLAY_STYLES: [(OverlayStyle, &str, &str); 3] = [
    (
        OverlayStyle::WithLiveText,
        "with_live_text",
        "With live text",
    ),
    (OverlayStyle::Minimal, "minimal", "Minimal"),
    (OverlayStyle::Off, "off", "Off"),
];

const OVERLAY_POSITIONS: [(OverlayPosition, &str, &str); 2] = [
    (OverlayPosition::Bottom, "bottom", "Bottom of the screen"),
    (OverlayPosition::Top, "top", "Top of the screen"),
];

/// The select value for a history limit: its entry count, or `"unlimited"`.
fn history_limit_value(limit: HistoryLimit) -> String {
    limit
        .max_entries()
        .map_or_else(|| "unlimited".to_owned(), |entries| entries.to_string())
}

/// The select value for `current` in `choices`.
fn choice_value<T: PartialEq>(choices: &[(T, &'static str, &str)], current: &T) -> &'static str {
    choices
        .iter()
        .find(|(value, _, _)| value == current)
        .map_or("", |(_, value, _)| value)
}

/// The setting a select value names in `choices`.
fn chosen<T: Copy>(choices: &[(T, &str, &str)], value: &str) -> Option<T> {
    choices
        .iter()
        .find(|(_, candidate, _)| *candidate == value)
        .map(|(setting, _, _)| *setting)
}

/// The live status text and the error a control shows for `status`.
pub(super) fn status_parts(status: SaveStatus) -> (Option<String>, Option<String>) {
    match status {
        SaveStatus::Idle => (Some(String::new()), None),
        SaveStatus::Saving => (Some("Saving…".to_owned()), None),
        SaveStatus::Saved => (Some("Saved".to_owned()), None),
        SaveStatus::Failed(message) => (Some(String::new()), Some(message)),
    }
}

/// A settings row whose control saves itself: "Saving…", "Saved", or why the change was not
/// saved appear beside the control.
#[component]
pub(super) fn AutosaveRow(
    id: String,
    title: String,
    description: Option<String>,
    setting: SettingKey,
    children: Element,
) -> Element {
    let management = use_context::<UiContext>().management;
    let (status, error) = status_parts(management.read().setting_status(&setting));
    rsx! {
        SettingRow { id, title, description, error, status, {children} }
    }
}

/// Explains, once per page, why settings cannot be changed right now.
#[component]
pub(super) fn SettingsBlockedReason() -> Element {
    let blocked = use_actions().blocked(UiOperation::SaveSettings);
    rsx! { DisabledReason { id: "settings-disabled-reason", reason: blocked } }
}

/// Shown until the desktop sends the settings.
#[component]
pub(super) fn SettingsLoading() -> Element {
    rsx! { p { role: "status", "Loading settings…" } }
}

/// General & Shortcuts: recording shortcuts, appearance and startup.
#[component]
pub(super) fn GeneralPage() -> Element {
    let actions = use_actions();
    let management = use_context::<UiContext>().management;
    // Turning toggle-to-record on waits for a shortcut before anything is saved.
    let mut choosing_toggle = use_signal(|| false);
    let navigator = use_navigator();
    let Some(settings) = management.read().shown_settings().cloned() else {
        return rsx! { SettingsLoading {} };
    };
    let has_toggle = settings.toggle_shortcut.is_some();
    let toggle_on = has_toggle || choosing_toggle();
    let restart_blocked = actions.blocked(UiOperation::RestartSetup);
    use_effect(move || actions.load_permissions());
    let paste_helper_setup = management
        .read()
        .permissions
        .as_ref()
        .and_then(|permissions| permissions.paste_helper_setup.clone());
    rsx! { div { class: "settings-stack",
        SettingsBlockedReason {}
        Section { id: "shortcuts", title: "Recording shortcuts",
            description: "Start a recording from any application. If a shortcut cannot be registered, your previous shortcuts stay active.",
            AutosaveRow { id: "hold-to-record", title: "Hold to record",
                description: "Hold the shortcut while you speak, then release it to transcribe.",
                setting: SettingKey::HoldShortcut,
                div { class: "shortcut-control",
                    Toggle { checked: settings.hold_shortcut_enabled,
                        onchange: move |enabled| { actions.change_setting(SettingKey::HoldShortcut, move |settings| settings.hold_shortcut_enabled = enabled); } }
                    if settings.hold_shortcut_enabled {
                        ShortcutCapture { id: "hold-shortcut", label: "hold-to-record shortcut",
                            shortcut: Some(settings.hold_shortcut.clone()),
                            oncapture: move |shortcut: Shortcut| { actions.change_setting(SettingKey::HoldShortcut, move |settings| settings.hold_shortcut = shortcut.clone()); } }
                    }
                }
            }
            AutosaveRow { id: "toggle-to-record", title: "Toggle to record",
                description: if !has_toggle && toggle_on {
                    "Choose a shortcut to turn this on."
                } else {
                    "Press once to start recording and again to stop and transcribe."
                },
                setting: SettingKey::ToggleShortcut,
                div { class: "shortcut-control",
                    Toggle { checked: toggle_on,
                        onchange: move |enabled: bool| {
                            if enabled {
                                choosing_toggle.set(true);
                            } else {
                                choosing_toggle.set(false);
                                if has_toggle {
                                    actions.change_setting(SettingKey::ToggleShortcut, |settings| settings.toggle_shortcut = None);
                                }
                            }
                        }
                    }
                    if toggle_on {
                        ShortcutCapture { id: "toggle-shortcut", label: "toggle-to-record shortcut",
                            shortcut: settings.toggle_shortcut.clone(),
                            oncapture: move |shortcut: Shortcut| {
                                if actions.change_setting(SettingKey::ToggleShortcut, move |settings| settings.toggle_shortcut = Some(shortcut.clone())) {
                                    choosing_toggle.set(false);
                                }
                            }
                        }
                    }
                }
            }
            AutosaveRow { id: "cancel-dictation", title: "Cancel dictation",
                description: "Stops a dictation while it records or transcribes, without pasting; it stays in History to transcribe again. The shortcut is active only during a dictation. Where the system cannot add shortcuts on demand, such as some Wayland desktops, use ✕ on the recording overlay or Cancel Recording in the tray menu.",
                setting: SettingKey::CancelShortcut,
                div { class: "shortcut-control",
                    Toggle { checked: settings.cancel_shortcut.is_some(),
                        onchange: move |enabled: bool| {
                            actions.change_setting(SettingKey::CancelShortcut, move |settings| {
                                settings.cancel_shortcut = enabled.then(Shortcut::default_cancel);
                            });
                        }
                    }
                    if settings.cancel_shortcut.is_some() {
                        ShortcutCapture { id: "cancel-shortcut", label: "cancel-dictation shortcut",
                            shortcut: settings.cancel_shortcut.clone(),
                            lone_keys: true,
                            oncapture: move |shortcut: Shortcut| { actions.change_setting(SettingKey::CancelShortcut, move |settings| settings.cancel_shortcut = Some(shortcut.clone())); } }
                    }
                }
            }
        }
        Section { id: "appearance", title: "Appearance",
            AutosaveRow { id: "theme", title: "Theme",
                description: "Match your system, or keep WhisperNow light or dark.",
                setting: SettingKey::Theme,
                Select { value: choice_value(&THEMES, &settings.theme),
                    onchange: move |event: FormEvent| if let Some(theme) = chosen(&THEMES, &event.value()) {
                        actions.change_setting(SettingKey::Theme, move |settings| settings.theme = theme);
                    },
                    for (_, value, label) in THEMES { option { value, "{label}" } }
                }
            }
            AutosaveRow { id: "overlay-style", title: "Recording overlay",
                description: "Show dictation progress near the cursor, with or without the words so far.",
                setting: SettingKey::OverlayStyle,
                Select { value: choice_value(&OVERLAY_STYLES, &settings.overlay_style),
                    onchange: move |event: FormEvent| if let Some(style) = chosen(&OVERLAY_STYLES, &event.value()) {
                        actions.change_setting(SettingKey::OverlayStyle, move |settings| settings.overlay_style = style);
                    },
                    for (_, value, label) in OVERLAY_STYLES { option { value, "{label}" } }
                }
            }
            AutosaveRow { id: "overlay-position", title: "Overlay position",
                description: "Where the overlay appears on the screen under the cursor.",
                setting: SettingKey::OverlayPosition,
                Select { value: choice_value(&OVERLAY_POSITIONS, &settings.overlay_position),
                    disabled: settings.overlay_style == OverlayStyle::Off,
                    onchange: move |event: FormEvent| if let Some(position) = chosen(&OVERLAY_POSITIONS, &event.value()) {
                        actions.change_setting(SettingKey::OverlayPosition, move |settings| settings.overlay_position = position);
                    },
                    for (_, value, label) in OVERLAY_POSITIONS { option { value, "{label}" } }
                }
            }
        }
        Section { id: "startup", title: "Startup", description: "Keep WhisperNow ready when you need it.",
            AutosaveRow { id: "start-minimized", title: "Start minimized",
                description: "Open in the background, without showing this window.",
                setting: SettingKey::StartMinimized,
                Toggle { checked: settings.start_minimized,
                    onchange: move |value| { actions.change_setting(SettingKey::StartMinimized, move |settings| settings.start_minimized = value); } }
            }
            AutosaveRow { id: "start-on-login", title: "Start on login",
                description: "Start WhisperNow when you sign in to this computer.",
                setting: SettingKey::StartOnLogin,
                Toggle { checked: settings.auto_start_on_login,
                    onchange: move |value| { actions.change_setting(SettingKey::StartOnLogin, move |settings| settings.auto_start_on_login = value); } }
            }
        }
        if let Some(setup) = paste_helper_setup {
            Section { id: "pasting", title: "Pasting",
                PasteHelperNotice { setup, scope: NoticeScope::Settings }
            }
        }
        if settings.first_run_complete {
            Section { id: "setup", title: "Setup",
                SettingRow { id: "run-setup-again", title: "Run setup again",
                    description: "Check permissions, the speech model and shortcuts, and try a test dictation. Your settings are kept.",
                    Button {
                        id: "run-setup-again",
                        disabled: restart_blocked.is_some(),
                        aria_describedby: restart_blocked.map(|_| "run-setup-again-disabled-reason"),
                        onclick: move |_| {
                            actions.restart_setup();
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

/// Audio: the input device, a microphone test, and microphone permission.
#[component]
pub(super) fn AudioPage() -> Element {
    let actions = use_actions();
    let management = use_context::<UiContext>().management;
    let Some(settings) = management.read().shown_settings().cloned() else {
        return rsx! { SettingsLoading {} };
    };
    let devices = management.read().devices.clone();
    let permission = management
        .read()
        .permission_status(PermissionKindDto::Microphone);
    let test = management.read().microphone_test.clone();
    let test_blocked = actions.blocked(UiOperation::TestMicrophone);
    let testing = test_blocked == Some(BlockedReason::CommandPending);
    rsx! { div { class: "settings-stack",
        SettingsBlockedReason {}
        Section { id: "microphone", title: "Microphone",
            description: "Choose the microphone you use for dictation.",
            div { class: "section-note",
                MicrophonePermissionNotice { status: permission }
            }
            AutosaveRow { id: "input-device", title: "Input device",
                description: "The system default follows the microphone your computer is using.",
                setting: SettingKey::InputDevice,
                Select { value: settings.input_device.clone().unwrap_or_default(),
                    onchange: move |event: FormEvent| {
                        let value = event.value();
                        let device = (!value.is_empty()).then_some(value);
                        actions.change_setting(SettingKey::InputDevice, move |settings| settings.input_device.clone_from(&device));
                    },
                    option { value: "", "System default" }
                    for device in devices { option { value: "{device.id}", "{device.display_name}" } }
                }
            }
            SettingRow { id: "audio-test-microphone", title: "Test microphone",
                description: "Listens for three seconds and shows how loud you sound. Nothing is kept.",
                Button {
                    disabled: test_blocked.is_some(),
                    aria_describedby: (test_blocked.is_some() && !testing).then_some("audio-test-disabled-reason"),
                    onclick: move |_| actions.test_microphone(),
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

/// Privacy & Data: what leaves this computer, what is kept, and removing it.
#[component]
pub(super) fn PrivacyPage() -> Element {
    let actions = use_actions();
    let management = use_context::<UiContext>().management;
    let Some(settings) = management.read().shown_settings().cloned() else {
        return rsx! { SettingsLoading {} };
    };
    let cleanup_on = settings.active_prompt_id.is_some();
    rsx! { div { class: "settings-stack",
        SettingsBlockedReason {}
        Section { id: "what-leaves", title: "What leaves this computer",
            div { class: "section-prose",
                p { "Speech recognition runs on this computer. Your recordings are never uploaded." }
                p {
                    if cleanup_on {
                        "AI cleanup is on: each transcript is sent to {cleanup_recipient(&settings)} before it is pasted."
                    } else {
                        "AI cleanup is off, so transcripts stay on this computer. When you turn it on, each transcript is sent to the provider you choose."
                    }
                }
                p { "Checking for updates asks GitHub whether a newer release exists." }
                p { "Transcripts are pasted through the clipboard. WhisperNow then puts back what was on it, unless you copied something new in the meantime. Clipboard managers and sync tools may still see the transcript." }
            }
        }
        Section { id: "transcript-history", title: "History",
            description: "The oldest dictations are removed once the limit is reached.",
            AutosaveRow { id: "history-limit", title: "Keep", setting: SettingKey::HistoryLimit,
                Select { value: history_limit_value(settings.history_limit),
                    onchange: move |event: FormEvent| {
                        if let Some(limit) = HistoryLimit::ALL
                            .into_iter()
                            .find(|limit| history_limit_value(*limit) == event.value())
                        {
                            actions.change_setting(SettingKey::HistoryLimit, move |settings| settings.history_limit = limit);
                        }
                    },
                    for limit in HistoryLimit::ALL {
                        option { value: history_limit_value(limit), "{limit.label()}" }
                    }
                }
            }
            AutosaveRow { id: "history-keep-audio", title: "Keep audio",
                description: "Recordings use about 2 MB per minute. Failed and cancelled dictations always keep their audio.",
                setting: SettingKey::KeepAudio,
                Toggle { checked: settings.keep_audio,
                    onchange: move |value| { actions.change_setting(SettingKey::KeepAudio, move |settings| settings.keep_audio = value); } }
            }
        }
    } }
}

/// Where the active provider sends a transcript, in words.
pub(crate) fn cleanup_recipient(settings: &Settings) -> String {
    if let Some(api_base) = settings.provider_settings(settings.llm_provider).api_base {
        let origin = api_base.origin();
        return match settings.llm_provider {
            LlmProvider::Custom => format!("the custom OpenAI-compatible provider at {origin}"),
            _ => format!("the configured API endpoint at {origin}"),
        };
    }

    match settings.llm_provider {
        LlmProvider::OpenAi => "the configured OpenAI provider".to_owned(),
        LlmProvider::Anthropic => "the configured Anthropic provider".to_owned(),
        LlmProvider::OpenRouter => "the configured OpenRouter provider".to_owned(),
        LlmProvider::Ollama => "the configured Ollama provider".to_owned(),
        LlmProvider::LocalCleanup => "the local cleanup model".to_owned(),
        LlmProvider::Gemini => "the configured Gemini provider".to_owned(),
        LlmProvider::Custom => "the custom OpenAI-compatible provider".to_owned(),
    }
}
