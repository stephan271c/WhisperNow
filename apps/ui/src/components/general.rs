use dioxus::prelude::*;
use whispernow_domain::{RecordingMode, SettingChange, Shortcut, ThemePreference};

use super::paste_helper_notice::PasteHelperNotice;
use super::settings::{AutosaveRow, SettingsLoading};
use super::shortcut::ShortcutCapture;
use super::ui::{Button, Section, Segmented, SegmentedOption, Select, SettingRow, Toggle};
use crate::UiContext;
use crate::blocked_reason::DisabledReason;
use crate::management::SettingKey;
use crate::operations::UiOperation;
use crate::route::Route;

const THEMES: [(ThemePreference, &str); 3] = [
    (ThemePreference::System, "Match system"),
    (ThemePreference::Light, "Light"),
    (ThemePreference::Dark, "Dark"),
];

pub(super) const RECORDING_MODES: [SegmentedOption<RecordingMode>; 2] = [
    SegmentedOption {
        value: RecordingMode::Hold,
        suffix: "hold",
        label: "Hold",
    },
    SegmentedOption {
        value: RecordingMode::Toggle,
        suffix: "toggle",
        label: "Toggle",
    },
];

#[component]
pub(super) fn GeneralPage() -> Element {
    let context = use_context::<UiContext>();
    use_effect(move || context.load_permissions());
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
    let recording_shortcut = settings.recording_shortcut.clone();
    let recording_mode = settings.recording_mode;
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
                            context.change_setting(SettingChange::RecordingShortcut { shortcut: recording_shortcut.clone(), mode });
                        },
                    }
                    ShortcutCapture { id: "recording-shortcut", label: "recording shortcut",
                        disabled,
                        shortcut: settings.recording_shortcut.clone(),
                        oncapture: move |shortcut: Shortcut| context.change_setting(SettingChange::RecordingShortcut { shortcut, mode: recording_mode }) }
                }
            }
            AutosaveRow { id: "cancel-dictation", title: "Cancel dictation",
                setting: SettingKey::CancelShortcut,
                div { class: "shortcut-control",
                    Toggle { checked: settings.cancel_shortcut.is_some(),
                        disabled,
                        onchange: move |enabled: bool| {
                            context.change_setting(SettingChange::CancelShortcut(enabled.then(Shortcut::default_cancel)));
                        }
                    }
                    if let Some(cancel) = settings.cancel_shortcut.clone() {
                        ShortcutCapture { id: "cancel-shortcut", label: "cancel-dictation shortcut",
                            disabled,
                            shortcut: cancel,
                            lone_keys: true,
                            oncapture: move |shortcut: Shortcut| context.change_setting(SettingChange::CancelShortcut(Some(shortcut))) }
                    }
                }
            }
        }
        Section { id: "appearance", title: "Appearance",
            AutosaveRow { id: "theme", title: "Theme", setting: SettingKey::Theme,
                Select { value: settings.theme,
                    options: Vec::from(THEMES.map(|(theme, label)| (theme, label.to_owned()))),
                    disabled,
                    onchange: move |theme: ThemePreference| {
                        context.change_setting(SettingChange::Theme(theme));
                    },
                }
            }
        }
        Section { id: "startup", title: "Startup",
            AutosaveRow { id: "start-minimized", title: "Start minimized",
                setting: SettingKey::StartMinimized,
                Toggle { checked: settings.start_minimized,
                    disabled,
                    onchange: move |value| context.change_setting(SettingChange::StartMinimized(value)) }
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
        if management.read().setup_complete() {
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
