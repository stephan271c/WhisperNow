use dioxus::prelude::*;
use whispernow_domain::ipc::{AudioDevice, PermissionKind, PermissionStatus};
use whispernow_domain::{AudioDeviceId, SettingChange};

use super::microphone_test::MicrophoneTest;
use super::settings::{AutosaveRow, SettingsLoading};
use super::ui::{Section, Select, SettingRow};
use crate::UiContext;
use crate::blocked_reason::DisabledReason;
use crate::management::SettingKey;

pub(super) fn device_options(devices: &[AudioDevice]) -> Vec<(Option<AudioDeviceId>, String)> {
    std::iter::once((None, "System default".to_owned()))
        .chain(
            devices
                .iter()
                .map(|device| (Some(device.id.clone()), device.display_name.clone())),
        )
        .collect()
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
pub(super) fn AudioPage() -> Element {
    let context = use_context::<UiContext>();
    let settings_blocked = context.settings_blocked();
    let disabled = settings_blocked.is_some();
    let management = context.management;
    let Some(settings) = management.read().shown_settings().cloned() else {
        return rsx! { SettingsLoading {} };
    };
    let devices = device_options(management.read().devices.as_deref().unwrap_or_default());
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
                Select { value: settings.input_device.clone(),
                    placeholder: settings.input_device.as_ref().map(|id| id.as_str().to_owned()),
                    options: devices,
                    disabled,
                    onchange: move |device| context.change_setting(SettingChange::InputDevice(device)),
                }
            }
            SettingRow { id: "audio-test-microphone", title: "Test microphone",
                MicrophoneTest { id_prefix: "audio-test" }
            }
        }
    } }
}
