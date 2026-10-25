use dioxus::prelude::*;
use whispernow_domain::{
    CleanupTarget, HistoryLimit, HistoryRetention, RemoteProvider, SettingChange, Settings,
};

use super::settings::{AutosaveRow, SettingsLoading};
use super::ui::{Section, Select, Toggle};
use crate::UiContext;
use crate::blocked_reason::DisabledReason;
use crate::management::SettingKey;

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
                    options: Vec::from(HistoryLimit::ALL.map(|limit| (limit, limit.label().to_owned()))),
                    disabled,
                    onchange: move |limit: HistoryLimit| {
                        context.change_setting(SettingChange::HistoryLimit(limit));
                    },
                }
            }
            AutosaveRow { id: "history-retention", title: "For",
                description: "Older entries are deleted.",
                setting: SettingKey::HistoryRetention,
                Select { value: settings.history_retention,
                    options: Vec::from(HistoryRetention::ALL.map(|retention| (retention, retention.label().to_owned()))),
                    disabled,
                    onchange: move |retention: HistoryRetention| {
                        context.change_setting(SettingChange::HistoryRetention(retention));
                    },
                }
            }
            AutosaveRow { id: "history-keep-audio", title: "Keep audio",
                description: "About 2 MB per minute.",
                setting: SettingKey::KeepAudio,
                Toggle { checked: settings.keep_audio,
                    disabled,
                    onchange: move |value| context.change_setting(SettingChange::KeepAudio(value)) }
            }
        }
    } }
}

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
