use dioxus::prelude::*;
use whispernow_domain::{
    ApiKey, LlmProvider, ModelAvailability, ModelFamily, ModelId, ProviderEndpoint,
    ProviderSettings,
};

use super::NoticeRegion;
use super::configuration::{AutosaveRow, SettingsLoading};
use super::models::{ModelRow, ModelRowKind};
use super::ui::{
    Button, ButtonVariant, Combobox, ConfirmButton, DraftText, Section, Segmented, Select,
    SettingRow, TextInput, Toggle,
};
use crate::blocked_reason::DisabledReason;
use crate::client;
use crate::{NoticeScope, SettingKey, UiContext, UiOperation};

/// Where cleanup runs: one of the app's own cleanup models, an LLM server on this computer,
/// or a hosted API.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CleanupSource {
    WhisperNow,
    LocalLlm,
    Api,
}

impl CleanupSource {
    const fn of(provider: LlmProvider) -> Self {
        match provider {
            LlmProvider::LocalCleanup => Self::WhisperNow,
            LlmProvider::Ollama | LlmProvider::Custom => Self::LocalLlm,
            LlmProvider::OpenAi
            | LlmProvider::Anthropic
            | LlmProvider::Gemini
            | LlmProvider::OpenRouter => Self::Api,
        }
    }

    /// The first is chosen when the user switches to this source.
    const fn providers(self) -> &'static [LlmProvider] {
        match self {
            Self::WhisperNow => &[LlmProvider::LocalCleanup],
            Self::LocalLlm => &[LlmProvider::Ollama, LlmProvider::Custom],
            Self::Api => &[
                LlmProvider::OpenAi,
                LlmProvider::Anthropic,
                LlmProvider::Gemini,
                LlmProvider::OpenRouter,
            ],
        }
    }
}

const SOURCES: [(CleanupSource, &str, &str); 3] = [
    (CleanupSource::WhisperNow, "whispernow", "WhisperNow model"),
    (CleanupSource::LocalLlm, "local", "Local LLM"),
    (CleanupSource::Api, "api", "API provider"),
];

#[component]
pub(super) fn AiCleanupPage() -> Element {
    let context = use_context::<UiContext>();
    let settings_blocked = context.settings_blocked();
    let disabled = settings_blocked.is_some();
    let management = context.management;
    let Some(settings) = management.read().shown_settings().cloned() else {
        return rsx! { SettingsLoading {} };
    };
    let provider = settings.llm_provider;
    let source = CleanupSource::of(provider);
    let pick = move |provider: LlmProvider| {
        context.change_setting(SettingKey::ActiveProvider, move |settings| {
            settings.llm_provider = provider;
        });
    };
    rsx! { div { class: "settings-stack",
        DisabledReason { id: "settings-disabled-reason", reason: settings_blocked }
        NoticeRegion { scope: NoticeScope::Models }
        Section { id: "ai-cleanup", title: "Cleanup",
            AutosaveRow { id: "ai-cleanup-enabled", title: "Use AI cleanup",
                setting: SettingKey::CleanupEnabled,
                Toggle { checked: settings.cleanup_enabled,
                    disabled,
                    onchange: move |enabled: bool| {
                        context.change_setting(SettingKey::CleanupEnabled, move |settings| {
                            settings.cleanup_enabled = enabled;
                        });
                    }
                }
            }
            SettingRow { id: "cleanup-source", title: "Run with",
                Segmented {
                    disabled,
                    value: source,
                    options: SOURCES.to_vec(),
                    onchange: move |source: CleanupSource| pick(source.providers()[0]),
                }
            }
            if source.providers().len() > 1 {
                AutosaveRow { id: "provider-name", title: "Provider", setting: SettingKey::ActiveProvider,
                    Select { value: provider,
                        options: source.providers().iter().map(|item| (*item, item.label().to_owned())).collect::<Vec<_>>(),
                        disabled,
                        onchange: pick,
                    }
                }
            }
            if source == CleanupSource::WhisperNow {
                AutosaveRow { id: "cleanup-uses-gpu", title: "Use GPU",
                    setting: SettingKey::CleanupUsesGpu,
                    Toggle { checked: settings.cleanup_uses_gpu,
                        disabled,
                        onchange: move |enabled: bool| {
                            context.change_setting(SettingKey::CleanupUsesGpu, move |settings| {
                                settings.cleanup_uses_gpu = enabled;
                            });
                        }
                    }
                }
            } else {
                ProviderRows { provider, provider_settings: settings.provider_settings(provider).clone(), disabled }
            }
        }
        if source == CleanupSource::WhisperNow {
            CleanupModels { chosen_model: settings.cleanup_model_id(), disabled }
        }
    } }
}

/// The model picker's form and its new-name field; tests fill and submit them as the user does.
const MODEL_FORM: &str = "provider-model-form";
const MODEL_NAME: &str = "provider-model-name";

#[component]
fn ProviderRows(
    provider: LlmProvider,
    provider_settings: ProviderSettings,
    disabled: bool,
) -> Element {
    let context = use_context::<UiContext>();
    let mut management = context.management;
    let credential_blocked = context
        .blocked(UiOperation::CredentialMutation)
        .or(context.settings_pending());
    let mut credential = use_signal(String::new);
    let mut new_model = use_signal(String::new);
    let models = provider_settings.listed_models();
    let active = provider_settings.model.clone();
    let key_saved = management
        .read()
        .settings
        .as_ref()
        .is_some_and(|view| view.settings.api_keys.contains(&provider));
    let saved_endpoint = management.read().settings.as_ref().and_then(|view| {
        view.settings
            .provider_settings(provider)
            .api_base
            .as_ref()
            .map(|endpoint| endpoint.expose_url().to_owned())
    });
    // The desktop asks with the saved endpoint and key, so saving either asks again.
    let offered = use_resource(use_reactive(
        (&provider, &saved_endpoint, &key_saved),
        |(provider, _, _)| async move {
            client::invoke::<_, Vec<String>>("list_provider_models", provider)
                .await
                .unwrap_or_default()
        },
    ));
    let offered = offered.read().clone().unwrap_or_default();
    rsx! {
        AutosaveRow { id: "provider-model", title: "Model", setting: SettingKey::ProviderModel,
            form { id: MODEL_FORM, class: "model-picker",
                onsubmit: move |event: FormEvent| {
                    event.prevent_default();
                    let model = new_model().trim().to_owned();
                    if !model.is_empty() {
                        context.change_setting(SettingKey::ProviderModel, move |settings| {
                            settings.provider_settings_mut(provider).choose_model(model);
                        });
                        new_model.set(String::new());
                    }
                },
                if !models.is_empty() {
                    Select { value: active.clone(), placeholder: "Choose a model",
                        options: models.iter().map(|model| (model.clone(), model.clone())).collect::<Vec<_>>(),
                        disabled,
                        onchange: move |model: String| {
                            context.change_setting(SettingKey::ProviderModel, move |settings| {
                                settings.provider_settings_mut(provider).choose_model(model);
                            });
                        },
                    }
                    Button { id: "provider-model-remove", variant: ButtonVariant::Ghost,
                        aria_label: "Remove {active}",
                        disabled: disabled || active.is_empty(),
                        onclick: move |_| {
                            let removed = active.clone();
                            context.change_setting(SettingKey::ProviderModel, move |settings| {
                                settings.provider_settings_mut(provider).remove_model(&removed);
                            });
                        },
                        "Remove"
                    }
                }
                Combobox { id: MODEL_NAME, label: "New model name", placeholder: "Model name",
                    value: new_model(), suggestions: offered, disabled,
                    oninput: move |typed: String| new_model.set(typed),
                }
                button { r#type: "submit", id: "provider-model-add", class: "btn btn-secondary btn-md", disabled, "Add model" }
            }
        }
        AutosaveRow { id: "provider-endpoint", title: "API endpoint",
            description: (provider != LlmProvider::Custom).then(|| "Leave empty for the default address.".to_owned()),
            setting: SettingKey::ProviderApiBase,
            DraftText { value: provider_settings.api_base.as_ref().map(ProviderEndpoint::expose_url).unwrap_or_default().to_owned(),
                disabled,
                oncommit: move |value: String| {
                    let value = value.trim();
                    match (!value.is_empty()).then(|| ProviderEndpoint::parse(value)).transpose() {
                        Ok(api_base) => {
                            context.change_setting(SettingKey::ProviderApiBase, move |settings| settings.provider_settings_mut(provider).api_base = api_base);
                            true
                        },
                        Err(error) => {
                            management.write().reject_setting(SettingKey::ProviderApiBase, error.to_string());
                            false
                        }
                    }
                },
            }
        }
        if provider != LlmProvider::Ollama {
            SettingRow { id: "provider-credential", title: "API key",
                description: "Stored in the system keychain.",
                div { class: "stacked-control",
                    TextInput { r#type: "password", autocomplete: "new-password", value: "{credential}",
                        disabled: credential_blocked.is_some(),
                        placeholder: if key_saved { "Key saved" } else { "Enter your API key" },
                        oninput: move |event: FormEvent| credential.set(event.value()),
                    }
                    div { class: "credential-actions",
                        Button {
                            id: "save-credential",
                            disabled: ApiKey::new(credential()).is_err() || credential_blocked.is_some(),
                            aria_describedby: credential_blocked.as_ref().map(|_| "credential-disabled-reason"),
                            onclick: move |_| if let Ok(key) = ApiKey::new(credential()) {
                                context.set_credential(provider, key);
                                credential.set(String::new());
                            },
                            if key_saved { "Replace key" } else { "Save key" }
                        }
                        if key_saved {
                            ConfirmButton { id: "remove-credential", label: "Remove key", confirm_label: "Remove saved key",
                                disabled: credential_blocked.is_some(),
                                onconfirm: move |()| context.clear_credential(provider),
                            }
                        }
                    }
                    DisabledReason { id: "credential-disabled-reason", reason: credential_blocked }
                }
            }
        }
    }
}

/// The app's own cleanup models, which run on this computer.
#[component]
fn CleanupModels(chosen_model: ModelId, disabled: bool) -> Element {
    let context = use_context::<UiContext>();
    let management = context.management;
    let models = management
        .read()
        .models
        .iter()
        .filter(|model| model.family == ModelFamily::Cleanup)
        .cloned()
        .collect::<Vec<_>>();
    let loaded = management.read().models_loaded;
    let reason = context
        .model_download_blocked()
        .or(context.model_delete_blocked());
    rsx! {
        Section { id: "cleanup-models", title: "WhisperNow models",
            div { class: "section-note",
                DisabledReason { id: "models-disabled-reason", reason }
            }
            if !loaded {
                p { class: "section-note", role: "status", "Loading models…" }
            }
            ul { class: "model-list",
                for model in models {
                    ModelRow {
                        key: "{model.id.as_str()}",
                        kind: ModelRowKind::Cleanup {
                            chosen: model.id == chosen_model
                                && model.availability == ModelAvailability::Downloaded,
                        },
                        settings_disabled: disabled,
                        model,
                    }
                }
            }
        }
    }
}
