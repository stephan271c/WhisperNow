use dioxus::prelude::*;
use whispernow_domain::{
    ApiKey, LlmProvider, ModelFamily, ProviderEndpoint, ProviderSettings, Settings,
};

use super::NoticeRegion;
use super::configuration::{AutosaveRow, SettingsLoading, choice_value, chosen};
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
                    value: choice_value(&SOURCES, &source),
                    options: SOURCES.iter().map(|(_, value, label)| (*value, *label)).collect::<Vec<_>>(),
                    onchange: move |value: &'static str| if let Some(source) = chosen(&SOURCES, value) {
                        pick(source.providers()[0]);
                    },
                }
            }
            if source.providers().len() > 1 {
                AutosaveRow { id: "provider-name", title: "Provider", setting: SettingKey::ActiveProvider,
                    Select { value: provider.as_str(),
                        options: source.providers().iter().map(|item| (item.as_str().to_owned(), item.label().to_owned())).collect::<Vec<_>>(),
                        disabled,
                        onchange: move |value: String| if let Some(provider) = LlmProvider::from_name(&value) {
                            pick(provider);
                        },
                    }
                }
            }
            if source != CleanupSource::WhisperNow {
                ProviderRows { settings: settings.clone(), provider, disabled }
            }
        }
        if source == CleanupSource::WhisperNow {
            CleanupModels { chosen_model: settings.cleanup_model_id().as_str().to_owned(), disabled }
        }
    } }
}

/// The model picker's form and its new-name field; tests fill and submit them as the user does.
const MODEL_FORM: &str = "provider-model-form";
const MODEL_NAME: &str = "provider-model-name";

/// The names the model picker lists: the saved ones, after the active model if it was never
/// saved. A change from the picker saves this list.
fn listed_models(settings: &ProviderSettings) -> Vec<String> {
    let mut models = settings.saved_models.clone();
    if !settings.model.is_empty() && !models.contains(&settings.model) {
        models.insert(0, settings.model.clone());
    }
    models
}

#[component]
fn ProviderRows(settings: Settings, provider: LlmProvider, disabled: bool) -> Element {
    let context = use_context::<UiContext>();
    let mut management = context.management;
    let credential_blocked = context
        .blocked(UiOperation::CredentialMutation)
        .or(context.settings_pending());
    let mut credential = use_signal(String::new);
    let mut new_model = use_signal(String::new);
    let provider_settings = settings.provider_settings(provider);
    let models = listed_models(provider_settings);
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
            client::invoke::<_, Vec<String>>("list_provider_models", Some(&provider))
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
                            let entry = settings.provider_settings_mut(provider);
                            entry.saved_models = listed_models(entry);
                            if !entry.saved_models.contains(&model) {
                                entry.saved_models.push(model.clone());
                            }
                            entry.model = model;
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
                                let entry = settings.provider_settings_mut(provider);
                                entry.saved_models = listed_models(entry);
                                entry.model = model;
                            });
                        },
                    }
                    Button { id: "provider-model-remove", variant: ButtonVariant::Ghost,
                        aria_label: "Remove {active}",
                        disabled: disabled || active.is_empty(),
                        onclick: move |_| {
                            let removed = active.clone();
                            context.change_setting(SettingKey::ProviderModel, move |settings| {
                                let entry = settings.provider_settings_mut(provider);
                                entry.saved_models = listed_models(entry);
                                entry.saved_models.retain(|saved| *saved != removed);
                                if entry.model == removed {
                                    entry.model = entry.saved_models.first().cloned().unwrap_or_default();
                                }
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

/// The app's own cleanup models, which run on this computer, and the CUDA pack that speeds them
/// up on NVIDIA graphics cards.
#[component]
fn CleanupModels(chosen_model: String, disabled: bool) -> Element {
    let context = use_context::<UiContext>();
    let management = context.management;
    let models = management
        .read()
        .models
        .iter()
        .filter(|model| model.family == ModelFamily::Cleanup)
        .cloned()
        .collect::<Vec<_>>();
    let cuda_pack = management
        .read()
        .models
        .iter()
        .find(|model| model.family == ModelFamily::CudaPack)
        .cloned();
    let loaded = management.read().models_loaded;
    let download_blocked = context.blocked(UiOperation::DownloadModel);
    let delete_blocked = context
        .blocked(UiOperation::ModelDelete)
        .or(context.phase_blocked());
    let reason = download_blocked.clone().or(delete_blocked.clone());
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
                        kind: ModelRowKind::Cleanup { chosen: model.id.as_str() == chosen_model },
                        settings_disabled: disabled,
                        download_blocked: download_blocked.clone(),
                        delete_blocked: delete_blocked.clone(),
                        model,
                    }
                }
            }
        }
        if let Some(pack) = cuda_pack {
            Section { id: "cuda-pack", title: "NVIDIA CUDA",
                description: "Faster cleanup on GeForce RTX 20, GTX 16 and newer cards.",
                ul { class: "model-list",
                    ModelRow {
                        key: "{pack.id.as_str()}", kind: ModelRowKind::Runtime, model: pack,
                        download_blocked: download_blocked.clone(), delete_blocked: delete_blocked.clone(),
                    }
                }
            }
        }
    }
}
