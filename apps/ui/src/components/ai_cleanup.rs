use dioxus::prelude::*;
use whispernow_domain::{ApiKey, LlmProvider, ModelFamily, ProviderEndpoint, Settings};

use super::NoticeRegion;
use super::actions::DisabledReason;
use super::configuration::{AutosaveRow, SettingsLoading, choice_value, chosen};
use super::models::{ModelRow, ModelRowKind};
use super::ui::{
    Button, ButtonSize, ButtonVariant, ConfirmButton, DraftText, Section, Segmented, Select,
    SettingRow, TextInput, Toggle,
};
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
                description: "If cleanup fails, the plain transcript is pasted.",
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
                        disabled,
                        onchange: move |event: FormEvent| if let Some(provider) = LlmProvider::from_name(&event.value()) {
                            pick(provider);
                        },
                        for item in source.providers() { option { value: item.as_str(), "{item.label()}" } }
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
    let key_saved = management
        .read()
        .settings
        .as_ref()
        .is_some_and(|view| view.settings.api_keys.contains(&provider));
    rsx! {
        AutosaveRow { id: "provider-model", title: "Model", setting: SettingKey::ProviderModel,
            DraftText { value: provider_settings.model.clone(),
                disabled,
                oncommit: move |model: String| {
                    let model = model.trim().to_owned();
                    context.change_setting(SettingKey::ProviderModel, move |settings| settings.provider_settings_mut(provider).model.clone_from(&model));
                    true
                },
            }
        }
        SettingRow { id: "provider-saved-models", title: "Saved models",
            div { class: "stacked-control",
                if !provider_settings.saved_models.is_empty() {
                    ul { class: "saved-models",
                        for saved in provider_settings.saved_models.clone() {
                            li { key: "{saved}",
                                span { class: "saved-model-name", "{saved}" }
                                Button { size: ButtonSize::Sm, aria_label: "Use {saved}",
                                    disabled: disabled || saved == provider_settings.model,
                                    onclick: { let model = saved.clone(); move |_| {
                                        let model = model.clone();
                                        context.change_setting(SettingKey::ProviderModel, move |settings| settings.provider_settings_mut(provider).model.clone_from(&model));
                                    } },
                                    "Use"
                                }
                                Button { size: ButtonSize::Sm, variant: ButtonVariant::Ghost, aria_label: "Remove {saved}",
                                    disabled,
                                    onclick: { let model = saved.clone(); move |_| {
                                        let model = model.clone();
                                        context.change_setting(SettingKey::ProviderModel, move |settings| {
                                            settings.provider_settings_mut(provider).saved_models.retain(|saved| *saved != model);
                                        });
                                    } },
                                    "Remove"
                                }
                            }
                        }
                    }
                }
                form { class: "inline-form",
                    onsubmit: move |event: FormEvent| {
                        event.prevent_default();
                        let model = new_model().trim().to_owned();
                        if !model.is_empty() {
                            context.change_setting(SettingKey::ProviderModel, move |settings| {
                            let saved = &mut settings.provider_settings_mut(provider).saved_models;
                            if !saved.contains(&model) {
                                saved.push(model.clone());
                            }
                        });
                            new_model.set(String::new());
                        }
                    },
                    TextInput { placeholder: "Model name",
                        disabled,
                        value: "{new_model}", oninput: move |event: FormEvent| new_model.set(event.value()) }
                    button { r#type: "submit", class: "btn btn-secondary btn-sm", disabled, "Add model" }
                }
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
                            aria_describedby: credential_blocked.map(|_| "credential-disabled-reason"),
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
    let reason = download_blocked.or(delete_blocked);
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
                        download_blocked,
                        delete_blocked,
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
                        download_blocked, delete_blocked,
                    }
                }
            }
        }
    }
}
