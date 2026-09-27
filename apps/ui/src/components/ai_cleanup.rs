//! AI cleanup: whether transcripts are rewritten, by which provider, with which prompt.

use dioxus::prelude::*;
use whispernow_api::{
    ApiKey, CleanupPrompt, LlmProvider, ModelAvailability, ProviderEndpoint, ProviderSettings,
    Settings,
};

use super::actions::{DisabledReason, UiActions, use_actions};
use super::configuration::{AutosaveRow, SettingsBlockedReason, SettingsLoading, status_parts};
use super::labels::format_bytes;
use super::models::DownloadProgress;
use super::ui::focus::{neighbour, take_focus};
use super::ui::{
    Banner, Button, ButtonSize, ButtonVariant, ConfirmButton, DraftText, Field, Section, Select,
    SettingRow, TextInput, Toggle, Tone,
};
use crate::{SettingKey, UiContext, UiOperation};

/// Text a prompt test starts with; the user can replace it.
const SAMPLE_TEXT: &str = "um so i think we should uh meet on tuesday to go over the the budget";

/// Queues `edit`; while the desktop has refused the provider the user chose, asks for it
/// again, since the edit may have made it usable.
fn change_then_follow_choice(
    actions: UiActions,
    chosen: Signal<Option<LlmProvider>>,
    key: SettingKey,
    edit: impl Fn(&mut Settings) + 'static,
) -> bool {
    let taken = actions.change_setting(key, edit);
    if taken && let Some(provider) = *chosen.peek() {
        choose_provider(actions, provider);
    }
    taken
}

fn choose_provider(actions: UiActions, provider: LlmProvider) -> bool {
    actions.change_setting(SettingKey::ActiveProvider, move |settings| {
        settings.llm_provider = provider;
    })
}

/// Edits the prompt `id`, if the settings still have it.
fn edit_prompt(
    id: &str,
    edit: impl Fn(&mut CleanupPrompt) + 'static,
) -> impl Fn(&mut Settings) + 'static {
    let id = id.to_owned();
    move |settings| {
        if let Some(prompt) = settings.prompts.iter_mut().find(|item| item.id == id) {
            edit(prompt);
        }
    }
}

/// AI cleanup: the on/off switch, the provider and the prompts.
#[component]
pub(super) fn AiCleanupPage() -> Element {
    let actions = use_actions();
    let mut management = use_context::<UiContext>().management;
    let Some(settings) = management.read().shown_settings().cloned() else {
        return rsx! { SettingsLoading {} };
    };
    // The provider the user picked last, shown below even while the desktop refuses it.
    let chosen = use_signal(|| None::<LlmProvider>);
    let first_prompt = settings.prompts.first().map(|item| item.id.clone());
    let active = settings.active_prompt_id.clone();
    rsx! { div { class: "settings-stack",
        SettingsBlockedReason {}
        Section { id: "ai-cleanup", title: "AI cleanup",
            description: "Rewrite each transcript with an AI provider before it is pasted. While this is on, the transcript is sent to the provider below.",
            AutosaveRow { id: "ai-cleanup-enabled", title: "Use AI cleanup",
                description: "If the provider fails or takes too long, the plain transcript is pasted.",
                setting: SettingKey::ActivePrompt,
                Toggle { checked: active.is_some(),
                    onchange: move |enabled: bool| {
                        if !enabled {
                            change_then_follow_choice(actions, chosen, SettingKey::ActivePrompt, |settings| {
                                settings.active_prompt_id = None;
                            });
                        } else if let Some(id) = first_prompt.clone() {
                            actions.change_setting(SettingKey::ActivePrompt, move |settings| {
                                settings.active_prompt_id = Some(id.clone());
                            });
                        } else {
                            management.write().reject_setting(SettingKey::ActivePrompt, "Add a prompt below first");
                        }
                    }
                }
            }
            if let Some(active) = active {
                SettingRow { id: "active-prompt", title: "Prompt",
                    description: "The instructions the provider follows.",
                    Select { value: active,
                        onchange: move |event: FormEvent| {
                            let id = event.value();
                            actions.change_setting(SettingKey::ActivePrompt, move |settings| {
                                settings.active_prompt_id = Some(id.clone());
                            });
                        },
                        for item in settings.prompts.iter() { option { value: "{item.id}", "{item.title}" } }
                    }
                }
            }
        }
        ProviderSection { settings: settings.clone(), chosen }
        PromptsSection { prompts: settings.prompts.clone() }
    } }
}

/// The provider, its model, endpoint and API key, and the time limit.
///
/// Picking a provider shows its fields. While AI cleanup is on, the desktop refuses a provider
/// without a model (or a custom provider without an endpoint): the refusal beside the picker
/// says what is missing, and the switch is asked for again once the fields below change or AI
/// cleanup is turned off.
#[component]
fn ProviderSection(settings: Settings, chosen: Signal<Option<LlmProvider>>) -> Element {
    let actions = use_actions();
    let mut management = use_context::<UiContext>().management;
    let credential_blocked = actions.blocked(UiOperation::CredentialMutation);
    let mut credential = use_signal(String::new);
    let mut new_model = use_signal(String::new);
    let mut chosen = chosen;
    let saved = management
        .read()
        .settings
        .as_ref()
        .map(|view| view.settings.llm_provider);
    // Once the desktop has saved the chosen provider, nothing is waiting any more.
    use_effect(use_reactive!(|saved| {
        if saved.is_some() && *chosen.peek() == saved {
            chosen.set(None);
        }
    }));
    let provider = chosen().unwrap_or(settings.llm_provider);
    let change = move |key: SettingKey, edit: Box<dyn Fn(&mut ProviderSettings)>| {
        change_then_follow_choice(actions, chosen, key, move |settings| {
            edit(settings.provider_settings_mut(provider));
        })
    };
    let provider_settings = settings.provider_settings(provider);
    let key_saved = management
        .read()
        .settings
        .as_ref()
        .is_some_and(|view| view.settings.api_keys.contains(&provider));
    rsx! {
        Section { id: "provider", title: "Provider",
            AutosaveRow { id: "provider-name", title: "Provider", setting: SettingKey::ActiveProvider,
                Select { value: provider.as_str(),
                    onchange: move |event: FormEvent| if let Some(provider) = LlmProvider::from_name(&event.value()) {
                        // The choice stays shown, so a refused provider's fields can be filled in.
                        chosen.set((saved != Some(provider)).then_some(provider));
                        choose_provider(actions, provider);
                    },
                    for item in LlmProvider::ALL { option { value: item.as_str(), "{item.label()}" } }
                }
            }
            if provider == LlmProvider::LocalCleanup {
                CleanupModelRows { chosen_model: settings.cleanup_model_id().to_owned() }
            } else {
            AutosaveRow { id: "provider-model", title: "Model",
                description: "Saved when you leave the field or press Enter.",
                setting: SettingKey::ProviderModel,
                DraftText { value: provider_settings.model.clone(),
                    oncommit: move |model: String| {
                        let model = model.trim().to_owned();
                        change(SettingKey::ProviderModel, Box::new(move |settings| settings.model.clone_from(&model)))
                    },
                }
            }
            SettingRow { id: "provider-saved-models", title: "Saved models",
                description: "Add model names here to switch between them quickly.",
                div { class: "stacked-control",
                    if !provider_settings.saved_models.is_empty() {
                        ul { class: "saved-models",
                            for saved in provider_settings.saved_models.clone() {
                                li { key: "{saved}",
                                    span { class: "saved-model-name", "{saved}" }
                                    Button { size: ButtonSize::Sm, aria_label: "Use {saved}",
                                        disabled: saved == provider_settings.model,
                                        onclick: { let model = saved.clone(); move |_| {
                                            let model = model.clone();
                                            change(SettingKey::ProviderModel, Box::new(move |settings| settings.model.clone_from(&model)));
                                        } },
                                        "Use"
                                    }
                                    Button { size: ButtonSize::Sm, variant: ButtonVariant::Ghost, aria_label: "Remove {saved}",
                                        onclick: { let model = saved.clone(); move |_| {
                                            let model = model.clone();
                                            actions.change_setting(SettingKey::ProviderModel, move |settings| {
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
                            if !model.is_empty() && actions.change_setting(SettingKey::ProviderModel, move |settings| {
                                let saved = &mut settings.provider_settings_mut(provider).saved_models;
                                if !saved.contains(&model) {
                                    saved.push(model.clone());
                                }
                            }) {
                                new_model.set(String::new());
                            }
                        },
                        TextInput { placeholder: "Model name",
                            value: "{new_model}", oninput: move |event: FormEvent| new_model.set(event.value()) }
                        button { r#type: "submit", class: "btn btn-secondary btn-sm", "Add model" }
                    }
                }
            }
            AutosaveRow { id: "provider-endpoint", title: "API endpoint",
                description: "Leave empty for the provider's usual address. Use HTTPS, or HTTP for a service on this computer.",
                setting: SettingKey::ProviderApiBase,
                DraftText { value: provider_settings.api_base.as_ref().map(ProviderEndpoint::expose_url).unwrap_or_default().to_owned(),
                    oncommit: move |value: String| {
                        let value = value.trim();
                        match (!value.is_empty()).then(|| ProviderEndpoint::parse(value)).transpose() {
                            Ok(api_base) => change(SettingKey::ProviderApiBase, Box::new(move |settings| settings.api_base.clone_from(&api_base))),
                            Err(error) => {
                                management.write().reject_setting(SettingKey::ProviderApiBase, error.to_string());
                                false
                            }
                        }
                    },
                }
            }
            SettingRow { id: "provider-credential", title: "API key",
                description: if key_saved { "A key is saved in the system keychain. Enter a new one to replace it." } else { "Stored in the system keychain, never in the settings file." },
                div { class: "stacked-control",
                    TextInput { r#type: "password", autocomplete: "new-password", value: "{credential}",
                        placeholder: if key_saved { "Key saved" } else { "Enter your API key" },
                        oninput: move |event: FormEvent| credential.set(event.value()),
                    }
                    div { class: "credential-actions",
                        Button {
                            disabled: ApiKey::new(credential()).is_err() || credential_blocked.is_some(),
                            aria_describedby: credential_blocked.map(|_| "credential-disabled-reason"),
                            onclick: move |_| if let Ok(key) = ApiKey::new(credential()) {
                                actions.set_credential(provider, key);
                                credential.set(String::new());
                            },
                            if key_saved { "Replace key" } else { "Save key" }
                        }
                        if key_saved {
                            ConfirmButton { id: "remove-credential", label: "Remove key", confirm_label: "Remove saved key",
                                disabled: credential_blocked.is_some(),
                                onconfirm: move |()| actions.clear_credential(provider),
                            }
                        }
                    }
                    DisabledReason { id: "credential-disabled-reason", reason: credential_blocked }
                }
            }
            }
            AutosaveRow { id: "llm-timeout", title: "Time limit (seconds)",
                description: "If the provider takes longer, the plain transcript is pasted.",
                setting: SettingKey::LlmTimeout,
                DraftText { r#type: "number", inputmode: "numeric", value: settings.llm_timeout_seconds.to_string(),
                    oncommit: move |value: String| if let Ok(seconds) = value.trim().parse() {
                        actions.change_setting(SettingKey::LlmTimeout, move |settings| settings.llm_timeout_seconds = seconds)
                    } else {
                        management.write().reject_setting(SettingKey::LlmTimeout, "Enter a whole number of seconds");
                        false
                    },
                }
            }
        }
    }
}

/// Which local cleanup model AI cleanup uses, and its install state and download. They run
/// inside the app, so they have no endpoint or key; a server of your own is a custom provider.
#[component]
fn CleanupModelRows(chosen_model: String) -> Element {
    let actions = use_actions();
    let management = use_context::<UiContext>().management;
    let models = management
        .read()
        .cleanup_models()
        .cloned()
        .collect::<Vec<_>>();
    let download_blocked = actions.blocked(UiOperation::DownloadModel);
    let Some(model) = models
        .iter()
        .find(|model| model.id.as_str() == chosen_model)
        .cloned()
    else {
        return rsx! {};
    };
    let id = model.id.clone();
    let name = model.display_name.clone();
    let download = management.read().download(&id).cloned();
    let installed = model.availability != ModelAvailability::NotInstalled;
    rsx! {
        AutosaveRow { id: "provider-cleanup-model-choice", title: "Cleanup model",
            description: "The larger model keeps more of what was said, but takes two to three times as long and needs a bigger download.",
            setting: SettingKey::ProviderModel,
            Select { value: chosen_model.clone(),
                onchange: move |event: FormEvent| {
                    let model = event.value();
                    actions.change_setting(SettingKey::ProviderModel, move |settings| {
                        settings.provider_settings_mut(LlmProvider::LocalCleanup).model.clone_from(&model);
                    });
                },
                for item in models {
                    option { key: "{item.id.as_str()}", value: item.id.as_str(),
                        "{item.display_name} · {format_bytes(item.size_bytes)}"
                    }
                }
            }
        }
        SettingRow { id: "provider-cleanup-model", title: if installed { "On this computer" } else { "Download" },
            description: if installed {
                "Installed. It runs on this computer, on the graphics card when one is available. To use a llama-server of your own, choose Custom provider."
            } else {
                "Download the model to use it. It runs on this computer, so transcripts never leave it. Until then, the plain transcript is pasted."
            },
            div { class: "stacked-control",
                if let Some(download) = download {
                    DownloadProgress { download, name: name.clone() }
                } else if !installed {
                    Button {
                        id: "provider-cleanup-model-download",
                        aria_label: "Download {name}",
                        disabled: download_blocked.is_some(),
                        aria_describedby: download_blocked.map(|_| "provider-cleanup-model-disabled-reason"),
                        onclick: move |_| actions.download_model(id.clone()),
                        "Download ({format_bytes(model.size_bytes)})"
                    }
                    DisabledReason { id: "provider-cleanup-model-disabled-reason", reason: download_blocked }
                }
            }
        }
    }
}

/// The saved prompts, a sample to test them on, and a form for a new one.
#[component]
fn PromptsSection(prompts: Vec<CleanupPrompt>) -> Element {
    let actions = use_actions();
    let management = use_context::<UiContext>().management;
    let mut sample = use_signal(|| SAMPLE_TEXT.to_owned());
    let mut new_title = use_signal(String::new);
    let mut new_prompt = use_signal(String::new);
    let (add_status, add_error) =
        status_parts(management.read().setting_status(&SettingKey::Prompts));
    let test_blocked = actions.blocked(UiOperation::TestPrompt);
    let focus = use_signal(|| None::<PromptFocus>);
    let mut new_prompt_group = use_signal(|| None::<std::rc::Rc<MountedData>>);
    use_effect(move || {
        if focus() == Some(PromptFocus::NewPrompt) {
            take_focus(new_prompt_group);
        }
    });
    let ids = prompts
        .iter()
        .map(|item| item.id.clone())
        .collect::<Vec<_>>();
    rsx! {
        Section { id: "prompts", title: "Prompts",
            description: "Each prompt tells the provider how to rewrite a transcript. Changes are saved when you leave a field.",
            Field { id: "prompt-sample", label: "Sample text for tests",
                description: "Test runs a prompt on this text with the provider above. Nothing is pasted.",
                TextInput { value: "{sample}", oninput: move |event: FormEvent| sample.set(event.value()) }
            }
            DisabledReason { id: "prompt-test-disabled-reason", reason: test_blocked }
            for (index, item) in prompts.iter().cloned().enumerate() {
                PromptEditor { key: "{item.id}", index, item, sample, test_blocked: test_blocked.is_some(),
                    focus, after_delete: neighbour(&ids, index).map_or(PromptFocus::NewPrompt, PromptFocus::Prompt) }
            }
            div { class: "prompt-new", role: "group", tabindex: "-1", aria_labelledby: "prompt-new-heading",
                onmounted: move |event| new_prompt_group.set(Some(event.data())),
                h3 { id: "prompt-new-heading", class: "prompt-heading", "New prompt" }
                Field { id: "prompt-new-name", label: "Name",
                    TextInput { value: "{new_title}", oninput: move |event: FormEvent| new_title.set(event.value()) }
                }
                Field { id: "prompt-new-instructions", label: "Instructions",
                    error: add_error, status: add_status,
                    textarea { id: "prompt-new-instructions", class: "text-input", rows: "4",
                        value: "{new_prompt}", oninput: move |event| new_prompt.set(event.value()) }
                }
                Button { id: "prompt-add",
                    onclick: move |_| {
                        let title = new_title().trim().to_owned();
                        let prompt = new_prompt().trim().to_owned();
                        let add = move |settings: &mut Settings| {
                            settings.prompts.push(CleanupPrompt {
                                id: settings.next_prompt_id(),
                                title: title.clone(),
                                prompt: prompt.clone(),
                                default_hash: None,
                            });
                        };
                        if actions.change_setting(SettingKey::Prompts, add) {
                            new_title.set(String::new());
                            new_prompt.set(String::new());
                        }
                    },
                    "Add prompt"
                }
            }
        }
    }
}

/// Where focus goes after a prompt is deleted.
#[derive(Clone, Debug, PartialEq)]
enum PromptFocus {
    Prompt(String),
    /// No prompt is left: the new-prompt form.
    NewPrompt,
}

/// One saved prompt: its name and instructions, a newer default, a test, and delete. Deleting
/// it moves focus to `after_delete`.
#[component]
fn PromptEditor(
    index: usize,
    item: CleanupPrompt,
    sample: Signal<String>,
    test_blocked: bool,
    focus: Signal<Option<PromptFocus>>,
    after_delete: PromptFocus,
) -> Element {
    let actions = use_actions();
    let mut article = use_signal(|| None::<std::rc::Rc<MountedData>>);
    let own_focus = PromptFocus::Prompt(item.id.clone());
    use_effect(move || {
        if focus() == Some(own_focus.clone()) {
            take_focus(article);
        }
    });
    let management = use_context::<UiContext>().management;
    let id = item.id.clone();
    let title = item.title.clone();
    let (name_status, name_error) = status_parts(
        management
            .read()
            .setting_status(&SettingKey::PromptTitle(id.clone())),
    );
    let (text_status, text_error) = status_parts(
        management
            .read()
            .setting_status(&SettingKey::PromptText(id.clone())),
    );
    let test = management
        .read()
        .prompt_test
        .clone()
        .filter(|test| test.prompt_id == id);
    let dom = format!("prompt-{index}");
    rsx! {
        article { class: "prompt-item", aria_labelledby: "{dom}-heading", tabindex: "-1",
            onmounted: move |event| article.set(Some(event.data())),
            h3 { id: "{dom}-heading", class: "prompt-heading", "{title}" }
            Field { id: "{dom}-name", label: "Name", error: name_error, status: name_status,
                DraftText { value: item.title.clone(), aria_label: "Name of prompt “{title}”",
                    oncommit: { let id = id.clone(); move |title: String| {
                        let title = title.trim().to_owned();
                        actions.change_setting(SettingKey::PromptTitle(id.clone()), edit_prompt(&id, move |item| item.title.clone_from(&title)))
                    } },
                }
            }
            Field { id: "{dom}-instructions", label: "Instructions", error: text_error, status: text_status,
                DraftText { multiline: true, value: item.prompt.clone(), aria_label: "Instructions for prompt “{title}”",
                    oncommit: { let id = id.clone(); move |prompt: String| {
                        let prompt = prompt.trim().to_owned();
                        actions.change_setting(SettingKey::PromptText(id.clone()), edit_prompt(&id, move |item| item.prompt.clone_from(&prompt)))
                    } },
                }
            }
            if item.newer_default().is_some() {
                Banner { tone: Tone::Info, title: "A newer version of this prompt is available",
                    p { "Your edited instructions stay until you choose." }
                    div { class: "card-actions",
                        Button { size: ButtonSize::Sm, aria_label: "Use new default for {title}",
                            onclick: { let id = id.clone(); move |_| {
                                actions.change_setting(SettingKey::PromptText(id.clone()), edit_prompt(&id, |item| {
                                    if let Some(newer) = item.newer_default() {
                                        newer.clone_into(&mut item.prompt);
                                    }
                                }));
                            } },
                            "Use new default"
                        }
                        Button { size: ButtonSize::Sm, variant: ButtonVariant::Ghost, aria_label: "Keep my version of {title}",
                            onclick: { let id = id.clone(); move |_| {
                                actions.change_setting(SettingKey::PromptText(id.clone()), edit_prompt(&id, CleanupPrompt::acknowledge_current_default));
                            } },
                            "Keep my version"
                        }
                    }
                }
            }
            div { class: "card-actions",
                Button { id: "{dom}-test", size: ButtonSize::Sm, aria_label: "Test {title}",
                    disabled: test_blocked,
                    aria_describedby: test_blocked.then_some("prompt-test-disabled-reason"),
                    onclick: { let id = id.clone(); move |_| actions.test_prompt(&id, &sample()) },
                    "Test"
                }
                ConfirmButton { id: "{dom}-delete", label: "Delete", confirm_label: "Delete prompt",
                    target: "“{title}”",
                    size: ButtonSize::Sm,
                    onconfirm: {
                        let id = id.clone();
                        let mut focus = focus;
                        move |()| if actions.change_setting(SettingKey::Prompts, { let id = id.clone(); move |settings| settings.delete_prompt(&id) }) {
                            focus.set(Some(after_delete.clone()));
                        }
                    },
                }
            }
            if let Some(test) = test {
                match test.outcome {
                    None => rsx! { p { class: "status-detail", role: "status", "Testing…" } },
                    Some(Ok(text)) => rsx! {
                        div { class: "prompt-test-result", role: "status",
                            p { class: "status-detail", "Result" }
                            p { class: "prompt-test-text", "{text}" }
                        }
                    },
                    Some(Err(message)) => rsx! {
                        p { class: "field-error-message", role: "alert", "{message}" }
                    },
                }
            }
        }
    }
}
