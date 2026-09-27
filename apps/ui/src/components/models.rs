//! Speech models: what each one costs and offers, and installing, choosing and removing them;
//! the local cleanup models, which AI cleanup can use; and the CUDA download that runs them
//! on NVIDIA GPUs.

use dioxus::prelude::*;
use whispernow_api::{
    ActiveDownloadDto, LlmProvider, ModelAvailability, ModelDto, ModelId, ModelStateDto,
    RelativeRating,
};

use super::NoticeRegion;
use super::actions::{DisabledReason, use_actions};
use super::labels::{self, format_bytes};
use super::ui::{
    Banner, Button, ButtonSize, ButtonVariant, ConfirmButton, ProgressBar, Section, Tone,
};
use crate::{NoticeScope, UiContext, UiOperation};

/// Speech models: install, choose and remove local models. Below them, the local cleanup models.
#[component]
pub(super) fn ModelsPage() -> Element {
    let context = use_context::<UiContext>();
    let actions = use_actions();
    let management = context.management;
    let snapshot = context.state.read().snapshot.clone();
    let active_id = snapshot
        .as_ref()
        .map(|snapshot| snapshot.selected_model_id.clone());
    let active_state = snapshot
        .as_ref()
        .map_or(ModelStateDto::Unloaded, |snapshot| snapshot.model_state);
    let models = ordered_models(
        &management
            .read()
            .speech_models()
            .cloned()
            .collect::<Vec<_>>(),
        active_id.as_ref(),
    );
    let cleanup_models = management
        .read()
        .cleanup_models()
        .cloned()
        .collect::<Vec<_>>();
    // Offered with the cleanup models it serves.
    let cuda_pack = management
        .read()
        .cuda_pack()
        .filter(|_| !cleanup_models.is_empty())
        .cloned();
    // The cleanup model AI cleanup uses, while its provider is the local cleanup model.
    let cleanup_in_use = management.read().settings.as_ref().and_then(|view| {
        (view.settings.llm_provider == LlmProvider::LocalCleanup)
            .then(|| view.settings.cleanup_model_id())
    });
    let loaded = management.read().models_loaded;
    let switch_blocked = actions.blocked(UiOperation::ModelSwitch);
    let download_blocked = actions.blocked(UiOperation::DownloadModel);
    let delete_blocked = actions.blocked(UiOperation::ModelDelete);
    let reason = switch_blocked.or(download_blocked).or(delete_blocked);
    let failed_active = (active_state == ModelStateDto::Failed)
        .then(|| active_id.clone())
        .flatten();
    rsx! { div { class: "settings-stack",
        NoticeRegion { scope: NoticeScope::Models }
        if let Some(model_id) = failed_active {
            Banner { tone: Tone::Error, title: "{management.read().model_display_name(model_id.as_str())} could not be loaded",
                p { "Dictation needs a working model. Reload it, or choose another one below." }
                Button {
                    id: "reload-model",
                    disabled: switch_blocked.is_some(),
                    aria_describedby: switch_blocked.map(|_| "models-disabled-reason"),
                    onclick: move |_| actions.switch_model(model_id.clone()),
                    "Reload model"
                }
            }
        }
        Section { id: "speech-models", title: "Speech models",
            description: "Models run on this computer, so your audio never leaves it. Larger models are usually more accurate but slower.",
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
                        kind: ModelRowKind::Speech { active: active_id.as_ref() == Some(&model.id) },
                        active_state,
                        model,
                    }
                }
            }
        }
        if !cleanup_models.is_empty() {
            Section { id: "cleanup-model", title: "Cleanup models",
                description: "Small models made for Clean Up & Format. One runs on this computer when AI cleanup uses the Local cleanup model provider, so your transcripts never leave it; choose which under AI cleanup. Fine-tunes of Liquid AI's LFM2.5-350M, under the LFM Open License v1.0, and of OpenBMB's MiniCPM5-1B, under Apache-2.0.",
                ul { class: "model-list",
                    for model in cleanup_models {
                        ModelRow {
                            key: "{model.id.as_str()}",
                            kind: ModelRowKind::Cleanup { in_use: cleanup_in_use == Some(model.id.as_str()) },
                            active_state,
                            model,
                        }
                    }
                }
            }
        }
        if let Some(pack) = cuda_pack {
            Section { id: "cuda-pack", title: "NVIDIA CUDA",
                description: "For NVIDIA graphics cards from the GeForce RTX 20 and GTX 16 series on, with a recent driver: the cleanup models run with CUDA, which is faster than Vulkan, which they use otherwise. Downloads llama.cpp's CUDA backend and NVIDIA's CUDA runtime, under NVIDIA's license, from llama.cpp's release on GitHub. Where CUDA cannot run, cleanup keeps using Vulkan or the processor. Deleting the download takes effect when WhisperNow restarts.",
                ul { class: "model-list",
                    ModelRow {
                        key: "{pack.id.as_str()}",
                        kind: ModelRowKind::Runtime,
                        active_state,
                        model: pack,
                    }
                }
            }
        }
    } }
}

/// The model in use first, then the others in catalogue order.
fn ordered_models(models: &[ModelDto], active_id: Option<&ModelId>) -> Vec<ModelDto> {
    let mut ordered = models.to_vec();
    ordered.sort_by_key(|model| Some(&model.id) != active_id);
    ordered
}

/// What a model row is for.
#[derive(Clone, Copy, PartialEq)]
enum ModelRowKind {
    /// A speech model; `active` is the one in use.
    Speech { active: bool },
    /// A cleanup model: installed and removed here, but chosen under AI cleanup; `in_use`
    /// marks the one AI cleanup uses.
    Cleanup { in_use: bool },
    /// The CUDA pack, which has no speed or accuracy of its own.
    Runtime,
}

/// One model: what it offers, and the action that fits its state.
#[component]
fn ModelRow(model: ModelDto, kind: ModelRowKind, active_state: ModelStateDto) -> Element {
    let actions = use_actions();
    let management = use_context::<UiContext>().management;
    let id = model.id.clone();
    let slug = id.as_str().replace('.', "-");
    let name = model.display_name.clone();
    let download = management.read().download(&id).cloned();
    let active = kind == ModelRowKind::Speech { active: true };
    let switch_blocked = actions.blocked(UiOperation::ModelSwitch);
    let download_blocked = actions.blocked(UiOperation::DownloadModel);
    let delete_blocked = actions.blocked(UiOperation::ModelDelete);
    let describe =
        |blocked: Option<crate::BlockedReason>| blocked.map(|_| "models-disabled-reason");
    rsx! {
        li { class: if active { "model-row active" } else { "model-row" },
            div { class: "model-summary",
                p { class: "model-name",
                    strong { id: "model-{slug}-name", "{name}" }
                    if active {
                        span { class: "pill pill-accent", "In use · {labels::model_state(active_state)}" }
                    }
                    if kind == (ModelRowKind::Cleanup { in_use: true }) {
                        span { class: "pill pill-accent", "Used by AI cleanup" }
                    }
                    if model.recommended {
                        span { class: "pill", "Recommended" }
                    }
                }
                p { class: "model-facts",
                    "{format_bytes(model.size_bytes)} download"
                    if let Some(languages) = languages(&model) {
                        " · {languages}"
                    }
                }
                if kind != ModelRowKind::Runtime {
                    p { class: "model-facts",
                        "Speed: {rating(model.speed)} · Accuracy: {rating(model.accuracy)}"
                    }
                }
            }
            div { class: "model-actions",
                if let Some(download) = download {
                    DownloadProgress { download, name: name.clone() }
                } else if model.availability == ModelAvailability::NotInstalled {
                    Button {
                        id: "model-{slug}-download",
                        aria_label: "Download {name}",
                        disabled: download_blocked.is_some(),
                        aria_describedby: describe(download_blocked),
                        onclick: { let id = id.clone(); move |_| actions.download_model(id.clone()) },
                        "Download"
                    }
                } else if kind == (ModelRowKind::Speech { active: false }) {
                    Button {
                        id: "model-{slug}-use",
                        variant: ButtonVariant::Primary,
                        aria_label: "Use {name}",
                        disabled: switch_blocked.is_some(),
                        aria_describedby: describe(switch_blocked),
                        onclick: { let id = id.clone(); move |_| actions.switch_model(id.clone()) },
                        "Use"
                    }
                }
                if model.availability == ModelAvailability::Downloaded && !active {
                    ConfirmButton {
                        id: "model-{slug}-delete",
                        label: "Delete",
                        confirm_label: "Delete download",
                        target: name.clone(),
                        size: ButtonSize::Md,
                        disabled: delete_blocked.is_some(),
                        onconfirm: { let id = id.clone(); move |()| actions.delete_model(id.clone()) },
                    }
                }
            }
        }
    }
}

/// A download's progress, speed and cancel control.
#[component]
pub(super) fn DownloadProgress(download: ActiveDownloadDto, name: String) -> Element {
    let actions = use_actions();
    let cancel_blocked = actions.blocked(UiOperation::CancelDownload);
    let max = download
        .total_bytes
        .unwrap_or(download.downloaded_bytes.max(1));
    let amount = download.total_bytes.map_or_else(
        || format!("{} downloaded", format_bytes(download.downloaded_bytes)),
        |total| {
            format!(
                "{} of {}",
                format_bytes(download.downloaded_bytes),
                format_bytes(total)
            )
        },
    );
    let progress = download.bytes_per_second.map_or_else(
        || amount.clone(),
        |rate| format!("{amount} · {}/s", format_bytes(rate)),
    );
    let model_id = download.model_id.clone();
    rsx! {
        div { class: "download-progress", role: "status",
            span { class: "download-phase", "{labels::download_phase(download.phase)}: {progress}" }
            ProgressBar { label: "{name} download progress", value: download.downloaded_bytes, max, value_text: progress.clone() }
        }
        Button {
            size: ButtonSize::Sm,
            aria_label: "Cancel download {name}",
            disabled: cancel_blocked.is_some(),
            onclick: move |_| actions.cancel_download(model_id.clone()),
            "Cancel download"
        }
    }
}

/// The languages a model declares, in words; `None` when it declares none.
pub(super) fn languages(model: &ModelDto) -> Option<String> {
    let named = model
        .languages
        .iter()
        .map(|code| match code.as_str() {
            "en" => "English".to_owned(),
            other => other.to_uppercase(),
        })
        .collect::<Vec<_>>()
        .join(", ");
    match (named.is_empty(), model.automatic_language_detection) {
        (true, false) => None,
        (true, true) => Some("Detects the spoken language".to_owned()),
        (false, false) => Some(named),
        (false, true) => Some(format!("{named}; detects the spoken language")),
    }
}

const fn rating(value: RelativeRating) -> &'static str {
    match value {
        RelativeRating::Low => "lower",
        RelativeRating::Medium => "medium",
        RelativeRating::High => "higher",
    }
}
