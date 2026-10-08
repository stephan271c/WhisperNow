use dioxus::prelude::*;
use whispernow_api::{
    ActiveDownloadDto, LlmProvider, ModelAvailability, ModelDto, ModelId, ModelStateDto, Rating,
    SpeechBenchmark, WerSource,
};

use super::NoticeRegion;
use super::actions::{DisabledReason, use_actions};
use super::labels::{self, format_bytes};
use super::ui::{
    Banner, Button, ButtonSize, ButtonVariant, ConfirmButton, ProgressBar, Section, Tone,
};
use crate::{NoticeScope, SettingKey, UiContext, UiOperation};

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
            description: "Run on this computer, so your audio never leaves it.",
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
                        kind: ModelRowKind::Speech {
                            state: (active_id.as_ref() == Some(&model.id)).then_some(active_state),
                        },
                        model,
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

#[derive(Clone, Copy, PartialEq)]
pub(super) enum ModelRowKind {
    /// `state` is the load state of the model in use; `None` for the others.
    Speech { state: Option<ModelStateDto> },
    /// `chosen` marks the model the local cleanup provider runs.
    Cleanup { chosen: bool },
    /// The CUDA pack, which has no speed or accuracy of its own.
    Runtime,
}

#[component]
pub(super) fn ModelRow(model: ModelDto, kind: ModelRowKind) -> Element {
    let actions = use_actions();
    let management = use_context::<UiContext>().management;
    let id = model.id.clone();
    let slug = id.as_str().replace('.', "-");
    let name = model.display_name.clone();
    let download = management.read().download(&id).cloned();
    let active_state = match kind {
        ModelRowKind::Speech { state } => state,
        _ => None,
    };
    let active = active_state.is_some();
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
                    if let Some(state) = active_state {
                        span { class: "pill pill-accent", "In use · {labels::model_state(state)}" }
                    }
                    if kind == (ModelRowKind::Cleanup { chosen: true }) {
                        span { class: "pill pill-accent", "Selected" }
                    }
                    if model.recommended {
                        span { class: "pill", "Recommended" }
                    }
                }
                div { class: "model-facts",
                    span { "{format_bytes(model.size_bytes)} download" }
                    ModelLanguages { languages: model.languages.clone() }
                }
                if kind != ModelRowKind::Runtime {
                    div { class: "model-ratings",
                        RatingBars { label: "Speed", value: model.speed,
                            detail: model.benchmark.map(|benchmark| speed_detail(&benchmark)) }
                        RatingBars { label: "Accuracy", value: model.accuracy,
                            detail: model.benchmark.map(|benchmark| accuracy_detail(&benchmark)) }
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
                } else if kind == (ModelRowKind::Speech { state: None }) {
                    Button {
                        id: "model-{slug}-use",
                        variant: ButtonVariant::Primary,
                        aria_label: "Use {name}",
                        disabled: switch_blocked.is_some(),
                        aria_describedby: describe(switch_blocked),
                        onclick: { let id = id.clone(); move |_| actions.switch_model(id.clone()) },
                        "Use"
                    }
                } else if kind == (ModelRowKind::Cleanup { chosen: false }) {
                    Button {
                        id: "model-{slug}-use",
                        variant: ButtonVariant::Primary,
                        aria_label: "Use {name}",
                        onclick: { let id = id.clone(); move |_| {
                            let model = id.as_str().to_owned();
                            actions.change_setting(SettingKey::ProviderModel, move |settings| {
                                settings.provider_settings_mut(LlmProvider::LocalCleanup).model.clone_from(&model);
                            });
                        } },
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

fn speed_detail(benchmark: &SpeechBenchmark) -> String {
    let speed = benchmark.speed;
    if speed < 10.0 {
        format!("{speed:.1}× real time")
    } else {
        format!("{speed:.0}× real time")
    }
}

fn accuracy_detail(benchmark: &SpeechBenchmark) -> String {
    let wer = benchmark.english_wer;
    match benchmark.wer_source {
        WerSource::Leaderboard => {
            format!("English word error rate: {wer:.2}% (Open ASR Leaderboard)")
        }
        WerSource::Estimated => format!(
            "English word error rate: about {wer:.1}% (estimated; not on the Open ASR Leaderboard)"
        ),
    }
}

/// `detail` is the measurement behind the rating, shown on hover.
#[component]
fn RatingBars(label: &'static str, value: Rating, detail: Option<String>) -> Element {
    let filled = value.get();
    let spoken = detail.as_ref().map_or_else(
        || format!("{label}: {filled} of {}", Rating::MAX),
        |detail| format!("{label}: {filled} of {}. {detail}", Rating::MAX),
    );
    rsx! {
        span { class: "rating", role: "img", title: detail, aria_label: "{spoken}",
            span { aria_hidden: "true", "{label}" }
            span { class: "rating-bars", aria_hidden: "true",
                for step in 1..=Rating::MAX {
                    span { class: if step <= filled { "rating-bar filled" } else { "rating-bar" } }
                }
            }
        }
    }
}

/// One language by name, or a count that opens the full list.
#[component]
pub(super) fn ModelLanguages(languages: Vec<String>) -> Element {
    let mut names: Vec<&str> = languages
        .iter()
        .map(|tag| labels::language_name(tag))
        .collect();
    names.sort_unstable();
    match names.as_slice() {
        [] => rsx! {},
        [name] => rsx! { span { "{name}" } },
        names => {
            let count = names.len();
            let list = names.join(", ");
            rsx! {
                details { class: "model-languages",
                    summary { "{count} languages" }
                    p { "{list}" }
                }
            }
        }
    }
}
