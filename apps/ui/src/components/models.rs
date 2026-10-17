use dioxus::prelude::*;
use whispernow_domain::ipc::{ActiveDownload, ModelState, ModelSummary};
use whispernow_domain::{
    LlmProvider, ModelAvailability, ModelId, Rating, SpeechBenchmark, WerSource,
};

use super::NoticeRegion;
use super::labels::{self, format_bytes};
use super::ui::{
    Banner, Button, ButtonSize, ButtonVariant, ConfirmButton, ProgressBar, Section, Tone,
};
use crate::blocked_reason::DisabledReason;
use crate::{BlockedReason, NoticeScope, SettingKey, UiContext, UiOperation};

#[component]
pub(super) fn ModelsPage() -> Element {
    let context = use_context::<UiContext>();
    let management = context.management;
    let snapshot = context.state.read().snapshot.clone();
    let active_id = snapshot
        .as_ref()
        .map(|snapshot| snapshot.selected_model_id.clone());
    let active_state = snapshot
        .as_ref()
        .map_or(ModelState::Unloaded, |snapshot| snapshot.model_state);
    let models = ordered_models(
        &management
            .read()
            .speech_models()
            .cloned()
            .collect::<Vec<_>>(),
        active_id.as_ref(),
    );
    let loaded = management.read().models_loaded;
    let switch_blocked = context
        .blocked(UiOperation::ModelSwitch)
        .or(context.settings_blocked());
    let download_blocked = context.blocked(UiOperation::DownloadModel);
    let delete_blocked = context
        .blocked(UiOperation::ModelDelete)
        .or(context.phase_blocked());
    let reason = switch_blocked
        .clone()
        .or(download_blocked.clone())
        .or(delete_blocked.clone());
    let failed_active = (active_state == ModelState::Failed)
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
                    aria_describedby: switch_blocked.as_ref().map(|_| "models-disabled-reason"),
                    onclick: move |_| context.switch_model(model_id.clone()),
                    "Reload model"
                }
            }
        }
        Section { id: "speech-models", title: "Speech models",
            description: "Runs locally.",
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
                        switch_blocked: switch_blocked.clone(),
                        download_blocked: download_blocked.clone(),
                        delete_blocked: delete_blocked.clone(),
                        model,
                    }
                }
            }
        }
    } }
}

/// The model in use first, then the others in catalogue order.
fn ordered_models(models: &[ModelSummary], active_id: Option<&ModelId>) -> Vec<ModelSummary> {
    let mut ordered = models.to_vec();
    ordered.sort_by_key(|model| Some(&model.id) != active_id);
    ordered
}

#[derive(Clone, Copy, PartialEq)]
pub(super) enum ModelRowKind {
    /// `state` is the load state of the model in use; `None` for the others.
    Speech { state: Option<ModelState> },
    /// `chosen` marks the model the local cleanup provider runs.
    Cleanup { chosen: bool },
    /// The CUDA pack.
    Runtime,
}

#[component]
pub(super) fn ModelRow(
    model: ModelSummary,
    kind: ModelRowKind,
    switch_blocked: Option<BlockedReason>,
    download_blocked: Option<BlockedReason>,
    delete_blocked: Option<BlockedReason>,
    #[props(default)] settings_disabled: bool,
) -> Element {
    let context = use_context::<UiContext>();
    let id = model.id.clone();
    let slug = id.as_str().replace('.', "-");
    let name = model.display_name.clone();
    let download = context.active_download(&id);
    let active_state = match kind {
        ModelRowKind::Speech { state } => state,
        _ => None,
    };
    let active = active_state.is_some();
    let describe =
        |blocked: &Option<crate::BlockedReason>| blocked.as_ref().map(|_| "models-disabled-reason");
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
                if matches!(kind, ModelRowKind::Speech { .. }) {
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
                        aria_describedby: describe(&download_blocked),
                        onclick: { let id = id.clone(); move |_| context.download_model(id.clone()) },
                        "Download"
                    }
                } else if kind == (ModelRowKind::Speech { state: None }) {
                    Button {
                        id: "model-{slug}-use",
                        variant: ButtonVariant::Primary,
                        aria_label: "Use {name}",
                        disabled: switch_blocked.is_some(),
                        aria_describedby: describe(&switch_blocked),
                        onclick: { let id = id.clone(); move |_| context.switch_model(id.clone()) },
                        "Use"
                    }
                } else if kind == (ModelRowKind::Cleanup { chosen: false }) {
                    Button {
                        disabled: settings_disabled,
                        id: "model-{slug}-use",
                        variant: ButtonVariant::Primary,
                        aria_label: "Use {name}",
                        onclick: { let id = id.clone(); move |_| {
                            let model = id.as_str().to_owned();
                            context.change_setting(SettingKey::ProviderModel, move |settings| {
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
                        onconfirm: { let id = id.clone(); move |()| context.delete_model(id.clone()) },
                    }
                }
            }
        }
    }
}

#[component]
pub(super) fn DownloadProgress(download: ActiveDownload, name: String) -> Element {
    let context = use_context::<UiContext>();
    let cancel_blocked = context.blocked(UiOperation::CancelDownload);
    let rate = download
        .bytes_per_second
        .map(|rate| format!("{}/s", format_bytes(rate)));
    let model_id = download.model_id.clone();
    rsx! {
        div { class: "download-progress", role: "status",
            span { class: "download-phase",
                "{labels::download_phase(download.phase)}"
                if let Some(rate) = rate { " · {rate}" }
            }
            ProgressBar { label: "{name} download progress", downloaded: download.downloaded_bytes, total: download.total_bytes }
        }
        Button {
            size: ButtonSize::Sm,
            aria_label: "Cancel download {name}",
            disabled: cancel_blocked.is_some(),
            onclick: move |_| context.cancel_download(model_id.clone()),
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
