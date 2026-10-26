use dioxus::prelude::*;
use whispernow_domain::ipc::{ActiveDownload, ModelState, ModelSummary};
use whispernow_domain::{
    CleanupBenchmark, ModelAvailability, ModelBenchmark, ModelId, Rating, SettingChange,
    SpeechBenchmark, WerSource,
};

use super::NoticeRegion;
use super::labels::{self, format_bytes};
use super::ui::{Banner, Button, ButtonSize, ButtonVariant, ConfirmButton, ProgressBar, Section};
use crate::UiContext;
use crate::blocked_reason::{BlockedReason, DisabledReason};
use crate::notice::{NoticeScope, Tone};
use crate::operations::UiOperation;

#[component]
pub(super) fn ModelsPage() -> Element {
    let context = use_context::<UiContext>();
    let management = context.management;
    let state = context.state;
    let active_id = use_memo(move || {
        state
            .read()
            .snapshot
            .as_ref()
            .map(|snapshot| snapshot.selected_model_id.clone())
    })();
    let active_state = use_memo(move || {
        state
            .read()
            .snapshot
            .as_ref()
            .map_or(ModelState::Unloaded, |snapshot| snapshot.model_state)
    })();
    let mut models = management
        .read()
        .speech_models()
        .cloned()
        .collect::<Vec<_>>();
    models.sort_by_key(|model| active_id.as_ref() != Some(&model.id));
    let loaded = management.read().models.is_some();
    let switch_blocked = context
        .blocked(UiOperation::ModelSwitch)
        .or(context.settings_blocked());
    let reason = switch_blocked
        .clone()
        .or(context.model_download_blocked())
        .or(context.model_delete_blocked());
    let failed_active = (active_state == ModelState::Failed)
        .then(|| active_id.clone())
        .flatten();
    rsx! { div { class: "settings-stack",
        NoticeRegion { scope: NoticeScope::Models }
        if let Some(model_id) = failed_active {
            Banner { tone: Tone::Error, title: "{management.read().model_display_name(&model_id)} could not be loaded",
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
                        model,
                    }
                }
            }
        }
    } }
}

#[derive(Clone, Copy, PartialEq)]
pub(super) enum ModelRowKind {
    /// `state` is the load state of the model in use; `None` for the others.
    Speech { state: Option<ModelState> },
    /// `chosen` marks the downloaded model the local cleanup provider runs.
    Cleanup { chosen: bool },
}

#[component]
pub(super) fn ModelRow(
    model: ModelSummary,
    kind: ModelRowKind,
    switch_blocked: Option<BlockedReason>,
    #[props(default)] settings_disabled: bool,
) -> Element {
    let context = use_context::<UiContext>();
    let download_blocked = context.model_download_blocked();
    let delete_blocked = context.model_delete_blocked();
    let id = model.id.clone();
    let slug = model_slug(&id);
    let name = model.display_name.clone();
    let download = use_memo(use_reactive!(|id| context.active_download(&id)))();
    let download_failed = use_memo(use_reactive!(|id| {
        context
            .state
            .read()
            .snapshot
            .as_ref()
            .is_some_and(|snapshot| snapshot.failed_download.as_ref() == Some(&id))
    }))();
    let active_state = match kind {
        ModelRowKind::Speech { state } => state,
        ModelRowKind::Cleanup { .. } => None,
    };
    let active = active_state.is_some();
    let describe =
        |blocked: &Option<BlockedReason>| blocked.as_ref().map(|_| "models-disabled-reason");
    let ratings = match model.benchmark {
        Some(ModelBenchmark::Speech(benchmark)) => rsx! {
            div { class: "model-ratings",
                RatingBars { label: "Speed", value: benchmark.speed(),
                    detail: speed_detail(&benchmark) }
                RatingBars { label: "Accuracy", value: benchmark.accuracy(),
                    detail: accuracy_detail(&benchmark) }
            }
        },
        Some(ModelBenchmark::Cleanup(benchmark)) => rsx! {
            div { class: "model-ratings",
                Measurement { label: "Memory", value: gigabytes(benchmark.memory_bytes),
                    detail: "The most GPU or system memory the loaded model takes" }
                RatingBars { label: "Accuracy", value: benchmark.accuracy(),
                    detail: factual_errors_detail(&benchmark) }
                Measurement { label: "Latency", value: latency(&benchmark),
                    detail: "To clean up a 100-word dictation, on an RTX 5070 GPU and on a Ryzen 9 8945HS CPU" }
            }
        },
        None => rsx! {},
    };
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
                ModelFacts { model: model.clone() }
                {ratings}
            }
            div { class: "model-actions",
                if let Some(download) = download {
                    DownloadProgress { download, name: name.clone() }
                } else if model.availability == ModelAvailability::NotInstalled {
                    if download_failed {
                        span { class: "download-phase", role: "status", "Download failed" }
                    }
                    Button {
                        id: "model-{slug}-download",
                        aria_label: "Download {name}",
                        disabled: download_blocked.is_some(),
                        aria_describedby: describe(&download_blocked),
                        onclick: { let id = id.clone(); move |_| context.download_model(id.clone()) },
                        if download_failed { "Retry download" } else { "Download" }
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
                        onclick: { let id = id.clone(); move |_| context.change_setting(SettingChange::CleanupModel(id.clone())) },
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
            ProgressBar { label: "{name} download progress", downloaded: download.downloaded_bytes, total: Some(download.total_bytes) }
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

fn factual_errors_detail(benchmark: &CleanupBenchmark) -> String {
    format!(
        "About {:.1} factual errors per 1,000 words (4,170 held-out dictations)",
        benchmark.factual_errors_per_1000_words
    )
}

fn gigabytes(bytes: u64) -> String {
    let tenths = (bytes + 50_000_000) / 100_000_000;
    format!("{}.{} GB", tenths / 10, tenths % 10)
}

fn latency(benchmark: &CleanupBenchmark) -> String {
    let seconds = |latency: std::time::Duration| {
        let tenths = (latency.as_millis() + 50) / 100;
        format!("{}.{} s", tenths / 10, tenths % 10)
    };
    format!(
        "{} GPU · {} CPU",
        seconds(benchmark.gpu_latency),
        seconds(benchmark.cpu_latency)
    )
}

/// A measurement shown as its number. `detail` says what was measured, on hover and to
/// screen readers.
#[component]
fn Measurement(label: &'static str, value: String, detail: &'static str) -> Element {
    rsx! {
        span { class: "rating", title: detail,
            span { "{label}" }
            span { class: "measurement", "{value}" }
            span { class: "visually-hidden", ". {detail}" }
        }
    }
}

#[component]
fn RatingBars(label: &'static str, value: Rating, detail: String) -> Element {
    let filled = value.get();
    let spoken = format!("{label}: {filled} of {}. {detail}", Rating::MAX);
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

pub(super) fn model_slug(id: &ModelId) -> String {
    id.as_str().replace('.', "-")
}

#[component]
pub(super) fn ModelFacts(model: ModelSummary, id: Option<String>) -> Element {
    let mut names: Vec<&str> = model
        .languages
        .iter()
        .map(|tag| labels::language_name(tag))
        .collect();
    names.sort_unstable();
    let languages = match names.as_slice() {
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
    };
    rsx! {
        div { id, class: "model-facts",
            span { "{format_bytes(model.size_bytes)} download" }
            {languages}
        }
    }
}
