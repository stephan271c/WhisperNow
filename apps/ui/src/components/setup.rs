//! Each step is saved as the user moves on, so setup resumes where it was left.

use std::rc::Rc;

use dioxus::prelude::*;
use whispernow_domain::ipc::{
    ModelSummary, Permission, PermissionKind, PermissionStatus, SetupTestResult,
};
use whispernow_domain::{
    ErrorCategory, ModelAvailability, ModelId, RecordingMode, SetupCheckpoint, SetupStep, Shortcut,
    ShortcutKey, ShortcutModifier,
};

use super::NoticeRegion;
use super::audio::device_options;
use super::general::RECORDING_MODES;
use super::labels;
use super::microphone_test::{level_fraction, percent};
use super::models::{DownloadProgress, ModelFacts, model_slug};
use super::paste_helper_notice::PasteHelperNotice;
use super::shortcut::{ShortcutCapture, ShortcutSummary};
use super::ui::focus::take_focus;
use super::ui::{Banner, Button, ButtonVariant, Field, Icon, IconName, Segmented, Select};
use crate::UiContext;
use crate::blocked_reason::{BlockedButton, BlockedReason, DisabledReason};
use crate::management::SetupTest;
use crate::notice::{NoticeScope, Tone};
use crate::operations::UiOperation;
use crate::route::{Route, SettingsRoute};

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

/// The step heading's id; focus moves to it on each new step.
const STEP_HEADING: &str = "setup-step-heading";

const fn step_title(step: Option<SetupStep>) -> &'static str {
    match step {
        Some(SetupStep::Welcome) => "Welcome to WhisperNow",
        Some(SetupStep::Permissions) => "Allow access",
        Some(SetupStep::Model) => "Choose a speech model",
        Some(SetupStep::Shortcuts) => "Choose recording shortcuts",
        Some(SetupStep::TryIt) => "Try it",
        None => "You're ready to dictate",
    }
}

#[component]
pub(super) fn SetupScreen() -> Element {
    let context = use_context::<UiContext>();
    let management = context.management;
    let setup = management.read().setup.clone();
    let step = management.read().unfinished_setup_step();
    let mut heading = use_signal(|| None::<Rc<MountedData>>);
    let mut focused_step = use_signal(|| step);
    // A new step, whether moved to here or by another window, takes focus to its heading.
    use_effect(use_reactive!(|step| {
        if *focused_step.peek() == step {
            return;
        }
        focused_step.set(step);
        take_focus(heading);
    }));
    let Some(setup) = setup else {
        return rsx! { p { role: "status", "Loading setup…" } };
    };
    rsx! { article {
        id: "setup-wizard",
        class: "settings-card setup-wizard",
        aria_labelledby: STEP_HEADING,
        if let Some(step) = step {
            SetupStepper { step }
        }
        h2 {
            id: STEP_HEADING,
            tabindex: "-1",
            onmounted: move |event| heading.set(Some(event.data())),
            "{step_title(step)}"
        }
        match step {
            Some(SetupStep::Welcome) => rsx! { WelcomeStep {} },
            Some(SetupStep::Permissions) => rsx! { PermissionsStep {} },
            Some(SetupStep::Model) => rsx! { ModelStep { selected: setup.model_id.clone() } },
            Some(SetupStep::Shortcuts) => rsx! { ShortcutsStep { checkpoint: setup.clone() } },
            Some(SetupStep::TryIt) => rsx! { TryItStep { checkpoint: setup.clone() } },
            None => rsx! { DoneStep {} },
        }
        NoticeRegion { scope: NoticeScope::Setup }
        if let Some(step) = step {
            SetupFooter { step }
        }
    } }
}

#[component]
fn SetupStepper(step: SetupStep) -> Element {
    rsx! { nav { aria_label: "Setup progress",
        ol { class: "setup-steps",
            for (index, shown) in SetupStep::ALL.into_iter().enumerate() {
                li {
                    class: if shown == step { "current" } else if shown < step { "complete" } else { "" },
                    aria_current: (shown == step).then_some("step"),
                    span { class: "setup-step-number",
                        if shown < step { Icon { name: IconName::CheckCircle } } else { "{index + 1}" }
                    }
                    span { "{labels::setup_step(shown)}" }
                    if shown < step {
                        span { class: "visually-hidden", " (done)" }
                    }
                }
            }
        }
    } }
}

#[component]
fn SetupFooter(step: SetupStep) -> Element {
    let context = use_context::<UiContext>();
    let navigator = context.navigator;
    let management = context.management;
    let model_downloading = management
        .read()
        .setup
        .as_ref()
        .is_some_and(|checkpoint| context.active_download(&checkpoint.model_id).is_some());
    let ready = management.read().setup_step_ready(model_downloading);
    let save_blocked = context
        .blocked(UiOperation::SaveSetup)
        .or(context.settings_pending());
    let complete_blocked = context
        .blocked(UiOperation::CompleteSetup)
        .or(context.settings_blocked());
    let earlier = SetupStep::ALL
        .into_iter()
        .rev()
        .find(|earlier| *earlier < step);
    let later = SetupStep::ALL.into_iter().find(|later| *later > step);
    let finishing = later.is_none();
    let primary_blocked = if finishing {
        complete_blocked
    } else {
        save_blocked.clone()
    };
    let hint = (!ready).then(|| step_hint(step)).flatten();
    let describedby = match (primary_blocked.is_some(), hint.is_some()) {
        (true, _) => Some("setup-disabled-reason"),
        (false, true) => Some("setup-step-hint"),
        (false, false) => None,
    };
    rsx! {
        if let Some(hint) = hint {
            p { id: "setup-step-hint", class: "field-help", "{hint}" }
        }
        div { class: "wizard-actions",
            Button {
                id: "setup-back",
                disabled: earlier.is_none() || save_blocked.is_some(),
                onclick: move |_| {
                    if let Some(earlier) = earlier {
                        context.move_setup(earlier);
                    }
                },
                "Back"
            }
            Button {
                id: "setup-finish-later",
                variant: ButtonVariant::Ghost,
                disabled: save_blocked.is_some(),
                onclick: move |_| {
                    context.finish_setup_later();
                    navigator.go(Route::Home);
                },
                "Finish later"
            }
            Button {
                id: "setup-continue",
                variant: ButtonVariant::Primary,
                disabled: !ready || primary_blocked.is_some(),
                aria_describedby: describedby,
                onclick: move |_| match later {
                    Some(later) => context.move_setup(later),
                    None => context.complete_setup(),
                },
                match step {
                    SetupStep::Welcome => "Get started",
                    SetupStep::TryIt => "Finish setup",
                    SetupStep::Permissions | SetupStep::Model | SetupStep::Shortcuts => "Continue",
                }
            }
        }
        DisabledReason { id: "setup-disabled-reason", reason: primary_blocked }
    }
}

const fn step_hint(step: SetupStep) -> Option<&'static str> {
    match step {
        SetupStep::Welcome | SetupStep::TryIt => None,
        SetupStep::Permissions => Some("Allow microphone access to continue."),
        SetupStep::Model => Some("Download a model, or choose one that is installed."),
        SetupStep::Shortcuts => {
            Some("Choose a recording shortcut that includes Control, Alt, Shift, or Meta.")
        }
    }
}

#[component]
fn WelcomeStep() -> Element {
    rsx! { div { class: "setup-hero",
        div { class: "setup-hero-icon", Icon { name: IconName::Mic } }
        p { "WhisperNow turns your speech into text in any app. Speech is recognized locally on this computer." }
    } }
}

#[component]
fn PermissionsStep() -> Element {
    let context = use_context::<UiContext>();
    let management = context.management;
    use_effect(move || context.load_permissions());
    let permissions = management.read().permissions.clone();
    let reason = context
        .state
        .read()
        .snapshot
        .is_none()
        .then_some(BlockedReason::Connecting);
    rsx! { div { class: "setup-content",
        match permissions {
            None => rsx! { p { role: "status", "Checking permissions…" } },
            Some(permissions) => rsx! {
                ul { class: "permission-cards",
                    for permission in permissions.permissions.clone() {
                        PermissionCard { key: "{permission_slug(permission.kind)}", permission, blocked: reason.is_some() }
                    }
                }
                if permissions.all_granted() {
                    Banner { tone: Tone::Success, p { "Everything WhisperNow needs is allowed." } }
                }
                if let Some(setup) = permissions.paste_helper_setup.clone() {
                    PasteHelperNotice { setup, scope: NoticeScope::Setup }
                }
            },
        }
        DisabledReason { id: "permission-disabled-reason", reason }
    } }
}

#[component]
fn PermissionCard(permission: Permission, blocked: bool) -> Element {
    let context = use_context::<UiContext>();
    let kind = permission.kind;
    let slug = permission_slug(kind);
    let (title, purpose) = permission_text(kind);
    let asking = context
        .management
        .read()
        .is_busy(UiOperation::RequestPermission(kind));
    let granted = permission.status == PermissionStatus::Granted;
    let (state, icon) = match permission.status {
        PermissionStatus::Granted => ("Granted", IconName::CheckCircle),
        PermissionStatus::NotDetermined => ("Waiting", IconName::Info),
        PermissionStatus::Denied => ("Not allowed", IconName::Warning),
        PermissionStatus::Unavailable => ("Not available", IconName::AlertCircle),
    };
    rsx! { li { id: "permission-{slug}", class: "permission-card {status_class(permission.status)}",
        div { class: "permission-text",
            h3 { id: "permission-{slug}-title", "{title}" }
            if let Some(purpose) = purpose {
                p { "{purpose}" }
            }
            if let Some(help) = permission_help(kind, permission.status) {
                p { class: "field-help", "{help}" }
            }
        }
        div { class: "permission-actions",
            p { id: "permission-{slug}-state", class: "permission-state", role: "status",
                Icon { name: icon }
                span { if asking { "Waiting for your answer…" } else { "{state}" } }
            }
            if !granted && permission.can_request {
                Button {
                    id: "permission-{slug}-grant",
                    variant: ButtonVariant::Primary,
                    aria_label: "Grant {title}",
                    disabled: blocked || asking,
                    aria_describedby: blocked.then_some("permission-disabled-reason"),
                    onclick: move |_| context.request_permission(kind),
                    if permission.status == PermissionStatus::Denied { "Ask again" } else { "Grant" }
                }
            }
            if !granted && permission.can_open_settings {
                Button {
                    id: "permission-{slug}-settings",
                    aria_label: "Open Settings for {title}",
                    onclick: move |_| context.open_permission_settings(kind),
                    "Open Settings"
                }
            }
        }
    } }
}

const fn permission_slug(kind: PermissionKind) -> &'static str {
    match kind {
        PermissionKind::Microphone => "microphone",
        PermissionKind::Accessibility => "accessibility",
        PermissionKind::GlobalShortcuts => "global-shortcuts",
        PermissionKind::RemoteDesktop => "remote-desktop",
        PermissionKind::PasteSupport => "paste-support",
    }
}

const fn status_class(status: PermissionStatus) -> &'static str {
    match status {
        PermissionStatus::Granted => "granted",
        PermissionStatus::NotDetermined => "waiting",
        PermissionStatus::Denied => "denied",
        PermissionStatus::Unavailable => "unavailable",
    }
}

const fn permission_text(kind: PermissionKind) -> (&'static str, Option<&'static str>) {
    match kind {
        PermissionKind::Microphone => ("Microphone", None),
        PermissionKind::Accessibility => (
            "Accessibility",
            Some("Lets WhisperNow paste the text into the app you are using."),
        ),
        PermissionKind::GlobalShortcuts => ("Recording shortcuts", None),
        PermissionKind::RemoteDesktop => (
            "Typing into other apps",
            Some(
                "Lets WhisperNow press the paste keys for you. Your desktop asks once to allow input control; nothing on your screen is shared.",
            ),
        ),
        PermissionKind::PasteSupport => ("Pasting into other apps", None),
    }
}

const fn permission_help(kind: PermissionKind, status: PermissionStatus) -> Option<&'static str> {
    match (kind, status) {
        (_, PermissionStatus::Granted) => None,
        (PermissionKind::Accessibility, _) => Some(
            "Open Settings, turn on WhisperNow under Privacy & Security › Accessibility, then come back.",
        ),
        (PermissionKind::Microphone, PermissionStatus::Denied) => {
            Some("Allow WhisperNow in your system's microphone privacy settings, then ask again.")
        }
        (PermissionKind::PasteSupport, _) => Some(
            "This X server does not accept simulated keys (XTEST), so WhisperNow copies the text for you to paste.",
        ),
        (
            PermissionKind::GlobalShortcuts | PermissionKind::RemoteDesktop,
            PermissionStatus::Unavailable,
        ) => Some(
            "Your desktop does not offer this. Shortcuts work while WhisperNow is focused, and the text is copied for you to paste.",
        ),
        (
            PermissionKind::GlobalShortcuts | PermissionKind::RemoteDesktop,
            PermissionStatus::Denied,
        ) => Some("Ask again and choose Allow when your desktop asks."),
        _ => None,
    }
}

#[component]
fn ModelStep(selected: ModelId) -> Element {
    let context = use_context::<UiContext>();
    let management = context.management;
    let mut show_all = use_signal(|| false);
    let download_blocked = context
        .model_download_blocked()
        .or(context.settings_pending());
    let models = management
        .read()
        .speech_models()
        .cloned()
        .collect::<Vec<_>>();
    let loaded = management.read().models.is_some();
    // The recommended model and the saved choice; the others wait behind "Show all".
    let (featured, others): (Vec<_>, Vec<_>) = models
        .into_iter()
        .partition(|model| model.recommended || model.id == selected);
    let hidden = others.len();
    rsx! { div { class: "setup-content",
        if !loaded {
            p { role: "status", "Loading models…" }
        }
        fieldset { class: "setup-models",
            legend { "Speech model" }
            for model in featured {
                ModelChoice { key: "{model.id.as_str()}", checked: model.id == selected, model, download_blocked: download_blocked.clone() }
            }
            if show_all() {
                div { id: "setup-more-models",
                    for model in others {
                        ModelChoice { key: "{model.id.as_str()}", checked: model.id == selected, model, download_blocked: download_blocked.clone() }
                    }
                }
            }
        }
        if hidden > 0 {
            Button {
                id: "setup-show-all-models",
                variant: ButtonVariant::Ghost,
                aria_expanded: if show_all() { "true" } else { "false" },
                aria_controls: "setup-more-models",
                onclick: move |_| show_all.toggle(),
                if show_all() { "Show fewer models" } else { "Show all models ({hidden} more)" }
            }
        }
        DisabledReason { id: "setup-models-disabled-reason", reason: download_blocked }
    } }
}

#[component]
fn ModelChoice(
    model: ModelSummary,
    checked: bool,
    download_blocked: Option<BlockedReason>,
) -> Element {
    let context = use_context::<UiContext>();
    let id = model.id.clone();
    let slug = model_slug(&id);
    let name = model.display_name.clone();
    let download = context.active_download(&id);
    let edit_blocked = context.settings_pending();
    let facts_id = format!("setup-model-{slug}-facts");
    let describedby = if edit_blocked.is_some() {
        format!("{facts_id} setup-models-disabled-reason")
    } else {
        facts_id.clone()
    };
    rsx! { div { class: "choice-row setup-model",
        label { class: "setup-model-choice",
            input { id: "setup-model-{slug}", r#type: "radio", name: "setup-model", value: model.id.as_str(), checked,
                disabled: edit_blocked.is_some(),
                aria_describedby: describedby,
                onchange: { let model_id = model.id.clone(); move |_| { context.save_setup_edit(|setup| setup.model_id = model_id.clone()); } }
            }
            span { class: "model-name",
                strong { "{name}" }
                if model.recommended { span { class: "pill", "Recommended" } }
            }
        }
        ModelFacts { id: facts_id, model: model.clone() }
        div { class: "model-actions",
            if let Some(download) = download {
                DownloadProgress { download, name: name.clone() }
            } else if model.availability == ModelAvailability::NotInstalled {
                Button {
                    id: "setup-model-{slug}-download",
                    aria_label: "Download {name}",
                    disabled: download_blocked.is_some(),
                    aria_describedby: download_blocked.as_ref().map(|_| "setup-models-disabled-reason"),
                    onclick: move |_| {
                        if context.save_setup_edit(|setup| setup.model_id = id.clone()) {
                            context.download_model(id.clone());
                        }
                    },
                    "Download"
                }
            } else {
                span { class: "pill",
                    "Installed"
                }
            }
        }
    } }
}

fn shortcut_conflict(shortcut: &Shortcut) -> Option<&'static str> {
    let only = |modifier: ShortcutModifier| shortcut.modifiers == [modifier].into();
    match shortcut.key {
        ShortcutKey::Space if only(ShortcutModifier::Meta) => {
            Some("Meta + Space opens the launcher or Spotlight on many desktops. Try another key.")
        }
        ShortcutKey::Tab if only(ShortcutModifier::Alt) => {
            Some("Alt + Tab switches windows on every desktop. Choose another key.")
        }
        _ => None,
    }
}

#[component]
fn ShortcutsStep(checkpoint: SetupCheckpoint) -> Element {
    let context = use_context::<UiContext>();
    let mut management = context.management;
    let conflict = shortcut_conflict(&checkpoint.recording_shortcut);
    let edit_blocked = context.settings_pending();
    let disabled = edit_blocked.is_some();
    rsx! { div { class: "setup-content",
        p { class: "field-help", "Hold records while you hold the keys; toggle starts on one press and stops on the next." }
        fieldset { class: "shortcut-settings",
            aria_describedby: disabled.then_some("setup-shortcuts-disabled-reason"),
            legend { "Recording shortcut" }
            Field { id: "setup-recording-mode", label: "Recording mode",
                Segmented {
                    disabled,
                    value: checkpoint.recording_mode,
                    options: RECORDING_MODES.to_vec(),
                    onchange: move |mode: RecordingMode| {
                        management.write().edit_setup(|setup| setup.recording_mode = mode);
                    },
                }
            }
            ShortcutCapture { id: "setup-recording-shortcut-capture", label: "recording shortcut", shortcut: checkpoint.recording_shortcut.clone(),
                disabled,
                oncapture: move |shortcut| { management.write().edit_setup(|setup| setup.recording_shortcut = shortcut); } }
            if let Some(conflict) = conflict {
                Banner { tone: Tone::Warning, p { "{conflict}" } }
            }
        }
        DisabledReason { id: "setup-shortcuts-disabled-reason", reason: edit_blocked }
    } }
}

#[component]
fn TryItStep(checkpoint: SetupCheckpoint) -> Element {
    let context = use_context::<UiContext>();
    let management = context.management;
    let model_name = management.read().model_display_name(&checkpoint.model_id);
    let model_ready = management.read().setup_model_ready();
    let download = context.active_download(&checkpoint.model_id);
    let devices = device_options(management.read().devices.as_deref().unwrap_or_default());
    let test = management.read().setup_test.clone();
    let running = matches!(test, Some(SetupTest::Running { .. }));
    let start_blocked = context
        .blocked(UiOperation::StartSetupTest)
        .or(context.phase_blocked());
    let stop_blocked = context.blocked(UiOperation::StopSetupTest);
    let settings_pending = context.settings_pending().is_some();
    rsx! { div { class: "setup-content",
        dl { id: "setup-summary", class: "setup-summary",
            div { dt { "Speech model" } dd { "{model_name}" } }
            ShortcutSummary { mode: checkpoint.recording_mode, shortcut: checkpoint.recording_shortcut.clone() }
        }
        Field { id: "setup-microphone", label: "Microphone",
            Select { value: checkpoint.input_device.clone(),
                placeholder: checkpoint.input_device.as_ref().map(|id| id.as_str().to_owned()),
                options: devices,
                disabled: running || settings_pending,
                onchange: move |device| { context.save_setup_edit(|setup| setup.input_device = device); },
            }
        }
        section { class: "setup-test", aria_labelledby: "setup-test-title",
            h3 { id: "setup-test-title", "Test dictation" }
            if model_ready {
                if running {
                    LiveLevel {}
                    p { class: "field-help", role: "status", "Listening… speak now, then press Stop." }
                    Button { id: "setup-test-stop", variant: ButtonVariant::Primary,
                        disabled: stop_blocked.is_some(),
                        onclick: move |_| context.stop_setup_test(),
                        Icon { name: IconName::Stop }
                        "Stop"
                    }
                } else {
                    BlockedButton { id: "setup-test-start", variant: ButtonVariant::Primary,
                        blocked: start_blocked,
                        onclick: move |_| context.start_setup_test(),
                        Icon { name: IconName::Mic }
                        if test.is_some() { "Test again" } else { "Start test" }
                    }
                }
            } else {
                p { class: "field-help", "The test is available once {model_name} has downloaded." }
                if let Some(download) = download {
                    DownloadProgress { download, name: model_name.clone() }
                }
            }
            div { id: "setup-test-result", class: "setup-test-result", role: "status", aria_live: "polite",
                if let Some(SetupTest::Finished(outcome)) = test {
                    TestDictationResult { outcome }
                }
            }
        }
    } }
}

#[component]
fn LiveLevel() -> Element {
    let state = use_context::<UiContext>().state;
    let mut rms = use_signal(|| 0.0);
    crate::client::use_audio_levels(move |level| {
        if state.peek().recording_session().is_some() {
            rms.set(level.rms);
        }
    });
    let recording = use_memo(move || state.read().recording_session().is_some());
    let level = percent(level_fraction(if recording() { rms() } else { 0.0 }));
    rsx! {
        div { id: "setup-test-level", class: "progress level-live", role: "meter",
            aria_label: "Microphone level", aria_valuemin: "0", aria_valuemax: "100", aria_valuenow: "{level}",
            svg { class: "progress-track", view_box: "0 0 100 4", preserve_aspect_ratio: "none",
                "aria-hidden": "true", "focusable": "false",
                rect { class: "progress-fill", x: "0", y: "0", width: "{level}", height: "4" }
            }
        }
    }
}

#[component]
fn TestDictationResult(outcome: SetupTestResult) -> Element {
    match outcome {
        SetupTestResult::Heard { text } => rsx! {
            Banner { tone: Tone::Success, title: "Perfect",
                p { class: "transcript-text", "“{text}”" }
            }
        },
        SetupTestResult::NoSpeech => rsx! {
            Banner { tone: Tone::Warning, title: "No speech detected",
                p { "Speak a little louder or closer to the microphone, or choose another microphone, then test again." }
            }
        },
        SetupTestResult::Cancelled => rsx! {
            Banner { tone: Tone::Info, title: "Test cancelled — try again",
                p { "Start the test when you're ready, then say a sentence." }
            }
        },
        SetupTestResult::Failed { error } => rsx! {
            Banner { tone: Tone::Error, title: "{error.summary}",
                p { "{setup_fix(error.category)}" }
            }
        },
    }
}

fn setup_fix(category: ErrorCategory) -> &'static str {
    match category {
        ErrorCategory::Audio => "Choose another microphone above, then test again.",
        ErrorCategory::Model | ErrorCategory::Recognition => {
            "Go back to the model step and download the speech model."
        }
        ErrorCategory::Permission => {
            "Check WhisperNow's permissions in your system settings, then test again."
        }
        _ => "Test again. If it keeps failing, restart WhisperNow.",
    }
}

#[component]
fn DoneStep() -> Element {
    let context = use_context::<UiContext>();
    let navigator = context.navigator;
    let management = context.management;
    let shortcut = management.read().settings.as_ref().map(|view| {
        let settings = &view.settings;
        (settings.recording_mode, settings.recording_shortcut.clone())
    });
    rsx! { div { class: "setup-hero",
        div { class: "setup-hero-icon complete", Icon { name: IconName::CheckCircle } }
        dl { id: "setup-done-shortcuts", class: "setup-summary",
            if let Some((mode, shortcut)) = shortcut {
                ShortcutSummary { mode, shortcut }
            }
        }
        p { class: "field-help", "Optional: AI cleanup can fix punctuation and remove filler words." }
        div { class: "wizard-actions",
            Button { id: "setup-ai-cleanup", onclick: move |_| navigator.go(Route::Settings(SettingsRoute::AiCleanup)),
                Icon { name: IconName::Sparkles }
                "Set up AI cleanup"
            }
            Button { id: "setup-done-home", variant: ButtonVariant::Primary, onclick: move |_| navigator.go(Route::Home), "Go to Home" }
        }
    } }
}
