//! First-run setup: permissions first, then a speech model, shortcuts and a test dictation.
//!
//! Each step is saved as the user moves on, so setup resumes where it was left, and focus
//! moves to the step's heading whenever the step changes.

use std::rc::Rc;

use dioxus::prelude::*;
use whispernow_api::{
    ModelAvailability, ModelDto, PermissionDto, PermissionKindDto, PermissionStatusDto,
    RecordingMode, RecoveryActionDto, SetupCheckpoint, SetupStep, SetupTestResultDto, Shortcut,
    ShortcutKey, ShortcutModifier,
};

use super::actions::{DisabledReason, use_actions};
use super::home::percent;
use super::labels::{self, format_bytes};
use super::models::{DownloadProgress, languages};
use super::ui::{Banner, Button, ButtonVariant, Icon, IconName, ShortcutLabel, Tone};
use super::{NoticeRegion, PasteHelperNotice, ShortcutCapture, use_navigator};
use crate::{
    BlockedReason, NoticeScope, Route, SettingsRoute, SetupEdit, SetupTest, UiContext, UiOperation,
};

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

/// The step heading's id; focus moves to it on each new step.
const STEP_HEADING: &str = "setup-step-heading";

/// The step before `step`; the first step has none and stays.
fn previous_step(step: SetupStep) -> SetupStep {
    SetupStep::ALL
        .into_iter()
        .rev()
        .find(|earlier| *earlier < step)
        .unwrap_or(step)
}

/// The step after `step`; the last step has none and stays.
fn next_step(step: SetupStep) -> SetupStep {
    SetupStep::ALL
        .into_iter()
        .find(|later| *later > step)
        .unwrap_or(step)
}

/// The step heading; `None` is the finished setup.
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
    let management = use_context::<UiContext>().management;
    let setup = management.read().setup.clone();
    let step = setup
        .as_ref()
        .filter(|setup| !setup.first_run_complete)
        .map(|setup| setup.checkpoint.step);
    let mut heading = use_signal(|| None::<Rc<MountedData>>);
    let mut focused_step = use_signal(|| step);
    // A new step, whether moved to here or by another window, takes focus to its heading.
    use_effect(use_reactive!(|step| {
        if *focused_step.peek() == step {
            return;
        }
        focused_step.set(step);
        if let Some(heading) = heading.peek().clone() {
            spawn(async move {
                let _ = heading.set_focus(true).await;
            });
        }
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
            Some(SetupStep::Model) => rsx! { ModelStep { selected: setup.checkpoint.model_id.clone() } },
            Some(SetupStep::Shortcuts) => rsx! { ShortcutsStep { checkpoint: setup.checkpoint.clone() } },
            Some(SetupStep::TryIt) => rsx! { TryItStep { checkpoint: setup.checkpoint.clone() } },
            None => rsx! { DoneStep {} },
        }
        if let Some(step) = step {
            NoticeRegion { scope: NoticeScope::Setup }
            SetupFooter { step }
        }
    } }
}

/// Where setup is; the current step is marked with `aria-current="step"`.
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

/// Back, the step's own action, and "Finish later", which keeps the progress for Home.
#[component]
fn SetupFooter(step: SetupStep) -> Element {
    let actions = use_actions();
    let navigator = use_navigator();
    let management = use_context::<UiContext>().management;
    let ready = management.read().setup_step_ready();
    let save_blocked = actions.blocked(UiOperation::SaveSetup);
    let complete_blocked = actions.blocked(UiOperation::CompleteSetup);
    let finishing = step == SetupStep::TryIt;
    let primary_blocked = if finishing {
        complete_blocked
    } else {
        save_blocked
    };
    let hint = (!ready).then(|| step_hint(step));
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
                disabled: step == SetupStep::Welcome || save_blocked.is_some(),
                onclick: move |_| actions.move_setup(previous_step(step)),
                "Back"
            }
            Button {
                id: "setup-finish-later",
                variant: ButtonVariant::Ghost,
                disabled: save_blocked.is_some(),
                onclick: move |_| {
                    actions.finish_setup_later();
                    navigator.go(Route::Home);
                },
                "Finish later"
            }
            Button {
                id: "setup-continue",
                variant: ButtonVariant::Primary,
                disabled: !ready || primary_blocked.is_some(),
                aria_describedby: describedby,
                onclick: move |_| {
                    if finishing {
                        actions.complete_setup();
                    } else {
                        actions.move_setup(next_step(step));
                    }
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

/// Why the step's action is not available yet.
const fn step_hint(step: SetupStep) -> &'static str {
    match step {
        SetupStep::Welcome => "Press Get started to begin.",
        SetupStep::Permissions => "Allow microphone access to continue.",
        SetupStep::Model => "Download a model, or choose one that is installed.",
        SetupStep::Shortcuts => {
            "Turn on at least one shortcut, complete it, and use different keys for each."
        }
        SetupStep::TryIt => "Finishing needs the speech model downloaded and a valid shortcut.",
    }
}

#[component]
fn WelcomeStep() -> Element {
    rsx! { div { class: "setup-hero",
        div { class: "setup-hero-icon", Icon { name: IconName::Mic } }
        p { "WhisperNow turns your speech into text in any app. Speech is recognized on this computer, so your audio stays here." }
        p { class: "field-help", "Setup takes a few minutes and saves your progress as you go." }
    } }
}

/// One card per permission this platform needs; after a request the cards are read again
/// about every second for up to two minutes, and setup moves on by itself once no card can
/// change any more and the microphone is granted. Leaving the step stops the polling.
#[component]
fn PermissionsStep() -> Element {
    let actions = use_actions();
    let management = use_context::<UiContext>().management;
    use_effect(move || actions.load_permissions());
    let permissions = management.read().permissions.clone();
    let reason = actions
        .blocked(UiOperation::RequestPermission)
        .filter(|reason| *reason != BlockedReason::CommandPending);
    rsx! { div { class: "setup-content",
        p { class: "field-help", "WhisperNow needs these to hear you and to type the text where you are working. Nothing leaves this computer." }
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
fn PermissionCard(permission: PermissionDto, blocked: bool) -> Element {
    let actions = use_actions();
    let kind = permission.kind;
    let slug = permission_slug(kind);
    let (title, purpose) = permission_text(kind);
    let asking = actions.permission_requested(kind);
    let granted = permission.status == PermissionStatusDto::Granted;
    let (state, icon) = match permission.status {
        PermissionStatusDto::Granted => ("Granted", IconName::CheckCircle),
        PermissionStatusDto::NotDetermined => ("Waiting", IconName::Info),
        PermissionStatusDto::Denied => ("Not allowed", IconName::Warning),
        PermissionStatusDto::Unavailable => ("Not available", IconName::AlertCircle),
    };
    rsx! { li { id: "permission-{slug}", class: "permission-card {status_class(permission.status)}",
        div { class: "permission-text",
            h3 { id: "permission-{slug}-title", "{title}" }
            p { "{purpose}" }
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
                    onclick: move |_| actions.request_permission(kind),
                    if permission.status == PermissionStatusDto::Denied { "Ask again" } else { "Grant" }
                }
            }
            if !granted && permission.can_open_settings {
                Button {
                    id: "permission-{slug}-settings",
                    aria_label: "Open Settings for {title}",
                    onclick: move |_| actions.open_permission_settings(kind),
                    "Open Settings"
                }
            }
        }
    } }
}

const fn permission_slug(kind: PermissionKindDto) -> &'static str {
    match kind {
        PermissionKindDto::Microphone => "microphone",
        PermissionKindDto::Accessibility => "accessibility",
        PermissionKindDto::GlobalShortcuts => "global-shortcuts",
        PermissionKindDto::RemoteDesktop => "remote-desktop",
        PermissionKindDto::PasteSupport => "paste-support",
    }
}

const fn status_class(status: PermissionStatusDto) -> &'static str {
    match status {
        PermissionStatusDto::Granted => "granted",
        PermissionStatusDto::NotDetermined => "waiting",
        PermissionStatusDto::Denied => "denied",
        PermissionStatusDto::Unavailable => "unavailable",
    }
}

/// Each permission's name and why dictation needs it, in plain words.
const fn permission_text(kind: PermissionKindDto) -> (&'static str, &'static str) {
    match kind {
        PermissionKindDto::Microphone => (
            "Microphone",
            "Lets WhisperNow hear you while you dictate. Audio is processed on this computer.",
        ),
        PermissionKindDto::Accessibility => (
            "Accessibility",
            "Lets WhisperNow paste the text into the app you are using.",
        ),
        PermissionKindDto::GlobalShortcuts => (
            "Recording shortcuts",
            "Lets your recording shortcut work in every app. Your desktop asks once whether to allow it.",
        ),
        PermissionKindDto::RemoteDesktop => (
            "Typing into other apps",
            "Lets WhisperNow press the paste keys for you. Your desktop asks once to allow input control; nothing on your screen is shared.",
        ),
        PermissionKindDto::PasteSupport => (
            "Pasting into other apps",
            "Checks that WhisperNow can press the paste keys in the app you are using.",
        ),
    }
}

/// What to do when a permission is refused or cannot be asked for.
const fn permission_help(
    kind: PermissionKindDto,
    status: PermissionStatusDto,
) -> Option<&'static str> {
    match (kind, status) {
        (_, PermissionStatusDto::Granted) => None,
        (PermissionKindDto::Accessibility, _) => Some(
            "Open Settings, turn on WhisperNow under Privacy & Security › Accessibility, then come back.",
        ),
        (PermissionKindDto::Microphone, PermissionStatusDto::Denied) => {
            Some("Allow WhisperNow in your system's microphone privacy settings, then ask again.")
        }
        (PermissionKindDto::PasteSupport, _) => Some(
            "This X server does not accept simulated keys (XTEST), so WhisperNow copies the text for you to paste.",
        ),
        (
            PermissionKindDto::GlobalShortcuts | PermissionKindDto::RemoteDesktop,
            PermissionStatusDto::Unavailable,
        ) => Some(
            "Your desktop does not offer this. Shortcuts work while WhisperNow is focused, and the text is copied for you to paste.",
        ),
        (
            PermissionKindDto::GlobalShortcuts | PermissionKindDto::RemoteDesktop,
            PermissionStatusDto::Denied,
        ) => Some("Ask again and choose Allow when your desktop asks."),
        _ => None,
    }
}

/// The recommended models, and the rest behind "Show all".
#[component]
fn ModelStep(selected: String) -> Element {
    let management = use_context::<UiContext>().management;
    let mut show_all = use_signal(|| false);
    let download_blocked = use_actions().blocked(UiOperation::DownloadModel);
    let models = management
        .read()
        .speech_models()
        .cloned()
        .collect::<Vec<_>>();
    let loaded = management.read().models_loaded;
    // The recommended model and the saved choice; the others wait behind "Show all".
    let (featured, others): (Vec<_>, Vec<_>) = models
        .into_iter()
        .partition(|model| model.recommended || model.id.as_str() == selected);
    let hidden = others.len();
    rsx! { div { class: "setup-content",
        p { class: "field-help", "Models run on this computer. The download continues in the background, so you can go on to the next step." }
        if !loaded {
            p { role: "status", "Loading models…" }
        }
        fieldset { class: "setup-models",
            legend { "Speech model" }
            for model in featured {
                ModelChoice { key: "{model.id.as_str()}", checked: model.id.as_str() == selected, model }
            }
            if show_all() {
                div { id: "setup-more-models",
                    for model in others {
                        ModelChoice { key: "{model.id.as_str()}", checked: model.id.as_str() == selected, model }
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
fn ModelChoice(model: ModelDto, checked: bool) -> Element {
    let actions = use_actions();
    let mut management = use_context::<UiContext>().management;
    let id = model.id.clone();
    let slug = id.as_str().replace('.', "-");
    let name = model.display_name.clone();
    let download = management.read().download(&id).cloned();
    let download_blocked = actions.blocked(UiOperation::DownloadModel);
    rsx! { div { class: "choice-row setup-model",
        label { class: "setup-model-choice",
            input { r#type: "radio", name: "setup-model", value: model.id.as_str(), checked,
                aria_describedby: "setup-model-{slug}-facts",
                onchange: { let model_id = model.id.clone(); move |_| management.write().edit_setup(SetupEdit::Model(model_id.clone())) }
            }
            span { class: "model-name",
                strong { "{name}" }
                if model.recommended { span { class: "pill", "Recommended" } }
            }
        }
        p { id: "setup-model-{slug}-facts", class: "model-facts",
            "{format_bytes(model.size_bytes)}"
            if let Some(languages) = languages(&model) { " · {languages}" }
        }
        div { class: "model-actions",
            if let Some(download) = download {
                DownloadProgress { download, name: name.clone() }
            } else if model.availability == ModelAvailability::NotInstalled {
                Button {
                    id: "setup-model-{slug}-download",
                    aria_label: "Download {name}",
                    disabled: download_blocked.is_some(),
                    aria_describedby: download_blocked.map(|_| "setup-models-disabled-reason"),
                    onclick: move |_| {
                        management.write().edit_setup(SetupEdit::Model(id.clone()));
                        actions.download_model(id.clone());
                    },
                    "Download"
                }
            } else {
                span { class: "pill",
                    if model.availability == ModelAvailability::Bundled { "Included" } else { "Installed" }
                }
            }
        }
    } }
}

/// Shortcuts that other software commonly takes, with advice for each.
fn shortcut_conflict(shortcut: &Shortcut) -> Option<&'static str> {
    let only = |modifier: ShortcutModifier| shortcut.modifiers == [modifier].into();
    match shortcut.key {
        ShortcutKey::Space if only(ShortcutModifier::Control) => Some(
            "Ctrl + Space often switches the input language or opens autocomplete. Add Shift or Alt if it does nothing.",
        ),
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
    let mut management = use_context::<UiContext>().management;
    // Turning toggle-to-record on waits for a shortcut before the checkpoint has one.
    let mut choosing_toggle = use_signal(|| false);
    let toggle_on = checkpoint.toggle_shortcut.is_some() || choosing_toggle();
    let hold_conflict = checkpoint
        .hold_shortcut_enabled
        .then(|| shortcut_conflict(&checkpoint.hold_shortcut))
        .flatten();
    let toggle_conflict = checkpoint
        .toggle_shortcut
        .as_ref()
        .and_then(shortcut_conflict);
    rsx! { div { class: "setup-content",
        p { class: "field-help", "Use one mode or both. Hold to record stops when you let go; toggle starts on one press and stops on the next." }
        fieldset { class: "shortcut-settings",
            legend { "Hold to record" }
            label { class: "checkbox-row",
                input { r#type: "checkbox", checked: checkpoint.hold_shortcut_enabled,
                    onchange: move |event| management.write().edit_setup(SetupEdit::HoldEnabled(event.checked()))
                }
                "Turn on hold to record"
            }
            if checkpoint.hold_shortcut_enabled {
                ShortcutCapture { id: "setup-hold-shortcut-capture", label: "hold-to-record shortcut", shortcut: Some(checkpoint.hold_shortcut.clone()),
                    oncapture: move |shortcut| management.write().edit_setup(SetupEdit::Shortcut(RecordingMode::Hold, shortcut)) }
            }
            if let Some(conflict) = hold_conflict {
                Banner { tone: Tone::Warning, p { "{conflict}" } }
            }
        }
        fieldset { class: "shortcut-settings",
            legend { "Toggle to record" }
            label { class: "checkbox-row",
                input { r#type: "checkbox", checked: toggle_on,
                    onchange: move |event| {
                        choosing_toggle.set(event.checked());
                        if !event.checked() {
                            management.write().edit_setup(SetupEdit::ToggleOff);
                        }
                    }
                }
                "Turn on toggle to record"
            }
            if toggle_on {
                ShortcutCapture { id: "setup-toggle-shortcut-capture", label: "toggle-to-record shortcut", shortcut: checkpoint.toggle_shortcut.clone(),
                    oncapture: move |shortcut| management.write().edit_setup(SetupEdit::Shortcut(RecordingMode::Toggle, shortcut)) }
            }
            if let Some(conflict) = toggle_conflict {
                Banner { tone: Tone::Warning, p { "{conflict}" } }
            }
        }
        aside { class: "setup-tips", aria_labelledby: "setup-tips-title",
            h3 { id: "setup-tips-title", "Tips" }
            ul {
                li { "Pick keys you don't use for anything else, such as Ctrl + Alt + a letter or an F key." }
                li { "Shortcuts with Meta (the Windows or Command key) are often taken by the desktop." }
                li { "If a shortcut does nothing after setup, change it in Settings; your previous one stays active until the new one works." }
            }
        }
    } }
}

/// The choices so far, a microphone to test with, and a test dictation.
#[component]
fn TryItStep(checkpoint: SetupCheckpoint) -> Element {
    let actions = use_actions();
    let context = use_context::<UiContext>();
    let management = context.management;
    let model_name = management
        .read()
        .model_display_name(checkpoint.model_id.as_str());
    let model_ready = management.read().setup_model_ready();
    let download = management.read().download(&checkpoint.model_id).cloned();
    let devices = management.read().devices.clone();
    let test = management.read().setup_test.clone();
    let running = matches!(test, Some(SetupTest::Running { .. }));
    let level = percent(context.state.read().audio_rms);
    let start_blocked = actions.blocked(UiOperation::StartSetupTest);
    let stop_blocked = actions.blocked(UiOperation::StopSetupTest);
    let selected = checkpoint.input_device.clone().unwrap_or_default();
    rsx! { div { class: "setup-content",
        p { class: "field-help", "Say a sentence to check that everything works. The test shows the text here; nothing is pasted or saved." }
        dl { id: "setup-summary", class: "setup-summary",
            div { dt { "Speech model" } dd { "{model_name}" } }
            div { dt { "Hold to record" }
                dd { if checkpoint.hold_shortcut_enabled { ShortcutLabel { shortcut: checkpoint.hold_shortcut.clone() } } else { "Off" } }
            }
            div { dt { "Toggle to record" }
                dd { if let Some(shortcut) = checkpoint.toggle_shortcut.clone() { ShortcutLabel { shortcut } } else { "Off" } }
            }
        }
        div { class: "field",
            label { r#for: "setup-microphone", "Microphone" }
            select { id: "setup-microphone", value: "{selected}", disabled: running,
                onchange: move |event| {
                    let value = event.value();
                    actions.save_setup_edit(SetupEdit::InputDevice((!value.is_empty()).then_some(value)));
                },
                option { value: "", "System default" }
                for device in devices { option { value: "{device.id}", "{device.display_name}" } }
            }
        }
        section { class: "setup-test", aria_labelledby: "setup-test-title",
            h3 { id: "setup-test-title", "Test dictation" }
            if model_ready {
                if running {
                    div { id: "setup-test-level", class: "progress level-live", role: "meter",
                        aria_label: "Microphone level", aria_valuemin: "0", aria_valuemax: "100", aria_valuenow: "{level}",
                        svg { class: "progress-track", view_box: "0 0 100 4", preserve_aspect_ratio: "none",
                            "aria-hidden": "true", "focusable": "false",
                            rect { class: "progress-fill", x: "0", y: "0", width: "{level}", height: "4" }
                        }
                    }
                    p { class: "field-help", role: "status", "Listening… speak now, then press Stop." }
                    Button { id: "setup-test-stop", variant: ButtonVariant::Primary,
                        disabled: stop_blocked.is_some(),
                        onclick: move |_| actions.stop_setup_test(),
                        Icon { name: IconName::Stop }
                        "Stop"
                    }
                } else {
                    Button { id: "setup-test-start", variant: ButtonVariant::Primary,
                        disabled: start_blocked.is_some(),
                        aria_describedby: start_blocked.map(|_| "setup-test-disabled-reason"),
                        onclick: move |_| actions.start_setup_test(),
                        Icon { name: IconName::Mic }
                        if test.is_some() { "Test again" } else { "Start test" }
                    }
                    DisabledReason { id: "setup-test-disabled-reason", reason: start_blocked }
                }
            } else {
                p { class: "field-help", "The test is available once {model_name} has downloaded." }
                if let Some(download) = download {
                    DownloadProgress { download, name: model_name.clone() }
                }
            }
            div { id: "setup-test-result", class: "setup-test-result", role: "status", aria_live: "polite",
                if let Some(SetupTest::Finished(outcome)) = test {
                    SetupTestResult { outcome }
                }
            }
        }
    } }
}

#[component]
fn SetupTestResult(outcome: SetupTestResultDto) -> Element {
    match outcome {
        SetupTestResultDto::Heard { text } => rsx! {
            Banner { tone: Tone::Success, title: "Perfect",
                p { class: "transcript-text", "“{text}”" }
            }
        },
        SetupTestResultDto::NoSpeech => rsx! {
            Banner { tone: Tone::Warning, title: "No speech detected",
                p { "Speak a little louder or closer to the microphone, or choose another microphone, then test again." }
            }
        },
        SetupTestResultDto::Cancelled => rsx! {
            Banner { tone: Tone::Info, title: "Test cancelled — try again",
                p { "Start the test when you're ready, then say a sentence." }
            }
        },
        SetupTestResultDto::Failed { error } => rsx! {
            Banner { tone: Tone::Error, title: "{error.summary}",
                p { "{recovery_fix(error.recovery)}" }
            }
        },
    }
}

/// How to fix a failed test, from the desktop's recovery hint.
const fn recovery_fix(recovery: Option<RecoveryActionDto>) -> &'static str {
    match recovery {
        Some(RecoveryActionDto::ChooseDevice) => {
            "Choose another microphone above, then test again."
        }
        Some(RecoveryActionDto::InstallModel) => {
            "Go back to the model step and download the speech model."
        }
        Some(RecoveryActionDto::OpenSettings) => {
            "Check WhisperNow's permissions in your system settings, then test again."
        }
        Some(
            RecoveryActionDto::Retry
            | RecoveryActionDto::CopyManually
            | RecoveryActionDto::CheckDestination,
        )
        | None => "Test again. If it keeps failing, restart WhisperNow.",
    }
}

/// Setup has finished: the shortcut reminder and optional AI cleanup.
#[component]
fn DoneStep() -> Element {
    let navigator = use_navigator();
    let management = use_context::<UiContext>().management;
    // The saved settings, or the checkpoint setup finished with until they load.
    let (hold, toggle) = {
        let management = management.read();
        if let Some(view) = management.settings.as_ref() {
            let settings = &view.settings;
            (
                settings
                    .hold_shortcut_enabled
                    .then(|| settings.hold_shortcut.clone()),
                settings.toggle_shortcut.clone(),
            )
        } else if let Some(checkpoint) = management.setup_checkpoint() {
            (
                checkpoint
                    .hold_shortcut_enabled
                    .then(|| checkpoint.hold_shortcut.clone()),
                checkpoint.toggle_shortcut.clone(),
            )
        } else {
            (None, None)
        }
    };
    rsx! { div { class: "setup-hero",
        div { class: "setup-hero-icon complete", Icon { name: IconName::CheckCircle } }
        p { "Use your shortcut in any app, speak, and your words appear where you are typing." }
        dl { id: "setup-done-shortcuts", class: "setup-summary",
            if let Some(shortcut) = hold {
                div { dt { "Hold to record" } dd { ShortcutLabel { shortcut } } }
            }
            if let Some(shortcut) = toggle {
                div { dt { "Toggle to record" } dd { ShortcutLabel { shortcut } } }
            }
        }
        p { class: "field-help", "Optional: AI cleanup can fix punctuation and remove filler words using a provider you choose." }
        div { class: "wizard-actions",
            Button { id: "setup-ai-cleanup", onclick: move |_| navigator.go(Route::Settings(SettingsRoute::AiCleanup)),
                Icon { name: IconName::Sparkles }
                "Set up AI cleanup"
            }
            Button { id: "setup-done-home", variant: ButtonVariant::Primary, onclick: move |_| navigator.go(Route::Home), "Go to Home" }
        }
    } }
}
