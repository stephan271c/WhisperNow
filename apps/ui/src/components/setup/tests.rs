use dioxus::prelude::*;
use whispernow_domain::AppError;
use whispernow_domain::ipc::{
    ActiveDownload, ErrorCategory, EventPayload, PasteHelperSetup, Permission, PermissionKind,
    PermissionStatus, PermissionsResponse, Phase, SaveSettingsResponse, SettingsView,
    SetupStateView, SetupTestResult,
};
use whispernow_domain::{
    DownloadPhase, ModelAvailability, ModelId, RecordingMode, SetupCheckpoint, SetupStep, Shortcut,
    ShortcutKey, ShortcutModifier,
};

use super::super::actions::PERMISSION_POLLS;
use super::super::tests::{loaded, model, shell};
use crate::ManagementState;
use crate::client::fake;
use crate::test_support::{Page, element, focused, snapshot, state_event, tag, text};

/// Setup saved at `step`, with its model in `availability`.
fn setup_at(step: SetupStep, availability: ModelAvailability) -> ManagementState {
    let mut management = loaded();
    management.models = vec![model("base", "Base", availability)];
    management.models_loaded = true;
    management.setup = Some(SetupStateView {
        first_run_complete: false,
        checkpoint: SetupCheckpoint {
            step,
            model_id: whispernow_domain::ModelId::new("base").expect("fixture model id"),
            input_device: None,
            recording_shortcut: Shortcut {
                modifiers: [
                    ShortcutModifier::Meta,
                    ShortcutModifier::Shift,
                    ShortcutModifier::Control,
                ]
                .into(),
                key: ShortcutKey::R,
            },
            recording_mode: RecordingMode::Toggle,
        },
    });
    management
}

fn home(management: ManagementState) -> Page {
    shell(snapshot(Phase::Idle), management)
}

/// Setup opened from the Home checklist.
fn opened(management: ManagementState) -> Page {
    let mut page = home(management);
    page.click("continue-setup");
    page
}

fn at(step: SetupStep) -> Page {
    opened(setup_at(step, ModelAvailability::Downloaded))
}

/// The desktop's answer to a checkpoint save that moved setup to `step`.
fn saved_at(page: &Page, step: SetupStep) -> SetupStateView {
    let management = page.context().management;
    let mut setup = management.read().setup.clone().expect("setup loaded");
    setup.checkpoint.step = step;
    setup
}

fn start_test(page: &mut Page) {
    page.click("setup-test-start");
    assert_eq!(page.sent(), ["start_test_dictation"]);
}

/// The step the latest checkpoint save asked for.
fn saved_step() -> SetupStep {
    fake::requests::<SetupCheckpoint>("save_setup_checkpoint")
        .last()
        .expect("a checkpoint save")
        .step
}

fn cards(microphone: PermissionStatus, shortcuts: PermissionStatus) -> PermissionsResponse {
    let card = |kind, status| Permission {
        kind,
        status,
        can_request: status != PermissionStatus::Granted,
        can_open_settings: false,
    };
    PermissionsResponse {
        permissions: vec![
            card(PermissionKind::Microphone, microphone),
            card(PermissionKind::GlobalShortcuts, shortcuts),
        ],
        paste_helper_setup: None,
    }
}

fn current_step(html: &str) -> usize {
    let stepper = &html[html.find("setup-steps").expect("stepper")..];
    let current = stepper
        .find("aria-current=\"step\"")
        .expect("one current step");
    stepper[..current].matches("<li").count()
}

fn error(code: &str, summary: &str, category: ErrorCategory) -> AppError {
    AppError::new(category, code, summary)
}

#[test]
fn setup_resumes_at_the_step_saved_before_a_restart() {
    // A new window reads the checkpoint the desktop saved; setup opens at that step.
    let setup = at(SetupStep::Shortcuts);
    let html = setup.html();
    assert_eq!(current_step(&html), 4, "Shortcuts is the fourth step");
    assert_eq!(html.matches("aria-current=\"step\"").count(), 1);
    assert!(tag(&html, "setup-recording-shortcut-capture").is_some());
    assert!(tag(&html, "setup-finish-later").is_some());
}

#[test]
fn granted_permissions_move_setup_on_after_a_request() {
    let mut setup = at(SetupStep::Permissions);
    assert_eq!(
        setup.sent(),
        ["get_permissions"],
        "the step reads the cards, and lists no microphones yet"
    );
    setup.answer(
        "get_permissions",
        Ok(cards(
            PermissionStatus::NotDetermined,
            PermissionStatus::NotDetermined,
        )),
    );
    assert!(
        tag(&setup.html(), "setup-continue")
            .expect("continue")
            .contains(" disabled="),
        "the microphone is needed before moving on"
    );

    setup.click("permission-microphone-grant");
    assert_eq!(setup.sent(), ["request_permission"]);
    assert_eq!(
        fake::requests::<PermissionKind>("request_permission")[0],
        PermissionKind::Microphone
    );
    setup.answer(
        "request_permission",
        Ok(cards(
            PermissionStatus::Granted,
            PermissionStatus::NotDetermined,
        )),
    );
    assert!(setup.sent().is_empty(), "one card still waits");
    assert!(
        !tag(&setup.html(), "setup-continue")
            .expect("continue")
            .contains(" disabled="),
        "with the microphone granted, the user may move on"
    );

    setup.wait();
    assert_eq!(fake::count("get_permissions"), 2, "the watch polls again");
    setup.wait();
    assert_eq!(
        fake::count("get_permissions"),
        2,
        "an unanswered poll is not repeated"
    );
    setup.sent();
    setup.answer(
        "get_permissions",
        Ok(cards(PermissionStatus::Granted, PermissionStatus::Granted)),
    );
    assert_eq!(
        setup.sent(),
        ["save_setup_checkpoint"],
        "setup moves on by itself"
    );
    assert_eq!(saved_step(), SetupStep::Model);
    setup.wait();
    assert_eq!(
        fake::count("get_permissions"),
        2,
        "granting everything ends the watch"
    );

    let saved = saved_at(&setup, SetupStep::Model);
    setup.answer("save_setup_checkpoint", Ok(saved));
    assert_eq!(current_step(&setup.html()), 3);
    assert_eq!(
        setup.sent(),
        ["list_audio_devices"],
        "microphones are listed once setup is past the permission step"
    );
}

fn requested(answer: PermissionsResponse) -> Page {
    let mut setup = at(SetupStep::Permissions);
    setup.answer(
        "get_permissions",
        Ok(cards(
            PermissionStatus::NotDetermined,
            PermissionStatus::NotDetermined,
        )),
    );
    setup.click("permission-microphone-grant");
    setup.sent();
    setup.answer("request_permission", Ok(answer));
    setup
}

#[test]
fn a_card_this_session_cannot_provide_does_not_hold_setup_back() {
    let mut setup = requested(cards(
        PermissionStatus::Granted,
        PermissionStatus::Unavailable,
    ));
    assert_eq!(
        fake::count("save_setup_checkpoint"),
        1,
        "setup moves on without the card nobody can grant"
    );
    assert_eq!(saved_step(), SetupStep::Model);
    setup.wait();
    assert_eq!(fake::count("get_permissions"), 1);
}

#[test]
fn a_final_denial_ends_the_watch_without_moving_on() {
    let mut denied = cards(PermissionStatus::Denied, PermissionStatus::Granted);
    denied.permissions[0].can_request = false;
    let mut setup = requested(denied);
    assert_eq!(
        fake::count("save_setup_checkpoint"),
        0,
        "the denied microphone keeps setup here"
    );
    setup.wait();
    assert_eq!(
        fake::count("get_permissions"),
        1,
        "a final denial ends the watch"
    );
}

#[test]
fn a_permission_watch_stops_reading_after_its_poll_limit() {
    let pending = cards(
        PermissionStatus::NotDetermined,
        PermissionStatus::NotDetermined,
    );
    let mut setup = requested(pending.clone());
    for poll in 1..=PERMISSION_POLLS {
        setup.wait();
        assert_eq!(fake::count("get_permissions"), 1 + poll as usize);
        setup.answer("get_permissions", Ok(pending.clone()));
    }
    setup.wait().wait();
    assert_eq!(
        fake::count("get_permissions"),
        1 + PERMISSION_POLLS as usize,
        "a spent watch stays over"
    );
    assert_eq!(fake::count("save_setup_checkpoint"), 0);
}

#[test]
fn a_new_step_moves_focus_to_its_heading() {
    let mut setup = at(SetupStep::Welcome);
    let html = setup.html();
    assert!(
        tag(&html, "setup-wizard")
            .expect("wizard")
            .contains("aria-labelledby=\"setup-step-heading\"")
    );
    assert!(
        tag(&html, "setup-step-heading")
            .expect("heading")
            .contains("tabindex=\"-1\"")
    );
    setup.mount("setup-step-heading");
    setup.settle();
    assert_eq!(
        focused(),
        None,
        "opening setup leaves focus on the page heading"
    );

    setup.click("setup-continue");
    let saved = saved_at(&setup, SetupStep::Permissions);
    setup.answer("save_setup_checkpoint", Ok(saved));
    assert_eq!(focused().as_deref(), Some("setup-step-heading"));

    setup.click("setup-back");
    assert!(
        setup.sent().contains(&"save_setup_checkpoint"),
        "back saves the step"
    );
    let saved = saved_at(&setup, SetupStep::Welcome);
    setup.answer("save_setup_checkpoint", Ok(saved));
    assert_eq!(focused().as_deref(), Some("setup-step-heading"));
}

#[test]
fn a_pending_model_checkpoint_keeps_the_selected_model_and_allows_download_cancellation() {
    let mut management = setup_at(SetupStep::Model, ModelAvailability::Downloaded);
    management.models.extend([
        model("small", "Small", ModelAvailability::Downloaded),
        model("large", "Large", ModelAvailability::NotInstalled),
        model("stream", "Stream", ModelAvailability::NotInstalled),
    ]);
    let mut setup = opened(management);
    setup.click("setup-show-all-models");
    setup.publish(state_event(Phase::Idle, |state| {
        state.active_downloads.push(ActiveDownload {
            model_id: ModelId::new("stream").expect("fixture model id"),
            phase: DownloadPhase::Downloading,
            downloaded_bytes: 25,
            total_bytes: Some(100),
            bytes_per_second: None,
        });
    }));
    setup.fire("setup-model-small", "change", "small");
    setup.click("setup-continue");
    let submitted = fake::requests::<SetupCheckpoint>("save_setup_checkpoint")[0].clone();
    assert_eq!(submitted.model_id.as_str(), "small");
    assert_eq!(submitted.step, SetupStep::Shortcuts);
    let html = setup.html();
    assert_eq!(current_step(&html), 3, "the old step waits for the answer");
    for id in [
        "setup-model-base",
        "setup-model-small",
        "setup-model-large-download",
    ] {
        assert!(
            tag(&html, id)
                .expect("model control")
                .contains(" disabled="),
            "{id}"
        );
    }
    assert!(
        tag(&html, "setup-model-base")
            .expect("model choice")
            .contains("setup-models-disabled-reason")
    );
    assert!(text(&html, "setup-models-disabled-reason").contains("Wait for the current change"));
    assert!(html.contains("aria-valuenow=\"25\""));

    setup.fire("setup-model-base", "change", "base");
    setup.click_named("Download Large");
    assert_eq!(fake::count("download_model"), 0);
    assert_eq!(
        setup
            .context()
            .management
            .read()
            .setup_checkpoint()
            .expect("checkpoint")
            .model_id,
        submitted.model_id,
    );
    setup.click_named("Cancel download Stream");
    assert_eq!(
        fake::requests::<ModelId>("cancel_download"),
        [ModelId::new("stream").expect("fixture model id")],
    );
    assert!(setup.context().management.read().settings_pending());

    setup.answer(
        "save_setup_checkpoint",
        Ok(SetupStateView {
            first_run_complete: false,
            checkpoint: submitted.clone(),
        }),
    );
    assert_eq!(current_step(&setup.html()), 4);
    assert_eq!(
        setup.context().management.read().setup_checkpoint(),
        Some(&submitted),
    );
}

#[test]
fn a_refused_model_checkpoint_unlocks_model_selection_and_download() {
    let mut management = setup_at(SetupStep::Model, ModelAvailability::Downloaded);
    management
        .models
        .push(model("small", "Small", ModelAvailability::NotInstalled));
    let mut setup = opened(management);
    setup.click("setup-show-all-models");
    setup.click("setup-continue");
    assert!(
        tag(&setup.html(), "setup-model-small-download")
            .expect("download")
            .contains(" disabled=")
    );
    setup.answer(
        "save_setup_checkpoint",
        Err::<SetupStateView, _>(error(
            "setup_save_failed",
            "Setup could not be saved",
            ErrorCategory::Settings,
        )),
    );
    let html = setup.html();
    assert_eq!(current_step(&html), 3);
    assert!(text(&html, "setup-wizard").contains("Setup could not be saved"));
    for id in [
        "setup-model-base",
        "setup-model-small",
        "setup-model-small-download",
    ] {
        assert!(
            !tag(&html, id)
                .expect("model control")
                .contains(" disabled="),
            "{id}"
        );
    }
    setup.click("setup-model-small-download");
    assert_eq!(
        fake::requests::<ModelId>("download_model"),
        [ModelId::new("small").expect("fixture model id")]
    );
    assert_eq!(
        setup
            .context()
            .management
            .read()
            .setup_checkpoint()
            .expect("checkpoint")
            .model_id
            .as_str(),
        "small"
    );
    setup.fire("setup-model-base", "change", "base");
    assert_eq!(
        setup
            .context()
            .management
            .read()
            .setup_checkpoint()
            .expect("checkpoint")
            .model_id
            .as_str(),
        "base"
    );
}

#[test]
fn a_pending_shortcut_checkpoint_keeps_the_submitted_keys_and_mode_and_ends_capture() {
    let mut setup = at(SetupStep::Shortcuts);
    setup.click("setup-recording-mode-hold");
    setup.click("setup-recording-shortcut-capture");
    setup.key_with_code(
        "setup-recording-shortcut-capture",
        Key::Character("t".to_owned()),
        Code::KeyT,
        Modifiers::CONTROL | Modifiers::ALT,
    );
    setup.click("setup-recording-shortcut-capture");
    assert!(
        tag(&setup.html(), "setup-recording-shortcut-capture")
            .expect("capture")
            .contains("aria-pressed=true")
    );
    setup.click("setup-continue");
    let submitted = fake::requests::<SetupCheckpoint>("save_setup_checkpoint")[0].clone();
    assert_eq!(submitted.recording_mode, RecordingMode::Hold);
    assert_eq!(
        submitted.recording_shortcut,
        Shortcut {
            modifiers: [ShortcutModifier::Control, ShortcutModifier::Alt].into(),
            key: ShortcutKey::T,
        }
    );
    let html = setup.html();
    for id in [
        "setup-recording-mode-hold",
        "setup-recording-mode-toggle",
        "setup-recording-shortcut-capture",
    ] {
        assert!(
            tag(&html, id)
                .expect("shortcut control")
                .contains(" disabled="),
            "{id}"
        );
    }
    assert!(
        tag(&html, "setup-recording-shortcut-capture")
            .expect("capture")
            .contains("aria-pressed=false")
    );
    assert!(html.contains("aria-describedby=\"setup-shortcuts-disabled-reason\""));
    assert!(text(&html, "setup-shortcuts-disabled-reason").contains("Wait for the current change"));
    setup.key_with_code(
        "setup-recording-shortcut-capture",
        Key::Character("r".to_owned()),
        Code::KeyR,
        Modifiers::CONTROL,
    );
    setup.act(|mut actions| {
        actions.management.write().edit_setup(|checkpoint| {
            checkpoint.recording_mode = RecordingMode::Toggle;
        });
    });
    assert_eq!(
        setup
            .context()
            .management
            .read()
            .setup_checkpoint()
            .expect("checkpoint")
            .recording_shortcut,
        submitted.recording_shortcut
    );
    assert_eq!(
        setup
            .context()
            .management
            .read()
            .setup_checkpoint()
            .expect("checkpoint")
            .recording_mode,
        submitted.recording_mode
    );
    setup.answer(
        "save_setup_checkpoint",
        Ok(SetupStateView {
            first_run_complete: false,
            checkpoint: submitted.clone(),
        }),
    );
    assert_eq!(current_step(&setup.html()), 5);
    assert_eq!(
        setup.context().management.read().setup_checkpoint(),
        Some(&submitted)
    );
    assert!(text(&setup.html(), "setup-summary").contains("Hold to record"));
}

#[test]
fn a_refused_shortcut_checkpoint_unlocks_keys_and_mode_for_the_next_save() {
    let mut setup = at(SetupStep::Shortcuts);
    setup.click("setup-continue");
    setup.answer(
        "save_setup_checkpoint",
        Err::<SetupStateView, _>(error(
            "setup_save_failed",
            "Setup could not be saved",
            ErrorCategory::Settings,
        )),
    );
    let html = setup.html();
    assert_eq!(current_step(&html), 4);
    for id in [
        "setup-recording-mode-hold",
        "setup-recording-mode-toggle",
        "setup-recording-shortcut-capture",
    ] {
        assert!(
            !tag(&html, id)
                .expect("shortcut control")
                .contains(" disabled="),
            "{id}"
        );
    }
    setup.click("setup-recording-mode-hold");
    setup.click("setup-recording-shortcut-capture");
    setup.key_with_code(
        "setup-recording-shortcut-capture",
        Key::Character("t".to_owned()),
        Code::KeyT,
        Modifiers::CONTROL,
    );
    setup.click("setup-continue");
    let requests = fake::requests::<SetupCheckpoint>("save_setup_checkpoint");
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[1].recording_mode, RecordingMode::Hold);
    assert_eq!(
        requests[1].recording_shortcut,
        Shortcut {
            modifiers: [ShortcutModifier::Control].into(),
            key: ShortcutKey::T,
        }
    );
}

#[test]
fn a_pending_microphone_checkpoint_rejects_a_later_change_and_unlocks_after_success() {
    let mut setup = at(SetupStep::TryIt);
    setup.act(|actions| {
        actions.save_setup_edit(|checkpoint| {
            checkpoint.input_device = Some("usb-mic".to_owned());
        });
    });
    let submitted = fake::requests::<SetupCheckpoint>("save_setup_checkpoint")[0].clone();
    assert_eq!(submitted.input_device.as_deref(), Some("usb-mic"));
    assert!(
        tag(&setup.html(), "setup-microphone")
            .expect("microphone")
            .contains(" disabled=")
    );
    setup.act(|actions| {
        actions.save_setup_edit(|checkpoint| {
            checkpoint.input_device = Some("headset".to_owned());
        });
    });
    assert_eq!(fake::count("save_setup_checkpoint"), 1);
    assert_eq!(
        setup.context().management.read().setup_checkpoint(),
        Some(&submitted)
    );
    setup.answer(
        "save_setup_checkpoint",
        Ok(SetupStateView {
            first_run_complete: false,
            checkpoint: submitted.clone(),
        }),
    );
    assert_eq!(
        setup.context().management.read().setup_checkpoint(),
        Some(&submitted)
    );
    assert!(
        !tag(&setup.html(), "setup-microphone")
            .expect("microphone")
            .contains(" disabled=")
    );
}

#[test]
fn the_summary_shows_the_recording_mode() {
    let summary = text(&at(SetupStep::TryIt).html(), "setup-summary");
    assert!(summary.contains("Press to start and stop"), "{summary}");
}

#[test]
fn try_it_shows_the_test_result_and_can_run_again() {
    for (result, expected) in [
        (
            SetupTestResult::Heard {
                text: "Hello world.".to_owned(),
            },
            "Hello world.",
        ),
        (SetupTestResult::NoSpeech, "No speech detected"),
    ] {
        let mut setup = at(SetupStep::TryIt);
        start_test(&mut setup);
        setup.answer("start_test_dictation", Ok(()));
        setup.publish(state_event(crate::test_support::RECORDING, |_| {}));
        setup.publish(EventPayload::AudioLevel {
            rms: 0.4,
            peak: 0.8,
            elapsed_ms: 1_000,
        });
        let html = setup.html();
        let meter = tag(&html, "setup-test-level").expect("live level");
        assert!(meter.contains("role=\"meter\""), "{meter}");
        assert!(meter.contains("aria-valuenow=\"40\""), "{meter}");
        let start = html.find(meter).expect("meter position") + meter.len();
        let meter_content = &html[start..];
        let graphic = &meter_content[..meter_content.find("</svg>").expect("meter graphic")];
        let fill = element(graphic, "<rect").expect("visible level fill");
        assert!(
            fill.contains("width=\"40\""),
            "RMS 0.4 fills forty percent of the meter: {fill}"
        );
        setup.click("setup-test-stop");
        assert_eq!(fake::count("stop_recording"), 1);
        setup.answer("stop_recording", Ok(()));
        setup.publish(state_event(Phase::Idle, |state| {
            state.setup_test = Some(result);
        }));
        assert!(text(&setup.html(), "setup-test-result").contains(expected));
        assert!(
            tag(&setup.html(), "setup-test-start").is_some(),
            "the test can run again"
        );
    }
}

#[test]
fn a_cancelled_test_reads_as_cancelled_not_as_no_speech() {
    let mut setup = at(SetupStep::TryIt);
    start_test(&mut setup);
    setup.answer("start_test_dictation", Ok(()));
    setup.publish(state_event(crate::test_support::RECORDING, |_| {}));
    setup.click("setup-test-stop");
    assert_eq!(fake::count("stop_recording"), 1);
    // An earlier dictation's transcript is still the last one; it is not the test's result.
    setup.publish(state_event(Phase::Idle, |state| {
        state.last_transcript = Some("Earlier.".to_owned());
        state.setup_test = Some(SetupTestResult::Cancelled);
    }));
    assert!(text(&setup.html(), "setup-test-result").contains("Test cancelled"));
    assert!(!text(&setup.html(), "setup-test-result").contains("Earlier."));

    // A stop sent as the test was cancelled fails; the reported result stands.
    setup.answer(
        "stop_recording",
        Err::<(), _>(error(
            "not_recording",
            "Nothing is recording",
            ErrorCategory::General,
        )),
    );
    assert!(text(&setup.html(), "setup-test-result").contains("Test cancelled"));
    assert!(tag(&setup.html(), "setup-test-start").is_some());
}

#[test]
fn try_it_reports_a_failed_start() {
    let mut setup = at(SetupStep::TryIt);
    start_test(&mut setup);
    setup.answer(
        "start_test_dictation",
        Err::<(), _>(error(
            "audio_start",
            "The microphone could not be opened",
            ErrorCategory::Audio,
        )),
    );
    assert!(
        text(&setup.html(), "setup-test-result").contains("The microphone could not be opened")
    );
    assert!(text(&setup.html(), "setup-test-result").contains("Choose another microphone"));
}

#[test]
fn finishing_setup_shows_a_refusal_then_saves_the_checkpoint_and_stops_permission_watch() {
    let mut management = setup_at(SetupStep::TryIt, ModelAvailability::NotInstalled);
    let checkpoint = management.setup.as_ref().expect("setup").checkpoint.clone();
    let mut settings = crate::test_support::editable_settings();
    settings.first_run_complete = false;
    settings.setup = Some(checkpoint.clone());
    management.install_settings(SettingsView::new(&settings));
    let mut setup = opened(management);
    assert!(tag(&setup.html(), "setup-test-start").is_none());

    setup.act(|actions| actions.open_permission_settings(PermissionKind::Microphone));
    setup.answer("open_permission_settings", Ok(()));
    setup.wait();
    assert_eq!(fake::count("get_permissions"), 1);
    setup.answer(
        "get_permissions",
        Ok(cards(
            PermissionStatus::Granted,
            PermissionStatus::NotDetermined,
        )),
    );

    setup.click("setup-continue");
    assert_eq!(
        fake::requests::<SetupCheckpoint>("complete_setup").as_slice(),
        std::slice::from_ref(&checkpoint)
    );
    let refusal = "Download the selected speech model first";
    setup.answer(
        "complete_setup",
        Err::<SaveSettingsResponse, _>(error("model_not_installed", refusal, ErrorCategory::Model)),
    );
    assert_eq!(text(&setup.html(), "page-heading"), "Set up WhisperNow");
    assert!(text(&setup.html(), "setup-wizard").contains(refusal));
    setup.wait();
    assert_eq!(fake::count("get_permissions"), 2);
    setup.answer(
        "get_permissions",
        Ok(cards(
            PermissionStatus::Granted,
            PermissionStatus::NotDetermined,
        )),
    );

    setup.click("setup-continue");
    assert_eq!(
        fake::requests::<SetupCheckpoint>("complete_setup"),
        [checkpoint.clone(), checkpoint.clone()]
    );
    settings.first_run_complete = true;
    settings.setup = None;
    settings.model_id = checkpoint.model_id;
    settings.input_device = checkpoint.input_device;
    settings.recording_shortcut = checkpoint.recording_shortcut;
    settings.recording_mode = checkpoint.recording_mode;
    setup.answer(
        "complete_setup",
        Ok(SaveSettingsResponse {
            settings: SettingsView::new(&settings),
            warnings: Vec::new(),
        }),
    );
    assert!(tag(&setup.html(), "setup-continue").is_none());
    setup.click("setup-done-home");
    assert_eq!(text(&setup.html(), "page-heading"), "Home");
    assert!(tag(&setup.html(), "setup-wizard").is_none());
    assert!(tag(&setup.html(), "setup-checklist").is_none());
    setup.wait().wait();
    assert_eq!(
        fake::count("get_permissions"),
        2,
        "finishing cancels the permission watch"
    );
}

#[test]
fn finish_later_keeps_the_progress_and_returns_home() {
    let mut setup = at(SetupStep::Shortcuts);
    setup.click("setup-finish-later");
    assert_eq!(
        setup.sent(),
        ["save_setup_checkpoint"],
        "the checkpoint is saved"
    );
    assert_eq!(saved_step(), SetupStep::Shortcuts);
    assert_eq!(text(&setup.html(), "page-heading"), "Home");
    let saved = saved_at(&setup, SetupStep::Shortcuts);
    setup.answer("save_setup_checkpoint", Ok(saved));

    setup.click("continue-setup");
    assert_eq!(
        current_step(&setup.html()),
        4,
        "setup continues where it was left"
    );
}

#[test]
fn settings_runs_setup_again_after_it_finished() {
    let mut management = setup_at(SetupStep::TryIt, ModelAvailability::Downloaded);
    let mut finished = management.setup.clone().expect("setup");
    finished.first_run_complete = true;
    management.setup = Some(finished);
    let mut window = home(management);
    assert!(
        tag(&window.html(), "continue-setup").is_none(),
        "finished setup leaves Home"
    );

    window.click("nav-settings");
    window.click("run-setup-again");
    assert_eq!(fake::count("restart_setup"), 1);
    let management = window.context().management;
    let mut restarted = management.read().setup.clone().expect("setup");
    restarted.first_run_complete = false;
    restarted.checkpoint.step = SetupStep::Welcome;
    window.answer("restart_setup", Ok(restarted));
    let html = window.html();
    assert_eq!(text(&html, "page-heading"), "Set up WhisperNow");
    assert_eq!(current_step(&html), 1);
}

#[test]
fn settings_explain_how_to_paste_without_the_portal_indicator() {
    let mut management = setup_at(SetupStep::TryIt, ModelAvailability::Downloaded);
    let mut finished = management.setup.clone().expect("setup");
    finished.first_run_complete = true;
    management.setup = Some(finished);
    let mut window = home(management);
    window.click("nav-settings");
    assert!(window.sent().contains(&"get_permissions"));
    assert!(tag(&window.html(), "paste-helper-notice").is_none());

    let command = "sudo sh /setup/install_input_helper.sh";
    let mut answer = cards(PermissionStatus::Granted, PermissionStatus::Granted);
    answer.paste_helper_setup = Some(PasteHelperSetup {
        command: Some(command.to_owned()),
    });
    window.answer("get_permissions", Ok(answer));
    assert_eq!(text(&window.html(), "paste-helper-command"), command);
    window.click("paste-helper-copy");
    assert_eq!(window.sent(), ["copy_text"]);
}
