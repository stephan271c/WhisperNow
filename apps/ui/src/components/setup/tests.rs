use dioxus::prelude::*;
use whispernow_api::{
    ApiErrorDto, AppStateDto, ErrorCategoryDto, EventPayload, ListModelsResponse,
    ModelAvailability, PasteHelperSetupDto, PermissionDto, PermissionKindDto, PermissionRequest,
    PermissionStatusDto, PermissionsResponse, PhaseDto, RecoveryActionDto, SetupCheckpoint,
    SetupStateView, SetupStep, SetupTestResultDto, Shortcut, ShortcutKey, ShortcutModifier,
    TranscriptResponse,
};

use super::super::actions::PERMISSION_POLLS;
use super::super::tests::{loaded, model, shell};
use crate::client::{ClientError, fake};
use crate::test_support::{Page, focused, snapshot, tag, text};
use crate::{ManagementState, SetupTest};

/// Setup saved at `step`, with its model in `availability`.
fn setup_at(step: SetupStep, availability: ModelAvailability) -> ManagementState {
    let mut management = loaded();
    management.install_models(ListModelsResponse {
        models: vec![model("base", "Base", availability)],
    });
    management.install_setup(SetupStateView {
        first_run_complete: false,
        checkpoint: SetupCheckpoint {
            step,
            model_id: whispernow_api::ModelId::new("base").expect("fixture model id"),
            input_device: None,
            hold_shortcut: Shortcut {
                modifiers: [
                    ShortcutModifier::Meta,
                    ShortcutModifier::Shift,
                    ShortcutModifier::Control,
                ]
                .into(),
                key: ShortcutKey::R,
            },
            hold_shortcut_enabled: true,
            toggle_shortcut: Some(Shortcut {
                modifiers: [ShortcutModifier::Alt].into(),
                key: ShortcutKey::F9,
            }),
        },
    });
    management
}

fn home(management: ManagementState) -> Page {
    shell(snapshot(PhaseDto::Idle), management)
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

/// Lets the permission watch's timer fire; `false` once the watch is over.
fn poll(page: &mut Page) -> bool {
    page.wait();
    page.context().management.read().permission_watch.is_some()
}

/// The desktop's answer to a checkpoint save that moved setup to `step`.
fn saved_at(page: &Page, step: SetupStep) -> SetupStateView {
    let management = page.context().management;
    let mut setup = management.read().setup.clone().expect("setup loaded");
    setup.checkpoint.step = step;
    setup
}

fn outcome(page: &Page) -> Option<SetupTestResultDto> {
    let management = page.context().management;
    match management.read().setup_test.clone() {
        Some(SetupTest::Finished(outcome)) => Some(outcome),
        _ => None,
    }
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

fn cards(microphone: PermissionStatusDto, shortcuts: PermissionStatusDto) -> PermissionsResponse {
    let card = |kind, status| PermissionDto {
        kind,
        status,
        can_request: status != PermissionStatusDto::Granted,
        can_open_settings: false,
    };
    PermissionsResponse {
        permissions: vec![
            card(PermissionKindDto::Microphone, microphone),
            card(PermissionKindDto::GlobalShortcuts, shortcuts),
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

fn state_event(phase: PhaseDto, change: impl FnOnce(&mut AppStateDto)) -> EventPayload {
    let mut state = snapshot(PhaseDto::Idle);
    state.phase = phase;
    change(&mut state);
    EventPayload::StateChanged { state }
}

fn error(code: &str, summary: &str, recovery: Option<RecoveryActionDto>) -> ApiErrorDto {
    ApiErrorDto {
        category: ErrorCategoryDto::Recognition,
        code: code.to_owned(),
        summary: summary.to_owned(),
        recovery,
    }
}

#[test]
fn setup_resumes_at_the_step_saved_before_a_restart() {
    // A new window reads the checkpoint the desktop saved; setup opens at that step.
    let setup = at(SetupStep::Shortcuts);
    let html = setup.html();
    assert_eq!(current_step(&html), 4, "Shortcuts is the fourth step");
    assert_eq!(html.matches("aria-current=\"step\"").count(), 1);
    assert!(tag(&html, "setup-hold-shortcut-capture").is_some());
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
            PermissionStatusDto::NotDetermined,
            PermissionStatusDto::NotDetermined,
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
        fake::requests::<PermissionRequest>("request_permission")[0].kind,
        PermissionKindDto::Microphone
    );
    setup.answer(
        "request_permission",
        Ok(cards(
            PermissionStatusDto::Granted,
            PermissionStatusDto::NotDetermined,
        )),
    );
    assert!(setup.sent().is_empty(), "one card still waits");
    assert!(
        !tag(&setup.html(), "setup-continue")
            .expect("continue")
            .contains(" disabled="),
        "with the microphone granted, the user may move on"
    );

    assert!(poll(&mut setup));
    assert!(poll(&mut setup));
    assert_eq!(
        setup.sent(),
        ["get_permissions"],
        "an unanswered poll is not repeated"
    );
    setup.answer(
        "get_permissions",
        Ok(cards(
            PermissionStatusDto::Granted,
            PermissionStatusDto::Granted,
        )),
    );
    assert_eq!(
        setup.sent(),
        ["save_setup_checkpoint"],
        "setup moves on by itself"
    );
    assert_eq!(saved_step(), SetupStep::Model);
    assert!(
        !poll(&mut setup),
        "the watch ends once everything is granted"
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

/// Setup on the permission step after the microphone was requested and answered with
/// `answer`; returns the harness and the answer's follow-up commands.
fn requested(answer: PermissionsResponse) -> (Page, Vec<&'static str>) {
    let mut setup = at(SetupStep::Permissions);
    setup.answer(
        "get_permissions",
        Ok(cards(
            PermissionStatusDto::NotDetermined,
            PermissionStatusDto::NotDetermined,
        )),
    );
    setup.click("permission-microphone-grant");
    setup.sent();
    setup.answer("request_permission", Ok(answer));
    let follow_up = setup.sent();
    (setup, follow_up)
}

#[test]
fn a_card_this_session_cannot_provide_does_not_hold_setup_back() {
    let (mut setup, follow_up) = requested(cards(
        PermissionStatusDto::Granted,
        PermissionStatusDto::Unavailable,
    ));
    assert_eq!(
        follow_up,
        ["save_setup_checkpoint"],
        "setup moves on without the card nobody can grant"
    );
    assert_eq!(saved_step(), SetupStep::Model);
    assert!(!poll(&mut setup));
}

#[test]
fn a_final_denial_ends_the_watch_without_moving_on() {
    let mut denied = cards(PermissionStatusDto::Denied, PermissionStatusDto::Granted);
    denied.permissions[0].can_request = false;
    let (mut setup, follow_up) = requested(denied);
    assert!(
        follow_up.is_empty(),
        "the denied microphone keeps setup here"
    );
    assert!(
        !poll(&mut setup),
        "no card can change any more, so nothing is read again"
    );
    assert!(setup.sent().is_empty());
}

#[test]
fn an_unanswered_permission_watch_gives_up_after_its_polls() {
    let (mut setup, _) = requested(cards(
        PermissionStatusDto::NotDetermined,
        PermissionStatusDto::NotDetermined,
    ));
    for _ in 1..PERMISSION_POLLS {
        assert!(poll(&mut setup));
    }
    assert!(!poll(&mut setup));
    assert!(!poll(&mut setup), "a spent watch stays over");
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
fn the_summary_shows_every_modifier_of_each_shortcut() {
    let html = at(SetupStep::TryIt).html();
    let summary = &html[html.find("id=\"setup-summary\"").expect("summary")..];
    let summary = &summary[..summary.find("</dl>").expect("end of summary")];
    let keys = summary
        .split("<kbd")
        .skip(1)
        .map(|key| {
            let key = &key[key.find('>').expect("kbd") + 1..];
            key[..key.find("</kbd>").expect("kbd end")].to_owned()
        })
        .collect::<Vec<_>>();
    assert_eq!(keys, ["Ctrl", "Shift", "Meta", "R", "Alt", "F9"]);
}

#[test]
fn try_it_shows_what_the_test_dictation_heard() {
    let mut setup = at(SetupStep::TryIt);
    start_test(&mut setup);
    setup.answer("start_test_dictation", Ok(()));
    setup.publish(state_event(crate::test_support::RECORDING, |_| {}));
    let html = setup.html();
    assert!(
        tag(&html, "setup-test-level")
            .expect("live level")
            .contains("role=\"meter\"")
    );
    assert!(tag(&html, "setup-test-stop").is_some());

    setup.publish(state_event(PhaseDto::Idle, |state| {
        state.setup_test = Some(SetupTestResultDto::Heard {
            text: "Hello world.".to_owned(),
        });
    }));
    assert_eq!(
        outcome(&setup),
        Some(SetupTestResultDto::Heard {
            text: "Hello world.".to_owned()
        })
    );
    let html = setup.html();
    assert!(text(&html, "setup-test-result").contains("Hello world."));
    assert!(
        tag(&html, "setup-test-start").is_some(),
        "the test can run again"
    );
}

#[test]
fn try_it_reports_no_speech_when_the_stop_heard_nothing() {
    let mut setup = at(SetupStep::TryIt);
    start_test(&mut setup);
    setup.answer("start_test_dictation", Ok(()));
    setup.publish(state_event(crate::test_support::RECORDING, |_| {}));

    setup.click("setup-test-stop");
    assert_eq!(setup.sent(), ["stop_recording"]);
    setup.answer("stop_recording", Ok(()));
    setup.publish(state_event(PhaseDto::Idle, |state| {
        state.setup_test = Some(SetupTestResultDto::NoSpeech);
    }));
    assert_eq!(outcome(&setup), Some(SetupTestResultDto::NoSpeech));
    assert!(!text(&setup.html(), "setup-test-result").trim().is_empty());
}

#[test]
fn a_cancelled_test_reads_as_cancelled_not_as_no_speech() {
    let mut setup = at(SetupStep::TryIt);
    start_test(&mut setup);
    setup.answer("start_test_dictation", Ok(()));
    setup.publish(state_event(crate::test_support::RECORDING, |_| {}));
    // An earlier dictation's transcript is still the last one; it is not the test's result.
    setup.publish(state_event(PhaseDto::Idle, |state| {
        state.last_transcript = Some(TranscriptResponse {
            raw_text: "earlier".to_owned(),
            final_text: "Earlier.".to_owned(),
            text_output_status: None,
            cleanup_failed: false,
            history_save_failed: false,
        });
        state.setup_test = Some(SetupTestResultDto::Cancelled);
    }));
    assert_eq!(outcome(&setup), Some(SetupTestResultDto::Cancelled));
    assert!(!text(&setup.html(), "setup-test-result").contains("Earlier."));

    // A stop sent as the test was cancelled fails; the reported result stands.
    let mut management = setup.context().management;
    management.write().fail_setup_test(&ClientError::Api(error(
        "not_recording",
        "Nothing is recording",
        None,
    )));
    setup.settle();
    assert_eq!(outcome(&setup), Some(SetupTestResultDto::Cancelled));
    assert!(tag(&setup.html(), "setup-test-start").is_some());
}

#[test]
fn try_it_shows_a_failure_with_its_fix() {
    let mut setup = at(SetupStep::TryIt);
    start_test(&mut setup);
    setup.answer(
        "start_test_dictation",
        Err::<(), _>(ClientError::Api(error(
            "audio_start",
            "The microphone could not be opened",
            Some(RecoveryActionDto::ChooseDevice),
        ))),
    );
    let Some(SetupTestResultDto::Failed { error }) = outcome(&setup) else {
        panic!("the test failed");
    };
    assert_eq!(error.summary, "The microphone could not be opened");
    assert_eq!(error.recovery, Some(RecoveryActionDto::ChooseDevice));
    assert!(
        text(&setup.html(), "setup-test-result").contains("The microphone could not be opened")
    );
}

#[test]
fn try_it_waits_for_the_model_download() {
    let html = opened(setup_at(SetupStep::TryIt, ModelAvailability::NotInstalled)).html();
    assert!(tag(&html, "setup-test-start").is_none());
    assert!(
        !tag(&html, "setup-continue")
            .expect("finish")
            .contains(" disabled="),
        "the desktop says why setup cannot finish yet"
    );
}

#[test]
fn finish_later_keeps_the_progress_and_returns_home() {
    let mut setup = at(SetupStep::Shortcuts);
    let html = setup.html();
    let footer = &html[html.find("wizard-actions").expect("footer")..];
    let footer = &footer[..footer.find("</div>").expect("end of footer")];
    assert_eq!(
        footer.matches("<button").count(),
        3,
        "Back, Finish later and Continue: {footer}"
    );

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
    management.install_setup(finished);
    let mut window = home(management);
    assert!(
        tag(&window.html(), "continue-setup").is_none(),
        "finished setup leaves Home"
    );

    window.click("nav-settings");
    window.click("run-setup-again");
    assert_eq!(window.sent(), ["get_permissions", "restart_setup"]);
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
    management.install_setup(finished);
    let mut window = home(management);
    window.click("nav-settings");
    assert!(window.sent().contains(&"get_permissions"));
    assert!(tag(&window.html(), "paste-helper-notice").is_none());

    let command = "sudo sh /setup/install_input_helper.sh";
    let mut answer = cards(PermissionStatusDto::Granted, PermissionStatusDto::Granted);
    answer.paste_helper_setup = Some(PasteHelperSetupDto {
        command: Some(command.to_owned()),
    });
    window.answer("get_permissions", Ok(answer));
    assert_eq!(text(&window.html(), "paste-helper-command"), command);
    window.click("paste-helper-copy");
    assert_eq!(window.sent(), ["write_clipboard"]);
}
