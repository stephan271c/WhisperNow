use whispernow_domain::ipc::{
    EventPayload, ManualUpdatePackage, ModelState, Phase, SettingsView, UpdateError, UpdatePhase,
    UpdateProgress, UpdateState,
};
use whispernow_domain::{AppError, ErrorCategory};
use whispernow_domain::{LlmProvider, Settings, Shortcut, ShortcutKey, ShortcutModifier};

use crate::components::cleanup_destination;
use crate::test_support::{
    editable_settings, history_entry, management_with, settings_view, show_history, snapshot,
    state_event,
};
use crate::{HomeStatus, HomeStatusKind, ManagementState, SettingKey, UiState};

fn ui_state(phase: Phase) -> UiState {
    let mut state = UiState::default();
    state.hydrate(snapshot(phase));
    state
}

fn update_state(phase: UpdatePhase) -> UpdateState {
    UpdateState {
        phase,
        notification_dismissed: false,
        restart_deferred: false,
    }
}

fn settings_html(update: UpdateState) -> String {
    let mut state = ui_state(Phase::Idle);
    state.apply_update_state(update);
    crate::test_support::Page::host(state, ManagementState::default(), crate::AppShell).html()
}

/// Editable settings whose AI cleanup goes to `provider`, at `api_base` when given.
fn cleanup_to(provider: LlmProvider, api_base: Option<&str>) -> Settings {
    let mut settings = editable_settings();
    settings.llm_provider = provider;
    settings.cleanup_enabled = true;
    settings
        .provider_settings_mut(provider)
        .choose_model("cleanup-test".to_owned());
    settings.provider_settings_mut(provider).api_base = api_base.map(|api_base| {
        whispernow_domain::ProviderEndpoint::parse(api_base).expect("a safe endpoint")
    });
    settings
}

fn notice_message(management: &ManagementState) -> Option<&str> {
    management
        .notice
        .as_ref()
        .map(|notice| notice.message.as_str())
}

#[test]
fn failed_refresh_keeps_loaded_rows() {
    let mut management = ManagementState::default();
    show_history(&mut management, vec![history_entry(10, "loaded")]);

    management.fail_history_refresh(&crate::client::connection_error());
    assert_eq!(
        management.history.entries(),
        [history_entry(10, "loaded")],
        "a transient failure keeps rows"
    );
    assert_eq!(
        notice_message(&management),
        Some("The desktop connection failed")
    );
}

fn start_minimized(settings: &mut Settings) {
    settings.start_minimized = true;
}

fn theme(theme: whispernow_domain::ThemePreference) -> impl Fn(&mut Settings) {
    move |settings| settings.theme = theme
}

#[test]
fn cleanup_disclosure_names_the_configured_provider() {
    let settings = cleanup_to(LlmProvider::Anthropic, None);
    assert_eq!(cleanup_destination(&settings).as_deref(), Some("Anthropic"));
}

#[test]
fn cleanup_disclosure_names_a_custom_endpoint_by_its_origin_without_its_path() {
    let settings = cleanup_to(
        LlmProvider::Anthropic,
        Some("https://gateway.example:8443/private-looking-path/v1"),
    );
    assert_eq!(
        cleanup_destination(&settings).as_deref(),
        Some("https://gateway.example:8443")
    );
}

#[test]
fn nothing_is_disclosed_when_cleanup_is_off_or_runs_on_this_computer() {
    let mut off = cleanup_to(LlmProvider::OpenAi, None);
    off.cleanup_enabled = false;
    for settings in [
        off,
        cleanup_to(LlmProvider::LocalCleanup, None),
        cleanup_to(LlmProvider::Ollama, None),
        cleanup_to(LlmProvider::Custom, Some("http://127.0.0.1:8080/v1")),
    ] {
        assert_eq!(
            cleanup_destination(&settings),
            None,
            "{:?}",
            settings.llm_provider
        );
    }
}

#[test]
fn audio_level_is_scoped_by_the_single_recording_phase() {
    let mut state = ui_state(Phase::Idle);
    state.apply_event(EventPayload::AudioLevel {
        rms: 0.4,
        peak: 0.8,
        elapsed_ms: 0,
    });
    assert!(state.audio_rms.abs() < f32::EPSILON);

    state.apply_event(state_event(crate::test_support::RECORDING, |_| {}));
    state.apply_event(EventPayload::AudioLevel {
        rms: 0.4,
        peak: 0.8,
        elapsed_ms: 0,
    });
    assert!((state.audio_rms - 0.4).abs() < f32::EPSILON);
    state.apply_event(state_event(crate::test_support::PROCESSING, |_| {}));
    assert!(state.audio_rms.abs() < f32::EPSILON);
}

#[test]
fn settings_show_update_download_progress() {
    let update = update_state(UpdatePhase::Downloading {
        version: "0.3.0".to_owned(),
        release_notes: None,
        progress: UpdateProgress {
            downloaded_bytes: 512,
            total_bytes: Some(1_024),
        },
    });

    let html = settings_html(update);
    assert!(html.contains("role=\"progressbar\""));
    assert!(html.contains("aria-label=\"Update download progress\""));
    assert!(html.contains("aria-valuenow=\"50\""));
}

#[test]
fn ready_update_banner_escapes_remote_release_notes_as_plain_text() {
    let update = update_state(UpdatePhase::ReadyToRestart {
        version: "0.3.0".to_owned(),
        release_notes: Some("Fixes\n<script>alert('remote')</script>".to_owned()),
        error: None,
    });

    let html = settings_html(update);
    assert!(html.contains("&#60;script&#62;"));
    assert!(html.contains("alert"));
    assert!(html.contains("remote"));
    assert!(html.contains("&#60;/script&#62;"));
    assert!(!html.contains("<script>"));
}

#[test]
fn dismissed_ready_update_keeps_a_secondary_restart_affordance() {
    let mut update = update_state(UpdatePhase::ReadyToRestart {
        version: "0.3.0".to_owned(),
        release_notes: None,
        error: None,
    });
    update.notification_dismissed = true;

    let html = settings_html(update);
    assert!(
        !html.contains("Dismiss"),
        "a dismissed offer stays dismissed"
    );
    assert!(html.contains("Restart to update"));
}

#[test]
fn a_manual_install_offers_the_release_page_and_keeps_it_after_dismissal() {
    let mut update = update_state(UpdatePhase::ManualInstallAvailable {
        version: "0.3.0".to_owned(),
        release_notes: None,
        package: ManualUpdatePackage::Deb,
        error: None,
    });

    let html = settings_html(update.clone());
    assert!(html.contains("Open release page"));
    assert!(html.contains("Dismiss"));
    assert!(!html.contains("Restart to update"));

    update.notification_dismissed = true;
    let dismissed = settings_html(update);
    assert!(!dismissed.contains("Dismiss"));
    assert!(dismissed.contains("Open release page"));
    assert!(!dismissed.contains("Restart to update"));
}

#[test]
fn a_recoverable_update_error_offers_another_check() {
    let offline = update_state(UpdatePhase::Error {
        error: UpdateError::Offline,
    });
    let offline = settings_html(offline);
    assert!(offline.contains("Try again"));
}

#[test]
fn home_status_explains_the_phase_or_the_problem_that_takes_priority() {
    let mut loading = ui_state(Phase::Idle);
    loading.snapshot.as_mut().expect("snapshot").model_state = ModelState::Loading;
    let mut recording = ui_state(crate::test_support::RECORDING);
    recording.snapshot.as_mut().expect("snapshot").model_state = ModelState::Loading;
    let mut failed = snapshot(crate::test_support::RECORDING);
    failed.error = Some(AppError::new(
        ErrorCategory::Audio,
        "microphone_unavailable",
        "Choose an available microphone",
    ));
    let mut audio_problem = UiState::default();
    audio_problem.hydrate(failed);
    let mut disconnected = audio_problem.clone();
    disconnected.event_stream_failed = true;
    let mut unreachable = UiState::default();
    unreachable.report_state_read_failure();

    for (state, kind, title, detail) in [
        (
            loading,
            HomeStatusKind::Waiting,
            "Loading speech model",
            None,
        ),
        (recording, HomeStatusKind::Recording, "Recording", None),
        (
            audio_problem,
            HomeStatusKind::Problem,
            "Dictation needs attention",
            Some("Choose an available microphone"),
        ),
        (
            disconnected,
            HomeStatusKind::Problem,
            "Live updates are unavailable",
            Some("Close and reopen this window to reconnect."),
        ),
        (
            unreachable,
            HomeStatusKind::Problem,
            "WhisperNow could not connect",
            Some("Close and reopen this window to retry."),
        ),
    ] {
        let status = HomeStatus::from_state(&state);
        assert_eq!(status.kind, kind, "{title}");
        assert_eq!(status.title, title);
        assert_eq!(status.detail.as_deref(), detail, "{title}");
    }
}

#[test]
fn a_saved_native_warning_keeps_the_desktops_substituted_shortcut() {
    let mut management = management_with(editable_settings());
    let substituted = management
        .shown_settings()
        .expect("loaded")
        .recording_shortcut
        .clone();
    assert!(
        management.begin_settings_update(SettingKey::RecordingShortcut, |settings| {
            settings.recording_shortcut = Shortcut {
                modifiers: [ShortcutModifier::Alt].into(),
                key: ShortcutKey::R,
            };
        })
    );
    let warning = AppError {
        category: ErrorCategory::General,
        code: "shortcut_registration_failed".to_owned(),
        summary: "The recording shortcut is taken by another app".to_owned(),
    };
    let mut returned = management.shown_settings().cloned().expect("loaded");
    returned.recording_shortcut = substituted.clone();

    management.finish_settings_update(Ok(whispernow_domain::ipc::SaveSettingsResponse {
        settings: settings_view(returned),
        warnings: vec![warning.summary.clone()],
    }));

    assert_eq!(
        management
            .settings
            .as_ref()
            .map(|settings| settings.settings.recording_shortcut.clone()),
        Some(substituted.clone()),
        "the save committed even though a side effect failed"
    );
    assert_eq!(
        management
            .shown_settings()
            .expect("loaded")
            .recording_shortcut,
        substituted
    );
    assert!(!management.settings_pending());
    assert!(notice_message(&management).is_some_and(|notice| notice.contains(&warning.summary)));
}

#[test]
fn an_unrelated_save_keeps_an_unresolved_native_settings_warning() {
    let mut management = management_with(editable_settings());
    assert!(management.begin_settings_update(SettingKey::StartMinimized, start_minimized));
    let request = management.shown_settings().cloned().expect("loaded");
    management.finish_settings_update(Ok(whispernow_domain::ipc::SaveSettingsResponse {
        settings: SettingsView::new(&request),
        warnings: vec!["The recording shortcut could not be activated".to_owned()],
    }));
    let warning = management.notice.clone();

    assert!(management.begin_settings_update(
        SettingKey::Theme,
        theme(whispernow_domain::ThemePreference::Dark),
    ));
    let request = management.shown_settings().cloned().expect("loaded");
    management.finish_settings_update(Ok(whispernow_domain::ipc::SaveSettingsResponse {
        settings: SettingsView::new(&request),
        warnings: Vec::new(),
    }));
    assert_eq!(management.notice, warning);
}

#[test]
fn theme_attribute_follows_a_theme_change_at_once_and_its_rollback() {
    let mut management = ManagementState::default();
    assert_eq!(management.theme_attribute(), None);

    let mut settings = editable_settings();
    settings.theme = whispernow_domain::ThemePreference::Dark;
    management.install_settings(settings_view(settings));
    assert_eq!(management.theme_attribute(), Some("dark"));

    assert!(management.begin_settings_update(
        SettingKey::Theme,
        theme(whispernow_domain::ThemePreference::Light),
    ));
    assert_eq!(management.theme_attribute(), Some("light"));

    management.finish_settings_update(Err(AppError::new(
        ErrorCategory::Settings,
        "settings_refused",
        "refused",
    )));
    assert_eq!(management.theme_attribute(), Some("dark"));

    assert!(management.begin_settings_update(
        SettingKey::Theme,
        theme(whispernow_domain::ThemePreference::System),
    ));
    assert_eq!(management.theme_attribute(), None);
}
