use whispernow_domain::AppError;
use whispernow_domain::ipc::{
    ClearDataItem, ClearDataResult, ErrorCategory, EventPayload, ManualUpdatePackage, ModelState,
    Phase, SettingsView, UpdateError, UpdatePhase, UpdateProgress, UpdateState,
};
use whispernow_domain::{
    ClearDataCategory, EntryId, HistoryCursor, HistoryEntry, HistoryPage, HistoryPageRequest,
    LlmProvider, Settings, Shortcut, ShortcutKey, ShortcutModifier, UtcMillis,
};

use crate::components::cleanup_destination;
use crate::test_support::{editable_settings, history_entry, show_history, snapshot, state_event};
use crate::{HomeStatus, HomeStatusKind, ManagementState, SaveStatus, SettingKey, UiState};

fn ui_state(phase: Phase) -> UiState {
    let mut state = UiState::default();
    state.hydrate(snapshot(phase));
    state
}

fn update_state(phase: UpdatePhase) -> UpdateState {
    UpdateState {
        phase,
        available_version: None,
        release_notes: None,
        progress: None,
        error: None,
        notification_dismissed: false,
        restart_deferred: false,
        manual_package: None,
    }
}

fn settings_html(update: UpdateState) -> String {
    let mut state = ui_state(Phase::Idle);
    state.hydrate_update(update);
    crate::test_support::Page::shell(state, ManagementState::default()).html()
}

/// Editable settings whose AI cleanup goes to `provider`, at `api_base` when given.
fn cleanup_to(provider: LlmProvider, api_base: Option<&str>) -> Settings {
    let mut settings = editable_settings();
    settings.llm_provider = provider;
    settings.cleanup_enabled = true;
    settings.provider_settings_mut(provider).api_base = api_base.map(|api_base| {
        whispernow_domain::ProviderEndpoint::parse(api_base).expect("a safe endpoint")
    });
    settings
}

fn history_window(entries: Vec<HistoryEntry>) -> HistoryPage {
    HistoryPage {
        entries,
        next_cursor: None,
    }
}

fn notice_message(management: &ManagementState) -> Option<&str> {
    management
        .notice
        .as_ref()
        .map(|notice| notice.message.as_str())
}

fn history_unavailable_error() -> AppError {
    AppError {
        category: ErrorCategory::HistoryUnavailable,
        code: "history_unavailable".to_owned(),
        summary: "History storage is unavailable".to_owned(),
    }
}

#[test]
fn history_copy_uses_the_pasted_text_of_the_entry() {
    let mut management = ManagementState::default();
    let mut cleaned = history_entry(11, "raw");
    cleaned.cleaned_text = Some("cleaned".to_owned());
    show_history(&mut management, vec![history_entry(10, "first"), cleaned]);

    assert_eq!(
        management.history_copy(EntryId(10)).as_deref(),
        Some("first")
    );
    assert_eq!(
        management.history_copy(EntryId(11)).as_deref(),
        Some("cleaned")
    );
    assert!(management.history_copy(EntryId(99)).is_none());
}

#[test]
fn a_history_read_for_a_filter_no_longer_shown_is_ignored() {
    let mut management = ManagementState::default();
    let (stale, _) = management.history_request();
    assert!(management.show_failed_history_only(true));
    let (latest, _) = management.history_request();

    management.finish_history_refresh(latest, history_window(vec![history_entry(11, "newest")]));
    management.finish_history_refresh(stale, history_window(vec![history_entry(10, "stale")]));
    management.fail_history_refresh(stale, &history_unavailable_error());

    assert!(management.history_loaded());
    assert!(management.history_unavailable.is_none());
    assert_eq!(management.history.len(), 1);
    assert_eq!(management.history[0].id, EntryId(11));
}

#[test]
fn loading_older_history_grows_the_window_that_later_refreshes_reread() {
    let mut management = ManagementState::default();
    let (request_id, _) = management.history_request();
    management.finish_history_refresh(
        request_id,
        HistoryPage {
            next_cursor: Some(HistoryCursor {
                created_at: UtcMillis(10),
                id: EntryId(10),
            }),
            ..history_window(vec![history_entry(10, "first page")])
        },
    );
    assert_eq!(management.history_window_pages(), 1);

    let (request_id, request) = management.older_history_request();
    assert_eq!(request.limit, 2 * HistoryPageRequest::DEFAULT_PAGE_SIZE);
    management.finish_history_refresh(
        request_id,
        history_window(vec![
            history_entry(10, "first page"),
            history_entry(9, "second page"),
        ]),
    );
    assert_eq!(management.history.len(), 2);
    assert!(management.history_next_cursor.is_none());

    assert_eq!(
        management.history_request().1.limit,
        2 * HistoryPageRequest::DEFAULT_PAGE_SIZE
    );
}

#[test]
fn failed_refresh_keeps_loaded_rows_but_unavailable_storage_degrades() {
    let mut management = ManagementState::default();
    show_history(&mut management, vec![history_entry(10, "loaded")]);

    let (request_id, _) = management.history_request();
    management.fail_history_refresh(request_id, &crate::client::connection_error());
    assert_eq!(
        management.history.len(),
        1,
        "a transient failure keeps rows"
    );
    assert_eq!(
        notice_message(&management),
        Some("The desktop connection failed")
    );
    assert!(management.history_unavailable.is_none());

    let (request_id, _) = management.history_request();
    management.fail_history_refresh(request_id, &history_unavailable_error());
    assert!(management.history.is_empty());
    assert!(!management.history_loaded());
    assert_eq!(
        management.history_unavailable.as_deref(),
        Some("History storage is unavailable")
    );
}

#[test]
fn partial_clear_data_does_not_hide_history_when_database_deletion_failed() {
    let mut management = ManagementState::default();
    let (request_id, _) = management.history_request();
    management.finish_history_refresh(
        request_id,
        HistoryPage {
            next_cursor: Some(HistoryCursor {
                created_at: UtcMillis(10),
                id: EntryId(10),
            }),
            ..history_window(vec![history_entry(10, "still in database")])
        },
    );

    management.apply_clear_result(ClearDataResult {
        items: vec![
            ClearDataItem {
                category: ClearDataCategory::History,
                path: "history.sqlite3".to_owned(),
                deleted: false,
            },
            ClearDataItem {
                category: ClearDataCategory::History,
                path: "recordings".to_owned(),
                deleted: true,
            },
        ],
        restart_required: false,
    });

    assert_eq!(management.history.len(), 1);
    assert_eq!(management.history[0].id, EntryId(10));
    assert!(management.history_next_cursor.is_some());
    assert!(management.clear_result.is_some());
}

fn loaded_management(settings: Settings) -> ManagementState {
    let mut management = ManagementState::default();
    management.install_settings(SettingsView {
        settings,
        warnings: Vec::new(),
    });
    management
}

fn start_on_login(settings: &mut Settings) {
    settings.auto_start_on_login = true;
}

fn theme(theme: whispernow_domain::ThemePreference) -> impl Fn(&mut Settings) {
    move |settings| settings.theme = theme
}

fn is_failed(status: &SaveStatus) -> bool {
    matches!(status, SaveStatus::Failed(_))
}

#[test]
fn a_pending_shortcut_save_blocks_another_edit_and_a_refusal_does_not_leak_into_it() {
    let mut management = loaded_management(editable_settings());
    let confirmed = management
        .shown_settings()
        .expect("loaded")
        .recording_shortcut
        .clone();
    let request = management
        .begin_settings_update(SettingKey::RecordingShortcut, |settings| {
            settings.recording_shortcut.key = ShortcutKey::R;
        })
        .expect("shortcut edit");
    assert_eq!(request.recording_shortcut.key, ShortcutKey::R);
    assert!(management.settings_pending());
    assert_eq!(
        management
            .shown_settings()
            .expect("loaded")
            .recording_shortcut,
        request.recording_shortcut
    );
    assert!(
        management
            .begin_settings_update(
                SettingKey::Theme,
                theme(whispernow_domain::ThemePreference::Dark),
            )
            .is_none()
    );
    assert_eq!(
        management.setting_status(&SettingKey::Theme),
        SaveStatus::Idle
    );

    management.finish_settings_update(Err("The shortcut is taken".to_owned()));
    assert!(!management.settings_pending());
    assert_eq!(
        management
            .shown_settings()
            .expect("loaded")
            .recording_shortcut,
        confirmed
    );
    let request = management
        .begin_settings_update(
            SettingKey::Theme,
            theme(whispernow_domain::ThemePreference::Dark),
        )
        .expect("theme edit after refusal");
    assert_eq!(request.recording_shortcut, confirmed);
    assert_eq!(request.theme, whispernow_domain::ThemePreference::Dark);
    management.finish_settings_update(Ok(whispernow_domain::ipc::SaveSettingsResponse {
        settings: SettingsView::new(&request),
        warnings: Vec::new(),
    }));
    assert_eq!(
        management.setting_status(&SettingKey::Theme),
        SaveStatus::Saved
    );
    assert!(is_failed(
        &management.setting_status(&SettingKey::RecordingShortcut)
    ));
}

#[test]
fn a_refused_change_rolls_back_and_keeps_its_error_beside_the_control() {
    let mut management = loaded_management(editable_settings());
    assert!(
        management
            .begin_settings_update(SettingKey::StartOnLogin, start_on_login)
            .is_some()
    );
    management.finish_settings_update(Err("refused".to_owned()));

    assert!(
        !management
            .shown_settings()
            .expect("loaded")
            .auto_start_on_login
    );
    assert!(is_failed(
        &management.setting_status(&SettingKey::StartOnLogin)
    ));
    assert!(!management.settings_pending());

    assert!(
        management
            .begin_settings_update(SettingKey::StartOnLogin, start_on_login)
            .is_some()
    );
    assert_eq!(
        management.setting_status(&SettingKey::StartOnLogin),
        SaveStatus::Saving,
        "a new attempt clears the old error"
    );
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
fn audio_level_is_scoped_by_the_single_recording_phase_and_clamped() {
    let mut state = ui_state(Phase::Idle);
    state.apply_event(EventPayload::AudioLevel {
        rms: 0.4,
        peak: 0.8,
        elapsed_ms: 0,
    });
    assert!(state.audio_rms.abs() < f32::EPSILON);

    state.apply_event(state_event(crate::test_support::RECORDING, |_| {}));
    state.apply_event(EventPayload::AudioLevel {
        rms: 2.0,
        peak: 2.0,
        elapsed_ms: 0,
    });
    assert!((state.audio_rms - 1.0).abs() < f32::EPSILON);
    state.apply_event(EventPayload::AudioLevel {
        rms: -1.0,
        peak: 0.0,
        elapsed_ms: 0,
    });
    assert!(state.audio_rms.abs() < f32::EPSILON);

    state.apply_event(EventPayload::AudioLevel {
        rms: 0.4,
        peak: 0.8,
        elapsed_ms: 0,
    });
    state.apply_event(state_event(crate::test_support::PROCESSING, |_| {}));
    assert!(state.audio_rms.abs() < f32::EPSILON);
}

#[test]
fn settings_show_update_download_progress() {
    let mut update = update_state(UpdatePhase::Downloading);
    update.available_version = Some("0.3.0".to_owned());
    update.progress = Some(UpdateProgress {
        downloaded_bytes: 512,
        total_bytes: Some(1_024),
    });

    let html = settings_html(update);
    assert!(html.contains("Downloading update to v0.3.0"));
    assert!(html.contains("Update download progress"));
    assert!(html.contains("0.5 KiB of 1.0 KiB downloaded"));
}

#[test]
fn ready_update_banner_escapes_remote_release_notes_as_plain_text() {
    let mut update = update_state(UpdatePhase::ReadyToRestart);
    update.available_version = Some("0.3.0".to_owned());
    update.release_notes = Some("Fixes\n<script>alert('remote')</script>".to_owned());

    let html = settings_html(update);
    assert!(html.contains("Restart to update to v0.3.0"));
    assert!(html.contains("What’s new"));
    assert!(html.contains("&#60;script&#62;"));
    assert!(html.contains("alert"));
    assert!(html.contains("remote"));
    assert!(html.contains("&#60;/script&#62;"));
    assert!(!html.contains("<script>"));
}

#[test]
fn dismissed_ready_update_keeps_a_secondary_restart_affordance() {
    let mut update = update_state(UpdatePhase::ReadyToRestart);
    update.available_version = Some("v0.3.0".to_owned());
    update.notification_dismissed = true;

    let html = settings_html(update);
    assert!(
        !html.contains("Dismiss"),
        "a dismissed offer stays dismissed"
    );
    assert!(html.contains("Update v0.3.0 is ready"));
    assert!(html.contains("Restart to update"));
}

#[test]
fn manual_install_offer_names_the_package_and_links_the_release_page_without_restart() {
    for (package, format) in [
        (ManualUpdatePackage::Deb, ".deb"),
        (ManualUpdatePackage::Rpm, ".rpm"),
    ] {
        let mut update = update_state(UpdatePhase::ManualInstallAvailable);
        update.available_version = Some("0.3.0".to_owned());
        update.manual_package = Some(package);

        let html = settings_html(update.clone());
        assert!(html.contains("WhisperNow v0.3.0 is available"));
        assert!(html.contains(format));
        assert!(html.contains("Open release page"));
        assert!(!html.contains("Restart to update"));

        update.notification_dismissed = true;
        let dismissed = settings_html(update);
        assert!(
            !dismissed.contains("Dismiss"),
            "a dismissed offer stays dismissed"
        );
        assert!(
            !dismissed.contains(format),
            "only the version and the link stay"
        );
        assert!(dismissed.contains("Open release page"));
    }
}

#[test]
fn manual_check_results_render_up_to_date_and_typed_errors() {
    let up_to_date = settings_html(update_state(UpdatePhase::UpToDate));
    assert!(up_to_date.contains("WhisperNow is up to date"));
    assert!(up_to_date.contains("No newer stable release is available"));

    let mut offline = update_state(UpdatePhase::Error);
    offline.error = Some(UpdateError::Offline);
    let offline = settings_html(offline);
    assert!(offline.contains("Update needs attention"));
    assert!(offline.contains("You appear to be offline"));
    assert!(offline.contains("Try again"));
}

#[test]
fn home_status_explains_the_phase_or_the_problem_that_takes_priority() {
    let mut loading = ui_state(Phase::Idle);
    loading.snapshot.as_mut().expect("snapshot").model_state = ModelState::Loading;
    let mut failed = snapshot(Phase::Idle);
    failed.error = Some(AppError::new(
        ErrorCategory::Audio,
        "microphone_unavailable",
        "Choose an available microphone",
    ));
    let mut audio_problem = UiState::default();
    audio_problem.hydrate(failed);
    let mut disconnected = ui_state(Phase::Idle);
    disconnected.report_event_stream_failure(crate::client::connection_error());

    for (state, kind, title, detail) in [
        (ui_state(Phase::Idle), HomeStatusKind::Ready, "Ready", None),
        (
            ui_state(crate::test_support::RECORDING),
            HomeStatusKind::Recording,
            "Recording",
            None,
        ),
        (
            ui_state(crate::test_support::PROCESSING),
            HomeStatusKind::Working,
            "Transcribing recording",
            None,
        ),
        (
            loading,
            HomeStatusKind::Waiting,
            "Loading speech model",
            None,
        ),
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
            ui_state(Phase::Paused),
            HomeStatusKind::Paused,
            "Paused",
            Some("Resume WhisperNow from the system tray to record again."),
        ),
        (
            ui_state(Phase::Updating),
            HomeStatusKind::Waiting,
            "Installing an update",
            Some("WhisperNow will be ready after the update finishes."),
        ),
        (
            ui_state(Phase::ClearingData),
            HomeStatusKind::Waiting,
            "Clearing application data",
            None,
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
    let mut management = loaded_management(editable_settings());
    let substituted = management
        .shown_settings()
        .expect("loaded")
        .recording_shortcut
        .clone();
    let request = management
        .begin_settings_update(SettingKey::RecordingShortcut, |settings| {
            settings.recording_shortcut = Shortcut {
                modifiers: [ShortcutModifier::Alt].into(),
                key: ShortcutKey::R,
            };
        })
        .expect("loaded settings can be saved");
    let warning = AppError {
        category: ErrorCategory::General,
        code: "shortcut_registration_failed".to_owned(),
        summary: "The recording shortcut is taken by another app".to_owned(),
    };
    let mut returned = request;
    returned.recording_shortcut = substituted.clone();

    management.finish_settings_update(Ok(whispernow_domain::ipc::SaveSettingsResponse {
        settings: SettingsView {
            settings: returned,
            warnings: Vec::new(),
        },
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
fn a_loaded_cleanup_warning_clears_after_provider_repair() {
    let mut settings = editable_settings();
    settings.cleanup_enabled = true;
    settings.llm_provider = LlmProvider::Anthropic;
    settings
        .provider_settings_mut(LlmProvider::Anthropic)
        .model
        .clear();
    let mut management = ManagementState::default();
    management.install_settings(SettingsView::new(&settings));
    assert!(notice_message(&management).is_some_and(|message| message.contains("needs a model")));

    let request = management
        .begin_settings_update(SettingKey::ProviderModel, |settings| {
            settings.provider_settings_mut(LlmProvider::Anthropic).model = "claude-test".to_owned();
        })
        .expect("provider repair");
    management.finish_settings_update(Ok(whispernow_domain::ipc::SaveSettingsResponse {
        settings: SettingsView::new(&request),
        warnings: Vec::new(),
    }));
    assert!(management.notice.is_none());
}

#[test]
fn an_unrelated_save_keeps_an_unresolved_native_settings_warning() {
    let mut management = loaded_management(editable_settings());
    let request = management
        .begin_settings_update(SettingKey::StartOnLogin, start_on_login)
        .expect("startup edit");
    management.finish_settings_update(Ok(whispernow_domain::ipc::SaveSettingsResponse {
        settings: SettingsView::new(&request),
        warnings: vec!["The OS startup registration failed".to_owned()],
    }));
    let warning = management.notice.clone();

    let request = management
        .begin_settings_update(
            SettingKey::Theme,
            theme(whispernow_domain::ThemePreference::Dark),
        )
        .expect("theme edit");
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
    management.install_settings(SettingsView {
        settings,
        warnings: Vec::new(),
    });
    assert_eq!(management.theme_attribute(), Some("dark"));

    assert!(
        management
            .begin_settings_update(
                SettingKey::Theme,
                theme(whispernow_domain::ThemePreference::Light),
            )
            .is_some()
    );
    assert_eq!(management.theme_attribute(), Some("light"));

    management.finish_settings_update(Err("refused".to_owned()));
    assert_eq!(management.theme_attribute(), Some("dark"));

    assert!(
        management
            .begin_settings_update(
                SettingKey::Theme,
                theme(whispernow_domain::ThemePreference::System),
            )
            .is_some()
    );
    assert_eq!(management.theme_attribute(), None);
}
