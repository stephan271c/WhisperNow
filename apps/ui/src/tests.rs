use dioxus::prelude::*;
use whispernow_api::{
    ActiveDownloadDto, ApiErrorDto, AppStateDto, AudioDeviceDto, ClearDataCategory,
    ClearDataResult, ClearDataResultItemDto, DownloadPhase, EntryId, ErrorCategoryDto,
    EventPayload, HistoryCursor, HistoryEntry, HistoryFilter, HistoryPage,
    ListAudioDevicesResponse, ListModelsResponse, LlmProvider, ManualUpdatePackageDto,
    ModelAvailability, ModelDto, ModelStateDto, PhaseDto, RecoveryActionDto, Settings,
    SettingsView, SetupCheckpoint, SetupStateView, SetupStep, Shortcut, ShortcutKey,
    ShortcutModifier, TextOutputStatusDto, TranscriptResponse, UpdateErrorCodeDto, UpdateErrorDto,
    UpdatePhaseDto, UpdateProgressDto, UpdateStateDto, UtcMillis,
};

use crate::client::fake;
use crate::components::UiActions;
use crate::components::cleanup_recipient;
use crate::test_support::{
    editable_settings, show_history, snapshot, test_context, use_test_context,
};
use crate::{
    AppShell, EventStreamHealth, HomeStatus, HomeStatusKind, ManagementState, SaveStatus,
    SettingKey, UiState,
};

#[component]
fn TestSurface(initial_state: UiState) -> Element {
    let state = use_signal(move || initial_state);
    let management = use_signal(ManagementState::default);
    use_test_context(state, management);
    rsx! { AppShell {} }
}

fn state_event(state: AppStateDto) -> EventPayload {
    EventPayload::StateChanged { state }
}

fn history_event() -> EventPayload {
    EventPayload::HistoryChanged
}

fn ui_state(phase: PhaseDto) -> UiState {
    let mut state = UiState::default();
    state.hydrate(snapshot(phase));
    state.mark_event_stream_listening();
    state
}

fn update_state(phase: UpdatePhaseDto) -> UpdateStateDto {
    UpdateStateDto {
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

fn settings_html(update: UpdateStateDto) -> String {
    let mut state = ui_state(PhaseDto::Idle);
    state.hydrate_update(update);
    dioxus_ssr::render_element(rsx! {
        TestSurface { initial_state: state }
    })
}

/// Editable settings whose AI cleanup goes to `provider`, at `api_base` when given.
fn cleanup_to(provider: LlmProvider, api_base: Option<&str>) -> Settings {
    let mut settings = editable_settings();
    settings.llm_provider = provider;
    settings.provider_settings_mut(provider).api_base = api_base.map(|api_base| {
        whispernow_api::ProviderEndpoint::parse(api_base).expect("a safe endpoint")
    });
    settings
}

fn model_id(id: &str) -> whispernow_api::ModelId {
    whispernow_api::ModelId::new(id).expect("test model id")
}

fn history_entry(id: u64, text: &str) -> HistoryEntry {
    crate::test_support::history_entry(id, text)
}

/// Installs a read of `entries`, the first page of all history.
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

fn history_unavailable_error() -> ApiErrorDto {
    ApiErrorDto {
        category: ErrorCategoryDto::HistoryUnavailable,
        code: "history_unavailable".to_owned(),
        summary: "History storage is unavailable".to_owned(),
        recovery: None,
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
    let (pages, stale) = management.history_request();
    assert!(management.show_failed_history_only(true));
    let (_, latest) = management.history_request();

    management.finish_history_refresh(
        pages,
        &latest,
        history_window(vec![history_entry(11, "newest")]),
    );
    management.finish_history_refresh(
        pages,
        &stale,
        history_window(vec![history_entry(10, "stale")]),
    );
    management.fail_history_refresh(&stale, Some(&history_unavailable_error()));

    assert!(management.history_loaded());
    assert!(management.history_unavailable.is_none());
    assert_eq!(management.history.len(), 1);
    assert_eq!(management.history[0].id, EntryId(11));
}

#[test]
fn loading_older_history_grows_the_window_that_later_refreshes_reread() {
    let mut management = ManagementState::default();
    management.finish_history_refresh(
        1,
        &HistoryFilter::default(),
        HistoryPage {
            next_cursor: Some(HistoryCursor {
                created_at: UtcMillis(10),
                id: EntryId(10),
            }),
            ..history_window(vec![history_entry(10, "first page")])
        },
    );
    assert_eq!(management.history_window_pages(), 1);

    let (pages, filter) = management.older_history_request();
    assert_eq!(pages, 2);
    management.finish_history_refresh(
        pages,
        &filter,
        history_window(vec![
            history_entry(10, "first page"),
            history_entry(9, "second page"),
        ]),
    );
    assert_eq!(management.history.len(), 2);
    assert!(management.history_next_cursor.is_none());

    assert_eq!(management.history_request().0, 2);
}

#[test]
fn failed_refresh_keeps_loaded_rows_but_unavailable_storage_degrades() {
    let mut management = ManagementState::default();
    show_history(&mut management, vec![history_entry(10, "loaded")]);

    management.fail_history_refresh(&HistoryFilter::default(), None);
    assert_eq!(
        management.history.len(),
        1,
        "a transient failure keeps rows"
    );
    assert_eq!(
        notice_message(&management),
        Some("History could not be loaded")
    );
    assert!(management.history_unavailable.is_none());

    management.fail_history_refresh(
        &HistoryFilter::default(),
        Some(&history_unavailable_error()),
    );
    assert!(management.history.is_empty());
    assert!(!management.history_loaded());
    assert_eq!(
        management.history_unavailable.as_deref(),
        Some("History storage is unavailable")
    );
}

#[test]
fn history_unavailability_preserves_independently_loaded_management_sections() {
    let mut management = ManagementState::default();
    management.install_settings(SettingsView {
        settings: editable_settings(),
        warnings: Vec::new(),
    });
    management.install_setup(SetupStateView {
        first_run_complete: true,
        checkpoint: SetupCheckpoint {
            step: SetupStep::TryIt,
            model_id: model_id("model"),
            input_device: Some("microphone".to_owned()),
            hold_shortcut: Shortcut {
                modifiers: [ShortcutModifier::Control].into(),
                key: ShortcutKey::Space,
            },
            hold_shortcut_enabled: true,
            toggle_shortcut: None,
        },
    });
    management.install_devices(ListAudioDevicesResponse {
        devices: vec![AudioDeviceDto {
            id: "microphone".to_owned(),
            display_name: "Microphone".to_owned(),
            is_default: true,
        }],
    });
    management.install_models(ListModelsResponse {
        models: vec![ModelDto {
            id: whispernow_api::ModelId::new("model").expect("fixture model id"),
            display_name: "Model".to_owned(),
            family: whispernow_api::ModelFamily::Whisper,
            availability: ModelAvailability::Downloaded,
            languages: vec!["en".to_owned()],
            automatic_language_detection: true,
            size_bytes: 116_204_861,
            speed: whispernow_api::RelativeRating::High,
            accuracy: whispernow_api::RelativeRating::Low,
            recommended: false,
        }],
    });
    show_history(&mut management, vec![history_entry(10, "stale")]);

    management.fail_history_refresh(
        &HistoryFilter::default(),
        Some(&history_unavailable_error()),
    );

    assert_eq!(
        management
            .settings
            .as_ref()
            .map(|view| view.settings.model_id.as_str()),
        Some("model")
    );
    assert!(
        management
            .setup
            .as_ref()
            .is_some_and(|setup| setup.first_run_complete)
    );
    assert_eq!(management.devices[0].id, "microphone");
    assert_eq!(management.models[0].id.as_str(), "model");
    assert!(management.history.is_empty());

    show_history(&mut management, Vec::new());
    assert!(management.history_unavailable.is_none());
}

#[test]
fn state_events_do_not_refresh_history_but_history_events_do() {
    fake::reset();
    let mut dom = VirtualDom::new(|| rsx! {});
    dom.rebuild_in_place();
    let actions = dom.in_scope(ScopeId::ROOT, || {
        let mut management = ManagementState::default();
        management.install_snapshot(&snapshot(crate::test_support::PROCESSING));
        UiActions::new(test_context(UiState::default(), management))
    });
    let mut idle = snapshot(PhaseDto::Idle);
    idle.last_transcript = Some(TranscriptResponse {
        raw_text: "new recording".to_owned(),
        final_text: "new recording".to_owned(),
        text_output_status: None,
        cleanup_failed: false,
        history_save_failed: false,
    });
    dom.in_scope(ScopeId::ROOT, || actions.apply_event(state_event(idle)));
    dom.process_events();
    assert!(fake::commands().is_empty());

    dom.in_scope(ScopeId::ROOT, || actions.apply_event(history_event()));
    dom.process_events();
    assert_eq!(fake::commands(), ["get_history"]);
    assert_eq!(
        fake::requests::<whispernow_api::HistoryPageRequest>("get_history")[0].filter,
        HistoryFilter::default()
    );
}

#[test]
fn partial_clear_data_does_not_hide_history_when_database_deletion_failed() {
    let mut management = ManagementState::default();
    management.finish_history_refresh(
        1,
        &HistoryFilter::default(),
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
            ClearDataResultItemDto {
                category: ClearDataCategory::History,
                path: "history.sqlite3".to_owned(),
                deleted: false,
            },
            ClearDataResultItemDto {
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

fn start_minimized(settings: &mut Settings) {
    settings.start_minimized = true;
}

fn start_on_login(settings: &mut Settings) {
    settings.auto_start_on_login = true;
}

fn theme(theme: whispernow_api::ThemePreference) -> impl Fn(&mut Settings) {
    move |settings| settings.theme = theme
}

fn is_failed(status: &SaveStatus) -> bool {
    matches!(status, SaveStatus::Failed(_))
}

#[test]
fn a_change_shows_at_once_and_changes_made_during_a_save_go_out_together_after_it() {
    let mut management = loaded_management(editable_settings());
    assert!(management.change_setting(SettingKey::StartMinimized, start_minimized));
    let first = management.next_settings_update().expect("first change");
    assert!(first.settings.start_minimized);

    assert!(management.change_setting(
        SettingKey::Theme,
        theme(whispernow_api::ThemePreference::Dark)
    ));
    assert!(management.change_setting(SettingKey::StartOnLogin, start_on_login));
    let shown = management.shown_settings().expect("loaded");
    assert!(shown.start_minimized && shown.auto_start_on_login);
    assert_eq!(shown.theme, whispernow_api::ThemePreference::Dark);
    assert_eq!(
        management.setting_status(&SettingKey::Theme),
        SaveStatus::Saving
    );
    assert!(
        management.next_settings_update().is_none(),
        "the later changes wait for the first answer"
    );

    management.finish_settings_update(Ok(whispernow_api::SaveSettingsResponse {
        settings: SettingsView {
            settings: first.settings,
            warnings: Vec::new(),
        },
        warnings: Vec::new(),
    }));
    assert_eq!(
        management.setting_status(&SettingKey::StartMinimized),
        SaveStatus::Saved
    );
    assert!(
        management
            .shown_settings()
            .expect("loaded")
            .auto_start_on_login,
        "the waiting changes stay shown"
    );
    let next = management
        .next_settings_update()
        .expect("the waiting changes");
    assert!(next.settings.start_minimized && next.settings.auto_start_on_login);
    assert_eq!(next.settings.theme, whispernow_api::ThemePreference::Dark);
}

#[test]
fn a_refused_change_rolls_back_and_keeps_its_error_beside_the_control() {
    let mut management = loaded_management(editable_settings());
    assert!(management.change_setting(SettingKey::StartOnLogin, start_on_login));
    assert!(management.next_settings_update().is_some());
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
    assert!(!management.settings_saving());

    assert!(management.change_setting(SettingKey::StartOnLogin, start_on_login));
    assert_eq!(
        management.setting_status(&SettingKey::StartOnLogin),
        SaveStatus::Saving,
        "a new attempt clears the old error"
    );
}

type Edit = Box<dyn Fn(&mut Settings)>;

fn add_prompt(title: &str, prompt: &str) -> Edit {
    let (title, prompt) = (title.to_owned(), prompt.to_owned());
    Box::new(move |settings| {
        settings.prompts.push(whispernow_api::CleanupPrompt {
            id: settings.next_prompt_id(),
            title: title.clone(),
            prompt: prompt.clone(),
            default_hash: None,
        });
    })
}

#[test]
fn new_prompts_get_distinct_generated_identifiers() {
    let mut management = loaded_management(editable_settings());
    for title in ["Email", "Email"] {
        assert!(management.change_setting(SettingKey::Prompts, add_prompt(title, "Tidy the text")));
    }
    let prompts = management.shown_settings().expect("loaded").prompts.clone();
    assert_eq!(prompts.len(), 2);
    assert!(prompts.iter().all(|prompt| !prompt.id.is_empty()));
    assert_ne!(prompts[0].id, prompts[1].id);
}

#[test]
fn a_prompt_test_needs_sample_text() {
    let mut management = loaded_management(editable_settings());
    assert!(management.begin_prompt_test("clean", "   ").is_none());
    assert!(
        management
            .prompt_test
            .as_ref()
            .is_some_and(|test| matches!(test.outcome, Some(Err(_))))
    );
    let request = management
        .begin_prompt_test("clean", "hello there")
        .expect("sample text is enough");
    assert_eq!(
        management.prompt_test.as_ref().map(|test| &test.outcome),
        Some(&None)
    );
    management.finish_prompt_test(
        &request,
        Ok(whispernow_api::TestPromptResponse {
            text: "Hello there.".to_owned(),
        }),
    );
    assert_eq!(
        management.prompt_test.and_then(|test| test.outcome),
        Some(Ok("Hello there.".to_owned()))
    );
}

#[test]
fn cleanup_disclosure_names_the_configured_provider() {
    let settings = cleanup_to(LlmProvider::Anthropic, None);
    assert_eq!(
        cleanup_recipient(&settings),
        "the configured Anthropic provider"
    );
}

#[test]
fn named_provider_disclosure_identifies_a_custom_recipient_without_claiming_direct_delivery() {
    let settings = cleanup_to(
        LlmProvider::Anthropic,
        Some("https://gateway.example:8443/private-looking-path/v1"),
    );
    let recipient = cleanup_recipient(&settings);
    assert_eq!(
        recipient,
        "the configured API endpoint at https://gateway.example:8443"
    );
    assert!(!recipient.contains("configured Anthropic provider"));
    assert!(!recipient.contains("private-looking-path"));
}

#[test]
fn custom_provider_disclosure_identifies_its_safe_endpoint_without_path_details() {
    let settings = cleanup_to(
        LlmProvider::Custom,
        Some("https://compatible.provider.example:8443/private-looking-path/v1"),
    );
    let recipient = cleanup_recipient(&settings);
    assert_eq!(
        recipient,
        "the custom OpenAI-compatible provider at https://compatible.provider.example:8443"
    );
    assert!(!recipient.contains("private-looking-path"));
}

#[test]
fn one_snapshot_reconstructs_phase_and_last_transcript_after_reload() {
    let mut snapshot = snapshot(PhaseDto::Idle);
    snapshot.last_transcript = Some(TranscriptResponse {
        raw_text: "raw dictation".to_owned(),
        final_text: "Finished dictation".to_owned(),
        text_output_status: Some(TextOutputStatusDto::PastedAndRestored),
        cleanup_failed: false,
        history_save_failed: false,
    });

    let mut state = UiState::default();
    state.hydrate(snapshot);

    assert_eq!(HomeStatus::from_state(&state).kind, HomeStatusKind::Ready);
    assert_eq!(
        state
            .snapshot
            .as_ref()
            .and_then(|snapshot| snapshot.last_transcript.as_ref())
            .map(|transcript| transcript.final_text.as_str()),
        Some("Finished dictation")
    );
}

#[test]
fn state_changed_replaces_the_entire_workflow_projection_without_reconciliation() {
    let mut state = ui_state(crate::test_support::RECORDING);
    state.apply_event(state_event(snapshot(crate::test_support::PROCESSING)));
    assert_eq!(
        state.snapshot.as_ref().map(|snapshot| &snapshot.phase),
        Some(&crate::test_support::PROCESSING)
    );

    state.apply_event(state_event(snapshot(PhaseDto::Idle)));
    assert_eq!(
        state.snapshot.as_ref().map(|snapshot| &snapshot.phase),
        Some(&PhaseDto::Idle)
    );
}

#[test]
fn audio_level_is_scoped_by_the_single_recording_phase_and_clamped() {
    let mut state = ui_state(PhaseDto::Idle);
    state.apply_event(EventPayload::AudioLevel {
        rms: 0.4,
        peak: 0.8,
        elapsed_ms: 0,
    });
    assert!(state.audio_rms.abs() < f32::EPSILON);

    state.apply_event(state_event(snapshot(crate::test_support::RECORDING)));
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
    state.apply_event(state_event(snapshot(crate::test_support::PROCESSING)));
    assert!(state.audio_rms.abs() < f32::EPSILON);
}

#[test]
fn update_snapshot_hydration_and_events_replace_the_complete_projection() {
    let mut state = ui_state(PhaseDto::Idle);
    let mut downloading = update_state(UpdatePhaseDto::Downloading);
    downloading.available_version = Some("0.3.0".to_owned());
    downloading.progress = Some(UpdateProgressDto {
        downloaded_bytes: 512,
        total_bytes: Some(1_024),
    });
    state.hydrate_update(downloading);

    let mut ready = update_state(UpdatePhaseDto::ReadyToRestart);
    ready.available_version = Some("0.3.0".to_owned());
    ready.release_notes = Some("Verified release".to_owned());
    state.apply_event(EventPayload::UpdateChanged {
        state: ready.clone(),
    });

    assert_eq!(state.update_snapshot, Some(ready));
}

#[test]
fn settings_show_update_download_progress() {
    let mut update = update_state(UpdatePhaseDto::Downloading);
    update.available_version = Some("0.3.0".to_owned());
    update.progress = Some(UpdateProgressDto {
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
    let mut update = update_state(UpdatePhaseDto::ReadyToRestart);
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
    let mut update = update_state(UpdatePhaseDto::ReadyToRestart);
    update.available_version = Some("v0.3.0".to_owned());
    update.notification_dismissed = true;

    let html = settings_html(update);
    assert!(
        !html.contains("Dismiss"),
        "a dismissed offer stays dismissed"
    );
    assert!(html.contains("Update v0.3.0 is ready"));
    assert!(html.contains("Restart to update"));
    assert!(!html.contains("The verified update is ready"));
}

#[test]
fn manual_install_offer_names_the_package_and_links_the_release_page_without_restart() {
    for (package, format) in [
        (ManualUpdatePackageDto::Deb, ".deb"),
        (ManualUpdatePackageDto::Rpm, ".rpm"),
    ] {
        let mut update = update_state(UpdatePhaseDto::ManualInstallAvailable);
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
    let up_to_date = settings_html(update_state(UpdatePhaseDto::UpToDate));
    assert!(up_to_date.contains("WhisperNow is up to date"));
    assert!(up_to_date.contains("No newer stable release is available"));

    let mut offline = update_state(UpdatePhaseDto::Error);
    offline.error = Some(UpdateErrorDto {
        code: UpdateErrorCodeDto::Offline,
        recoverable: true,
    });
    let offline = settings_html(offline);
    assert!(offline.contains("Update needs attention"));
    assert!(offline.contains("You appear to be offline"));
    assert!(offline.contains("Try again"));
}

#[test]
fn three_app_phases_drive_the_home_status() {
    let kind = |phase| HomeStatus::from_state(&ui_state(phase)).kind;
    assert_eq!(kind(PhaseDto::Idle), HomeStatusKind::Ready);
    assert_eq!(
        kind(crate::test_support::RECORDING),
        HomeStatusKind::Recording
    );
    assert_eq!(
        kind(crate::test_support::PROCESSING),
        HomeStatusKind::Working
    );
}

#[test]
fn ordinary_state_error_drives_the_home_problem() {
    let mut snapshot = snapshot(PhaseDto::Idle);
    snapshot.error = Some(ApiErrorDto {
        category: ErrorCategoryDto::Audio,
        code: "microphone_unavailable".to_owned(),
        summary: "Choose an available microphone".to_owned(),
        recovery: Some(RecoveryActionDto::ChooseDevice),
    });
    let mut state = UiState::default();
    state.hydrate(snapshot);
    let status = HomeStatus::from_state(&state);
    assert_eq!(status.kind, HomeStatusKind::Problem);
    assert_eq!(
        status.detail.as_deref(),
        Some("Choose an available microphone")
    );
}

#[test]
fn idle_no_audio_and_no_speech_outcomes_preserve_exact_retry_messages() {
    const OUTCOMES: [(&str, ErrorCategoryDto, &str); 2] = [
        (
            "no_microphone_audio",
            ErrorCategoryDto::Audio,
            "No microphone audio was detected; check the selected input and try again.",
        ),
        (
            "no_speech",
            ErrorCategoryDto::Recognition,
            "No speech was detected; check the microphone and try again.",
        ),
    ];

    for (code, category, summary) in OUTCOMES {
        let mut snapshot = snapshot(PhaseDto::Idle);
        snapshot.error = Some(ApiErrorDto {
            category,
            code: code.to_owned(),
            summary: summary.to_owned(),
            recovery: Some(RecoveryActionDto::Retry),
        });
        let mut state = UiState::default();
        state.hydrate(snapshot);

        assert_eq!(
            state.snapshot.as_ref().map(|snapshot| &snapshot.phase),
            Some(&PhaseDto::Idle),
            "{code} must not leave the batch workflow in Processing"
        );
        let status = HomeStatus::from_state(&state);
        assert_eq!(status.kind, HomeStatusKind::Problem);
        assert_eq!(status.detail.as_deref(), Some(summary));

        // The recording bar shows the desktop's overlay outcome instead; see
        // `components::overlay::tests`.
        let rendered = dioxus_ssr::render_element(rsx! {
            TestSurface { initial_state: state.clone() }
        });
        assert!(
            rendered.contains(summary),
            "settings UI must render the exact {code} retry message"
        );
    }
}

#[test]
fn model_loading_still_takes_priority_over_idle() {
    let mut snapshot = snapshot(PhaseDto::Idle);
    snapshot.model_state = ModelStateDto::Loading;
    let mut state = UiState::default();
    state.hydrate(snapshot);
    assert!(state.is_loading_model());
    assert_eq!(HomeStatus::from_state(&state).title, "Loading speech model");
}

#[test]
fn state_snapshot_is_sufficient_for_download_management() {
    let model_id = whispernow_api::ModelId::new("parakeet-tdt-0.6b-v3").expect("fixture model id");
    let mut snapshot = snapshot(PhaseDto::Idle);
    snapshot.active_downloads.push(ActiveDownloadDto {
        model_id: model_id.clone(),
        phase: DownloadPhase::Downloading,
        downloaded_bytes: 512,
        total_bytes: Some(1_024),
        bytes_per_second: None,
    });

    let mut management = ManagementState::default();
    management.install_snapshot(&snapshot);
    let download = management.download(&model_id).expect("active download");
    assert_eq!(download.downloaded_bytes, 512);
    assert_eq!(download.total_bytes, Some(1_024));

    snapshot.active_downloads[0].phase = DownloadPhase::ValidatingArchive;
    snapshot.active_downloads[0].downloaded_bytes = 1_024;
    assert!(!management.apply_event(&state_event(snapshot.clone())));
    let download = management.download(&model_id).expect("updated download");
    assert_eq!(download.phase, DownloadPhase::ValidatingArchive);
    assert_eq!(download.downloaded_bytes, 1_024);
}

#[test]
fn event_stream_failure_remains_visible_without_an_extra_backend_event() {
    let mut state = ui_state(PhaseDto::Idle);
    state.report_event_stream_failure(None);
    assert_eq!(state.event_stream_health, EventStreamHealth::Failed);
    assert_eq!(HomeStatus::from_state(&state).kind, HomeStatusKind::Problem);
}

#[test]
fn dashboard_status_never_claims_ready_when_workflow_is_unavailable() {
    for (phase, kind) in [
        (PhaseDto::Paused, HomeStatusKind::Paused),
        (PhaseDto::Updating, HomeStatusKind::Waiting),
    ] {
        let snapshot = snapshot(phase);
        let mut state = UiState::default();
        state.hydrate(snapshot);
        let status = HomeStatus::from_state(&state);
        assert_eq!(status.kind, kind);
        assert!(status.detail.is_some());
    }
}

#[test]
fn a_saved_settings_warning_is_shown_beside_the_installed_settings() {
    let mut management = loaded_management(editable_settings());
    assert!(
        management.change_setting(SettingKey::HoldShortcut, |settings| {
            settings.hold_shortcut = Shortcut {
                modifiers: [ShortcutModifier::Alt].into(),
                key: ShortcutKey::R,
            };
        })
    );
    let request = management
        .next_settings_update()
        .expect("loaded settings can be saved");
    let warning = ApiErrorDto {
        category: ErrorCategoryDto::General,
        code: "shortcut_registration_failed".to_owned(),
        summary: "The recording shortcut is taken by another app".to_owned(),
        recovery: None,
    };

    management.finish_settings_update(Ok(whispernow_api::SaveSettingsResponse {
        settings: SettingsView {
            settings: request.settings,
            warnings: Vec::new(),
        },
        warnings: vec![warning.clone()],
    }));

    assert_eq!(
        management
            .settings
            .as_ref()
            .map(|settings| settings.settings.hold_shortcut.key),
        Some(ShortcutKey::R),
        "the save committed even though a side effect failed"
    );
    assert!(notice_message(&management).is_some_and(|notice| notice.contains(&warning.summary)));
}

#[test]
fn theme_attribute_follows_a_theme_change_at_once_and_its_rollback() {
    let mut management = ManagementState::default();
    assert_eq!(management.theme_attribute(), None);

    let mut settings = editable_settings();
    settings.theme = whispernow_api::ThemePreference::Dark;
    management.install_settings(SettingsView {
        settings,
        warnings: Vec::new(),
    });
    assert_eq!(management.theme_attribute(), Some("dark"));

    assert!(management.change_setting(
        SettingKey::Theme,
        theme(whispernow_api::ThemePreference::Light)
    ));
    assert_eq!(management.theme_attribute(), Some("light"));

    assert!(management.next_settings_update().is_some());
    management.finish_settings_update(Err("refused".to_owned()));
    assert_eq!(management.theme_attribute(), Some("dark"));

    assert!(management.change_setting(
        SettingKey::Theme,
        theme(whispernow_api::ThemePreference::System)
    ));
    assert_eq!(management.theme_attribute(), None);
}
