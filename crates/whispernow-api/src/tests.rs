use serde_json::{json, to_value};

use crate::{
    ApiErrorDto, ApiKey, AppStateDto, DeleteHistoryEntryRequest, ErrorCategoryDto, EventPayload,
    GetHistoryAudioRequest, LlmProvider, ModelStateDto, OverlayStyle, PhaseDto, RecordingMode,
    RecoveryActionDto, SetApiKeyRequest, SettingsView, TextOutputStatusDto, TranscriptResponse,
    UpdatePhaseDto, UpdateProgressDto, UpdateStateDto,
};
use whispernow_domain::{
    HistoryFilter, HistoryLimit, HistoryPageRequest, ModelId, OverlayPosition, Settings,
};

#[test]
fn history_requests_carry_no_storage_authority() {
    assert_eq!(
        serde_json::from_value::<HistoryPageRequest>(json!({"before": null, "limit": 5}))
            .expect("filter defaults")
            .filter,
        HistoryFilter::default()
    );
    assert!(
        serde_json::from_value::<GetHistoryAudioRequest>(json!({"id": 4, "path": "/tmp/x.wav"}))
            .is_err(),
        "a request carries no storage authority"
    );
    assert!(
        serde_json::from_value::<DeleteHistoryEntryRequest>(json!({"id": 8, "unexpected": true}))
            .is_err()
    );
}

#[test]
fn settings_view_names_saved_keys() {
    let persisted = Settings {
        history_limit: HistoryLimit::Last100,
        api_keys: [LlmProvider::OpenAi].into(),
        ..Settings::default()
    };
    let encoded = to_value(SettingsView::new(&persisted)).expect("serialize settings view");
    assert_eq!(encoded["settings"]["api_keys"], json!(["openai"]));
    assert_eq!(encoded["settings"]["history_limit"], "100");
}

#[test]
fn credential_mutation_is_write_only_and_debug_redacted() {
    let request = SetApiKeyRequest {
        provider: LlmProvider::OpenRouter,
        key: ApiKey::new("write-only-secret").expect("a key"),
    };
    let encoded = serde_json::to_value(&request).expect("serialize credential mutation");
    assert_eq!(encoded["key"], "write-only-secret");
    assert!(!format!("{request:?}").contains("write-only-secret"));
}

#[test]
fn partial_transcript_event_round_trips_and_is_not_broadcast() {
    let event = EventPayload::PartialTranscript {
        session: 4,
        committed_text: "so far so good".to_owned(),
    };
    assert_eq!(event.event_name(), crate::PARTIAL_TRANSCRIPT_EVENT);
    let value = to_value(&event).expect("serialize partial transcript event");
    assert_eq!(
        value,
        json!({
            "type": "partial_transcript",
            "payload": {"session": 4, "committedText": "so far so good"}
        })
    );
    assert_eq!(
        serde_json::from_value::<EventPayload>(value).expect("deserialize partial transcript"),
        event
    );
    assert!(!crate::EVENT_NAMES.contains(&crate::PARTIAL_TRANSCRIPT_EVENT));
    assert!(crate::RECORDING_BAR_EVENT_NAMES.contains(&crate::PARTIAL_TRANSCRIPT_EVENT));
    assert!(
        crate::EVENT_NAMES
            .iter()
            .all(|name| crate::RECORDING_BAR_EVENT_NAMES.contains(name))
    );
}

#[test]
fn phase_carries_the_recording_mode_and_the_failure() {
    let recording = PhaseDto::Recording {
        mode: RecordingMode::Hold,
        session: 3,
    };
    assert_eq!(
        to_value(&recording).expect("serialize recording phase"),
        json!({"phase": "recording", "mode": "hold", "session": 3})
    );
    let failed = PhaseDto::Error {
        error: ApiErrorDto {
            category: ErrorCategoryDto::Output,
            code: "paste_failed".to_owned(),
            summary: "Text was not pasted".to_owned(),
            recovery: Some(RecoveryActionDto::CopyManually),
        },
    };
    let value = to_value(&failed).expect("serialize failed phase");
    assert_eq!(value["phase"], "error");
    assert_eq!(value["error"]["recovery"], "copy_manually");
    assert_eq!(
        serde_json::from_value::<PhaseDto>(value).expect("deserialize failed phase"),
        failed
    );
    assert!(recording.is_busy());
    assert!(!failed.is_busy());
}

#[test]
fn history_changed_event_is_a_content_free_signal() {
    let event = EventPayload::HistoryChanged;
    assert_eq!(event.event_name(), crate::HISTORY_CHANGED_EVENT);
    let value = to_value(&event).expect("serialize history change event");
    assert_eq!(value, json!({"type": "history_changed"}));
    assert_eq!(
        serde_json::from_value::<EventPayload>(value).expect("round-trip history change event"),
        event
    );
}

#[test]
fn state_changed_event_carries_the_whole_snapshot() {
    let event = EventPayload::StateChanged {
        state: AppStateDto {
            phase: PhaseDto::Processing { cleaning_up: true },
            selected_model_id: ModelId::new("model").expect("fixture model id"),
            model_state: ModelStateDto::Ready,
            last_transcript: Some(TranscriptResponse {
                raw_text: "raw".to_owned(),
                final_text: "final".to_owned(),
                text_output_status: Some(TextOutputStatusDto::PastedAndRestored),
                cleanup_failed: false,
                history_save_failed: false,
            }),
            error: None,
            active_downloads: Vec::new(),
            setup_test: Some(crate::SetupTestResultDto::Heard {
                text: "hello".to_owned(),
            }),
            overlay_style: OverlayStyle::WithLiveText,
            overlay_position: OverlayPosition::Bottom,
        },
    };
    let value = to_value(event).expect("serialize event");
    assert_eq!(value["type"], "state_changed");
    assert_eq!(
        value["payload"]["state"]["phase"],
        json!({ "phase": "processing", "cleaningUp": true })
    );
    assert_eq!(value["payload"]["state"]["overlayStyle"], "with_live_text");
    assert_eq!(
        value["payload"]["state"]["setupTest"],
        json!({ "outcome": "heard", "text": "hello" })
    );
    assert_eq!(
        value["payload"]["state"]["lastTranscript"]["finalText"],
        "final"
    );
}

#[test]
fn update_changed_event_round_trips() {
    let state = UpdateStateDto {
        phase: UpdatePhaseDto::Downloading,
        available_version: Some("2.4.0".to_owned()),
        release_notes: Some("Fixes and improvements".to_owned()),
        progress: Some(UpdateProgressDto {
            downloaded_bytes: 512,
            total_bytes: Some(1_024),
        }),
        error: None,
        notification_dismissed: false,
        restart_deferred: false,
        manual_package: None,
    };
    let event = EventPayload::UpdateChanged { state };
    assert_eq!(event.event_name(), crate::UPDATE_CHANGED_EVENT);

    let value = to_value(&event).expect("serialize update event");
    assert_eq!(value["type"], "update_changed");
    assert_eq!(value["payload"]["state"]["phase"], "downloading");
    assert_eq!(
        value["payload"]["state"]["progress"],
        json!({"downloadedBytes": 512, "totalBytes": 1024})
    );
    assert_eq!(value["payload"]["state"]["restartDeferred"], false);
    assert_eq!(
        serde_json::from_value::<EventPayload>(value).expect("round-trip update event"),
        event
    );
}
