use crate::AppError;
use serde_json::{json, to_value};

use super::{
    AppState, ErrorCategory, EventPayload, LlmProvider, ModelState, Phase, RecordingTrigger,
    SettingsView, UpdatePhase, UpdateProgress, UpdateState,
};
use crate::{HistoryLimit, ModelId, Settings};

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
fn phase_carries_the_recording_trigger_and_the_failure() {
    let recording = Phase::Recording {
        trigger: RecordingTrigger::Hold,
        session: 3,
    };
    assert_eq!(
        to_value(&recording).expect("serialize recording phase"),
        json!({"phase": "recording", "trigger": "hold", "session": 3})
    );
    let failed = Phase::Error {
        error: AppError {
            category: ErrorCategory::Output,
            code: "paste_failed".to_owned(),
            summary: "Text was not pasted".to_owned(),
        },
    };
    let value = to_value(&failed).expect("serialize failed phase");
    assert_eq!(value["phase"], "error");
    assert_eq!(
        serde_json::from_value::<Phase>(value).expect("deserialize failed phase"),
        failed
    );
    assert!(recording.is_busy());
    assert!(!failed.is_busy());
}

#[test]
fn state_changed_event_carries_the_whole_snapshot() {
    let event = EventPayload::StateChanged {
        state: AppState {
            phase: Phase::Processing { cleaning_up: true },
            selected_model_id: ModelId::new("model").expect("fixture model id"),
            model_state: ModelState::Ready,
            last_transcript: Some("final".to_owned()),
            error: None,
            active_downloads: Vec::new(),
            setup_test: Some(super::SetupTestResult::Heard {
                text: "hello".to_owned(),
            }),
        },
    };
    let value = to_value(event).expect("serialize event");
    assert_eq!(value["type"], "state_changed");
    assert_eq!(
        value["payload"]["state"]["phase"],
        json!({ "phase": "processing", "cleaningUp": true })
    );
    assert_eq!(
        value["payload"]["state"]["setupTest"],
        json!({ "outcome": "heard", "text": "hello" })
    );
    assert_eq!(value["payload"]["state"]["lastTranscript"], "final");
}

#[test]
fn update_changed_event_round_trips() {
    let state = UpdateState {
        phase: UpdatePhase::Downloading {
            version: "2.4.0".to_owned(),
            release_notes: Some("Fixes and improvements".to_owned()),
            progress: UpdateProgress {
                downloaded_bytes: 512,
                total_bytes: Some(1_024),
            },
        },
        notification_dismissed: false,
        restart_deferred: false,
    };
    let event = EventPayload::UpdateChanged { state };
    assert_eq!(event.event_name(), super::UPDATE_CHANGED_EVENT);

    let value = to_value(&event).expect("serialize update event");
    assert_eq!(value["type"], "update_changed");
    assert_eq!(value["payload"]["state"]["phase"]["phase"], "downloading");
    assert_eq!(
        value["payload"]["state"]["phase"]["progress"],
        json!({"downloadedBytes": 512, "totalBytes": 1024})
    );
    assert_eq!(value["payload"]["state"]["restartDeferred"], false);
    assert_eq!(
        serde_json::from_value::<EventPayload>(value).expect("round-trip update event"),
        event
    );
}
