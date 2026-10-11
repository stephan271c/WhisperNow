use crate::AppError;
use serde_json::to_value;

use super::{
    AppState, ErrorCategory, EventPayload, ModelState, Phase, RecordingTrigger, UpdatePhase,
    UpdateProgress, UpdateState,
};
use crate::ModelId;

#[test]
fn phase_carries_the_recording_trigger_and_the_failure() {
    let recording = Phase::Recording {
        trigger: RecordingTrigger::Hold,
        session: 3,
    };
    let failed = Phase::Error {
        error: AppError {
            category: ErrorCategory::Output,
            code: "paste_failed".to_owned(),
            summary: "Text was not pasted".to_owned(),
        },
    };
    assert!(recording.is_busy());
    assert!(!failed.is_busy());
    for phase in [recording, failed] {
        let value = to_value(&phase).expect("serialize phase");
        assert_eq!(
            serde_json::from_value::<Phase>(value).expect("round-trip phase"),
            phase
        );
    }
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
    let value = to_value(&event).expect("serialize event");
    assert_eq!(
        serde_json::from_value::<EventPayload>(value).expect("round-trip state event"),
        event
    );
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
    assert_eq!(
        serde_json::from_value::<EventPayload>(value).expect("round-trip update event"),
        event
    );
}
