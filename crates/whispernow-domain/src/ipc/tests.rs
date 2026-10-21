use serde_json::to_value;

use super::{
    AppState, EventPayload, ModelState, Phase, RecordingTrigger, UpdatePhase, UpdateProgress,
    UpdateState,
};
use crate::{AppError, ErrorCategory, ModelId};

#[test]
fn phase_carries_the_recording_trigger() {
    let recording = Phase::Recording {
        trigger: RecordingTrigger::Hold,
        session: 3,
    };
    let failed = Phase::Error;
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
            error: Some(AppError {
                category: ErrorCategory::Output,
                code: "paste_failed".to_owned(),
                summary: "Text was not pasted".to_owned(),
            }),
            active_downloads: Vec::new(),
            setup_test: Some(super::SetupTestResult::Heard {
                text: "hello".to_owned(),
            }),
            cleanup_device: Some(super::CleanupDevice::Gpu {
                name: "NVIDIA GeForce RTX 5070".to_owned(),
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

    let value = to_value(&event).expect("serialize update event");
    assert_eq!(
        serde_json::from_value::<EventPayload>(value).expect("round-trip update event"),
        event
    );
}

#[test]
fn every_event_carries_its_variant_in_the_type_tag() {
    let state = AppState {
        phase: Phase::Idle,
        selected_model_id: ModelId::new("model").expect("fixture model id"),
        model_state: ModelState::Ready,
        last_transcript: None,
        error: None,
        active_downloads: Vec::new(),
        setup_test: None,
        cleanup_device: None,
    };
    let update = UpdateState {
        phase: UpdatePhase::Idle,
        notification_dismissed: false,
        restart_deferred: false,
    };
    let events = [
        (EventPayload::StateChanged { state }, "state_changed"),
        (
            EventPayload::AudioLevel {
                rms: 0.25,
                peak: 0.5,
                elapsed_ms: 300,
            },
            "audio_level",
        ),
        (
            EventPayload::UpdateChanged { state: update },
            "update_changed",
        ),
        (EventPayload::HistoryChanged, "history_changed"),
        (
            EventPayload::NavigationRequested {
                section: super::MainWindowSection::History,
            },
            "navigation_requested",
        ),
    ];
    for (event, tag) in events {
        let value = to_value(&event).expect("serialize event");
        assert_eq!(value["type"], tag);
        assert_eq!(
            serde_json::from_value::<EventPayload>(value).expect("round-trip event"),
            event
        );
    }
}
