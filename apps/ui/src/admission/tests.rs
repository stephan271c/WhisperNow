use whispernow_api::{AppStateDto, ModelId, ModelStateDto, PhaseDto};

use super::{BlockedReason, UiOperation, blocked};
use crate::ManagementState;

fn snapshot(phase: PhaseDto) -> AppStateDto {
    AppStateDto {
        phase,
        selected_model_id: ModelId::new("model").expect("fixture model id"),
        model_state: ModelStateDto::Ready,
        last_transcript: None,
        error: None,
        active_downloads: Vec::new(),
        overlay_style: whispernow_api::OverlayStyle::default(),
        overlay_position: whispernow_api::OverlayPosition::default(),
        setup_test: None,
    }
}

fn loaded_history() -> ManagementState {
    let mut management = ManagementState::default();
    management.history_rows_filter = Some(whispernow_api::HistoryFilter::default());
    management
}

#[test]
fn workflow_changes_wait_for_an_idle_desktop() {
    let management = loaded_history();
    for (phase, reason) in [
        (crate::test_support::PROCESSING, BlockedReason::Dictating),
        (PhaseDto::Paused, BlockedReason::Paused),
        (PhaseDto::Updating, BlockedReason::UpdateInstalling),
    ] {
        let state = snapshot(phase);
        for operation in [
            UiOperation::StartSetupTest,
            UiOperation::TestMicrophone,
            UiOperation::SaveSettings,
            UiOperation::ModelSwitch,
        ] {
            assert_eq!(
                blocked(operation, Some(&state), &management),
                Some(reason),
                "{operation:?}",
            );
        }
        for operation in [
            UiOperation::DownloadModel,
            UiOperation::ListDataLocations,
            UiOperation::ChangeHistory,
        ] {
            assert_eq!(
                blocked(operation, Some(&state), &management),
                None,
                "the desktop answers {operation:?} itself",
            );
        }
    }
}

#[test]
fn nothing_is_sent_before_the_first_desktop_snapshot() {
    assert_eq!(
        blocked(UiOperation::SaveSettings, None, &loaded_history()),
        Some(BlockedReason::Connecting)
    );
}

#[test]
fn own_calls_in_flight_disable_controls_until_they_are_answered() {
    let mut management = loaded_history();
    let idle = snapshot(PhaseDto::Idle);
    management.begin_busy(UiOperation::CheckForUpdates);
    for operation in [
        UiOperation::SaveSettings,
        UiOperation::ModelSwitch,
        UiOperation::ChangeHistory,
        UiOperation::DismissUpdate,
        UiOperation::UpdateInstallRestart,
    ] {
        assert_eq!(
            blocked(operation, Some(&idle), &management),
            Some(BlockedReason::CommandPending),
            "{operation:?}",
        );
    }
    assert_eq!(
        blocked(UiOperation::ListDataLocations, Some(&idle), &management),
        None,
        "listing data locations does not wait for other changes",
    );

    management.end_busy(UiOperation::CheckForUpdates);
    assert_eq!(
        blocked(UiOperation::SaveSettings, Some(&idle), &management),
        None
    );
}
