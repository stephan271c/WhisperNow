use whispernow_domain::ipc::Phase;

use super::{BlockedReason, UiOperation, blocked};
use crate::ManagementState;
use crate::test_support::snapshot;

fn loaded_history() -> ManagementState {
    let mut management = ManagementState::default();
    management.history_rows_filter = Some(whispernow_domain::HistoryFilter::default());
    management
}

#[test]
fn workflow_changes_wait_for_an_idle_desktop() {
    let management = loaded_history();
    for (phase, reason) in [
        (crate::test_support::PROCESSING, BlockedReason::Dictating),
        (Phase::Paused, BlockedReason::Paused),
        (Phase::Updating, BlockedReason::UpdateInstalling),
    ] {
        let state = snapshot(phase);
        for operation in [
            UiOperation::StartSetupTest,
            UiOperation::TestMicrophone,
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
        blocked(UiOperation::ModelSwitch, None, &loaded_history()),
        Some(BlockedReason::Connecting)
    );
}

#[test]
fn own_calls_in_flight_disable_controls_until_they_are_answered() {
    let mut management = loaded_history();
    let idle = snapshot(Phase::Idle);
    management.begin_busy(UiOperation::CheckForUpdates);
    for operation in [
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
        blocked(UiOperation::ModelSwitch, Some(&idle), &management),
        None
    );
}
