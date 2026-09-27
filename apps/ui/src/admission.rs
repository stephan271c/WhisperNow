//! Which controls this window keeps disabled.
//!
//! A control is disabled while its own call, or a change it would act on, is in flight, and a
//! control that changes the workflow is disabled while the desktop dictates, is paused or
//! installs an update. The desktop refuses anything else it cannot do, and says why.

use whispernow_api::{AppStateDto, PhaseDto};

use crate::{ManagementState, NoticeScope};

/// Why a control is disabled.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlockedReason {
    /// No state snapshot has arrived from the desktop yet.
    Connecting,
    /// A dictation is recording or being processed.
    Dictating,
    Paused,
    UpdateInstalling,
    /// A change this window sent is still in flight.
    CommandPending,
    /// This window is still loading history.
    HistoryLoading,
}

impl BlockedReason {
    #[must_use]
    pub const fn message(self) -> &'static str {
        match self {
            Self::Connecting => "WhisperNow is still connecting",
            Self::Dictating => "Wait for the current dictation to finish",
            Self::Paused => "Resume WhisperNow first",
            Self::UpdateInstalling => "Wait for the current update to finish installing",
            Self::CommandPending => "Wait for the current change to finish",
            Self::HistoryLoading => "History is still loading",
        }
    }
}

/// A control the UI can disable; while its call is in flight it is busy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UiOperation {
    /// Any settings control; changes queue behind one another rather than refusing.
    SaveSettings,
    /// Runs a saved prompt on sample text.
    TestPrompt,
    CredentialMutation,
    DownloadModel,
    CancelDownload,
    SaveSetup,
    CompleteSetup,
    /// Starts setup again from Settings once it has finished.
    RestartSetup,
    /// Asks the system for a permission setup needs.
    RequestPermission,
    OpenPermissionSettings,
    /// Reads the permission cards.
    LoadPermissions,
    /// Lists the microphones.
    ListDevices,
    /// Setup's test dictation.
    StartSetupTest,
    StopSetupTest,
    /// Deleting one history entry or clearing them all.
    ChangeHistory,
    OpenRecordingsFolder,
    ModelSwitch,
    ModelDelete,
    UpdateInstallRestart,
    /// Lists the application data a clear would remove.
    ListDataLocations,
    ClearData,
    CheckForUpdates,
    DismissUpdate,
    OpenUpdateReleasePage,
    StopRecording,
    CancelRecording,
    /// Copy, open a main-window section, or dismiss an outcome from the overlay.
    OverlayAction,
    TestMicrophone,
}

impl UiOperation {
    /// Whether this call changes what later controls act on, so others wait for it.
    const fn blocks_workflow_changes(self) -> bool {
        matches!(
            self,
            Self::CredentialMutation
                | Self::DownloadModel
                | Self::CancelDownload
                | Self::SaveSetup
                | Self::CompleteSetup
                | Self::RestartSetup
                | Self::CheckForUpdates
                | Self::DismissUpdate
                | Self::ChangeHistory
                | Self::ModelSwitch
                | Self::ModelDelete
                | Self::ListDataLocations
                | Self::ClearData
                | Self::UpdateInstallRestart
                | Self::TestMicrophone
        )
    }

    const fn is_update_command(self) -> bool {
        matches!(
            self,
            Self::CheckForUpdates
                | Self::DismissUpdate
                | Self::OpenUpdateReleasePage
                | Self::UpdateInstallRestart
        )
    }
}

impl UiOperation {
    /// Whether the desktop does this only while no dictation is in flight, nothing is paused
    /// and no update installs.
    const fn needs_idle(self) -> bool {
        matches!(
            self,
            Self::SaveSettings
                | Self::CredentialMutation
                | Self::CompleteSetup
                | Self::RestartSetup
                | Self::StartSetupTest
                | Self::TestMicrophone
                | Self::ModelSwitch
                | Self::ModelDelete
                | Self::ClearData
        )
    }
}

impl UiOperation {
    /// Where a failure of this control is explained.
    pub(crate) const fn notice_scope(self) -> NoticeScope {
        match self {
            Self::SaveSettings | Self::TestPrompt | Self::CredentialMutation => {
                NoticeScope::Settings
            }
            Self::DownloadModel | Self::CancelDownload | Self::ModelSwitch | Self::ModelDelete => {
                NoticeScope::Models
            }
            Self::SaveSetup
            | Self::CompleteSetup
            | Self::RestartSetup
            | Self::RequestPermission
            | Self::OpenPermissionSettings
            | Self::LoadPermissions
            | Self::StartSetupTest
            | Self::StopSetupTest => NoticeScope::Setup,
            Self::ChangeHistory | Self::OpenRecordingsFolder => NoticeScope::History,
            Self::UpdateInstallRestart
            | Self::CheckForUpdates
            | Self::DismissUpdate
            | Self::OpenUpdateReleasePage => NoticeScope::Updates,
            Self::ListDataLocations | Self::ClearData => NoticeScope::Data,
            Self::TestMicrophone => NoticeScope::Home,
            Self::ListDevices
            | Self::StopRecording
            | Self::CancelRecording
            | Self::OverlayAction => NoticeScope::Global,
        }
    }
}

/// Why `operation` is disabled right now, or `None` when it may be sent.
#[must_use]
pub fn blocked(
    operation: UiOperation,
    snapshot: Option<&AppStateDto>,
    management: &ManagementState,
) -> Option<BlockedReason> {
    let Some(snapshot) = snapshot else {
        return Some(BlockedReason::Connecting);
    };
    let phase = match snapshot.phase {
        PhaseDto::Recording { .. } | PhaseDto::Processing { .. } => Some(BlockedReason::Dictating),
        PhaseDto::Paused => Some(BlockedReason::Paused),
        PhaseDto::Updating => Some(BlockedReason::UpdateInstalling),
        PhaseDto::Idle | PhaseDto::Error { .. } => None,
    };
    operation
        .needs_idle()
        .then_some(phase)
        .flatten()
        .or_else(|| locally_blocked(operation, management))
}

fn locally_blocked(operation: UiOperation, management: &ManagementState) -> Option<BlockedReason> {
    let pending = |busy: bool| busy.then_some(BlockedReason::CommandPending);
    // One change at a time from this window: a later control would act on data the pending
    // change is about to replace.
    let change_in_flight = pending(
        management.settings_saving()
            || management
                .busy
                .iter()
                .any(|busy| busy.blocks_workflow_changes()),
    );
    match operation {
        UiOperation::UpdateInstallRestart
        | UiOperation::CheckForUpdates
        | UiOperation::DismissUpdate
        | UiOperation::OpenUpdateReleasePage => {
            pending(management.busy.iter().any(|busy| busy.is_update_command()))
        }
        UiOperation::ChangeHistory => (!management.history_loaded())
            .then_some(BlockedReason::HistoryLoading)
            .or(change_in_flight),
        // A settings change queues behind the ones still saving.
        UiOperation::SaveSettings => pending(
            management
                .busy
                .iter()
                .any(|busy| busy.blocks_workflow_changes()),
        ),
        // The test runs the saved prompt, so it waits for edits still saving.
        UiOperation::TestPrompt => {
            pending(management.settings_saving() || management.is_busy(UiOperation::TestPrompt))
        }
        UiOperation::CredentialMutation
        | UiOperation::DownloadModel
        | UiOperation::CancelDownload
        | UiOperation::SaveSetup
        | UiOperation::CompleteSetup
        | UiOperation::ModelSwitch
        | UiOperation::ModelDelete
        | UiOperation::RestartSetup
        | UiOperation::ClearData => change_in_flight,
        UiOperation::StartSetupTest => change_in_flight.or(pending(management.is_busy(operation))),
        // A stop returns only after its dictation's output, so it blocks only another stop:
        // cancel and the outcome it publishes stay usable meanwhile. A permission request may be
        // waiting on a system dialog, so there is one at a time.
        UiOperation::StopRecording
        | UiOperation::CancelRecording
        | UiOperation::OverlayAction
        | UiOperation::OpenRecordingsFolder
        | UiOperation::TestMicrophone
        | UiOperation::RequestPermission
        | UiOperation::OpenPermissionSettings
        | UiOperation::LoadPermissions
        | UiOperation::ListDevices
        | UiOperation::ListDataLocations
        | UiOperation::StopSetupTest => pending(management.is_busy(operation)),
    }
}

#[cfg(test)]
mod tests;
