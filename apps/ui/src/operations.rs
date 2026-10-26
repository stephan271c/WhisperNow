use whispernow_domain::ipc::PermissionKind;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum UiOperation {
    LoadSettings,
    CredentialMutation,
    DownloadModel,
    CancelDownload,
    SaveSetup,
    CompleteSetup,
    RestartSetup,
    OpenPermissionSettings,
    RequestPermission(PermissionKind),
    SetUpPasteHelper,
    StartSetupTest,
    StopSetupTest,
    /// Deleting one history entry or clearing them all.
    ChangeHistory,
    OpenRecordingsFolder,
    ModelSwitch,
    ModelDelete,
    UpdateInstallRestart,
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
    pub(crate) const fn changes_settings(self) -> bool {
        matches!(
            self,
            Self::LoadSettings
                | Self::CredentialMutation
                | Self::ModelSwitch
                | Self::SaveSetup
                | Self::CompleteSetup
                | Self::RestartSetup
                | Self::ClearData
        )
    }
}
