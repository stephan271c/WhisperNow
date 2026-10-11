use crate::NoticeScope;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UiOperation {
    LoadSettings,
    CredentialMutation,
    DownloadModel,
    CancelDownload,
    SaveSetup,
    CompleteSetup,
    RestartSetup,
    RequestPermission,
    OpenPermissionSettings,
    LoadPermissions,
    ListDevices,
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
    pub(crate) const fn notice_scope(self) -> NoticeScope {
        match self {
            Self::LoadSettings | Self::CredentialMutation => NoticeScope::Settings,
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
