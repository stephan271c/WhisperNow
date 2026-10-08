use serde::{Deserialize, Serialize};
use whispernow_domain::{
    ApiKey, ClearDataCategory, DownloadPhase, EntryId, LlmProvider, ModelAvailability, ModelFamily,
    ModelId, RecordingMode, RelativeRating, Settings, SetupCheckpoint,
};

/// No API key crosses: `settings.api_keys` only names the providers that have one.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsView {
    pub settings: Settings,
    /// What is not in effect although the settings ask for it, worded for the user.
    pub warnings: Vec<String>,
}

impl SettingsView {
    #[must_use]
    pub fn new(settings: &Settings) -> Self {
        Self {
            settings: settings.clone(),
            warnings: Vec::new(),
        }
    }
}

/// The desktop keeps what the window does not edit: the providers with API keys, the model, setup's
/// progress and the window geometry.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct UpdateSettingsRequest {
    pub settings: Settings,
}

/// A warning never means the save failed: the settings file already holds `settings`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SaveSettingsResponse {
    pub settings: SettingsView,
    pub warnings: Vec<ApiErrorDto>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SetupStateView {
    pub first_run_complete: bool,
    pub checkpoint: SetupCheckpoint,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SetApiKeyRequest {
    pub provider: LlmProvider,
    pub key: ApiKey,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ClearApiKeyRequest {
    pub provider: LlmProvider,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioDeviceDto {
    pub id: String,
    pub display_name: String,
    pub is_default: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionKindDto {
    /// Microphone access, on every platform.
    Microphone,
    /// macOS Accessibility, which sends the paste keystroke.
    Accessibility,
    /// The Wayland `GlobalShortcuts` portal consent for the recording shortcuts.
    GlobalShortcuts,
    /// The Wayland `RemoteDesktop` portal consent for the paste keystroke.
    RemoteDesktop,
    /// X11 keystroke injection (XTEST) for the paste keystroke.
    PasteSupport,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionStatusDto {
    Granted,
    /// Not asked for yet, or the platform cannot tell until it is asked.
    NotDetermined,
    Denied,
    /// This session cannot provide it; dictation still works, and the transcript is copied
    /// instead of pasted where the missing capability is the paste keystroke.
    Unavailable,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PermissionDto {
    pub kind: PermissionKindDto,
    pub status: PermissionStatusDto,
    /// Whether `request_permission` can ask for it from inside the app.
    pub can_request: bool,
    /// Whether `open_permission_settings` can show the system page that grants it.
    pub can_open_settings: bool,
}

/// The permissions this platform needs, in the order setup shows them.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PermissionsResponse {
    pub permissions: Vec<PermissionDto>,
    /// Present when Wayland paste falls back to the `RemoteDesktop` portal
    /// because the paste helper is not set up.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paste_helper_setup: Option<PasteHelperSetupDto>,
}

/// Without the paste helper, Wayland paste goes through the `RemoteDesktop`
/// portal, and GNOME shows its remote-control indicator for as long as the app runs.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PasteHelperSetupDto {
    /// The command that sets the helper up, when this build carries its install
    /// script: an `AppImage` or a source build. A package installs it itself.
    pub command: Option<String>,
}

impl PermissionsResponse {
    #[must_use]
    pub fn all_granted(&self) -> bool {
        self.permissions
            .iter()
            .all(|permission| permission.status == PermissionStatusDto::Granted)
    }

    #[must_use]
    pub fn get(&self, kind: PermissionKindDto) -> Option<&PermissionDto> {
        self.permissions
            .iter()
            .find(|permission| permission.kind == kind)
    }
}

/// Asks for, or opens the system settings page of, one permission.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PermissionRequest {
    pub kind: PermissionKindDto,
}

/// The levels heard during one bounded microphone test; it never contains audio.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MicrophoneTestResponse {
    /// RMS level of each consecutive window, oldest first, each in `0.0..=1.0`.
    pub levels: Vec<f32>,
    /// The loudest sample magnitude heard, in `0.0..=1.0`.
    pub peak: f32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ListAudioDevicesResponse {
    pub devices: Vec<AudioDeviceDto>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelDto {
    pub id: ModelId,
    pub display_name: String,
    pub family: ModelFamily,
    pub availability: ModelAvailability,
    /// The languages the model names, as BCP-47 tags; empty when it names none.
    pub languages: Vec<String>,
    pub automatic_language_detection: bool,
    pub size_bytes: u64,
    pub speed: RelativeRating,
    pub accuracy: RelativeRating,
    /// The model new installations start with.
    pub recommended: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ListModelsResponse {
    pub models: Vec<ModelDto>,
}

/// Loads one entry's kept audio as WAV bytes for playback.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetHistoryAudioRequest {
    pub id: EntryId,
}

/// Deleting an entry that no longer exists succeeds without a change.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DeleteHistoryEntryRequest {
    pub id: EntryId,
}

/// One existing piece of application data that clearing its category removes.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DataLocationDto {
    pub category: ClearDataCategory,
    pub path: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ClearDataRequest {
    pub categories: Vec<ClearDataCategory>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClearDataResult {
    pub items: Vec<ClearDataResultItemDto>,
    pub restart_required: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClearDataResultItemDto {
    pub category: ClearDataCategory,
    pub path: String,
    pub deleted: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdatePhaseDto {
    Disabled,
    Idle,
    Checking,
    Downloading,
    ReadyToRestart,
    Installing,
    /// A newer signed release exists, but this installation's package manager owns upgrades.
    ManualInstallAvailable,
    UpToDate,
    Error,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ManualUpdatePackageDto {
    Deb,
    Rpm,
}

/// Artifact identity and location stay native-only.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateProgressDto {
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdateErrorCodeDto {
    Offline,
    ServerUnavailable,
    MalformedMetadata,
    UnsupportedTarget,
    InvalidSignature,
    DownloadInterrupted,
    InstallFailed,
    InsufficientPrivileges,
    ReadOnlyFilesystem,
    ReleasePageUnavailable,
    Internal,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateErrorDto {
    pub code: UpdateErrorCodeDto,
    pub recoverable: bool,
}

/// The native controller owns all download and install material.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateStateDto {
    pub phase: UpdatePhaseDto,
    pub available_version: Option<String>,
    pub release_notes: Option<String>,
    pub progress: Option<UpdateProgressDto>,
    pub error: Option<UpdateErrorDto>,
    pub notification_dismissed: bool,
    /// A restart was requested and waits for the dictation in flight to finish.
    pub restart_deferred: bool,
    /// The package to download while the phase is `ManualInstallAvailable`.
    pub manual_package: Option<ManualUpdatePackageDto>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelStateDto {
    Unloaded,
    Downloading,
    Loading,
    Ready,
    Failed,
}

/// The status is deliberately closed: native error strings, target identity, clipboard
/// contents, and transcript bodies must never cross the event/snapshot boundary through this
/// field.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TextOutputStatusDto {
    PastedAndRestored,
    PastedClipboardNotRestored,
    RestoreSkippedClipboardChanged,
    PasteFailed,
    Cancelled,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ActiveDownloadDto {
    pub model_id: ModelId,
    pub phase: DownloadPhase,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
    /// Average transfer rate since the download started; `None` until it can be measured.
    pub bytes_per_second: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AppStateDto {
    pub phase: PhaseDto,
    pub selected_model_id: ModelId,
    pub model_state: ModelStateDto,
    pub last_transcript: Option<TranscriptResponse>,
    /// The latest failure a surface should explain; a failed dictation also shows it in `phase`.
    pub error: Option<ApiErrorDto>,
    pub active_downloads: Vec<ActiveDownloadDto>,
    /// How setup's latest test dictation ended; cleared when the next test starts.
    pub setup_test: Option<SetupTestResultDto>,
}

/// Every surface derives its view from this one value.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(
    tag = "phase",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum PhaseDto {
    Idle,
    Recording {
        mode: RecordingMode,
        session: u64,
    },
    /// The captured audio is being transcribed, or cleaned up when `cleaning_up`.
    Processing {
        cleaning_up: bool,
    },
    Paused,
    /// An update is being installed and the application will restart.
    Updating,
    /// The latest dictation failed or was not delivered. Stays until dismissed or the next
    /// recording starts.
    Error {
        error: ApiErrorDto,
    },
}

impl PhaseDto {
    #[must_use]
    pub const fn is_busy(&self) -> bool {
        matches!(self, Self::Recording { .. } | Self::Processing { .. })
    }
}

/// The test reports only here: its transcript is not the last transcript and its failure is never
/// the application's current problem.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(
    tag = "outcome",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum SetupTestResultDto {
    Heard {
        text: String,
    },
    /// The microphone heard nothing, or nothing that sounded like speech.
    NoSpeech,
    /// The user cancelled the test, or the application interrupted it.
    Cancelled,
    /// Boxed so a snapshot stays small.
    Failed {
        error: Box<ApiErrorDto>,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MainWindowSectionDto {
    History,
    Settings,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OpenMainWindowRequest {
    pub section: MainWindowSectionDto,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptResponse {
    pub raw_text: String,
    pub final_text: String,
    pub text_output_status: Option<TextOutputStatusDto>,
    /// AI cleanup fell back to the recognized transcript.
    pub cleanup_failed: bool,
    pub history_save_failed: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCategoryDto {
    /// Anything the windows have no particular answer for.
    General,
    NotFound,
    Audio,
    Permission,
    Model,
    Recognition,
    Output,
    /// The history store cannot be read, so none of its entries can be shown.
    HistoryUnavailable,
}

impl ErrorCategoryDto {
    #[must_use]
    pub const fn default_recovery(self) -> Option<RecoveryActionDto> {
        match self {
            Self::NotFound => None,
            Self::Audio => Some(RecoveryActionDto::ChooseDevice),
            Self::Permission => Some(RecoveryActionDto::OpenSettings),
            Self::Model => Some(RecoveryActionDto::InstallModel),
            Self::Output => Some(RecoveryActionDto::CopyManually),
            Self::General | Self::Recognition | Self::HistoryUnavailable => {
                Some(RecoveryActionDto::Retry)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryActionDto {
    Retry,
    ChooseDevice,
    InstallModel,
    OpenSettings,
    CopyManually,
    CheckDestination,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiErrorDto {
    pub category: ErrorCategoryDto,
    pub code: String,
    pub summary: String,
    pub recovery: Option<RecoveryActionDto>,
}

impl ApiErrorDto {
    #[must_use]
    pub fn new(category: ErrorCategoryDto, code: &str, summary: &str) -> Self {
        Self {
            category,
            code: code.to_owned(),
            summary: summary.to_owned(),
            recovery: category.default_recovery(),
        }
    }
}

/// State and update changes carry authoritative snapshots; audio levels are the only independently
/// throttled realtime signal.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(
    tag = "type",
    content = "payload",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum EventPayload {
    StateChanged {
        state: AppStateDto,
    },
    AudioLevel {
        rms: f32,
        peak: f32,
        /// Captured audio so far, which the overlay shows as elapsed time.
        elapsed_ms: u64,
    },
    UpdateChanged {
        state: UpdateStateDto,
    },
    /// History may have changed; listeners re-fetch what they show.
    HistoryChanged,
    NavigationRequested {
        section: MainWindowSectionDto,
    },
}

impl EventPayload {
    #[must_use]
    pub const fn event_name(&self) -> &'static str {
        match self {
            EventPayload::StateChanged { .. } => crate::STATE_CHANGED_EVENT,
            EventPayload::AudioLevel { .. } => crate::AUDIO_LEVEL_EVENT,
            EventPayload::UpdateChanged { .. } => crate::UPDATE_CHANGED_EVENT,
            EventPayload::HistoryChanged => crate::HISTORY_CHANGED_EVENT,
            EventPayload::NavigationRequested { .. } => crate::NAVIGATION_REQUESTED_EVENT,
        }
    }
}
