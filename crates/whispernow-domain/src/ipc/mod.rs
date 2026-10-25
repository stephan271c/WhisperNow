mod command_names;

pub use command_names::COMMAND_NAMES;

use crate::{
    ApiKey, AppError, AudioDeviceId, ClearDataCategory, DownloadPhase, ErrorCategory, LlmProvider,
    ModelAvailability, ModelBenchmark, ModelFamily, ModelId, Settings,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// No API key crosses: `settings.api_keys` only names the providers that have one.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsView {
    pub settings: Settings,
    pub start_on_login: bool,
    /// What is not in effect although the settings ask for it, worded for the user.
    pub warnings: Vec<String>,
}

impl SettingsView {
    #[must_use]
    pub fn new(settings: &Settings) -> Self {
        Self {
            settings: settings.clone(),
            start_on_login: false,
            warnings: settings.cleanup_warning().into_iter().collect(),
        }
    }
}

/// A warning never means the save failed: the settings file already holds `settings`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SaveSettingsResponse {
    pub settings: SettingsView,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SetApiKeyRequest {
    pub provider: LlmProvider,
    pub key: ApiKey,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioDevice {
    pub id: AudioDeviceId,
    pub display_name: String,
    pub is_default: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionKind {
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
pub enum PermissionStatus {
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
pub struct Permission {
    pub kind: PermissionKind,
    pub status: PermissionStatus,
    /// Whether `request_permission` can ask for it from inside the app.
    pub can_request: bool,
    /// Whether `open_permission_settings` can show the system page that grants it.
    pub can_open_settings: bool,
}

/// The permissions this platform needs, in the order setup shows them.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PermissionsResponse {
    pub permissions: Vec<Permission>,
    /// Present when Wayland paste has no paste helper set up.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paste_helper_setup: Option<PasteHelperSetup>,
}

/// Without the paste helper, Wayland paste goes through the `RemoteDesktop`
/// portal, and GNOME shows its remote-control indicator for as long as the app runs.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PasteHelperSetup {
    /// The command that sets the helper up, when this build carries its install
    /// script: an `AppImage` or a source build. A package installs it itself.
    pub command: Option<String>,
}

impl PermissionsResponse {
    #[must_use]
    pub fn all_granted(&self) -> bool {
        self.permissions
            .iter()
            .all(|permission| permission.status == PermissionStatus::Granted)
    }

    #[must_use]
    pub fn get(&self, kind: PermissionKind) -> Option<&Permission> {
        self.permissions
            .iter()
            .find(|permission| permission.kind == kind)
    }
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

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelSummary {
    pub id: ModelId,
    pub display_name: String,
    pub family: ModelFamily,
    pub availability: ModelAvailability,
    /// BCP-47 tags of the languages a speech model transcribes; empty for other models.
    pub languages: Vec<String>,
    pub size_bytes: u64,
    /// The model's measurements; the UI rates it from them.
    pub benchmark: Option<ModelBenchmark>,
    /// The model new installations start with.
    pub recommended: bool,
}

/// One existing piece of application data that clearing its category removes.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DataLocation {
    pub category: ClearDataCategory,
    #[serde(serialize_with = "serialize_display_path")]
    pub path: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClearDataResult {
    pub items: Vec<ClearDataItem>,
    pub restart_required: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClearDataItem {
    #[serde(flatten)]
    pub location: DataLocation,
    pub deleted: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(
    tag = "phase",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum UpdatePhase {
    Disabled,
    Idle,
    Checking,
    Downloading {
        version: String,
        release_notes: Option<String>,
        progress: UpdateProgress,
    },
    ReadyToRestart {
        version: String,
        release_notes: Option<String>,
        error: Option<UpdateError>,
    },
    Installing {
        version: String,
        release_notes: Option<String>,
    },
    /// A newer signed release exists, but this installation's package manager owns upgrades.
    ManualInstallAvailable {
        version: String,
        release_notes: Option<String>,
        package: ManualUpdatePackage,
        error: Option<UpdateError>,
    },
    UpToDate,
    Error {
        error: UpdateError,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ManualUpdatePackage {
    Deb,
    Rpm,
}

/// Artifact identity and location stay native-only.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateProgress {
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdateError {
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

impl UpdateError {
    #[must_use]
    pub const fn is_recoverable(self) -> bool {
        !matches!(self, Self::InvalidSignature | Self::MalformedMetadata)
    }
}

/// The native controller owns all download and install material.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateState {
    pub phase: UpdatePhase,
    pub notification_dismissed: bool,
    /// A restart was requested and waits for the dictation in flight to finish.
    pub restart_deferred: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelState {
    Unloaded,
    Downloading,
    Loading,
    Ready,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ActiveDownload {
    pub model_id: ModelId,
    pub phase: DownloadPhase,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    /// Average transfer rate since the download started; `None` until it can be measured.
    pub bytes_per_second: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CleanupDevice {
    Cpu,
    Gpu { name: String },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AppState {
    pub phase: Phase,
    pub selected_model_id: ModelId,
    pub model_state: ModelState,
    pub last_transcript: Option<String>,
    /// The latest failure a surface should explain. Always present in `Phase::Error`.
    pub error: Option<AppError>,
    pub active_downloads: Vec<ActiveDownload>,
    /// The model whose latest download failed, until it is downloaded again.
    pub failed_download: Option<ModelId>,
    /// How setup's latest test dictation ended; cleared when the next test starts.
    pub setup_test: Option<SetupTestResult>,
    /// `None` until the local cleanup model loads, and again once it is freed.
    pub cleanup_device: Option<CleanupDevice>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordingTrigger {
    Hold,
    Toggle,
    /// The first-run test dictation: its transcript is shown in setup only, so it is never
    /// pasted, and neither it nor its audio is kept in history.
    SetupTest,
}

impl RecordingTrigger {
    #[must_use]
    pub const fn is_setup_test(self) -> bool {
        matches!(self, Self::SetupTest)
    }
}

/// Every surface derives its view from this one value.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(
    tag = "phase",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum Phase {
    Idle,
    Recording {
        trigger: RecordingTrigger,
        session: u64,
    },
    /// The captured audio is being transcribed, or cleaned up when `cleaning_up`.
    Processing {
        cleaning_up: bool,
    },
    Paused,
    ClearingData,
    /// An update is being installed and the application will restart.
    Updating,
    /// The latest dictation failed or was not delivered, and `AppState::error` says why. Stays
    /// until dismissed or the next recording starts.
    Error,
}

impl Phase {
    #[must_use]
    pub const fn is_busy(&self) -> bool {
        matches!(self, Self::Recording { .. } | Self::Processing { .. })
    }

    #[must_use]
    pub const fn shortcuts_enabled(&self) -> bool {
        !matches!(self, Self::Paused | Self::ClearingData | Self::Updating)
    }

    pub fn require_idle(&self) -> Result<(), AppError> {
        match self {
            Self::Idle | Self::Error => Ok(()),
            Self::Recording { .. } => Err(conflict(
                "workflow_busy",
                "Stop the current recording first",
            )),
            Self::Processing { .. } => Err(conflict(
                "workflow_busy",
                "Wait for the current transcription to finish",
            )),
            Self::Paused => Err(conflict("application_paused", "Resume WhisperNow first")),
            Self::ClearingData => Err(conflict(
                "application_data_clearing",
                "Wait for application data to finish clearing",
            )),
            Self::Updating => Err(conflict(
                "update_installing",
                "An application update is being installed",
            )),
        }
    }
}

fn conflict(code: &str, summary: &str) -> AppError {
    AppError::new(ErrorCategory::General, code, summary)
}

/// The test reports only here: its transcript is not the last transcript and its failure is never
/// the application's current problem.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(
    tag = "outcome",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum SetupTestResult {
    Heard {
        text: String,
    },
    /// The microphone heard nothing, or nothing that sounded like speech.
    NoSpeech,
    /// The user cancelled the test, or the application interrupted it.
    Cancelled,
    /// Boxed so a snapshot stays small.
    Failed {
        error: Box<AppError>,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MainWindowSection {
    History,
    Settings,
    Audio,
    SpeechModels,
    AiCleanup,
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
        state: AppState,
    },
    AudioLevel {
        rms: f32,
        peak: f32,
        /// Captured audio so far, which the overlay shows as elapsed time.
        elapsed_ms: u64,
    },
    UpdateChanged {
        state: UpdateState,
    },
    /// History may have changed; listeners re-fetch what they show.
    HistoryChanged,
    /// A download ended or a model was deleted; listeners re-fetch the model list.
    ModelsChanged,
    NavigationRequested {
        section: MainWindowSection,
    },
}

pub const EVENT: &str = "whispernow://event";

#[cfg(test)]
mod tests;

fn serialize_display_path<S: serde::Serializer>(
    path: &Path,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&path.to_string_lossy())
}
