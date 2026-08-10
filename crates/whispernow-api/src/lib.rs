//! Desktop command/event contracts shared by the native process and
//! the bundled webviews.

mod command_names;
mod dto;

pub use command_names::COMMAND_NAMES;
pub use dto::*;
pub use whispernow_domain::{
    ApiKey, CleanupPrompt, ClearDataCategory, DownloadPhase, EndpointError, EntryId, EntryStatus,
    HistoryCursor, HistoryEntry, HistoryFilter, HistoryLimit, HistoryPage, HistoryPageRequest,
    LLM_TIMEOUT_SECONDS_RANGE, LlmProvider, ModelAvailability, ModelFamily, ModelId,
    OverlayPosition, OverlayStyle, ProviderEndpoint, ProviderSettings, RecordingMode,
    RelativeRating, Settings, SettingsError, SetupCheckpoint, SetupStep, Shortcut, ShortcutKey,
    ShortcutModifier, ThemePreference, UtcMillis, VocabularyReplacement,
};

pub const STATE_CHANGED_EVENT: &str = "whispernow://state-changed";
pub const AUDIO_LEVEL_EVENT: &str = "whispernow://audio-level";
pub const UPDATE_CHANGED_EVENT: &str = "whispernow://update-changed";
pub const HISTORY_CHANGED_EVENT: &str = "whispernow://history-changed";
/// Carries transcript text, so the desktop emits it only to the recording-bar window and only
/// that window subscribes to it.
pub const PARTIAL_TRANSCRIPT_EVENT: &str = "whispernow://partial-transcript";
pub const NAVIGATION_REQUESTED_EVENT: &str = "whispernow://navigation-requested";
/// Content-free events broadcast to every window.
pub const EVENT_NAMES: [&str; 5] = [
    STATE_CHANGED_EVENT,
    AUDIO_LEVEL_EVENT,
    UPDATE_CHANGED_EVENT,
    HISTORY_CHANGED_EVENT,
    NAVIGATION_REQUESTED_EVENT,
];
/// Everything the recording bar subscribes to: the broadcasts and the partial transcript.
pub const RECORDING_BAR_EVENT_NAMES: [&str; 6] = [
    STATE_CHANGED_EVENT,
    AUDIO_LEVEL_EVENT,
    UPDATE_CHANGED_EVENT,
    HISTORY_CHANGED_EVENT,
    NAVIGATION_REQUESTED_EVENT,
    PARTIAL_TRANSCRIPT_EVENT,
];

#[cfg(test)]
mod tests;
