//! Desktop command/event contracts shared by the native process and
//! the bundled webviews.

mod command_names;
mod dto;

pub use command_names::COMMAND_NAMES;
pub use dto::*;
pub use whispernow_domain::{
    ApiKey, ClearDataCategory, DownloadPhase, EndpointError, EntryId, EntryStatus, HistoryCursor,
    HistoryEntry, HistoryFilter, HistoryLimit, HistoryPage, HistoryPageRequest, LlmProvider,
    ModelAvailability, ModelFamily, ModelId, ProviderEndpoint, ProviderSettings, RecordingMode,
    RelativeRating, Settings, SettingsError, SetupCheckpoint, SetupStep, Shortcut, ShortcutKey,
    ShortcutModifier, ThemePreference, UtcMillis, VocabularyReplacement,
};

pub const STATE_CHANGED_EVENT: &str = "whispernow://state-changed";
pub const AUDIO_LEVEL_EVENT: &str = "whispernow://audio-level";
pub const UPDATE_CHANGED_EVENT: &str = "whispernow://update-changed";
pub const HISTORY_CHANGED_EVENT: &str = "whispernow://history-changed";
pub const NAVIGATION_REQUESTED_EVENT: &str = "whispernow://navigation-requested";
pub const EVENT_NAMES: [&str; 5] = [
    STATE_CHANGED_EVENT,
    AUDIO_LEVEL_EVENT,
    UPDATE_CHANGED_EVENT,
    HISTORY_CHANGED_EVENT,
    NAVIGATION_REQUESTED_EVENT,
];

#[cfg(test)]
mod tests;
