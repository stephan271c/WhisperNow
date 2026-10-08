use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::ModelId;

/// Never reused, so an entry's audio file name is unique.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct EntryId(pub u64);

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct UtcMillis(pub i64);

impl UtcMillis {
    #[must_use]
    pub fn now() -> Self {
        match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(duration) => Self(i64::try_from(duration.as_millis()).unwrap_or(i64::MAX)),
            Err(error) => Self(-i64::try_from(error.duration().as_millis()).unwrap_or(i64::MAX)),
        }
    }

    #[must_use]
    pub const fn get(self) -> i64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub enum HistoryLimit {
    #[default]
    #[serde(rename = "20")]
    Last20,
    #[serde(rename = "100")]
    Last100,
    #[serde(rename = "1000")]
    Last1000,
    #[serde(rename = "unlimited")]
    Unlimited,
}

impl HistoryLimit {
    pub const ALL: [Self; 4] = [Self::Last20, Self::Last100, Self::Last1000, Self::Unlimited];

    #[must_use]
    pub const fn max_entries(self) -> Option<u32> {
        match self {
            Self::Last20 => Some(20),
            Self::Last100 => Some(100),
            Self::Last1000 => Some(1_000),
            Self::Unlimited => None,
        }
    }

    /// The smallest choice that keeps at least `entries` (`None` meaning all), for importing
    /// older settings.
    #[must_use]
    pub fn at_least(entries: Option<u32>) -> Self {
        Self::ALL
            .into_iter()
            .find(|limit| match (limit.max_entries(), entries) {
                (None, _) => true,
                (Some(_), None) => false,
                (Some(kept), Some(wanted)) => kept >= wanted,
            })
            .unwrap_or(Self::Unlimited)
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Last20 => "Last 20",
            Self::Last100 => "Last 100",
            Self::Last1000 => "Last 1,000",
            Self::Unlimited => "Unlimited",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EntryStatus {
    /// Transcribed and pasted.
    Ok,
    /// Transcription failed, including its automatic retry.
    Failed,
    Cancelled,
    /// Transcribed, but the text never reached the focused application.
    NotDelivered,
}

impl EntryStatus {
    /// Failed and cancelled dictations always keep their audio, so the speech is not lost.
    #[must_use]
    pub const fn always_keeps_audio(self) -> bool {
        matches!(self, Self::Failed | Self::Cancelled)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct HistoryEntry {
    pub id: EntryId,
    pub created_at: UtcMillis,
    pub duration_millis: Option<u64>,
    /// The recognized transcript; absent when recognition never produced one.
    pub raw_text: Option<String>,
    /// The text as pasted, after vocabulary replacements and AI cleanup, when it differs from
    /// the transcript.
    pub cleaned_text: Option<String>,
    pub status: EntryStatus,
    /// A plain-language explanation of a failure or an undelivered paste.
    pub error: Option<String>,
    pub has_audio: bool,
    pub asr_model_id: Option<ModelId>,
    pub llm_model: Option<String>,
}

impl HistoryEntry {
    #[must_use]
    pub fn text(&self) -> Option<&str> {
        self.cleaned_text.as_deref().or(self.raw_text.as_deref())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NewHistoryEntry {
    pub created_at: UtcMillis,
    pub duration_millis: Option<u64>,
    pub raw_text: Option<String>,
    pub cleaned_text: Option<String>,
    pub status: EntryStatus,
    pub error: Option<String>,
    pub asr_model_id: Option<ModelId>,
    pub llm_model: Option<String>,
}

/// Where a newest-first listing continues.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct HistoryCursor {
    pub created_at: UtcMillis,
    pub id: EntryId,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct HistoryFilter {
    /// Keeps entries whose transcript or cleaned text contains this text, ignoring ASCII case.
    /// Blank text does not filter.
    pub query: String,
    /// A cancelled dictation is the user's choice, not a failure.
    pub failed_only: bool,
}

impl HistoryFilter {
    pub const MAX_QUERY_BYTES: usize = 256;

    #[must_use]
    pub fn search_text(&self) -> Option<&str> {
        let query = self.query.trim();
        (!query.is_empty()).then_some(query)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct HistoryPageRequest {
    pub before: Option<HistoryCursor>,
    pub limit: u32,
    #[serde(default)]
    pub filter: HistoryFilter,
}

impl HistoryPageRequest {
    pub const DEFAULT_PAGE_SIZE: u32 = 30;

    #[must_use]
    pub fn first_page(filter: HistoryFilter) -> Self {
        Self {
            before: None,
            limit: Self::DEFAULT_PAGE_SIZE,
            filter,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct HistoryPage {
    pub entries: Vec<HistoryEntry>,
    pub next_cursor: Option<HistoryCursor>,
}

#[cfg(test)]
mod tests {
    use super::{EntryStatus, HistoryLimit};

    #[test]
    fn history_limit_imports_to_the_smallest_choice_that_keeps_everything() {
        assert_eq!(HistoryLimit::at_least(Some(1)), HistoryLimit::Last20);
        assert_eq!(HistoryLimit::at_least(Some(20)), HistoryLimit::Last20);
        assert_eq!(HistoryLimit::at_least(Some(21)), HistoryLimit::Last100);
        assert_eq!(HistoryLimit::at_least(Some(1_000)), HistoryLimit::Last1000);
        assert_eq!(HistoryLimit::at_least(Some(5_000)), HistoryLimit::Unlimited);
        assert_eq!(HistoryLimit::at_least(None), HistoryLimit::Unlimited);
        assert_eq!(
            serde_json::to_string(&HistoryLimit::Last100).expect("encode"),
            r#""100""#
        );
    }

    #[test]
    fn only_failed_and_cancelled_entries_always_keep_audio() {
        assert!(EntryStatus::Failed.always_keeps_audio());
        assert!(EntryStatus::Cancelled.always_keeps_audio());
        assert!(!EntryStatus::Ok.always_keeps_audio());
        assert!(!EntryStatus::NotDelivered.always_keeps_audio());
    }
}
