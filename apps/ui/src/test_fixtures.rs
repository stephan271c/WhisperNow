use whispernow_domain::{EntryId, EntryStatus, HistoryEntry, UtcMillis};

/// A pasted dictation `id` saying `text`, made `id` milliseconds after the epoch.
pub(crate) fn history_entry(id: i64, text: &str) -> HistoryEntry {
    HistoryEntry {
        id: EntryId(id),
        created_at: UtcMillis(id),
        duration_millis: None,
        raw_text: Some(text.to_owned()),
        cleaned_text: None,
        status: EntryStatus::Ok,
        error: None,
        audio_kept: false,
    }
}
