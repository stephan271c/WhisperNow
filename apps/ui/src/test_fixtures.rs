use whispernow_domain::ipc::{AppState, ModelState, Phase, SettingsView};
use whispernow_domain::{EntryId, EntryStatus, HistoryEntry, ModelId, Settings, UtcMillis};

use crate::ManagementState;

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

/// A snapshot in `phase` with the tiny Whisper model ready.
pub(crate) fn snapshot(phase: Phase) -> AppState {
    AppState {
        phase,
        selected_model_id: ModelId::new("sherpa-onnx-whisper-tiny").expect("fixture model id"),
        model_state: ModelState::Ready,
        last_transcript: None,
        error: None,
        active_downloads: Vec::new(),
        setup_test: None,
        failed_download: None,
        cleanup_device: None,
    }
}

pub(crate) fn settings_view(settings: Settings) -> SettingsView {
    SettingsView {
        start_on_login: false,
        settings,
        warnings: Vec::new(),
    }
}

pub(crate) fn management_with(settings: Settings) -> ManagementState {
    let mut management = ManagementState::default();
    management.install_settings(settings_view(settings));
    management
}
