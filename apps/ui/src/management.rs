//! Management-screen projection shared by the settings, models, setup, and history screens.
//!
//! The per-screen transitions live in submodules; this module owns the notice and the busy
//! flags.

mod draft;
mod history;
mod models;
mod settings;
mod setup;

use std::collections::BTreeMap;

use dioxus::core::Task;
use whispernow_api::{
    ActiveDownloadDto, AudioDeviceDto, ClearDataResult, DataLocationDto, EventPayload,
    HistoryCursor, HistoryEntry, HistoryFilter, ListAudioDevicesResponse, MicrophoneTestResponse,
    ModelDto, PermissionsResponse, Settings, SettingsView, SetupStateView,
};

use crate::components::ui::Tone;
use crate::{Notice, NoticeAction, NoticeScope, UiOperation};

pub(crate) use draft::SetupEdit;
pub(crate) use history::HistoryCopied;
pub(crate) use settings::{PromptTest, SaveStatus, SettingKey};
pub(crate) use setup::SetupTest;

#[derive(Clone, Default, PartialEq)]
pub(crate) struct ManagementState {
    /// The settings the desktop last saved; controls show [`Self::shown_settings`].
    pub settings: Option<SettingsView>,
    /// The settings with the edits the desktop has not saved yet.
    edited: Option<Settings>,
    /// The controls whose edits are with the desktop.
    sending: Vec<SettingKey>,
    /// The controls edited since, sent together once the desktop answers.
    unsent: Vec<SettingKey>,
    /// Why each control's latest change was not saved.
    setting_errors: BTreeMap<SettingKey, String>,
    /// The controls whose edits the desktop saved last.
    saved_settings: Vec<SettingKey>,
    pub prompt_test: Option<PromptTest>,
    pub setup: Option<SetupStateView>,
    pub devices: Vec<AudioDeviceDto>,
    /// Whether the microphones have been listed since the window opened.
    pub devices_listed: bool,
    pub models: Vec<ModelDto>,
    pub models_loaded: bool,
    /// The downloads the latest snapshot reported.
    pub downloads: Vec<ActiveDownloadDto>,
    pub history: Vec<HistoryEntry>,
    pub history_unavailable: Option<String>,
    pub history_next_cursor: Option<HistoryCursor>,
    /// The search and failed filter the latest window request reads with.
    pub history_filter: HistoryFilter,
    /// The filter the rows in `history` were read with; `None` until a window is installed
    /// and once it can no longer be read.
    pub(crate) history_rows_filter: Option<HistoryFilter>,
    /// Why the rows for the current search could not be read; shown in place of the list.
    pub history_search_failed: Option<String>,
    /// The search text as typed, applied once typing pauses.
    pub history_search: String,
    /// Pages in the loaded history window; "Load more" grows it by one.
    history_pages: u32,
    /// Whether the latest history read failed; returning to idle reads it again.
    history_refresh_failed: bool,
    /// The latest history copy's outcome, shown on its entry's Copy button.
    pub history_copied: Option<HistoryCopied>,
    /// The latest history copy, which clears its outcome after a moment.
    pub(crate) history_copy: Option<Task>,
    /// The application data a clear would remove, once the user asked to review it.
    pub data_locations: Option<Vec<DataLocationDto>>,
    pub clear_result: Option<ClearDataResult>,
    /// Controls whose call is in flight, once per call.
    pub(crate) busy: Vec<UiOperation>,
    /// The permission whose request is waiting for the system's answer.
    pub(crate) requested_permission: Option<whispernow_api::PermissionKindDto>,
    pub notice: Option<Notice>,
    next_notice: u64,
    /// The levels of the latest microphone test.
    pub microphone_test: Option<MicrophoneTestResponse>,
    /// The permission cards the desktop last reported.
    pub permissions: Option<PermissionsResponse>,
    /// The polls setup runs after a permission request until no card can change any more.
    pub(crate) permission_watch: Option<Task>,
    /// Setup's test dictation and its result.
    pub setup_test: Option<SetupTest>,
}

impl ManagementState {
    /// Replaces the current notice.
    pub fn notify(&mut self, tone: Tone, scope: NoticeScope, message: impl Into<String>) {
        self.notify_with_action(tone, scope, message, None);
    }

    /// Replaces the current notice with one that offers a follow-up link.
    pub fn notify_with_action(
        &mut self,
        tone: Tone,
        scope: NoticeScope,
        message: impl Into<String>,
        action: Option<NoticeAction>,
    ) {
        self.next_notice = self.next_notice.saturating_add(1);
        self.notice = Some(Notice {
            id: self.next_notice,
            tone,
            message: message.into(),
            action,
            scope,
        });
    }

    /// Reports a save, naming every side effect the desktop could not apply.
    pub(crate) fn notify_saved(
        &mut self,
        scope: NoticeScope,
        saved: &str,
        warnings: &[whispernow_api::ApiErrorDto],
    ) {
        let tone = if warnings.is_empty() {
            Tone::Success
        } else {
            Tone::Warning
        };
        self.notify(tone, scope, settings::saved_notice(saved, warnings));
    }

    /// Dismisses the notice with `id`; a newer notice stays.
    pub fn dismiss_notice(&mut self, id: u64) {
        if self.notice.as_ref().is_some_and(|notice| notice.id == id) {
            self.notice = None;
        }
    }

    /// Dismisses the current notice when it belongs to `scope`.
    pub fn clear_notice_scope(&mut self, scope: NoticeScope) {
        if self
            .notice
            .as_ref()
            .is_some_and(|notice| notice.scope == scope)
        {
            self.notice = None;
        }
    }

    /// Marks `operation`'s call as in flight.
    pub(crate) fn begin_busy(&mut self, operation: UiOperation) {
        self.busy.push(operation);
    }

    /// Marks one of `operation`'s calls as answered.
    pub(crate) fn end_busy(&mut self, operation: UiOperation) {
        if let Some(index) = self.busy.iter().position(|busy| *busy == operation) {
            self.busy.remove(index);
        }
    }

    /// Whether a call of `operation` is in flight.
    #[must_use]
    pub fn is_busy(&self, operation: UiOperation) -> bool {
        self.busy.contains(&operation)
    }

    pub fn install_devices(&mut self, devices: ListAudioDevicesResponse) {
        self.devices = devices.devices;
        self.devices_listed = true;
    }

    /// Applies a process event to the management projection; returns whether a download
    /// ended, so the models must be read again. A history change is content-free; the window
    /// answers it by reading history again.
    pub fn apply_event(&mut self, event: &EventPayload) -> bool {
        match event {
            EventPayload::StateChanged { state } => self.install_snapshot(state),
            _ => false,
        }
    }

    /// Shows what a clear removed; the window then reads everything again.
    pub fn apply_clear_result(&mut self, result: ClearDataResult) {
        self.data_locations = None;
        let failed = result.items.iter().any(|item| !item.deleted);
        let restart = result.restart_required;
        self.clear_result = Some(result);
        if failed {
            self.notify(
                Tone::Warning,
                NoticeScope::Data,
                "Some application data could not be removed",
            );
        } else if restart {
            self.notify(
                Tone::Info,
                NoticeScope::Data,
                "Application data removed. Restart WhisperNow before dictating again.",
            );
        } else {
            self.notify(Tone::Success, NoticeScope::Data, "Application data removed");
        }
    }
}
