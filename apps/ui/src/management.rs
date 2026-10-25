mod history;
mod models;
mod settings;
mod setup;

use std::collections::BTreeMap;

use dioxus::core::Task;
use whispernow_domain::ipc::{
    AudioDevice, ClearDataResult, DataLocation, MicrophoneTestResponse, ModelSummary,
    PermissionsResponse, SettingsView,
};
use whispernow_domain::{HistoryFilter, SetupCheckpoint};

use crate::notice::{Notice, NoticeScope, Tone};
use crate::operations::UiOperation;

pub(crate) use history::HistoryRows;
pub(crate) use settings::SettingKey;
pub(crate) use setup::SetupTest;

pub(crate) fn replace_task(slot: &mut Option<Task>, next: Option<Task>) {
    if let Some(previous) = std::mem::replace(slot, next) {
        previous.cancel();
    }
}

#[derive(Clone, Default, PartialEq)]
pub(crate) struct ManagementState {
    pub(crate) settings: Option<SettingsView>,
    pending_save: Option<settings::PendingSave>,
    setting_errors: BTreeMap<SettingKey, String>,
    /// The checkpoint setup shows, including choices not saved yet.
    pub(crate) setup: Option<SetupCheckpoint>,
    pub(crate) devices: Option<Vec<AudioDevice>>,
    pub(crate) models: Option<Vec<ModelSummary>>,
    pub(crate) history: HistoryRows,
    pub(crate) history_filter: HistoryFilter,
    history_extra_pages: u32,
    /// The window read in flight; a newer read cancels it, so an older answer never lands.
    pub(crate) history_read: Option<Task>,
    /// The application data a clear would remove, once the user asked to review it.
    pub(crate) data_locations: Option<Vec<DataLocation>>,
    pub(crate) clear_result: Option<ClearDataResult>,
    /// Controls whose call is in flight, once per call.
    pub(crate) busy: Vec<UiOperation>,
    pub(crate) notice: Option<Notice>,
    next_notice: u64,
    pub(crate) microphone_test: Option<MicrophoneTestResponse>,
    pub(crate) microphone_test_task: Option<Task>,
    pub(crate) permissions: Option<PermissionsResponse>,
    pub(crate) permission_watch: Option<Task>,
    pub(crate) setup_test: Option<SetupTest>,
}

impl ManagementState {
    pub(crate) fn notify(&mut self, tone: Tone, scope: NoticeScope, message: impl Into<String>) {
        self.next_notice = self.next_notice.saturating_add(1);
        self.notice = Some(Notice {
            id: self.next_notice,
            tone,
            message: message.into(),
            scope,
        });
    }

    pub(crate) fn notify_saved(&mut self, scope: NoticeScope, saved: &str, warnings: &[String]) {
        self.notify(
            Tone::Warning,
            scope,
            format!(
                "{saved}, but not everything took effect. {}",
                warnings.join(" ")
            ),
        );
    }

    /// Dismisses the notice with `id`; a newer notice stays.
    pub(crate) fn dismiss_notice(&mut self, id: u64) {
        if self.notice.as_ref().is_some_and(|notice| notice.id == id) {
            self.notice = None;
        }
    }

    pub(crate) fn clear_notice_scope(&mut self, scope: NoticeScope) {
        if self
            .notice
            .as_ref()
            .is_some_and(|notice| notice.scope == scope)
        {
            self.notice = None;
        }
    }

    pub(crate) fn begin_busy(&mut self, operation: UiOperation) -> bool {
        if operation.changes_settings() && self.settings_pending() {
            return false;
        }
        self.busy.push(operation);
        true
    }

    pub(crate) fn end_busy(&mut self, operation: UiOperation) {
        if let Some(index) = self.busy.iter().position(|busy| *busy == operation) {
            self.busy.remove(index);
        }
    }

    #[must_use]
    pub(crate) fn is_busy(&self, operation: UiOperation) -> bool {
        self.busy.contains(&operation)
    }

    pub(crate) fn apply_clear_result(&mut self, result: ClearDataResult) {
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
