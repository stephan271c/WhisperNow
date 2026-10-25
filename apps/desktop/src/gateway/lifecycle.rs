use std::sync::Arc;

use super::{Failure, Gateway, settings::shortcut_error};
use crate::{
    services::{Phase, RecordingTrigger},
    shortcuts::{ShortcutAction, ShortcutRegistrationError},
};

impl Gateway {
    /// The shortcut backend stopped delivering events. A recording a shortcut started is
    /// cancelled, since its release can no longer arrive; history records it.
    pub(crate) fn shortcut_capability_lost(&self, error: ShortcutRegistrationError) {
        self.publish_error(&shortcut_error(error));
        self.shortcuts.capability_lost(error);
        let shortcut_recording = matches!(
            self.backend.state.lock().phase,
            Phase::Recording {
                trigger: RecordingTrigger::Hold | RecordingTrigger::Toggle,
                ..
            }
        );
        if shortcut_recording && let Err(error) = self.backend.dictation.cancel() {
            tracing::debug!(
                code = error.code,
                "the shortcut recording was not cancelled"
            );
        }
    }

    pub(crate) fn shortcut_action(&self, action: ShortcutAction) {
        let sent = match action {
            ShortcutAction::Pressed(trigger) => self.backend.dictation.shortcut_pressed(trigger),
            ShortcutAction::HoldReleased => self.backend.dictation.hold_released(),
            ShortcutAction::CancelDictation => self.backend.dictation.cancel(),
        };
        if let Err(error) = sent {
            tracing::warn!(code = error.code, "a shortcut was ignored");
        }
    }

    /// A press that arrives after the dictation ended has nothing to cancel, so failures are
    /// only logged.
    pub(crate) async fn cancel_dictation(&self) {
        if let Err(error) = self
            .run_command("cancel_dictation", Failure::Local, |backend| {
                backend.dictation.cancel()
            })
            .await
        {
            tracing::debug!(code = error.code, "the dictation was not cancelled");
        }
    }

    /// The tray has no place to show a failure, so a failed copy is only logged.
    pub(crate) async fn copy_last_transcript_from_tray(&self) {
        if let Err(error) = self
            .run_command("copy_last_transcript", Failure::Local, |backend| {
                backend.copy_last_transcript()
            })
            .await
        {
            tracing::warn!(code = error.code, "the last transcript was not copied");
        }
    }

    /// Registration may wait on the event-loop thread, so it runs on a worker and is never
    /// awaited here.
    pub(crate) fn observe_dictation(&self, active: bool) {
        if self.shortcuts.observe_dictation(active) {
            let shortcuts = Arc::clone(&self.shortcuts);
            tauri::async_runtime::spawn_blocking(move || shortcuts.sync_cancel_shortcut());
        }
    }

    /// A registrar failure is a warning; the phase has already changed and is not rolled back.
    pub(super) fn sync_shortcuts(&self) {
        if let Err(error) = self.shortcuts.sync_lifecycle() {
            tracing::warn!(?error, "recording shortcuts could not follow the lifecycle");
            self.publish_error(&shortcut_error(error));
        }
    }

    /// Returns false when shutdown had already begun.
    pub(crate) fn begin_shutdown(&self) -> bool {
        self.shutdown.begin()
    }

    /// The caller must have received `true` from [`Self::begin_shutdown`].
    pub(crate) fn finish_shutdown(&self) {
        self.shortcuts.shutdown();
        self.backend.request_shutdown();
        drop(
            self.commands
                .write()
                .expect("command barrier lock poisoned"),
        );
        self.backend.shutdown();
        self.shutdown.complete();
    }

    pub(crate) fn shutdown_complete(&self) -> bool {
        self.shutdown.is_complete()
    }

    pub(crate) fn shortcuts_enabled(&self) -> bool {
        !self.shutdown.has_begun() && self.backend.state.lock().phase.shortcuts_enabled()
    }
}
