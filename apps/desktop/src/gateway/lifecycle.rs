use std::sync::{Arc, atomic::Ordering};
use whispernow_domain::AppError;

use super::{Failure, Gateway, settings::shortcut_error};
use crate::{
    services::{Phase, RecordingTrigger},
    shortcuts::{ShortcutAction, ShortcutRegistrationError},
};

impl Gateway {
    /// The shortcut backend stopped delivering events. A recording a shortcut started is
    /// cancelled, since its release can no longer arrive; history keeps its audio.
    pub(crate) fn shortcut_capability_lost(&self, error: ShortcutRegistrationError) {
        self.publish_error(&shortcut_error(error));
        self.shortcuts.capability_lost(error);
        let shortcut_recording = matches!(
            self.services.state.lock().phase,
            Phase::Recording {
                trigger: RecordingTrigger::Hold | RecordingTrigger::Toggle,
                ..
            }
        );
        if shortcut_recording && let Err(error) = self.services.dictation.cancel() {
            tracing::debug!(
                code = error.code,
                "the shortcut recording was not cancelled"
            );
        }
    }

    pub(crate) fn shortcut_action(&self, action: ShortcutAction) {
        let sent = match action {
            ShortcutAction::Pressed(trigger) => self.services.dictation.shortcut_pressed(trigger),
            ShortcutAction::HoldReleased => self.services.dictation.hold_released(),
            ShortcutAction::CancelDictation => self.services.dictation.cancel(),
        };
        if let Err(error) = sent {
            tracing::debug!(code = error.code, "a shortcut was ignored");
        }
    }

    /// A press that arrives after the dictation ended has nothing to cancel, so failures are
    /// only logged.
    pub(crate) async fn cancel_dictation(&self) {
        if let Err(error) = self
            .run_command("cancel_dictation", Failure::Local, |services| {
                services.dictation.cancel()
            })
            .await
        {
            tracing::debug!(code = error.code, "the dictation was not cancelled");
        }
    }

    /// The tray offers the copy only where the session can copy, and has no place to show a
    /// failure, so a failed copy is only logged.
    pub(crate) async fn copy_last_transcript_from_tray(&self) {
        if let Err(error) = self
            .run_command("copy_last_transcript", Failure::Local, |services| {
                services.copy_last_transcript()
            })
            .await
        {
            tracing::warn!(code = error.code, "the last transcript was not copied");
        }
    }

    /// Follows a published snapshot: the cancel shortcut is registered while a
    /// dictation runs. Registration may wait on the event-loop thread, so it
    /// runs on a worker and is never awaited here.
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

    /// Refuses new commands and cancels the running ones. Returns false when shutdown had
    /// already begun.
    pub(crate) fn begin_shutdown(&self) -> bool {
        if self.shutdown_begun.swap(true, Ordering::AcqRel) {
            return false;
        }
        self.shutdown.cancel();
        true
    }

    /// Stops shortcuts, waits for the running commands, and releases services.
    /// Runs once, and only after [`Self::begin_shutdown`].
    pub(crate) fn finish_shutdown(&self) {
        if !self.shutdown_begun.load(Ordering::Acquire)
            || self.shutdown_claimed.swap(true, Ordering::AcqRel)
        {
            return;
        }
        self.shortcuts.shutdown();
        self.services.request_shutdown();
        drop(
            self.commands
                .write()
                .expect("command barrier lock poisoned"),
        );
        self.services.shutdown();
        self.shutdown_complete.store(true, Ordering::Release);
    }

    pub(crate) fn shutdown_complete(&self) -> bool {
        self.shutdown_complete.load(Ordering::Acquire)
    }

    pub(crate) fn shortcuts_enabled(&self) -> bool {
        !self.shutdown_begun.load(Ordering::Acquire)
            && self.services.state.lock().phase.shortcuts_enabled()
    }

    pub(crate) fn refuse_during_maintenance(&self) -> Result<(), AppError> {
        self.services.state.refuse_during_maintenance()
    }
}
