use std::sync::Arc;
use whispernow_domain::AppError;

use whispernow_domain::ipc::{
    ClearDataResult, ErrorCategory, SaveSettingsResponse, SetApiKeyRequest, SettingsView,
};
use whispernow_domain::{ClearDataCategory, LlmProvider, Settings, SetupCheckpoint};

use super::{Failure, Gateway, run_blocking};
use crate::shortcuts::{
    RecordingBinding, ShortcutBinding, ShortcutRegistrationError, cancel_binding, recording_binding,
};

impl Gateway {
    pub(crate) async fn settings_view(&self) -> Result<SettingsView, AppError> {
        let view = self
            .run_command("get_settings", Failure::Published, |services| {
                Ok(services.settings.get())
            })
            .await?;
        Ok(self.finish_view(view))
    }

    /// Adds what the desktop reports now to a view. The UI replaces its whole view with each
    /// one it receives, so every view sent to it passes through here.
    pub(super) fn finish_view(&self, mut view: SettingsView) -> SettingsView {
        if self.shortcuts_enabled() && !self.shortcuts.registration_active() {
            let summary = match self.shortcuts.failure() {
                Some(ShortcutRegistrationError::PermissionDenied) => {
                    "The compositor did not grant one of the configured recording shortcuts"
                }
                Some(
                    ShortcutRegistrationError::PortalUnavailable
                    | ShortcutRegistrationError::SessionClosed,
                ) => {
                    "The compositor shortcut portal is unavailable; tray, settings, and manual recording remain available"
                }
                _ => "A configured recording shortcut is unavailable; choose another combination",
            };
            view.warnings.push(summary.to_owned());
        }
        if view.settings.cancel_shortcut.is_some()
            && self.shortcuts.cancel_registration_failure().is_some()
        {
            view.warnings.push("The cancel shortcut could not be activated during the last dictation; choose another key, or cancel from the recording bar or the tray".to_owned());
        }
        match self.autostart.is_enabled() {
            Some(enabled) => view.start_on_login = enabled,
            None => view
                .warnings
                .push("The OS startup registration could not be inspected".to_owned()),
        }
        view
    }

    pub(crate) async fn set_api_key(
        &self,
        request: SetApiKeyRequest,
    ) -> Result<SettingsView, AppError> {
        let view = self
            .run_command("set_api_key", Failure::Local, move |services| {
                services.settings.set_api_key(&request)
            })
            .await?;
        Ok(self.finish_view(view))
    }

    pub(crate) async fn clear_api_key(
        &self,
        provider: LlmProvider,
    ) -> Result<SettingsView, AppError> {
        let view = self
            .run_command("clear_api_key", Failure::Local, move |services| {
                services.settings.clear_api_key(provider)
            })
            .await?;
        Ok(self.finish_view(view))
    }

    pub(crate) async fn save_setup_checkpoint(
        &self,
        request: SetupCheckpoint,
    ) -> Result<SettingsView, AppError> {
        let view = self
            .run_command(
                "save_setup_checkpoint",
                Failure::Published,
                move |services| services.save_setup_checkpoint(request),
            )
            .await?;
        Ok(self.finish_view(view))
    }

    pub(crate) async fn restart_setup(&self) -> Result<SettingsView, AppError> {
        let view = self
            .run_command("restart_setup", Failure::Local, |services| {
                services.settings.restart_setup()
            })
            .await?;
        Ok(self.finish_view(view))
    }

    pub(crate) async fn update_settings(
        &self,
        request: Settings,
    ) -> Result<SaveSettingsResponse, AppError> {
        let _native_settings = self.native_settings.lock().await;
        let saved = self
            .saved_settings("update_settings", Failure::Local)
            .await?;
        let mut edited = request;
        let (refused, registered) = self.register_shortcuts(&saved, &mut edited).await;
        let saving = self
            .run_command("update_settings", Failure::Local, move |services| {
                services.update_settings(edited)
            })
            .await;
        let (before, mut response) = match saving {
            Ok(saved) => saved,
            Err(error) => {
                if registered {
                    self.restore_shortcuts(&saved).await;
                }
                return Err(error);
            }
        };
        response
            .warnings
            .extend(refused.map(|warning| warning.summary));
        if before.cancel_shortcut != response.settings.settings.cancel_shortcut {
            self.apply_cancel_shortcut(cancel_binding(&response.settings.settings))
                .await;
        }
        Ok(SaveSettingsResponse {
            settings: self.finish_view(response.settings),
            warnings: response.warnings,
        })
    }

    pub(super) async fn complete_setup(
        &self,
        mut checkpoint: SetupCheckpoint,
    ) -> Result<SaveSettingsResponse, AppError> {
        let _native_settings = self.native_settings.lock().await;
        let saved = self
            .saved_settings("complete_setup", Failure::Published)
            .await?;
        let mut chosen = saved.clone();
        chosen
            .recording_shortcut
            .clone_from(&checkpoint.recording_shortcut);
        chosen.recording_mode = checkpoint.recording_mode;
        let (refused, registered) = self.register_shortcuts(&saved, &mut chosen).await;
        checkpoint.recording_shortcut = chosen.recording_shortcut;
        checkpoint.recording_mode = chosen.recording_mode;
        let completing = self
            .run_command("complete_setup", Failure::Published, move |services| {
                services.complete_setup(&checkpoint)
            })
            .await;
        let settings = match completing {
            Ok(settings) => settings,
            Err(error) => {
                if registered {
                    self.restore_shortcuts(&saved).await;
                }
                return Err(error);
            }
        };
        let warnings = refused.map(|warning| warning.summary).into_iter().collect();
        Ok(SaveSettingsResponse {
            settings: self.finish_view(settings),
            warnings,
        })
    }

    async fn saved_settings(
        &self,
        operation: &'static str,
        failure: Failure,
    ) -> Result<Settings, AppError> {
        self.run_command(operation, failure, |services| Ok(services.settings.load()))
            .await
    }

    /// Registers the recording shortcut `chosen` names in place of `saved`'s, before `chosen`
    /// is saved. A refusal leaves `saved`'s registered and writes it into `chosen`. Shortcuts
    /// that break a rule are left to the save, which refuses them unless `saved` broke the
    /// rule too; either way they register nothing. Returns the refusal, and whether
    /// `chosen`'s shortcut replaced `saved`'s.
    async fn register_shortcuts(
        &self,
        saved: &Settings,
        chosen: &mut Settings,
    ) -> (Option<AppError>, bool) {
        if saved.recording_shortcut == chosen.recording_shortcut
            && saved.recording_mode == chosen.recording_mode
        {
            return (None, false);
        }
        let Ok(recording) = recording_binding(chosen) else {
            return (None, false);
        };
        match self.apply_shortcuts(recording).await {
            None => (None, true),
            Some(mut refused) => {
                chosen
                    .recording_shortcut
                    .clone_from(&saved.recording_shortcut);
                chosen.recording_mode = saved.recording_mode;
                "The recording shortcut could not be activated; the previous shortcut is still in use"
                    .clone_into(&mut refused.summary);
                (Some(refused), false)
            }
        }
    }

    /// Suspends or restores global shortcuts while the settings UI captures a
    /// new combination. Serialized with settings saves so a capture never
    /// observes a half-applied shortcut update.
    pub(crate) async fn set_shortcut_capture(&self, capturing: bool) -> Result<(), AppError> {
        let _native_settings = self.native_settings.lock().await;
        let shortcuts = Arc::clone(&self.shortcuts);
        let capture_result = run_blocking(move || {
            if capturing {
                shortcuts.suspend_for_capture()
            } else {
                shortcuts.resume_after_capture()
            }
        })
        .await?;
        if let Err(error) = capture_result {
            // Shortcuts that cannot come back after a capture stay off.
            if !capturing {
                self.publish_error(&shortcut_error(error));
            }
            return Err(shortcut_error(error));
        }
        Ok(())
    }

    /// Registers `saved`'s shortcut again after a save that registered another failed.
    async fn restore_shortcuts(&self, saved: &Settings) {
        let restored = match recording_binding(saved) {
            Ok(recording) => self.apply_shortcuts(recording).await,
            Err(error) => Some(shortcut_error(error)),
        };
        if let Some(error) = restored {
            tracing::warn!(
                code = error.code,
                "the saved shortcuts could not be registered again"
            );
        }
    }

    /// Registers `recording` in place of the current shortcut, which stays registered on a
    /// refusal.
    async fn apply_shortcuts(&self, recording: RecordingBinding) -> Option<AppError> {
        let shortcuts = Arc::clone(&self.shortcuts);
        match run_blocking(move || shortcuts.apply_update(recording)).await {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(shortcut_error(error)),
            Err(error) => Some(error),
        }
    }

    /// Swaps the cancel shortcut. It is registered only while a dictation
    /// runs, so a refusal surfaces then, as a settings diagnostic.
    async fn apply_cancel_shortcut(&self, binding: Option<ShortcutBinding>) {
        let shortcuts = Arc::clone(&self.shortcuts);
        if let Err(error) = run_blocking(move || shortcuts.set_cancel_binding(binding)).await {
            tracing::warn!(
                code = error.code,
                "the cancel shortcut could not be applied"
            );
        }
    }

    pub(crate) async fn set_start_on_login(
        &self,
        enabled: bool,
    ) -> Result<SaveSettingsResponse, AppError> {
        let _native_settings = self.native_settings.lock().await;
        self.apply_autostart(enabled).await?;
        Ok(SaveSettingsResponse {
            settings: self.settings_view().await?,
            warnings: Vec::new(),
        })
    }

    async fn apply_autostart(&self, enabled: bool) -> Result<(), AppError> {
        let registration = Arc::clone(&self.autostart);
        run_blocking(move || registration.set_enabled(enabled)).await?
    }

    /// Clears all application data, including the start-on-login registration. A
    /// registration that cannot be removed is published as an error; the data is
    /// already gone.
    pub(super) async fn clear_data(&self) -> Result<ClearDataResult, AppError> {
        let _native_settings = self.native_settings.lock().await;
        let result = self
            .run_command("clear_data", Failure::Published, |services| {
                services.clear_data(&ClearDataCategory::ALL)
            })
            .await?;
        if let Err(warning) = self.apply_autostart(false).await {
            self.publish_error(&warning);
        }
        Ok(result)
    }
}

pub(crate) fn shortcut_error(error: ShortcutRegistrationError) -> AppError {
    AppError {
        category: ErrorCategory::Settings,
        code: match error {
            ShortcutRegistrationError::ConflictOrUnsupported => "shortcut_registration_failed",
            ShortcutRegistrationError::PermissionDenied => "wayland_shortcut_permission_denied",
            ShortcutRegistrationError::PortalUnavailable => "wayland_shortcut_portal_unavailable",
            ShortcutRegistrationError::SessionClosed => "wayland_shortcut_session_closed",
        }
        .to_owned(),
        summary: "The recording shortcut could not be activated; choose a different combination"
            .to_owned(),
    }
}
