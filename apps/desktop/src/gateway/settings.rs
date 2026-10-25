use std::sync::Arc;
use whispernow_domain::{AppError, ErrorCategory};

use whispernow_domain::ipc::{
    ClearDataResult, SaveSettingsResponse, SetApiKeyRequest, SettingsView,
};
use whispernow_domain::{
    ClearDataCategory, LlmProvider, RecordingMode, SettingChange, Settings, SetupCheckpoint,
    Shortcut,
};

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
        Ok(self.finish_view(view).await)
    }

    /// The UI replaces its whole view with each one it receives, so every view sent to it passes
    /// through here.
    pub(super) async fn finish_view(&self, mut view: SettingsView) -> SettingsView {
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
                Some(ShortcutRegistrationError::ConflictOrUnsupported) | None => {
                    "A configured recording shortcut is unavailable; choose another combination"
                }
            };
            view.warnings.push(summary.to_owned());
        }
        if view.settings.cancel_shortcut.is_some()
            && self.shortcuts.cancel_registration_failure().is_some()
        {
            view.warnings.push("The cancel shortcut could not be activated during the last dictation; choose another key, or cancel from the recording bar or the tray".to_owned());
        }
        let registration = Arc::clone(&self.autostart);
        match run_blocking(move || registration.is_enabled())
            .await
            .and_then(|result| result)
        {
            Ok(enabled) => view.start_on_login = enabled,
            Err(error) => view.warnings.push(error.summary),
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
        Ok(self.finish_view(view).await)
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
        Ok(self.finish_view(view).await)
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
        Ok(self.finish_view(view).await)
    }

    pub(crate) async fn restart_setup(&self) -> Result<SettingsView, AppError> {
        let view = self
            .run_command("restart_setup", Failure::Local, |services| {
                services.settings.restart_setup()
            })
            .await?;
        Ok(self.finish_view(view).await)
    }

    /// Registers a new recording shortcut before it is saved and swaps the cancel shortcut
    /// after; the services run the change's other side effects.
    pub(crate) async fn update_settings(
        &self,
        mut change: SettingChange,
    ) -> Result<SaveSettingsResponse, AppError> {
        let _native_settings = self.native_settings.lock().await;
        let saved = self
            .saved_settings("update_settings", Failure::Local)
            .await?;
        let swap = match &mut change {
            SettingChange::RecordingShortcut { shortcut, mode } => {
                self.register_shortcuts(&saved, shortcut, mode).await?
            }
            _ => ShortcutSwap::Unchanged,
        };
        let swaps_cancel = matches!(change, SettingChange::CancelShortcut(_));
        let saving = self
            .run_command("update_settings", Failure::Local, move |services| {
                services.update_settings(&change)
            })
            .await;
        let mut response = match saving {
            Ok(saved) => saved,
            Err(error) => {
                if swap == ShortcutSwap::Registered {
                    self.restore_shortcuts(&saved).await;
                }
                return Err(error);
            }
        };
        if swap == ShortcutSwap::Refused {
            response.warnings.push(SHORTCUT_REFUSED.to_owned());
        }
        if swaps_cancel {
            self.apply_cancel_shortcut(cancel_binding(&response.settings.settings))
                .await;
        }
        Ok(SaveSettingsResponse {
            settings: self.finish_view(response.settings).await,
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
        let swap = self
            .register_shortcuts(
                &saved,
                &mut checkpoint.recording_shortcut,
                &mut checkpoint.recording_mode,
            )
            .await?;
        let completing = self
            .run_command("complete_setup", Failure::Published, move |services| {
                services.complete_setup(&checkpoint)
            })
            .await;
        let settings = match completing {
            Ok(settings) => settings,
            Err(error) => {
                if swap == ShortcutSwap::Registered {
                    self.restore_shortcuts(&saved).await;
                }
                return Err(error);
            }
        };
        let warnings = (swap == ShortcutSwap::Refused)
            .then(|| SHORTCUT_REFUSED.to_owned())
            .into_iter()
            .collect();
        Ok(SaveSettingsResponse {
            settings: self.finish_view(settings).await,
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

    /// Registers the chosen recording `shortcut` and `mode` in place of `saved`'s, before they
    /// are saved. A refusal leaves `saved`'s registered and writes them back. Shortcuts that
    /// break a rule are left to the save, which refuses them; they register nothing.
    async fn register_shortcuts(
        &self,
        saved: &Settings,
        shortcut: &mut Shortcut,
        mode: &mut RecordingMode,
    ) -> Result<ShortcutSwap, AppError> {
        if saved.recording_shortcut == *shortcut && saved.recording_mode == *mode {
            return Ok(ShortcutSwap::Unchanged);
        }
        let chosen = Settings {
            recording_shortcut: shortcut.clone(),
            recording_mode: *mode,
            ..saved.clone()
        };
        let Some(recording) = recording_binding(&chosen) else {
            return Ok(ShortcutSwap::Unchanged);
        };
        match self.apply_shortcuts(recording).await? {
            Ok(()) => Ok(ShortcutSwap::Registered),
            Err(error) => {
                tracing::warn!(?error, "the new recording shortcut was refused");
                shortcut.clone_from(&saved.recording_shortcut);
                *mode = saved.recording_mode;
                Ok(ShortcutSwap::Refused)
            }
        }
    }

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

    async fn restore_shortcuts(&self, saved: &Settings) {
        let Some(recording) = recording_binding(saved) else {
            tracing::warn!("the saved shortcuts break a rule; none is registered again");
            return;
        };
        match self.apply_shortcuts(recording).await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                tracing::warn!(?error, "the saved shortcuts could not be registered again");
            }
            Err(error) => {
                tracing::warn!(
                    code = error.code,
                    "the saved shortcuts could not be registered again"
                );
            }
        }
    }

    /// Registers `recording` in place of the current shortcut, which stays registered on a
    /// refusal. The outer error is a registration worker that stopped.
    async fn apply_shortcuts(
        &self,
        recording: RecordingBinding,
    ) -> Result<Result<(), ShortcutRegistrationError>, AppError> {
        let shortcuts = Arc::clone(&self.shortcuts);
        run_blocking(move || shortcuts.apply_update(recording)).await
    }

    /// The cancel shortcut is registered only while a dictation runs, so a refusal surfaces
    /// then, as a settings diagnostic.
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

    /// A start-on-login registration that cannot be removed is published as an error; the data
    /// is already gone.
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ShortcutSwap {
    Unchanged,
    Registered,
    /// The new shortcut was refused, so the saved one stays registered and is saved again.
    Refused,
}

const SHORTCUT_REFUSED: &str =
    "The recording shortcut could not be activated; the previous shortcut is still in use";

pub(crate) fn shortcut_error(error: ShortcutRegistrationError) -> AppError {
    let code = match error {
        ShortcutRegistrationError::ConflictOrUnsupported => "shortcut_registration_failed",
        ShortcutRegistrationError::PermissionDenied => "wayland_shortcut_permission_denied",
        ShortcutRegistrationError::PortalUnavailable => "wayland_shortcut_portal_unavailable",
        ShortcutRegistrationError::SessionClosed => "wayland_shortcut_session_closed",
    };
    AppError::new(
        ErrorCategory::Settings,
        code,
        "The recording shortcut could not be activated; choose a different combination",
    )
}
