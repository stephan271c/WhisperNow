//! Settings commands whose native side effects (shortcuts, start-on-login) follow the save,
//! and the probes run from the settings and Home screens.

use std::sync::Arc;

use whispernow_api::{
    ApiErrorDto, ClearDataCategory, ClearDataRequest, ClearDataResult, ErrorCategoryDto,
    SaveSettingsResponse, Settings, SettingsView, SetupCheckpoint, UpdateSettingsRequest,
};

use whispernow_domain::WindowGeometry;

use super::{Failure, Gateway, run_blocking};
use crate::{
    autostart::{self, AutostartError},
    services::PersistedShortcuts,
    shortcuts::{
        ShortcutBinding, ShortcutBindings, ShortcutRegistrationError, cancel_binding,
        recording_bindings,
    },
};

impl Gateway {
    /// Reads the saved settings and adds warnings for native side effects that are not
    /// currently in effect: an inactive recording shortcut, a cancel shortcut the last
    /// dictation could not register, start-on-login drift, and the last lifecycle failure.
    pub(crate) async fn settings_view(&self) -> Result<SettingsView, ApiErrorDto> {
        let mut view = self
            .run_command("get_settings", Failure::Published, |services, _| {
                services.settings.get()
            })
            .await?;
        if self.shortcuts_enabled() && !self.shortcuts.registration_active() {
            let summary = match self.shortcuts.registration_diagnostic() {
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
            Ok(actual) if actual != view.settings.auto_start_on_login => {
                view.warnings.push("The OS startup registration differs from the saved setting; turn the setting off and on again to repair it".to_owned());
            }
            Err(_) => view
                .warnings
                .push("The OS startup registration could not be inspected".to_owned()),
            Ok(_) => {}
        }
        if let Some(warning) = self.lifecycle_warning() {
            view.warnings.push(warning);
        }
        Ok(view)
    }

    /// Registers the edited recording shortcuts, saves the edit, then applies the other
    /// native side effects of what changed: the cancel shortcut and the start-on-login
    /// registration. A side effect that fails comes back as a warning next to the saved
    /// settings. A refused shortcut leaves the previous one active, and the previous one is
    /// saved in its place, so the saved settings name the shortcuts in effect.
    ///
    /// A refusal belongs to the control that asked, so it never becomes the
    /// application's current problem.
    pub(crate) async fn update_settings(
        &self,
        request: UpdateSettingsRequest,
    ) -> Result<SaveSettingsResponse, ApiErrorDto> {
        let _native_settings = self.native_settings.lock().await;
        let saved = self
            .saved_settings("update_settings", Failure::Local)
            .await?;
        let mut edited = request.settings;
        let (refused, registered) = self.register_shortcuts(&saved, &mut edited).await;
        let saving = self
            .run_command("update_settings", Failure::Local, move |services, _| {
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
        response.warnings.extend(refused);
        let effects = NativeEffects::between(&before, &response.settings.settings);
        response.warnings.extend(
            self.apply_native_settings(&response.settings.settings, effects)
                .await,
        );
        Ok(response)
    }

    /// Finishes setup like a settings save: validate, register the chosen
    /// shortcuts, persist, then apply the saved start-on-login setting.
    pub(super) async fn complete_setup(
        &self,
        mut checkpoint: SetupCheckpoint,
    ) -> Result<SaveSettingsResponse, ApiErrorDto> {
        let _native_settings = self.native_settings.lock().await;
        let saved = self
            .saved_settings("complete_setup", Failure::Published)
            .await?;
        let mut chosen = saved.clone();
        chosen.hold_shortcut.clone_from(&checkpoint.hold_shortcut);
        chosen.hold_shortcut_enabled = checkpoint.hold_shortcut_enabled;
        chosen
            .toggle_shortcut
            .clone_from(&checkpoint.toggle_shortcut);
        let (refused, registered) = self.register_shortcuts(&saved, &mut chosen).await;
        checkpoint.hold_shortcut = chosen.hold_shortcut;
        checkpoint.hold_shortcut_enabled = chosen.hold_shortcut_enabled;
        checkpoint.toggle_shortcut = chosen.toggle_shortcut;
        let completing = self
            .run_command("complete_setup", Failure::Published, move |services, _| {
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
        let mut warnings: Vec<_> = refused.into_iter().collect();
        warnings.extend(
            self.apply_native_settings(&settings.settings, NativeEffects::SETUP)
                .await,
        );
        Ok(SaveSettingsResponse { settings, warnings })
    }

    async fn saved_settings(
        &self,
        operation: &'static str,
        failure: Failure,
    ) -> Result<Settings, ApiErrorDto> {
        self.run_command(operation, failure, |services, _| {
            Ok(services.settings.get()?.settings)
        })
        .await
    }

    /// Registers the recording shortcuts `chosen` names in place of `saved`'s, before
    /// `chosen` is saved. A refusal leaves `saved`'s registered and writes them into
    /// `chosen`. Shortcuts that break a rule are left to the save, which refuses them unless
    /// `saved` broke the rule too; either way they register nothing. Returns the refusal, and
    /// whether `chosen`'s shortcuts replaced `saved`'s.
    async fn register_shortcuts(
        &self,
        saved: &Settings,
        chosen: &mut Settings,
    ) -> (Option<ApiErrorDto>, bool) {
        let previous = PersistedShortcuts::of(saved);
        if previous == PersistedShortcuts::of(chosen) {
            return (None, false);
        }
        let Ok(bindings) = recording_bindings(chosen) else {
            return (None, false);
        };
        match self.apply_shortcuts(Ok(bindings)).await {
            None => (None, true),
            Some(mut refused) => {
                previous.apply_to(chosen);
                "The recording shortcut could not be activated; the previous shortcut is still in use"
                    .clone_into(&mut refused.summary);
                (Some(refused), false)
            }
        }
    }

    /// Suspends or restores global shortcuts while the settings UI captures a
    /// new combination. Serialized with settings saves so a capture never
    /// observes a half-applied shortcut update.
    pub(crate) async fn set_shortcut_capture(&self, capturing: bool) -> Result<(), ApiErrorDto> {
        let _native_settings = self.native_settings.lock().await;
        let shortcuts = Arc::clone(&self.shortcuts);
        let changed = run_blocking(move || {
            if capturing {
                shortcuts.suspend_for_capture()
            } else {
                shortcuts.resume_after_capture()
            }
        })
        .await?;
        if let Err(error) = changed {
            // Shortcuts that cannot come back after a capture stay off.
            if !capturing {
                self.publish_error(&shortcut_error(error));
            }
            return Err(shortcut_error(error));
        }
        Ok(())
    }

    /// Applies the saved `settings`' `effects` other than the recording shortcuts, which are
    /// registered before the save. The caller holds `native_settings`; every native call runs
    /// on a blocking worker without any state lock held. Failures are returned as warnings.
    async fn apply_native_settings(
        &self,
        settings: &Settings,
        effects: NativeEffects,
    ) -> Option<ApiErrorDto> {
        if effects.cancel_shortcut {
            self.apply_cancel_shortcut(cancel_binding(settings)).await;
        }
        if effects.autostart {
            return self.apply_autostart(settings.auto_start_on_login).await;
        }
        None
    }

    /// Registers `saved`'s shortcuts again after a save that registered others failed.
    async fn restore_shortcuts(&self, saved: &Settings) {
        if let Some(error) = self.apply_shortcuts(recording_bindings(saved)).await {
            tracing::warn!(
                code = error.code,
                "the saved shortcuts could not be registered again"
            );
        }
    }

    /// Registers `bindings` in place of the current ones, which stay registered on a refusal.
    async fn apply_shortcuts(
        &self,
        bindings: Result<ShortcutBindings, ShortcutRegistrationError>,
    ) -> Option<ApiErrorDto> {
        let shortcuts = Arc::clone(&self.shortcuts);
        match run_blocking(move || shortcuts.apply_update(bindings)).await {
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

    async fn apply_autostart(&self, desired: bool) -> Option<ApiErrorDto> {
        let registration = Arc::clone(&self.autostart);
        match run_blocking(move || autostart::reconcile(registration.as_ref(), desired)).await {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(autostart_error(error)),
            Err(error) => Some(error),
        }
    }

    /// Clears application data; clearing settings also removes the start-on-login
    /// registration. A registration that cannot be removed is published as an
    /// error; the data is already gone.
    pub(super) async fn clear_data(
        &self,
        request: ClearDataRequest,
    ) -> Result<ClearDataResult, ApiErrorDto> {
        let _native_settings = self.native_settings.lock().await;
        let clears_settings = request.categories.contains(&ClearDataCategory::Settings);
        let result = self
            .run_command("clear_data", Failure::Published, move |services, _| {
                services.clear_data(&request)
            })
            .await?;
        if clears_settings && let Some(warning) = self.apply_autostart(false).await {
            self.publish_error(&warning);
        }
        Ok(result)
    }

    pub(crate) fn save_window_geometry(&self, geometry: WindowGeometry) -> Result<(), ApiErrorDto> {
        self.services.settings.save_window_geometry(geometry)
    }
}

/// The native side effects a saved change needs applied after the save.
#[derive(Clone, Copy)]
struct NativeEffects {
    cancel_shortcut: bool,
    autostart: bool,
}

impl NativeEffects {
    /// Setup chooses only the recording shortcuts, so the cancel shortcut keeps
    /// the binding it already has.
    const SETUP: Self = Self {
        cancel_shortcut: false,
        autostart: true,
    };

    fn between(before: &Settings, after: &Settings) -> Self {
        Self {
            cancel_shortcut: before.cancel_shortcut != after.cancel_shortcut,
            autostart: before.auto_start_on_login != after.auto_start_on_login,
        }
    }
}

pub(crate) fn autostart_error(error: AutostartError) -> ApiErrorDto {
    ApiErrorDto {
        category: ErrorCategoryDto::General,
        code: match error {
            AutostartError::Query => "autostart_query_failed",
            AutostartError::Update => "autostart_update_failed",
            AutostartError::Verification => "autostart_verification_failed",
        }
        .to_owned(),
        summary: "The start-on-login registration could not be synchronized".to_owned(),
        recovery: Some(whispernow_api::RecoveryActionDto::OpenSettings),
    }
}

pub(crate) fn shortcut_error(error: ShortcutRegistrationError) -> ApiErrorDto {
    ApiErrorDto {
        category: ErrorCategoryDto::General,
        code: match error {
            ShortcutRegistrationError::ConflictOrUnsupported => "shortcut_registration_failed",
            ShortcutRegistrationError::PermissionDenied => "wayland_shortcut_permission_denied",
            ShortcutRegistrationError::PortalUnavailable => "wayland_shortcut_portal_unavailable",
            ShortcutRegistrationError::SessionClosed => "wayland_shortcut_session_closed",
        }
        .to_owned(),
        summary: "The recording shortcut could not be activated; choose a different combination"
            .to_owned(),
        recovery: Some(whispernow_api::RecoveryActionDto::OpenSettings),
    }
}
