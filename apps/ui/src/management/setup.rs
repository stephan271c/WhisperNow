//! First-run setup: checkpoint transitions, the permission watch and the test dictation.

use whispernow_api::{
    ApiErrorDto, AppStateDto, ErrorCategoryDto, ModelAvailability, PermissionKindDto,
    PermissionStatusDto, PermissionsResponse, SaveSettingsResponse, SetupCheckpoint,
    SetupStateView, SetupStep, SetupTestResultDto,
};

use super::ManagementState;
use crate::components::ui::Tone;
use crate::{NoticeScope, client::ClientError};

/// Setup's test dictation, from the request to its result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SetupTest {
    /// Asked for or listening; `active` once a snapshot showed the dictation running.
    Running {
        active: bool,
    },
    Finished(SetupTestResultDto),
}

/// Whether no card can change any more while setup watches: each is granted, unavailable to
/// this session, or refused with nothing setup can do about it.
fn permissions_settled(permissions: &PermissionsResponse) -> bool {
    permissions
        .permissions
        .iter()
        .all(|permission| match permission.status {
            PermissionStatusDto::Granted | PermissionStatusDto::Unavailable => true,
            PermissionStatusDto::Denied => !permission.can_request && !permission.can_open_settings,
            PermissionStatusDto::NotDetermined => false,
        })
}

impl ManagementState {
    pub fn install_setup(&mut self, setup: SetupStateView) {
        self.setup = Some(setup);
    }

    #[must_use]
    pub fn setup_checkpoint(&self) -> Option<&SetupCheckpoint> {
        self.setup.as_ref().map(|setup| &setup.checkpoint)
    }

    pub fn setup_move(&mut self, step: SetupStep) -> Option<SetupCheckpoint> {
        let mut checkpoint = self.setup.as_ref()?.checkpoint.clone();
        checkpoint.step = step;
        Some(checkpoint)
    }

    /// Whether microphones may be listed: listing can ask for microphone access, so setup waits
    /// until it has moved past its permission step.
    #[must_use]
    pub fn may_list_devices(&self) -> bool {
        self.setup.as_ref().is_some_and(|setup| {
            setup.first_run_complete
                || !matches!(
                    setup.checkpoint.step,
                    SetupStep::Welcome | SetupStep::Permissions
                )
        })
    }

    #[must_use]
    pub fn setup_step_ready(&self) -> bool {
        let Some(setup) = self.setup.as_ref() else {
            return false;
        };
        match setup.checkpoint.step {
            // The desktop checks the choices when setup finishes.
            SetupStep::Welcome | SetupStep::TryIt => true,
            SetupStep::Permissions => {
                self.permission_status(PermissionKindDto::Microphone)
                    == Some(PermissionStatusDto::Granted)
            }
            // The download carries on in the background while setup moves on.
            SetupStep::Model => self.setup_model_ready() || self.setup_model_downloading(),
            SetupStep::Shortcuts => setup.checkpoint.shortcuts_ready(),
        }
    }

    /// Whether the checkpoint's model is installed.
    #[must_use]
    pub fn setup_model_ready(&self) -> bool {
        self.setup.as_ref().is_some_and(|setup| {
            self.models.iter().any(|model| {
                model.id == setup.checkpoint.model_id
                    && model.availability != ModelAvailability::NotInstalled
            })
        })
    }

    /// Whether the checkpoint's model is downloading.
    #[must_use]
    pub fn setup_model_downloading(&self) -> bool {
        self.setup
            .as_ref()
            .is_some_and(|setup| self.download(&setup.checkpoint.model_id).is_some())
    }

    /// The saved checkpoint with this window's unsaved choices, for "Finish later".
    #[must_use]
    pub fn finish_setup_later_request(&self) -> Option<SetupCheckpoint> {
        let setup = self.setup.as_ref()?;
        (!setup.first_run_complete).then(|| setup.checkpoint.clone())
    }

    pub fn finish_setup_later(&mut self, response: SetupStateView) {
        self.finish_setup_checkpoint(response);
        self.notify(
            Tone::Info,
            NoticeScope::Setup,
            "Setup saved. Continue it from Home whenever you're ready.",
        );
    }

    pub fn finish_setup_checkpoint(&mut self, response: SetupStateView) {
        self.setup = Some(response);
    }

    /// Installs the fresh checkpoint of setup started again from Settings.
    pub fn restart_setup(&mut self, response: SetupStateView) {
        self.permissions = None;
        self.end_permission_watch();
        self.setup_test = None;
        if let Some(settings) = &mut self.settings {
            settings.settings.first_run_complete = response.first_run_complete;
        }
        self.finish_setup_checkpoint(response);
    }

    #[must_use]
    pub fn complete_setup_request(&self) -> Option<SetupCheckpoint> {
        let setup = self.setup.as_ref()?;
        (!setup.first_run_complete && setup.checkpoint.step == SetupStep::TryIt)
            .then(|| setup.checkpoint.clone())
    }

    pub fn finish_setup(&mut self, saved: SaveSettingsResponse) {
        let SaveSettingsResponse {
            settings: response,
            warnings,
        } = saved;
        if !warnings.is_empty() {
            self.notify_saved(NoticeScope::Setup, "Setup finished", &warnings);
        }
        let first_run_complete = response.settings.first_run_complete;
        self.settings = Some(response);
        if let Some(setup) = &mut self.setup {
            setup.first_run_complete = first_run_complete;
        }
        self.end_permission_watch();
    }

    /// What the desktop reports for `kind`; `None` before the cards load or when this platform
    /// does not need it.
    #[must_use]
    pub fn permission_status(&self, kind: PermissionKindDto) -> Option<PermissionStatusDto> {
        self.permissions
            .as_ref()
            .and_then(|permissions| permissions.get(kind))
            .map(|permission| permission.status)
    }

    /// Installs the permission cards. A watch ends once no card can change any more; when the
    /// microphone is then granted, the returned checkpoint moves setup on to the model step.
    /// A card this session cannot provide does not hold setup back: dictation works without
    /// it.
    pub fn install_permissions(
        &mut self,
        permissions: PermissionsResponse,
    ) -> Option<SetupCheckpoint> {
        let settled = permissions_settled(&permissions)
            && self.permission_watch.is_some()
            && self.on_permissions_step();
        let microphone = permissions
            .get(PermissionKindDto::Microphone)
            .map(|permission| permission.status);
        self.permissions = Some(permissions);
        if !settled {
            return None;
        }
        self.end_permission_watch();
        (microphone == Some(PermissionStatusDto::Granted))
            .then(|| self.setup_move(SetupStep::Model))
            .flatten()
    }

    fn end_permission_watch(&mut self) {
        if let Some(watch) = self.permission_watch.take() {
            watch.cancel();
        }
    }

    fn on_permissions_step(&self) -> bool {
        self.setup.as_ref().is_some_and(|setup| {
            !setup.first_run_complete && setup.checkpoint.step == SetupStep::Permissions
        })
    }

    /// Whether the test dictation has been asked for and has not finished.
    #[must_use]
    pub fn setup_test_running(&self) -> bool {
        matches!(self.setup_test, Some(SetupTest::Running { .. }))
    }

    pub fn begin_setup_test(&mut self) {
        self.setup_test = Some(SetupTest::Running { active: false });
    }

    /// The test could not start, or its stop failed: the failure is its result, unless a
    /// snapshot already reported how the test ended.
    pub fn fail_setup_test(&mut self, error: &ClientError) {
        if !self.setup_test_running() {
            return;
        }
        self.setup_test = Some(SetupTest::Finished(match error {
            ClientError::Api(error) => SetupTestResultDto::Failed {
                error: Box::new(error.clone()),
            },
            ClientError::Transport => SetupTestResultDto::Failed {
                error: Box::new(ApiErrorDto::new(
                    ErrorCategoryDto::General,
                    "desktop_unreachable",
                    "The desktop connection failed",
                )),
            },
        }));
    }

    /// Follows the test dictation through the snapshots: once it has run and the workflow is
    /// idle again, the snapshot's setup-test result is its result, however it was stopped. A
    /// test that ended without one, such as one interrupted by shutdown, reads as cancelled.
    pub(super) fn observe_setup_test(&mut self, snapshot: &AppStateDto) {
        let Some(SetupTest::Running { active }) = &mut self.setup_test else {
            return;
        };
        if snapshot.phase.is_busy() {
            *active = true;
            return;
        }
        if !*active {
            return;
        }
        let outcome = snapshot
            .setup_test
            .clone()
            .unwrap_or(SetupTestResultDto::Cancelled);
        self.setup_test = Some(SetupTest::Finished(outcome));
    }
}
