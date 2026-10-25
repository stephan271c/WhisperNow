use whispernow_domain::AppError;
use whispernow_domain::ipc::{
    AppState, PermissionKind, PermissionStatus, PermissionsResponse, SaveSettingsResponse,
    SettingsView, SetupTestResult,
};
use whispernow_domain::{ModelAvailability, SetupCheckpoint, SetupStep};

use super::{ManagementState, replace_task};
use crate::NoticeScope;
use crate::components::ui::Tone;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SetupTest {
    /// Asked for or listening; `active` once a snapshot showed the dictation running.
    Running {
        active: bool,
    },
    Finished(SetupTestResult),
}

/// Whether no card can change any more while setup watches: each is granted, unavailable to
/// this session, or refused with nothing setup can do about it.
fn permissions_settled(permissions: &PermissionsResponse) -> bool {
    permissions
        .permissions
        .iter()
        .all(|permission| match permission.status {
            PermissionStatus::Granted | PermissionStatus::Unavailable => true,
            PermissionStatus::Denied => !permission.can_request && !permission.can_open_settings,
            PermissionStatus::NotDetermined => false,
        })
}

impl ManagementState {
    /// Applies `edit` to the checkpoint setup shows and returns it; `None` while a settings save
    /// is pending, which leaves the checkpoint as it was.
    pub fn edit_setup(
        &mut self,
        edit: impl FnOnce(&mut SetupCheckpoint),
    ) -> Option<SetupCheckpoint> {
        if self.settings_pending() {
            return None;
        }
        let setup = self
            .setup
            .as_mut()
            .expect("setup controls render after loading");
        edit(setup);
        Some(setup.clone())
    }

    #[must_use]
    pub fn checkpoint(&self) -> SetupCheckpoint {
        self.setup
            .clone()
            .expect("setup controls render after loading")
    }

    #[must_use]
    pub fn setup_move(&self, step: SetupStep) -> SetupCheckpoint {
        SetupCheckpoint {
            step,
            ..self.checkpoint()
        }
    }

    #[must_use]
    pub fn setup_complete(&self) -> bool {
        self.settings
            .as_ref()
            .is_some_and(|view| view.settings.setup.is_complete())
    }

    #[must_use]
    pub fn unfinished_setup_step(&self) -> Option<SetupStep> {
        self.setup
            .as_ref()
            .filter(|_| !self.setup_complete())
            .map(|setup| setup.step)
    }

    /// Whether microphones may be listed: listing can ask for microphone access, so setup waits
    /// until it has moved past its permission step.
    #[must_use]
    pub fn may_list_devices(&self) -> bool {
        self.setup.is_some()
            && !matches!(
                self.unfinished_setup_step(),
                Some(SetupStep::Welcome | SetupStep::Permissions)
            )
    }

    #[must_use]
    pub fn setup_step_ready(&self, model_downloading: bool) -> bool {
        let Some(setup) = self.setup.as_ref() else {
            return false;
        };
        match setup.step {
            // The desktop checks the choices when setup finishes.
            SetupStep::Welcome | SetupStep::TryIt => true,
            SetupStep::Permissions => {
                self.permission_status(PermissionKind::Microphone)
                    == Some(PermissionStatus::Granted)
            }
            // The download carries on in the background while setup moves on.
            SetupStep::Model => self.setup_model_ready() || model_downloading,
            SetupStep::Shortcuts => setup.shortcuts_ready(),
        }
    }

    #[must_use]
    pub fn setup_model_ready(&self) -> bool {
        self.setup.as_ref().is_some_and(|setup| {
            self.models.iter().any(|model| {
                model.id == setup.model_id && model.availability != ModelAvailability::NotInstalled
            })
        })
    }

    pub fn install_setup(&mut self, view: SettingsView) {
        self.setup = Some(
            view.settings
                .setup
                .checkpoint()
                .cloned()
                .unwrap_or_default(),
        );
        self.install_settings(view);
    }

    pub fn finish_setup_later(&mut self, view: SettingsView) {
        self.install_setup(view);
        self.notify(
            Tone::Info,
            NoticeScope::Setup,
            "Setup saved. Continue it from Home whenever you're ready.",
        );
    }

    pub fn restart_setup(&mut self, view: SettingsView) {
        self.permissions = None;
        self.end_permission_watch();
        self.setup_test = None;
        self.install_setup(view);
    }

    pub fn finish_setup(&mut self, saved: SaveSettingsResponse) {
        let SaveSettingsResponse { settings, warnings } = saved;
        if !warnings.is_empty() {
            self.notify_saved(NoticeScope::Setup, "Setup finished", &warnings);
        }
        self.install_settings(settings);
        self.end_permission_watch();
    }

    #[must_use]
    pub fn permission_status(&self, kind: PermissionKind) -> Option<PermissionStatus> {
        self.permissions
            .as_ref()
            .and_then(|permissions| permissions.get(kind))
            .map(|permission| permission.status)
    }

    /// A watch ends once no card can change any more; when the microphone is then granted, the
    /// returned checkpoint moves setup on to the model step. A card this session cannot provide
    /// does not hold setup back: dictation works without it.
    pub fn install_permissions(
        &mut self,
        permissions: PermissionsResponse,
    ) -> Option<SetupCheckpoint> {
        let settled = permissions_settled(&permissions)
            && self.permission_watch.is_some()
            && self.on_permissions_step();
        let microphone = permissions
            .get(PermissionKind::Microphone)
            .map(|permission| permission.status);
        self.permissions = Some(permissions);
        if !settled {
            return None;
        }
        self.end_permission_watch();
        (microphone == Some(PermissionStatus::Granted)).then(|| self.setup_move(SetupStep::Model))
    }

    fn end_permission_watch(&mut self) {
        replace_task(&mut self.permission_watch, None);
    }

    fn on_permissions_step(&self) -> bool {
        self.unfinished_setup_step() == Some(SetupStep::Permissions)
    }

    /// The test could not start, or its stop failed: the failure is its result, unless a
    /// snapshot already reported how the test ended.
    pub fn fail_setup_test(&mut self, error: &AppError) {
        if !matches!(self.setup_test, Some(SetupTest::Running { .. })) {
            return;
        }
        self.setup_test = Some(SetupTest::Finished(SetupTestResult::Failed {
            error: Box::new(error.clone()),
        }));
    }

    /// Follows the test dictation through the snapshots: once it has run and the workflow is
    /// idle again, the snapshot's setup-test result is its result, however it was stopped. A
    /// test that ended without one, such as one interrupted by shutdown, reads as cancelled.
    pub(crate) fn observe_setup_test(&mut self, snapshot: &AppState) {
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
            .unwrap_or(SetupTestResult::Cancelled);
        self.setup_test = Some(SetupTest::Finished(outcome));
    }
}
