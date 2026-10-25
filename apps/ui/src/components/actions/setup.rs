use dioxus::core::spawn_forever;
use dioxus::prelude::*;
use whispernow_domain::AppError;
use whispernow_domain::ipc::{PermissionKind, PermissionStatus, PermissionsResponse, SettingsView};
use whispernow_domain::{SetupCheckpoint, SetupStep};

use crate::client;
use crate::management::replace_task;
use crate::{ManagementState, UiContext};

use crate::{NoticeScope, SetupTest, UiOperation};

const PERMISSION_POLL_MS: u32 = 1_000;

pub(crate) const PERMISSION_POLLS: u32 = 120;

impl UiContext {
    pub(crate) fn move_setup(self, step: SetupStep) {
        let request = self.management.peek().setup_move(step);
        self.save_checkpoint(request, ManagementState::install_setup);
    }

    pub(crate) fn finish_setup_later(self) {
        let request = self.management.peek().checkpoint();
        self.save_checkpoint(request, ManagementState::finish_setup_later);
    }

    fn save_checkpoint(
        self,
        request: SetupCheckpoint,
        install: impl FnOnce(&mut ManagementState, SettingsView) + 'static,
    ) {
        self.run(
            UiOperation::SaveSetup,
            client::invoke("save_setup_checkpoint", request),
            move |mut actions, answer| match answer {
                Ok(view) => {
                    install(&mut actions.management.write(), view);
                    actions.list_devices_once_allowed();
                }
                Err(error) => actions.report(NoticeScope::Setup, &error),
            },
        );
    }

    /// Applies `edit` and saves it; `false` while a settings save is pending, which leaves the
    /// checkpoint as it was.
    pub(crate) fn save_setup_edit(mut self, edit: impl FnOnce(&mut SetupCheckpoint)) -> bool {
        let Some(request) = self.management.write().edit_setup(edit) else {
            return false;
        };
        self.save_checkpoint(request, ManagementState::install_setup);
        true
    }

    pub(crate) fn complete_setup(self) {
        let request = self.management.peek().checkpoint();
        self.run(
            UiOperation::CompleteSetup,
            client::invoke("complete_setup", request),
            move |mut actions, answer| match answer {
                Ok(saved) => actions.management.write().finish_setup(saved),
                Err(error) => actions.report(NoticeScope::Setup, &error),
            },
        );
    }

    pub(crate) fn restart_setup(self) {
        self.run(
            UiOperation::RestartSetup,
            client::call("restart_setup"),
            move |mut actions, answer| match answer {
                Ok(view) => actions.management.write().restart_setup(view),
                Err(error) => actions.report(NoticeScope::Setup, &error),
            },
        );
    }

    pub(crate) fn list_devices_once_allowed(self) {
        let allowed = {
            let management = self.management;
            let management = management.peek();
            management.may_list_devices() && !management.devices_listed
        };
        if allowed {
            self.list_devices();
        }
    }

    pub(super) fn list_devices(self) {
        self.run_unless_busy(
            UiOperation::ListDevices,
            client::call("list_audio_devices"),
            move |actions, answer| match answer {
                Ok(devices) => {
                    let mut management = actions.management;
                    let mut management = management.write();
                    management.devices = devices;
                    management.devices_listed = true;
                }
                Err(error) => actions.report(NoticeScope::Global, &error),
            },
        );
    }

    /// Reads the permission cards unless a read is already waiting for its answer. A read
    /// that fails is followed by the next poll.
    pub(crate) fn load_permissions(self) {
        self.run_unless_busy(
            UiOperation::LoadPermissions,
            client::call("get_permissions"),
            |actions, answer| {
                if let Ok(permissions) = answer {
                    actions.install_permissions(permissions);
                }
            },
        );
    }

    /// Installs the cards; every card granted after a request moves setup on by itself.
    pub(crate) fn install_permissions(mut self, permissions: PermissionsResponse) {
        let request = self.management.write().install_permissions(permissions);
        if let Some(request) = request {
            self.save_checkpoint(request, ManagementState::install_setup);
        }
    }

    /// Asks for `kind`, then watches the cards until none can change any more. macOS shows one
    /// prompt at a time and drops the other, so Accessibility is asked once the microphone is
    /// answered.
    pub(crate) fn request_permission(self, kind: PermissionKind) {
        self.run(
            UiOperation::RequestPermission(kind),
            client::invoke("request_permission", kind),
            move |actions, answer: Result<PermissionsResponse, AppError>| match answer {
                Ok(permissions) => {
                    let accessibility_next = kind == PermissionKind::Microphone
                        && permissions.permissions.iter().any(|card| {
                            card.kind == PermissionKind::Accessibility
                                && card.can_request
                                && card.status != PermissionStatus::Granted
                        });
                    actions.install_permissions(permissions);
                    if accessibility_next {
                        actions.request_permission(PermissionKind::Accessibility);
                    }
                }
                Err(error) => actions.report(NoticeScope::Setup, &error),
            },
        );
        self.watch_permissions();
    }

    /// Opens the system page that grants `kind`, then watches the cards for the grant.
    pub(crate) fn open_permission_settings(self, kind: PermissionKind) {
        self.run_reporting(
            UiOperation::OpenPermissionSettings,
            client::invoke("open_permission_settings", kind),
            NoticeScope::Setup,
        );
        self.watch_permissions();
    }

    /// Reads the cards about every [`PERMISSION_POLL_MS`] until a read finds none can change
    /// any more, for at most [`PERMISSION_POLLS`] reads; a hidden window reads nothing. A new
    /// watch replaces the last one.
    fn watch_permissions(mut self) {
        let watch = spawn_forever(async move {
            for _ in 0..PERMISSION_POLLS {
                client::delay(PERMISSION_POLL_MS).await;
                if !client::document_hidden() {
                    self.load_permissions();
                }
            }
            self.management.write().permission_watch = None;
        });
        replace_task(&mut self.management.write().permission_watch, Some(watch));
    }

    /// Starts setup's test dictation; the overlay's stop, a shortcut or [`Self::stop_setup_test`]
    /// ends it. The test's own result line explains why it did not run or finish.
    pub(crate) fn start_setup_test(mut self) {
        self.management.write().setup_test = Some(SetupTest::Running { active: false });
        self.run(
            UiOperation::StartSetupTest,
            client::call("start_test_dictation"),
            Self::fail_setup_test,
        );
    }

    pub(crate) fn stop_setup_test(self) {
        self.run(
            UiOperation::StopSetupTest,
            client::call("stop_recording"),
            Self::fail_setup_test,
        );
    }

    fn fail_setup_test(mut self, answer: Result<(), AppError>) {
        if let Err(error) = answer {
            self.management.write().fail_setup_test(&error);
        }
    }
}
