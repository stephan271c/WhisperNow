use dioxus::core::spawn_forever;
use dioxus::prelude::*;
use whispernow_domain::AppError;
use whispernow_domain::ipc::{PermissionKind, PermissionsResponse, SetupStateView};
use whispernow_domain::{SetupCheckpoint, SetupStep};

use crate::UiContext;
use crate::client;

use crate::{SetupTest, UiOperation};

const PERMISSION_POLL_MS: u32 = 1_000;

pub(crate) const PERMISSION_POLLS: u32 = 120;

impl UiContext {
    pub(crate) fn move_setup(mut self, step: SetupStep) {
        let request = self.management.write().setup_move(step);
        if let Some(request) = request {
            self.save_setup_checkpoint(request);
        }
    }

    fn save_setup_checkpoint(self, request: SetupCheckpoint) {
        self.save_checkpoint(request, |mut actions, setup| {
            actions.management.write().setup = Some(setup);
        });
    }

    pub(crate) fn finish_setup_later(self) {
        let request = self.management.peek().finish_setup_later_request();
        if let Some(request) = request {
            self.save_checkpoint(request, |mut actions, setup| {
                actions.management.write().finish_setup_later(setup);
            });
        }
    }

    fn save_checkpoint(
        self,
        request: SetupCheckpoint,
        finish: impl FnOnce(Self, SetupStateView) + 'static,
    ) {
        let operation = UiOperation::SaveSetup;
        self.run(
            operation,
            async move { client::invoke("save_setup_checkpoint", Some(&request)).await },
            move |actions, answer| match answer {
                Ok(setup) => {
                    finish(actions, setup);
                    actions.list_devices_once_allowed();
                }
                Err(error) => actions.report(operation, &error),
            },
        );
    }

    pub(crate) fn save_setup_edit(mut self, edit: impl FnOnce(&mut SetupCheckpoint)) {
        if self.management.peek().settings_pending() {
            return;
        }
        self.management.write().edit_setup(edit);
        let request = self
            .management
            .peek()
            .setup_checkpoint()
            .expect("setup is loaded")
            .clone();
        self.save_setup_checkpoint(request);
    }

    pub(crate) fn complete_setup(self) {
        if self.management.peek().settings_pending() {
            return;
        }
        let operation = UiOperation::CompleteSetup;
        let Some(request) = self.management.peek().complete_setup_request() else {
            return;
        };
        self.run(
            operation,
            async move { client::invoke("complete_setup", Some(&request)).await },
            move |mut actions, answer| match answer {
                Ok(saved) => actions.management.write().finish_setup(saved),
                Err(error) => actions.report(operation, &error),
            },
        );
    }

    pub(crate) fn restart_setup(self) {
        if self.management.peek().settings_pending() {
            return;
        }
        let operation = UiOperation::RestartSetup;
        self.run(
            operation,
            client::call("restart_setup"),
            move |mut actions, answer| match answer {
                Ok(setup) => actions.management.write().restart_setup(setup),
                Err(error) => actions.report(operation, &error),
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
        let operation = UiOperation::ListDevices;
        if self.management.peek().is_busy(operation) {
            return;
        }
        self.run(
            operation,
            client::call("list_audio_devices"),
            move |actions, answer| match answer {
                Ok(devices) => {
                    let mut management = actions.management;
                    let mut management = management.write();
                    management.devices = devices;
                    management.devices_listed = true;
                }
                Err(error) => actions.report(operation, &error),
            },
        );
    }

    /// Reads the permission cards unless a read is already waiting for its answer. A read
    /// that fails is followed by the next poll.
    pub(crate) fn load_permissions(self) {
        let operation = UiOperation::LoadPermissions;
        if self.management.peek().is_busy(operation) {
            return;
        }
        self.run(
            operation,
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
            self.save_setup_checkpoint(request);
        }
    }

    /// Asks for `kind`, then watches the cards until none can change any more.
    pub(crate) fn request_permission(mut self, kind: PermissionKind) {
        let operation = UiOperation::RequestPermission;
        self.management.write().requested_permission = Some(kind);
        self.run(
            operation,
            async move { client::invoke("request_permission", Some(&kind)).await },
            move |mut actions, answer| {
                actions.management.write().requested_permission = None;
                match answer {
                    Ok(permissions) => actions.install_permissions(permissions),
                    Err(error) => actions.report(operation, &error),
                }
            },
        );
        self.watch_permissions();
    }

    /// Opens the system page that grants `kind`, then watches the cards for the grant.
    pub(crate) fn open_permission_settings(self, kind: PermissionKind) {
        let operation = UiOperation::OpenPermissionSettings;
        self.run(
            operation,
            async move { client::invoke::<_, ()>("open_permission_settings", Some(&kind)).await },
            move |actions, answer| {
                if let Err(error) = answer {
                    actions.report(operation, &error);
                }
            },
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
        let previous = self.management.write().permission_watch.replace(watch);
        if let Some(previous) = previous {
            previous.cancel();
        }
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
