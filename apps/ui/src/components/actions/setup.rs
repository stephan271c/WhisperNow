use dioxus::core::spawn_forever;
use dioxus::prelude::*;
use whispernow_domain::AppError;
use whispernow_domain::ipc::{PermissionKind, PermissionStatus, PermissionsResponse, SettingsView};
use whispernow_domain::{SetupCheckpoint, SetupStep};

use crate::UiContext;
use crate::client;
use crate::management::{BackgroundRead, ManagementState, SetupTest, replace_task};
use crate::notice::NoticeScope;
use crate::operations::UiOperation;

const PERMISSION_POLL_MS: u32 = 1_000;

pub(crate) const PERMISSION_POLLS: u32 = 120;

/// Whether no card can change any more while a watch runs: each is granted, unavailable to
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
                Ok(saved) => {
                    actions.management.write().finish_setup(saved);
                    actions.end_permission_watch();
                }
                Err(error) => actions.report(NoticeScope::Setup, &error),
            },
        );
    }

    pub(crate) fn restart_setup(self) {
        self.run(
            UiOperation::RestartSetup,
            client::call("restart_setup"),
            move |mut actions, answer| match answer {
                Ok(view) => {
                    actions.management.write().restart_setup(view);
                    actions.end_permission_watch();
                }
                Err(error) => actions.report(NoticeScope::Setup, &error),
            },
        );
    }

    pub(crate) fn list_devices_once_allowed(self) {
        let allowed = {
            let management = self.management;
            let management = management.peek();
            management.may_list_devices() && management.devices.is_none()
        };
        if allowed {
            self.list_devices();
        }
    }

    pub(super) fn list_devices(self) {
        self.read_once(
            BackgroundRead::Devices,
            client::call("list_audio_devices"),
            move |actions, answer| match answer {
                Ok(devices) => {
                    let mut management = actions.management;
                    management.write().devices = Some(devices);
                }
                Err(error) => actions.report(NoticeScope::Global, &error),
            },
        );
    }

    /// Reads the permission cards unless a read is already waiting for its answer. A read
    /// that fails is followed by the next poll.
    pub(crate) fn load_permissions(self) {
        self.read_once(
            BackgroundRead::Permissions,
            client::call("get_permissions"),
            |actions, answer| {
                if let Ok(permissions) = answer {
                    actions.install_permissions(permissions);
                }
            },
        );
    }

    /// Installs the cards unless they are shown already. A watch ends once no card can change
    /// any more, and may then move setup on.
    pub(crate) fn install_permissions(mut self, permissions: PermissionsResponse) {
        let watch_settled = permissions_settled(&permissions) && self.end_permission_watch();
        if self.management.peek().permissions.as_ref() != Some(&permissions) {
            self.management.write().permissions = Some(permissions);
        }
        if !watch_settled {
            return;
        }
        let request = self.management.peek().settled_permissions_move();
        if let Some(request) = request {
            self.save_checkpoint(request, ManagementState::install_setup);
        }
    }

    /// Returns whether a watch was running.
    fn end_permission_watch(mut self) -> bool {
        let Some(watch) = self.background.write().permission_watch.take() else {
            return false;
        };
        watch.cancel();
        true
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

    /// Installs the paste helper through the desktop's password dialog; the app then restarts.
    /// A failure is explained in `scope`.
    pub(crate) fn set_up_paste_helper(self, scope: NoticeScope) {
        self.run_reporting(
            UiOperation::SetUpPasteHelper,
            client::call("set_up_paste_helper"),
            scope,
        );
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
            self.background.write().permission_watch = None;
        });
        replace_task(&mut self.background.write().permission_watch, Some(watch));
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
