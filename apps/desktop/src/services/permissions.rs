//! Permission facts become the cards and actions setup offers on each platform.

use std::sync::atomic::{AtomicU8, Ordering};
use whispernow_domain::{AppError, ErrorCategory};

use whispernow_domain::ipc::{
    PasteHelperSetup, Permission, PermissionKind, PermissionStatus, PermissionsResponse,
};

use super::NativeServices;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SessionFacts {
    #[cfg_attr(all(not(test), not(target_os = "macos")), expect(dead_code))]
    MacOs { accessibility_trusted: bool },
    #[cfg_attr(all(not(test), not(target_os = "windows")), expect(dead_code))]
    Windows,
    #[cfg_attr(
        all(not(test), any(target_os = "macos", target_os = "windows")),
        expect(dead_code)
    )]
    X11 { paste_injection_available: bool },
    #[cfg_attr(
        all(not(test), any(target_os = "macos", target_os = "windows")),
        expect(dead_code)
    )]
    Wayland {
        global_shortcuts: PermissionStatus,
        paste: WaylandPaste,
    },
}

/// Who presses the paste keystroke in a native Wayland session.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum WaylandPaste {
    /// The paste helper's virtual keyboard, which needs no consent.
    #[cfg_attr(
        all(not(test), any(target_os = "macos", target_os = "windows")),
        expect(dead_code)
    )]
    Helper,
    /// The `RemoteDesktop` portal, with its consent and the command that sets
    /// the helper up instead.
    #[cfg_attr(
        all(not(test), any(target_os = "macos", target_os = "windows")),
        expect(dead_code)
    )]
    Portal {
        consent: PermissionStatus,
        setup_command: Option<String>,
    },
}

pub(crate) fn permission_states(
    microphone: PermissionStatus,
    session: SessionFacts,
) -> PermissionsResponse {
    let settings_page = matches!(session, SessionFacts::MacOs { .. } | SessionFacts::Windows);
    let mut permissions = vec![Permission {
        kind: PermissionKind::Microphone,
        status: microphone,
        can_request: true,
        can_open_settings: settings_page,
    }];
    let mut paste_helper_setup = None;
    match session {
        SessionFacts::MacOs {
            accessibility_trusted,
        } => permissions.push(Permission {
            kind: PermissionKind::Accessibility,
            // macOS reports only whether the app is trusted; until it is, the
            // user grants it in System Settings.
            status: if accessibility_trusted {
                PermissionStatus::Granted
            } else {
                PermissionStatus::NotDetermined
            },
            can_request: false,
            can_open_settings: true,
        }),
        SessionFacts::Windows => {}
        SessionFacts::X11 {
            paste_injection_available,
        } => permissions.push(Permission {
            kind: PermissionKind::PasteSupport,
            status: if paste_injection_available {
                PermissionStatus::Granted
            } else {
                PermissionStatus::Unavailable
            },
            can_request: false,
            can_open_settings: false,
        }),
        SessionFacts::Wayland {
            global_shortcuts,
            paste,
        } => {
            permissions.push(portal_card(
                PermissionKind::GlobalShortcuts,
                global_shortcuts,
            ));
            permissions.push(match paste {
                WaylandPaste::Helper => Permission {
                    kind: PermissionKind::PasteSupport,
                    status: PermissionStatus::Granted,
                    can_request: false,
                    can_open_settings: false,
                },
                WaylandPaste::Portal {
                    consent,
                    setup_command,
                } => {
                    paste_helper_setup = Some(PasteHelperSetup {
                        command: setup_command,
                    });
                    portal_card(PermissionKind::RemoteDesktop, consent)
                }
            });
        }
    }
    PermissionsResponse {
        permissions,
        paste_helper_setup,
    }
}

fn portal_card(kind: PermissionKind, status: PermissionStatus) -> Permission {
    Permission {
        kind,
        status,
        // An unavailable portal cannot be asked; a granted one needs no asking.
        can_request: matches!(
            status,
            PermissionStatus::NotDetermined | PermissionStatus::Denied
        ),
        can_open_settings: false,
    }
}

impl SessionFacts {
    pub(super) fn collect(services: &NativeServices, global_shortcuts: PermissionStatus) -> Self {
        #[cfg(target_os = "macos")]
        {
            let _ = (services, global_shortcuts);
            Self::MacOs {
                accessibility_trusted: whispernow_platform::accessibility_trusted(),
            }
        }
        #[cfg(target_os = "windows")]
        {
            let _ = (services, global_shortcuts);
            Self::Windows
        }
        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        {
            if services.output.native_wayland() {
                let paste = match services.output.remote_desktop_consent() {
                    Some(consent) => WaylandPaste::Portal {
                        consent,
                        setup_command: services.paste_helper_setup_command(),
                    },
                    None => WaylandPaste::Helper,
                };
                Self::Wayland {
                    global_shortcuts,
                    paste,
                }
            } else {
                #[cfg(target_os = "linux")]
                let paste_injection_available =
                    whispernow_platform::x11_paste_injection_available();
                #[cfg(not(target_os = "linux"))]
                let paste_injection_available = false;
                Self::X11 {
                    paste_injection_available,
                }
            }
        }
    }
}

/// The microphone answer the last capture showed. No platform API here reports
/// microphone permission without asking, so the answer comes from listening.
#[derive(Default)]
pub(crate) struct MicrophoneAccess(AtomicU8);

impl MicrophoneAccess {
    const GRANTED: u8 = 1;
    const DENIED: u8 = 2;

    pub(crate) fn status(&self) -> PermissionStatus {
        match self.0.load(Ordering::Acquire) {
            Self::GRANTED => PermissionStatus::Granted,
            Self::DENIED => PermissionStatus::Denied,
            _ => PermissionStatus::NotDetermined,
        }
    }

    /// Records what a capture attempt showed. A failure other than a refused
    /// permission, such as a missing device, says nothing about permission.
    pub(crate) fn observe<T>(&self, result: &Result<T, AppError>) {
        let observed = match result {
            Ok(_) => Self::GRANTED,
            Err(error) if error.category == ErrorCategory::Permission => Self::DENIED,
            Err(_) => return,
        };
        self.0.store(observed, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use whispernow_domain::ipc::{
        ErrorCategory, PermissionKind as Kind, PermissionStatus as Status,
    };

    use super::{
        AppError, MicrophoneAccess, PasteHelperSetup, SessionFacts, WaylandPaste, permission_states,
    };

    fn cards(microphone: Status, session: SessionFacts) -> Vec<(Kind, Status, bool, bool)> {
        permission_states(microphone, session)
            .permissions
            .into_iter()
            .map(|card| {
                (
                    card.kind,
                    card.status,
                    card.can_request,
                    card.can_open_settings,
                )
            })
            .collect()
    }

    #[test]
    fn macos_asks_for_the_microphone_and_accessibility_in_system_settings() {
        assert_eq!(
            cards(
                Status::NotDetermined,
                SessionFacts::MacOs {
                    accessibility_trusted: false,
                },
            ),
            [
                (Kind::Microphone, Status::NotDetermined, true, true),
                (Kind::Accessibility, Status::NotDetermined, false, true),
            ]
        );
        assert!(
            permission_states(
                Status::Granted,
                SessionFacts::MacOs {
                    accessibility_trusted: true,
                },
            )
            .all_granted()
        );
    }

    #[test]
    fn windows_needs_only_the_microphone() {
        assert_eq!(
            cards(Status::Denied, SessionFacts::Windows),
            [(Kind::Microphone, Status::Denied, true, true)]
        );
    }

    #[test]
    fn x11_checks_that_the_paste_keystroke_can_be_sent() {
        assert_eq!(
            cards(
                Status::Granted,
                SessionFacts::X11 {
                    paste_injection_available: false,
                },
            ),
            [
                (Kind::Microphone, Status::Granted, true, false),
                (Kind::PasteSupport, Status::Unavailable, false, false),
            ]
        );
        assert!(
            permission_states(
                Status::Granted,
                SessionFacts::X11 {
                    paste_injection_available: true,
                },
            )
            .all_granted()
        );
    }

    #[test]
    fn wayland_asks_each_portal_until_it_answers() {
        assert_eq!(
            cards(
                Status::NotDetermined,
                SessionFacts::Wayland {
                    global_shortcuts: Status::Denied,
                    paste: WaylandPaste::Portal {
                        consent: Status::NotDetermined,
                        setup_command: None,
                    },
                },
            ),
            [
                (Kind::Microphone, Status::NotDetermined, true, false),
                (Kind::GlobalShortcuts, Status::Denied, true, false),
                (Kind::RemoteDesktop, Status::NotDetermined, true, false),
            ]
        );
        let states = permission_states(
            Status::NotDetermined,
            SessionFacts::Wayland {
                global_shortcuts: Status::Granted,
                paste: WaylandPaste::Portal {
                    consent: Status::Unavailable,
                    setup_command: None,
                },
            },
        );
        let shortcuts = states.get(Kind::GlobalShortcuts).expect("shortcut card");
        assert!(!shortcuts.can_request, "a granted portal needs no asking");
        let paste = states.get(Kind::RemoteDesktop).expect("paste card");
        assert!(!paste.can_request, "an unavailable portal cannot be asked");
        assert!(!states.all_granted());
    }

    #[test]
    fn wayland_paste_through_the_portal_offers_the_helper_setup() {
        let command = "sudo sh /setup/install_input_helper.sh".to_owned();
        assert_eq!(
            permission_states(
                Status::NotDetermined,
                SessionFacts::Wayland {
                    global_shortcuts: Status::NotDetermined,
                    paste: WaylandPaste::Portal {
                        consent: Status::Granted,
                        setup_command: Some(command.clone()),
                    },
                },
            )
            .paste_helper_setup,
            Some(PasteHelperSetup {
                command: Some(command)
            })
        );
    }

    #[test]
    fn the_wayland_paste_helper_needs_no_consent() {
        let session = SessionFacts::Wayland {
            global_shortcuts: Status::Granted,
            paste: WaylandPaste::Helper,
        };
        let states = permission_states(Status::Granted, session.clone());
        assert_eq!(
            cards(Status::Granted, session),
            [
                (Kind::Microphone, Status::Granted, true, false),
                (Kind::GlobalShortcuts, Status::Granted, false, false),
                (Kind::PasteSupport, Status::Granted, false, false),
            ]
        );
        assert!(states.all_granted());
        assert_eq!(states.paste_helper_setup, None);
    }

    #[test]
    fn only_a_capture_result_changes_the_microphone_answer() {
        let error = |category| AppError::new(category, "fixture", "");
        let access = MicrophoneAccess::default();
        assert_eq!(access.status(), Status::NotDetermined);
        access.observe(&Err::<(), _>(error(ErrorCategory::Permission)));
        assert_eq!(access.status(), Status::Denied);
        access.observe(&Err::<(), _>(error(ErrorCategory::Audio)));
        assert_eq!(
            access.status(),
            Status::Denied,
            "a device failure is no answer"
        );
        access.observe(&Ok::<_, AppError>(()));
        assert_eq!(access.status(), Status::Granted);
    }
}
