//! Permission facts become the cards and actions setup offers on each platform.

use std::sync::atomic::{AtomicU8, Ordering};
use whispernow_domain::{AppError, ErrorCategory};

use whispernow_domain::ipc::{
    PasteHelperSetup, Permission, PermissionKind, PermissionStatus, PermissionsResponse,
};

use super::NativeServices;

/// The desktop session, which decides which permissions dictation needs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PermissionSession {
    MacOs,
    Windows,
    /// X11 on Linux: the paste keystroke needs the XTEST extension.
    X11,
    /// Native Wayland: shortcuts need portal consent, and so does the paste
    /// keystroke unless the paste helper presses it.
    Wayland,
}

/// Who presses the paste keystroke in a native Wayland session.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum WaylandPaste {
    /// The paste helper's virtual keyboard, which needs no consent.
    Helper,
    /// The `RemoteDesktop` portal, with its consent and the command that sets
    /// the helper up instead.
    Portal {
        consent: PermissionStatus,
        setup_command: Option<String>,
    },
}

#[derive(Clone)]
pub(crate) struct PermissionFacts {
    session: PermissionSession,
    microphone: PermissionStatus,
    accessibility_trusted: bool,
    paste_injection_available: bool,
    global_shortcuts: PermissionStatus,
    wayland_paste: WaylandPaste,
}

pub(crate) fn permission_states(facts: PermissionFacts) -> PermissionsResponse {
    let session = facts.session;
    let settings_page = matches!(
        session,
        PermissionSession::MacOs | PermissionSession::Windows
    );
    let mut permissions = vec![Permission {
        kind: PermissionKind::Microphone,
        status: facts.microphone,
        can_request: true,
        can_open_settings: settings_page,
    }];
    let mut paste_helper_setup = None;
    match session {
        PermissionSession::MacOs => permissions.push(Permission {
            kind: PermissionKind::Accessibility,
            // macOS reports only whether the app is trusted; until it is, the
            // user grants it in System Settings.
            status: if facts.accessibility_trusted {
                PermissionStatus::Granted
            } else {
                PermissionStatus::NotDetermined
            },
            can_request: false,
            can_open_settings: true,
        }),
        PermissionSession::Windows => {}
        PermissionSession::X11 => permissions.push(Permission {
            kind: PermissionKind::PasteSupport,
            status: if facts.paste_injection_available {
                PermissionStatus::Granted
            } else {
                PermissionStatus::Unavailable
            },
            can_request: false,
            can_open_settings: false,
        }),
        PermissionSession::Wayland => {
            permissions.push(portal_card(
                PermissionKind::GlobalShortcuts,
                facts.global_shortcuts,
            ));
            permissions.push(match facts.wayland_paste {
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

impl PermissionFacts {
    pub(super) fn collect(services: &NativeServices, global_shortcuts: PermissionStatus) -> Self {
        let session = if cfg!(target_os = "macos") {
            PermissionSession::MacOs
        } else if cfg!(target_os = "windows") {
            PermissionSession::Windows
        } else if services.output.native_wayland() {
            PermissionSession::Wayland
        } else {
            PermissionSession::X11
        };
        let mut facts = Self {
            session,
            microphone: services.microphone_access.status(),
            accessibility_trusted: false,
            paste_injection_available: false,
            global_shortcuts: PermissionStatus::NotDetermined,
            wayland_paste: WaylandPaste::Helper,
        };
        match session {
            PermissionSession::MacOs => {
                #[cfg(target_os = "macos")]
                {
                    facts.accessibility_trusted = whispernow_platform::accessibility_trusted();
                }
            }
            PermissionSession::Windows => {}
            PermissionSession::X11 => {
                #[cfg(target_os = "linux")]
                {
                    facts.paste_injection_available =
                        whispernow_platform::x11_paste_injection_available();
                }
            }
            PermissionSession::Wayland => {
                facts.global_shortcuts = global_shortcuts;
                if let Some(consent) = services.output.remote_desktop_consent() {
                    facts.wayland_paste = WaylandPaste::Portal {
                        consent,
                        setup_command: services.paste_helper_setup_command(),
                    };
                }
            }
        }
        facts
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
        AppError, MicrophoneAccess, PasteHelperSetup, PermissionFacts, PermissionSession,
        WaylandPaste, permission_states,
    };

    fn facts_on(session: PermissionSession) -> PermissionFacts {
        PermissionFacts {
            session,
            microphone: Status::NotDetermined,
            accessibility_trusted: false,
            paste_injection_available: false,
            global_shortcuts: Status::NotDetermined,
            wayland_paste: WaylandPaste::Portal {
                consent: Status::NotDetermined,
                setup_command: None,
            },
        }
    }

    fn cards(facts: &PermissionFacts) -> Vec<(Kind, Status, bool, bool)> {
        permission_states(facts.clone())
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
        let mut facts = facts_on(PermissionSession::MacOs);
        assert_eq!(
            cards(&facts),
            [
                (Kind::Microphone, Status::NotDetermined, true, true),
                (Kind::Accessibility, Status::NotDetermined, false, true),
            ]
        );
        facts.microphone = Status::Granted;
        facts.accessibility_trusted = true;
        assert!(permission_states(facts).all_granted());
    }

    #[test]
    fn windows_needs_only_the_microphone() {
        let mut facts = facts_on(PermissionSession::Windows);
        facts.microphone = Status::Denied;
        assert_eq!(
            cards(&facts),
            [(Kind::Microphone, Status::Denied, true, true)]
        );
    }

    #[test]
    fn x11_checks_that_the_paste_keystroke_can_be_sent() {
        let mut facts = facts_on(PermissionSession::X11);
        facts.microphone = Status::Granted;
        assert_eq!(
            cards(&facts),
            [
                (Kind::Microphone, Status::Granted, true, false),
                (Kind::PasteSupport, Status::Unavailable, false, false),
            ]
        );
        facts.paste_injection_available = true;
        assert!(permission_states(facts).all_granted());
    }

    #[test]
    fn wayland_asks_each_portal_until_it_answers() {
        let mut facts = facts_on(PermissionSession::Wayland);
        facts.global_shortcuts = Status::Denied;
        assert_eq!(
            cards(&facts),
            [
                (Kind::Microphone, Status::NotDetermined, true, false),
                (Kind::GlobalShortcuts, Status::Denied, true, false),
                (Kind::RemoteDesktop, Status::NotDetermined, true, false),
            ]
        );
        facts.global_shortcuts = Status::Granted;
        facts.wayland_paste = WaylandPaste::Portal {
            consent: Status::Unavailable,
            setup_command: None,
        };
        let states = permission_states(facts);
        let shortcuts = states.get(Kind::GlobalShortcuts).expect("shortcut card");
        assert!(!shortcuts.can_request, "a granted portal needs no asking");
        let paste = states.get(Kind::RemoteDesktop).expect("paste card");
        assert!(!paste.can_request, "an unavailable portal cannot be asked");
        assert!(!states.all_granted());
    }

    #[test]
    fn wayland_paste_through_the_portal_offers_the_helper_setup() {
        let mut facts = facts_on(PermissionSession::Wayland);
        let command = "sudo sh /setup/install_input_helper.sh".to_owned();
        facts.wayland_paste = WaylandPaste::Portal {
            consent: Status::Granted,
            setup_command: Some(command.clone()),
        };
        assert_eq!(
            permission_states(facts).paste_helper_setup,
            Some(PasteHelperSetup {
                command: Some(command)
            })
        );
    }

    #[test]
    fn the_wayland_paste_helper_needs_no_consent() {
        let mut facts = facts_on(PermissionSession::Wayland);
        facts.microphone = Status::Granted;
        facts.global_shortcuts = Status::Granted;
        facts.wayland_paste = WaylandPaste::Helper;
        let states = permission_states(facts.clone());
        assert_eq!(
            cards(&facts),
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
