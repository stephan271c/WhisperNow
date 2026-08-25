//! The first-run permission cards. Each platform needs a different set: a
//! [`PermissionProbe`] answers what the system reports, and
//! [`permission_states`] turns those answers into the cards setup shows and the
//! actions this platform allows for each.

use std::sync::atomic::{AtomicU8, Ordering};

use whispernow_api::{
    ApiErrorDto, PasteHelperSetupDto, PermissionDto, PermissionKindDto, PermissionStatusDto,
    PermissionsResponse,
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
        consent: PermissionStatusDto,
        setup_command: Option<String>,
    },
}

/// What the system reports for each permission. Only the answers the session
/// needs are asked for, so a probe never touches another platform's API.
pub(crate) trait PermissionProbe {
    fn session(&self) -> PermissionSession;
    fn microphone(&self) -> PermissionStatusDto;
    /// Whether macOS trusts the app to send the paste keystroke.
    fn accessibility_trusted(&self) -> bool;
    /// Whether the X server accepts synthetic keystrokes.
    fn paste_injection_available(&self) -> bool;
    fn global_shortcuts_portal(&self) -> PermissionStatusDto;
    fn wayland_paste(&self) -> WaylandPaste;
}

/// The permission cards for the probe's session, in the order setup shows them.
pub(crate) fn permission_states(probe: &dyn PermissionProbe) -> PermissionsResponse {
    let session = probe.session();
    let settings_page = matches!(
        session,
        PermissionSession::MacOs | PermissionSession::Windows
    );
    let mut permissions = vec![PermissionDto {
        kind: PermissionKindDto::Microphone,
        status: probe.microphone(),
        // Listening briefly is how every platform asks for the microphone.
        can_request: true,
        can_open_settings: settings_page,
    }];
    let mut paste_helper_setup = None;
    match session {
        PermissionSession::MacOs => permissions.push(PermissionDto {
            kind: PermissionKindDto::Accessibility,
            // macOS reports only whether the app is trusted; until it is, the
            // user grants it in System Settings.
            status: if probe.accessibility_trusted() {
                PermissionStatusDto::Granted
            } else {
                PermissionStatusDto::NotDetermined
            },
            can_request: false,
            can_open_settings: true,
        }),
        PermissionSession::Windows => {}
        PermissionSession::X11 => permissions.push(PermissionDto {
            kind: PermissionKindDto::PasteSupport,
            status: if probe.paste_injection_available() {
                PermissionStatusDto::Granted
            } else {
                PermissionStatusDto::Unavailable
            },
            can_request: false,
            can_open_settings: false,
        }),
        PermissionSession::Wayland => {
            permissions.push(portal_card(
                PermissionKindDto::GlobalShortcuts,
                probe.global_shortcuts_portal(),
            ));
            permissions.push(match probe.wayland_paste() {
                WaylandPaste::Helper => PermissionDto {
                    kind: PermissionKindDto::PasteSupport,
                    status: PermissionStatusDto::Granted,
                    can_request: false,
                    can_open_settings: false,
                },
                WaylandPaste::Portal {
                    consent,
                    setup_command,
                } => {
                    paste_helper_setup = Some(PasteHelperSetupDto {
                        command: setup_command,
                    });
                    portal_card(PermissionKindDto::RemoteDesktop, consent)
                }
            });
        }
    }
    PermissionsResponse {
        permissions,
        paste_helper_setup,
    }
}

fn portal_card(kind: PermissionKindDto, status: PermissionStatusDto) -> PermissionDto {
    PermissionDto {
        kind,
        status,
        // An unavailable portal cannot be asked; a granted one needs no asking.
        can_request: matches!(
            status,
            PermissionStatusDto::NotDetermined | PermissionStatusDto::Denied
        ),
        can_open_settings: false,
    }
}

/// The running system, as the desktop services see it.
pub(super) struct SystemPermissions<'a> {
    pub(super) services: &'a NativeServices,
    pub(super) global_shortcuts: PermissionStatusDto,
}

impl PermissionProbe for SystemPermissions<'_> {
    fn session(&self) -> PermissionSession {
        if cfg!(target_os = "macos") {
            PermissionSession::MacOs
        } else if cfg!(target_os = "windows") {
            PermissionSession::Windows
        } else if self.services.output.native_wayland() {
            PermissionSession::Wayland
        } else {
            PermissionSession::X11
        }
    }

    fn microphone(&self) -> PermissionStatusDto {
        self.services.microphone_access.status()
    }

    fn accessibility_trusted(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            whispernow_platform::accessibility_trusted()
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }

    fn paste_injection_available(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            whispernow_platform::x11_paste_injection_available()
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    fn global_shortcuts_portal(&self) -> PermissionStatusDto {
        self.global_shortcuts
    }

    fn wayland_paste(&self) -> WaylandPaste {
        match self.services.output.remote_desktop_consent() {
            None => WaylandPaste::Helper,
            Some(consent) => WaylandPaste::Portal {
                consent,
                setup_command: self.services.paste_helper_setup_command(),
            },
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

    pub(crate) fn status(&self) -> PermissionStatusDto {
        match self.0.load(Ordering::Acquire) {
            Self::GRANTED => PermissionStatusDto::Granted,
            Self::DENIED => PermissionStatusDto::Denied,
            _ => PermissionStatusDto::NotDetermined,
        }
    }

    /// Records what a capture attempt showed. A failure other than a refused
    /// permission, such as a missing device, says nothing about permission.
    pub(crate) fn observe<T>(&self, result: &Result<T, ApiErrorDto>) {
        let observed = match result {
            Ok(_) => Self::GRANTED,
            Err(error) if error.code == "microphone_permission_denied" => Self::DENIED,
            Err(_) => return,
        };
        self.0.store(observed, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use whispernow_api::{
        ErrorCategoryDto, PermissionKindDto as Kind, PermissionStatusDto as Status,
    };

    use super::{
        ApiErrorDto, MicrophoneAccess, PasteHelperSetupDto, PermissionProbe, PermissionSession,
        WaylandPaste, permission_states,
    };

    struct FakeProbe {
        session: PermissionSession,
        microphone: Status,
        trusted: bool,
        injection: bool,
        shortcuts: Status,
        paste: WaylandPaste,
    }

    impl FakeProbe {
        const fn on(session: PermissionSession) -> Self {
            Self {
                session,
                microphone: Status::NotDetermined,
                trusted: false,
                injection: false,
                shortcuts: Status::NotDetermined,
                paste: WaylandPaste::Portal {
                    consent: Status::NotDetermined,
                    setup_command: None,
                },
            }
        }
    }

    impl PermissionProbe for FakeProbe {
        fn session(&self) -> PermissionSession {
            self.session
        }
        fn microphone(&self) -> Status {
            self.microphone
        }
        fn accessibility_trusted(&self) -> bool {
            assert_eq!(self.session, PermissionSession::MacOs, "macOS only");
            self.trusted
        }
        fn paste_injection_available(&self) -> bool {
            assert_eq!(self.session, PermissionSession::X11, "X11 only");
            self.injection
        }
        fn global_shortcuts_portal(&self) -> Status {
            assert_eq!(self.session, PermissionSession::Wayland, "Wayland only");
            self.shortcuts
        }
        fn wayland_paste(&self) -> WaylandPaste {
            assert_eq!(self.session, PermissionSession::Wayland, "Wayland only");
            self.paste.clone()
        }
    }

    /// Each card as (kind, status, can request, can open settings).
    fn cards(probe: &FakeProbe) -> Vec<(Kind, Status, bool, bool)> {
        permission_states(probe)
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
        let mut probe = FakeProbe::on(PermissionSession::MacOs);
        assert_eq!(
            cards(&probe),
            [
                (Kind::Microphone, Status::NotDetermined, true, true),
                (Kind::Accessibility, Status::NotDetermined, false, true),
            ]
        );
        probe.microphone = Status::Granted;
        probe.trusted = true;
        assert!(permission_states(&probe).all_granted());
    }

    #[test]
    fn windows_needs_only_the_microphone() {
        let mut probe = FakeProbe::on(PermissionSession::Windows);
        probe.microphone = Status::Denied;
        assert_eq!(
            cards(&probe),
            [(Kind::Microphone, Status::Denied, true, true)]
        );
    }

    #[test]
    fn x11_checks_that_the_paste_keystroke_can_be_sent() {
        let mut probe = FakeProbe::on(PermissionSession::X11);
        probe.microphone = Status::Granted;
        assert_eq!(
            cards(&probe),
            [
                (Kind::Microphone, Status::Granted, true, false),
                (Kind::PasteSupport, Status::Unavailable, false, false),
            ]
        );
        probe.injection = true;
        assert!(permission_states(&probe).all_granted());
    }

    #[test]
    fn wayland_asks_each_portal_until_it_answers() {
        let mut probe = FakeProbe::on(PermissionSession::Wayland);
        probe.shortcuts = Status::Denied;
        assert_eq!(
            cards(&probe),
            [
                (Kind::Microphone, Status::NotDetermined, true, false),
                (Kind::GlobalShortcuts, Status::Denied, true, false),
                (Kind::RemoteDesktop, Status::NotDetermined, true, false),
            ]
        );
        probe.shortcuts = Status::Granted;
        probe.paste = WaylandPaste::Portal {
            consent: Status::Unavailable,
            setup_command: None,
        };
        let states = permission_states(&probe);
        let shortcuts = states.get(Kind::GlobalShortcuts).expect("shortcut card");
        assert!(!shortcuts.can_request, "a granted portal needs no asking");
        let paste = states.get(Kind::RemoteDesktop).expect("paste card");
        assert!(!paste.can_request, "an unavailable portal cannot be asked");
        assert!(!states.all_granted());
    }

    #[test]
    fn wayland_paste_through_the_portal_offers_the_helper_setup() {
        let mut probe = FakeProbe::on(PermissionSession::Wayland);
        let command = "sudo sh /setup/install_input_helper.sh".to_owned();
        probe.paste = WaylandPaste::Portal {
            consent: Status::Granted,
            setup_command: Some(command.clone()),
        };
        assert_eq!(
            permission_states(&probe).paste_helper_setup,
            Some(PasteHelperSetupDto {
                command: Some(command)
            })
        );
    }

    #[test]
    fn the_wayland_paste_helper_needs_no_consent() {
        let mut probe = FakeProbe::on(PermissionSession::Wayland);
        probe.microphone = Status::Granted;
        probe.shortcuts = Status::Granted;
        probe.paste = WaylandPaste::Helper;
        let states = permission_states(&probe);
        assert_eq!(
            cards(&probe),
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
        let error = |code: &str| ApiErrorDto {
            category: ErrorCategoryDto::Audio,
            code: code.to_owned(),
            summary: String::new(),
            recovery: None,
        };
        let access = MicrophoneAccess::default();
        assert_eq!(access.status(), Status::NotDetermined);
        access.observe(&Err::<(), _>(error("microphone_permission_denied")));
        assert_eq!(access.status(), Status::Denied);
        access.observe(&Err::<(), _>(error("audio_start")));
        assert_eq!(
            access.status(),
            Status::Denied,
            "a device failure is no answer"
        );
        access.observe(&Ok::<_, ApiErrorDto>(()));
        assert_eq!(access.status(), Status::Granted);
    }
}
