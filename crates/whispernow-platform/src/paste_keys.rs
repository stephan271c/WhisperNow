/// The paste keystroke the focused application expects.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PasteKeys {
    ControlV,
    /// `Ctrl+Shift+V`, for terminals, where `Ctrl+V` types a control character.
    ControlShiftV,
    /// `Shift+Insert`, for the legacy Windows console.
    #[cfg(target_os = "windows")]
    ShiftInsert,
}

impl PasteKeys {
    /// The keys for a Linux application, named by a part of its X11 `WM_CLASS`
    /// or by its AT-SPI name, such as `gnome-terminal-server`,
    /// `org.gnome.Ptyxis` or `org.wezfurlong.wezterm`.
    #[cfg(target_os = "linux")]
    pub(crate) fn for_application(name: &str) -> Self {
        const TERMINALS: [&str; 14] = [
            "alacritty",
            "console",
            "foot",
            "ghostty",
            "kgx",
            "kitty",
            "konsole",
            "ptyxis",
            "terminator",
            "tilix",
            "warp",
            "wezterm",
            "wezterm-gui",
            "xterm",
        ];
        let name = name.to_ascii_lowercase();
        let name = name.rsplit('.').next().unwrap_or_default();
        if name.contains("terminal") || TERMINALS.contains(&name) {
            Self::ControlShiftV
        } else {
            Self::ControlV
        }
    }

    /// The keys for a Windows application, named by its top-level window class.
    #[cfg(target_os = "windows")]
    pub(crate) fn for_window_class(class_name: &[u16]) -> Self {
        match String::from_utf16_lossy(class_name).as_str() {
            "CASCADIA_HOSTING_WINDOW_CLASS" => Self::ControlShiftV,
            "ConsoleWindowClass" => Self::ShiftInsert,
            _ => Self::ControlV,
        }
    }
}

#[cfg(all(test, any(target_os = "linux", target_os = "windows")))]
mod tests {
    use super::PasteKeys;

    #[cfg(target_os = "linux")]
    #[test]
    fn terminals_are_recognized_by_application_name() {
        for (application, expected) in [
            ("org.gnome.Ptyxis", PasteKeys::ControlShiftV),
            ("gnome-terminal-server", PasteKeys::ControlShiftV),
            ("firefox", PasteKeys::ControlV),
        ] {
            assert_eq!(
                PasteKeys::for_application(application),
                expected,
                "{application}"
            );
        }
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn terminals_are_recognized_by_window_class() {
        for (class, expected) in [
            ("CASCADIA_HOSTING_WINDOW_CLASS", PasteKeys::ControlShiftV),
            ("ConsoleWindowClass", PasteKeys::ShiftInsert),
            ("Notepad", PasteKeys::ControlV),
        ] {
            let class_name: Vec<u16> = class.encode_utf16().collect();
            assert_eq!(
                PasteKeys::for_window_class(&class_name),
                expected,
                "{class}"
            );
        }
    }
}
