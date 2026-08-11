//! The paste keystroke the focused application expects.

/// The paste keystroke the focused application expects.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PasteKeys {
    /// `Ctrl+V`.
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
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::PasteKeys;

    #[test]
    fn terminals_are_recognized_by_application_name() {
        for terminal in [
            "gnome-terminal-server",
            "Ptyxis",
            "org.gnome.Ptyxis",
            "org.gnome.Console",
            "com.mitchellh.ghostty",
            "kitty",
            "org.wezfurlong.wezterm",
            "xfce4-terminal",
            "XTerm",
        ] {
            assert_eq!(
                PasteKeys::for_application(terminal),
                PasteKeys::ControlShiftV,
                "{terminal}"
            );
        }
        for application in ["code", "obsidian", "org.gnome.Nautilus", "firefox", ""] {
            assert_eq!(
                PasteKeys::for_application(application),
                PasteKeys::ControlV,
                "{application}"
            );
        }
    }
}
