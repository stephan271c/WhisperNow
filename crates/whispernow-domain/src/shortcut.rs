use std::collections::BTreeSet;

use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordingMode {
    Hold,
    /// Stops on a second press or the stop button.
    Toggle,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ShortcutModifier {
    Control,
    Alt,
    Shift,
    /// The Super, Windows or Command key.
    Meta,
}

impl ShortcutModifier {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Control => "ctrl",
            Self::Alt => "alt",
            Self::Shift => "shift",
            Self::Meta => "meta",
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Control => "Ctrl",
            Self::Alt => "Alt",
            Self::Shift => "Shift",
            Self::Meta => "Meta",
        }
    }

    /// The modifier's name in a desktop portal's shortcut trigger.
    #[must_use]
    pub const fn portal_name(self) -> &'static str {
        match self {
            Self::Control => "CTRL",
            Self::Alt => "ALT",
            Self::Shift => "SHIFT",
            Self::Meta => "LOGO",
        }
    }

    /// The Linux input codes of the left and right keys.
    #[must_use]
    pub const fn evdev(self) -> &'static [u16] {
        match self {
            Self::Control => &[29, 97],
            Self::Alt => &[56, 100],
            Self::Shift => &[42, 54],
            Self::Meta => &[125, 126],
        }
    }

    /// Reads a stored modifier name, including the aliases older files used.
    #[must_use]
    pub fn from_name(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "ctrl" | "control" => Some(Self::Control),
            "alt" => Some(Self::Alt),
            "shift" => Some(Self::Shift),
            "meta" | "super" | "cmd" => Some(Self::Meta),
            _ => None,
        }
    }
}

impl Serialize for ShortcutModifier {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ShortcutModifier {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        Self::from_name(&name).ok_or_else(|| D::Error::custom("unknown shortcut modifier"))
    }
}

/// Declares [`ShortcutKey`] from one row per key: the name settings store, the
/// `KeyboardEvent.code` the settings window captures, the label it shows, the XKB keysym a
/// desktop portal takes and the Linux input codes the input helper watches.
macro_rules! shortcut_keys {
    ($($variant:ident => $name:literal, $code:literal, $label:literal, $keysym:literal, [$($evdev:literal),+];)*) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub enum ShortcutKey {
            $($variant,)*
        }

        impl ShortcutKey {
            pub const ALL: &'static [Self] = &[$(Self::$variant),*];

            #[must_use]
            pub const fn name(self) -> &'static str {
                match self { $(Self::$variant => $name,)* }
            }

            #[must_use]
            pub const fn code(self) -> &'static str {
                match self { $(Self::$variant => $code,)* }
            }

            #[must_use]
            pub const fn label(self) -> &'static str {
                match self { $(Self::$variant => $label,)* }
            }

            #[must_use]
            pub const fn keysym(self) -> &'static str {
                match self { $(Self::$variant => $keysym,)* }
            }

            /// The Linux input codes of the key; a letter or digit is its QWERTY position.
            #[must_use]
            pub const fn evdev(self) -> &'static [u16] {
                match self { $(Self::$variant => &[$($evdev),+],)* }
            }
        }
    };
}

shortcut_keys! {
    A => "a", "KeyA", "A", "a", [30];
    B => "b", "KeyB", "B", "b", [48];
    C => "c", "KeyC", "C", "c", [46];
    D => "d", "KeyD", "D", "d", [32];
    E => "e", "KeyE", "E", "e", [18];
    F => "f", "KeyF", "F", "f", [33];
    G => "g", "KeyG", "G", "g", [34];
    H => "h", "KeyH", "H", "h", [35];
    I => "i", "KeyI", "I", "i", [23];
    J => "j", "KeyJ", "J", "j", [36];
    K => "k", "KeyK", "K", "k", [37];
    L => "l", "KeyL", "L", "l", [38];
    M => "m", "KeyM", "M", "m", [50];
    N => "n", "KeyN", "N", "n", [49];
    O => "o", "KeyO", "O", "o", [24];
    P => "p", "KeyP", "P", "p", [25];
    Q => "q", "KeyQ", "Q", "q", [16];
    R => "r", "KeyR", "R", "r", [19];
    S => "s", "KeyS", "S", "s", [31];
    T => "t", "KeyT", "T", "t", [20];
    U => "u", "KeyU", "U", "u", [22];
    V => "v", "KeyV", "V", "v", [47];
    W => "w", "KeyW", "W", "w", [17];
    X => "x", "KeyX", "X", "x", [45];
    Y => "y", "KeyY", "Y", "y", [21];
    Z => "z", "KeyZ", "Z", "z", [44];
    Digit0 => "0", "Digit0", "0", "0", [11];
    Digit1 => "1", "Digit1", "1", "1", [2];
    Digit2 => "2", "Digit2", "2", "2", [3];
    Digit3 => "3", "Digit3", "3", "3", [4];
    Digit4 => "4", "Digit4", "4", "4", [5];
    Digit5 => "5", "Digit5", "5", "5", [6];
    Digit6 => "6", "Digit6", "6", "6", [7];
    Digit7 => "7", "Digit7", "7", "7", [8];
    Digit8 => "8", "Digit8", "8", "8", [9];
    Digit9 => "9", "Digit9", "9", "9", [10];
    F1 => "f1", "F1", "F1", "F1", [59];
    F2 => "f2", "F2", "F2", "F2", [60];
    F3 => "f3", "F3", "F3", "F3", [61];
    F4 => "f4", "F4", "F4", "F4", [62];
    F5 => "f5", "F5", "F5", "F5", [63];
    F6 => "f6", "F6", "F6", "F6", [64];
    F7 => "f7", "F7", "F7", "F7", [65];
    F8 => "f8", "F8", "F8", "F8", [66];
    F9 => "f9", "F9", "F9", "F9", [67];
    F10 => "f10", "F10", "F10", "F10", [68];
    F11 => "f11", "F11", "F11", "F11", [87];
    F12 => "f12", "F12", "F12", "F12", [88];
    F13 => "f13", "F13", "F13", "F13", [183];
    F14 => "f14", "F14", "F14", "F14", [184];
    F15 => "f15", "F15", "F15", "F15", [185];
    F16 => "f16", "F16", "F16", "F16", [186];
    F17 => "f17", "F17", "F17", "F17", [187];
    F18 => "f18", "F18", "F18", "F18", [188];
    F19 => "f19", "F19", "F19", "F19", [189];
    F20 => "f20", "F20", "F20", "F20", [190];
    F21 => "f21", "F21", "F21", "F21", [191];
    F22 => "f22", "F22", "F22", "F22", [192];
    F23 => "f23", "F23", "F23", "F23", [193];
    F24 => "f24", "F24", "F24", "F24", [194];
    Space => "space", "Space", "Space", "space", [57];
    Enter => "enter", "Enter", "Enter", "Return", [28, 96];
    Tab => "tab", "Tab", "Tab", "Tab", [15];
    Escape => "escape", "Escape", "Esc", "Escape", [1];
    Backspace => "backspace", "Backspace", "Backspace", "BackSpace", [14];
    Delete => "delete", "Delete", "Delete", "Delete", [111];
    Insert => "insert", "Insert", "Insert", "Insert", [110];
    Home => "home", "Home", "Home", "Home", [102];
    End => "end", "End", "End", "End", [107];
    PageUp => "pageup", "PageUp", "Page Up", "Page_Up", [104];
    PageDown => "pagedown", "PageDown", "Page Down", "Page_Down", [109];
    ArrowUp => "arrowup", "ArrowUp", "Up", "Up", [103];
    ArrowDown => "arrowdown", "ArrowDown", "Down", "Down", [108];
    ArrowLeft => "arrowleft", "ArrowLeft", "Left", "Left", [105];
    ArrowRight => "arrowright", "ArrowRight", "Right", "Right", [106];
}

impl ShortcutKey {
    #[must_use]
    pub fn from_name(value: &str) -> Option<Self> {
        let value = value.trim();
        Self::ALL
            .iter()
            .copied()
            .find(|key| key.name().eq_ignore_ascii_case(value))
    }

    #[must_use]
    pub fn from_code(code: &str) -> Option<Self> {
        if code == "NumpadEnter" {
            return Some(Self::Enter);
        }
        Self::ALL.iter().copied().find(|key| key.code() == code)
    }

    #[must_use]
    pub const fn may_stand_alone(self) -> bool {
        matches!(
            self,
            Self::Escape
                | Self::F1
                | Self::F2
                | Self::F3
                | Self::F4
                | Self::F5
                | Self::F6
                | Self::F7
                | Self::F8
                | Self::F9
                | Self::F10
                | Self::F11
                | Self::F12
                | Self::F13
                | Self::F14
                | Self::F15
                | Self::F16
                | Self::F17
                | Self::F18
                | Self::F19
                | Self::F20
                | Self::F21
                | Self::F22
                | Self::F23
                | Self::F24
        )
    }
}

impl Serialize for ShortcutKey {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.name())
    }
}

impl<'de> Deserialize<'de> for ShortcutKey {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        Self::from_name(&name).ok_or_else(|| D::Error::custom("unknown shortcut key"))
    }
}

/// The modifiers iterate in display order.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Shortcut {
    #[serde(default)]
    pub modifiers: BTreeSet<ShortcutModifier>,
    pub key: ShortcutKey,
}

impl Default for Shortcut {
    /// Hold-to-record's default, Ctrl + Space.
    fn default() -> Self {
        Self {
            modifiers: BTreeSet::from([ShortcutModifier::Control]),
            key: ShortcutKey::Space,
        }
    }
}

impl Shortcut {
    #[must_use]
    pub const fn default_cancel() -> Self {
        Self {
            modifiers: BTreeSet::new(),
            key: ShortcutKey::Escape,
        }
    }

    #[must_use]
    pub fn is_recording_shortcut(&self) -> bool {
        !self.modifiers.is_empty()
    }

    /// Whether this can be the cancel shortcut. It is registered only while a dictation
    /// runs, so besides a recording-shortcut combination it may be a key that types no text,
    /// Escape or a function key, alone.
    #[must_use]
    pub fn is_cancel_shortcut(&self) -> bool {
        self.is_recording_shortcut() || self.key.may_stand_alone()
    }
}
