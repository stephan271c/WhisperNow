use std::collections::BTreeSet;

use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordingMode {
    #[default]
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
/// `KeyboardEvent.code` the settings window captures, the label it shows and the XKB keysym a
/// desktop portal takes.
macro_rules! shortcut_keys {
    ($($variant:ident => $name:literal, $code:literal, $label:literal, $keysym:literal;)*) => {
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

        }
    };
}

shortcut_keys! {
    A => "a", "KeyA", "A", "a";
    B => "b", "KeyB", "B", "b";
    C => "c", "KeyC", "C", "c";
    D => "d", "KeyD", "D", "d";
    E => "e", "KeyE", "E", "e";
    F => "f", "KeyF", "F", "f";
    G => "g", "KeyG", "G", "g";
    H => "h", "KeyH", "H", "h";
    I => "i", "KeyI", "I", "i";
    J => "j", "KeyJ", "J", "j";
    K => "k", "KeyK", "K", "k";
    L => "l", "KeyL", "L", "l";
    M => "m", "KeyM", "M", "m";
    N => "n", "KeyN", "N", "n";
    O => "o", "KeyO", "O", "o";
    P => "p", "KeyP", "P", "p";
    Q => "q", "KeyQ", "Q", "q";
    R => "r", "KeyR", "R", "r";
    S => "s", "KeyS", "S", "s";
    T => "t", "KeyT", "T", "t";
    U => "u", "KeyU", "U", "u";
    V => "v", "KeyV", "V", "v";
    W => "w", "KeyW", "W", "w";
    X => "x", "KeyX", "X", "x";
    Y => "y", "KeyY", "Y", "y";
    Z => "z", "KeyZ", "Z", "z";
    Digit0 => "0", "Digit0", "0", "0";
    Digit1 => "1", "Digit1", "1", "1";
    Digit2 => "2", "Digit2", "2", "2";
    Digit3 => "3", "Digit3", "3", "3";
    Digit4 => "4", "Digit4", "4", "4";
    Digit5 => "5", "Digit5", "5", "5";
    Digit6 => "6", "Digit6", "6", "6";
    Digit7 => "7", "Digit7", "7", "7";
    Digit8 => "8", "Digit8", "8", "8";
    Digit9 => "9", "Digit9", "9", "9";
    F1 => "f1", "F1", "F1", "F1";
    F2 => "f2", "F2", "F2", "F2";
    F3 => "f3", "F3", "F3", "F3";
    F4 => "f4", "F4", "F4", "F4";
    F5 => "f5", "F5", "F5", "F5";
    F6 => "f6", "F6", "F6", "F6";
    F7 => "f7", "F7", "F7", "F7";
    F8 => "f8", "F8", "F8", "F8";
    F9 => "f9", "F9", "F9", "F9";
    F10 => "f10", "F10", "F10", "F10";
    F11 => "f11", "F11", "F11", "F11";
    F12 => "f12", "F12", "F12", "F12";
    F13 => "f13", "F13", "F13", "F13";
    F14 => "f14", "F14", "F14", "F14";
    F15 => "f15", "F15", "F15", "F15";
    F16 => "f16", "F16", "F16", "F16";
    F17 => "f17", "F17", "F17", "F17";
    F18 => "f18", "F18", "F18", "F18";
    F19 => "f19", "F19", "F19", "F19";
    F20 => "f20", "F20", "F20", "F20";
    F21 => "f21", "F21", "F21", "F21";
    F22 => "f22", "F22", "F22", "F22";
    F23 => "f23", "F23", "F23", "F23";
    F24 => "f24", "F24", "F24", "F24";
    Space => "space", "Space", "Space", "space";
    Enter => "enter", "Enter", "Enter", "Return";
    Tab => "tab", "Tab", "Tab", "Tab";
    Escape => "escape", "Escape", "Esc", "Escape";
    Backspace => "backspace", "Backspace", "Backspace", "BackSpace";
    Delete => "delete", "Delete", "Delete", "Delete";
    Insert => "insert", "Insert", "Insert", "Insert";
    Home => "home", "Home", "Home", "Home";
    End => "end", "End", "End", "End";
    PageUp => "pageup", "PageUp", "Page Up", "Page_Up";
    PageDown => "pagedown", "PageDown", "Page Down", "Page_Down";
    ArrowUp => "arrowup", "ArrowUp", "Up", "Up";
    ArrowDown => "arrowdown", "ArrowDown", "Down", "Down";
    ArrowLeft => "arrowleft", "ArrowLeft", "Left", "Left";
    ArrowRight => "arrowright", "ArrowRight", "Right", "Right";
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
