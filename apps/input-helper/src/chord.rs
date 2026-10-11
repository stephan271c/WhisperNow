//! Whether a key event releases the watched chord.

use std::collections::{HashMap, HashSet};

// Linux input codes from `linux/input-event-codes.h`.
const KEY_LEFTCTRL: u16 = 29;
const KEY_RIGHTCTRL: u16 = 97;
const KEY_LEFTALT: u16 = 56;
const KEY_RIGHTALT: u16 = 100;
const KEY_LEFTMETA: u16 = 125;
const KEY_RIGHTMETA: u16 = 126;
const MAX_KEY_CODE: u16 = 0x2ff;

/// Ctrl, Alt and Super, left and right. A chord must hold one of these, so the
/// helper can never report plain or shifted typing.
const COMMAND_MODIFIERS: [u16; 6] = [
    KEY_LEFTCTRL,
    KEY_RIGHTCTRL,
    KEY_LEFTALT,
    KEY_RIGHTALT,
    KEY_LEFTMETA,
    KEY_RIGHTMETA,
];
const MAX_GROUPS: usize = 5;
const MAX_ALTERNATIVES: usize = 2;

/// Keys held together, each group satisfied by any one of its key codes (a
/// modifier's left or right key).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Chord {
    groups: Vec<Vec<u16>>,
}

impl Chord {
    /// Parses Linux input codes: `+` joins required groups, `,` joins alternatives.
    /// For example, `29,97+57` is either Ctrl key held with Space.
    pub fn parse(spec: &str) -> Result<Self, String> {
        let invalid = || format!("invalid chord `{spec}`");
        let groups = spec
            .split('+')
            .map(|group| {
                let codes = group
                    .split(',')
                    .map(|code| code.parse::<u16>().map_err(|_| invalid()))
                    .collect::<Result<Vec<_>, _>>()?;
                let valid = !codes.is_empty()
                    && codes.len() <= MAX_ALTERNATIVES
                    && codes.iter().all(|code| (1..=MAX_KEY_CODE).contains(code));
                valid.then_some(codes).ok_or_else(invalid)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let command_modifier = groups
            .iter()
            .any(|group| group.iter().all(|code| COMMAND_MODIFIERS.contains(code)));
        if !(2..=MAX_GROUPS).contains(&groups.len()) || !command_modifier {
            return Err(invalid());
        }
        Ok(Self { groups })
    }

    fn held(&self, down: &HashSet<u16>) -> bool {
        self.groups
            .iter()
            .all(|group| group.iter().any(|code| down.contains(code)))
    }
}

pub struct ChordTracker {
    chord: Option<Chord>,
    down: HashMap<u32, HashSet<u16>>,
    held: bool,
}

impl ChordTracker {
    pub fn new(chord: Option<Chord>) -> Self {
        Self {
            chord,
            down: HashMap::new(),
            held: false,
        }
    }

    fn all_down(&self) -> HashSet<u16> {
        self.down.values().flatten().copied().collect()
    }

    /// Applies one key event from `device` and reports whether it released the chord.
    /// Auto-repeat (`value` 2) changes nothing.
    pub fn key(&mut self, device: u32, code: u16, value: i32) -> bool {
        match value {
            1 => {
                self.down.entry(device).or_default().insert(code);
                let down = self.all_down();
                if self.chord.as_ref().is_some_and(|chord| chord.held(&down)) {
                    self.held = true;
                }
                false
            }
            0 => {
                if let Some(keys) = self.down.get_mut(&device) {
                    keys.remove(&code);
                }
                self.released()
            }
            _ => false,
        }
    }

    pub fn holds_any(&self, codes: &[u16]) -> bool {
        self.down
            .values()
            .flatten()
            .any(|code| codes.contains(code))
    }

    /// Forgets `device`'s keys: it was unplugged or its events overflowed.
    pub fn forget(&mut self, device: u32) -> bool {
        self.down.remove(&device);
        self.released()
    }

    fn released(&mut self) -> bool {
        let down = self.all_down();
        let released = self.held && !self.chord.as_ref().is_some_and(|chord| chord.held(&down));
        if released {
            self.held = false;
        }
        released
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPACE: u16 = 57;
    const A: u16 = 30;

    fn tracker() -> ChordTracker {
        ChordTracker::new(Some(Chord::parse("29,97+57").expect("chord")))
    }

    #[test]
    fn chords_need_ctrl_alt_or_super_so_typing_is_never_reported() {
        assert!(Chord::parse("29,97+57").is_ok());
        assert!(Chord::parse("125+42+30").is_ok());
        // Shift with a letter is typing; a key alone is typing.
        assert!(Chord::parse("42,54+30").is_err());
        assert!(Chord::parse("30").is_err());
        // A group mixing a command modifier with a letter is not a modifier.
        assert!(Chord::parse("29,30+57").is_err());
        assert!(Chord::parse("29+57+30+31+32+33").is_err());
        assert!(Chord::parse("29+768").is_err());
        assert!(Chord::parse("zz=29+57").is_err());
    }

    #[test]
    fn releasing_either_key_of_a_held_chord_reports_it_once() {
        let mut keys = tracker();
        assert!(!keys.key(0, KEY_LEFTCTRL, 1));
        assert!(!keys.key(0, SPACE, 1));
        assert!(!keys.key(0, SPACE, 2));
        assert!(keys.key(0, KEY_LEFTCTRL, 0));
        assert!(!keys.key(0, SPACE, 0));

        keys.key(0, KEY_LEFTCTRL, 1);
        keys.key(0, SPACE, 1);
        assert!(keys.key(0, SPACE, 0));
        assert!(!keys.key(0, KEY_LEFTCTRL, 0));
    }

    #[test]
    fn keys_released_without_the_whole_chord_held_report_nothing() {
        let mut keys = tracker();
        keys.key(0, KEY_LEFTCTRL, 1);
        keys.key(0, A, 1);
        assert!(!keys.key(0, A, 0));
        assert!(!keys.key(0, KEY_LEFTCTRL, 0));
        keys.key(0, SPACE, 1);
        assert!(!keys.key(0, SPACE, 0));
    }

    #[test]
    fn either_side_and_any_keyboard_hold_a_modifier() {
        let mut keys = tracker();
        keys.key(0, KEY_LEFTCTRL, 1);
        keys.key(1, KEY_RIGHTCTRL, 1);
        keys.key(1, SPACE, 1);
        assert!(!keys.key(0, KEY_LEFTCTRL, 0));
        assert!(keys.forget(1));
    }
}
