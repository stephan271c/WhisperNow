//! Which watched chords a key event releases.

use std::collections::{HashMap, HashSet};

/// Ctrl, Alt and Super, left and right. A chord must hold one of these, so the
/// helper can never report plain or shifted typing.
const COMMAND_MODIFIERS: [u16; 6] = [29, 97, 56, 100, 125, 126];
const MAX_CHORDS: usize = 8;
const MAX_GROUPS: usize = 5;
const MAX_ALTERNATIVES: usize = 2;
/// `KEY_MAX` in `linux/input-event-codes.h`.
const MAX_KEY_CODE: u16 = 0x2ff;

/// Keys held together, each group satisfied by any one of its key codes (a
/// modifier's left or right key).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Chord {
    pub id: u32,
    groups: Vec<Vec<u16>>,
}

impl Chord {
    /// Parses `<id hex>=<code>[,<code>]+<code>...`, for example `8003e=29,97+57`.
    pub fn parse(spec: &str) -> Result<Self, String> {
        let invalid = || format!("invalid chord `{spec}`");
        let (id, groups) = spec.split_once('=').ok_or_else(invalid)?;
        let id = u32::from_str_radix(id, 16).map_err(|_| invalid())?;
        let groups = groups
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
        Ok(Self { id, groups })
    }

    fn held(&self, down: &HashSet<u16>) -> bool {
        self.groups
            .iter()
            .all(|group| group.iter().any(|code| down.contains(code)))
    }
}

pub fn parse_chords(specs: &[String]) -> Result<Vec<Chord>, String> {
    if specs.is_empty() || specs.len() > MAX_CHORDS {
        return Err(format!("expected 1 to {MAX_CHORDS} chords"));
    }
    specs.iter().map(|spec| Chord::parse(spec)).collect()
}

/// Key state across every keyboard, and which chords are fully held.
pub struct ChordTracker {
    chords: Vec<Chord>,
    down: HashMap<u32, HashSet<u16>>,
    held: HashSet<u32>,
}

impl ChordTracker {
    pub fn new(chords: Vec<Chord>) -> Self {
        Self {
            chords,
            down: HashMap::new(),
            held: HashSet::new(),
        }
    }

    fn all_down(&self) -> HashSet<u16> {
        self.down.values().flatten().copied().collect()
    }

    /// Applies one key event from `device` and returns the chords it released.
    /// Auto-repeat (`value` 2) changes nothing.
    pub fn key(&mut self, device: u32, code: u16, value: i32) -> Vec<u32> {
        match value {
            1 => {
                self.down.entry(device).or_default().insert(code);
                let down = self.all_down();
                for chord in &self.chords {
                    if chord.held(&down) {
                        self.held.insert(chord.id);
                    }
                }
                Vec::new()
            }
            0 => {
                if let Some(keys) = self.down.get_mut(&device) {
                    keys.remove(&code);
                }
                self.released()
            }
            _ => Vec::new(),
        }
    }

    /// Whether any of `codes` is held on any keyboard.
    pub fn holds_any(&self, codes: &[u16]) -> bool {
        self.down
            .values()
            .flatten()
            .any(|code| codes.contains(code))
    }

    /// Forgets `device`'s keys: it was unplugged or its events overflowed.
    pub fn forget(&mut self, device: u32) -> Vec<u32> {
        self.down.remove(&device);
        self.released()
    }

    fn released(&mut self) -> Vec<u32> {
        let down = self.all_down();
        let released = self
            .chords
            .iter()
            .filter(|chord| self.held.contains(&chord.id) && !chord.held(&down))
            .map(|chord| chord.id)
            .collect::<Vec<_>>();
        for id in &released {
            self.held.remove(id);
        }
        released
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CTRL: u16 = 29;
    const RIGHT_CTRL: u16 = 97;
    const SPACE: u16 = 57;
    const A: u16 = 30;

    fn tracker() -> ChordTracker {
        ChordTracker::new(vec![Chord::parse("2a=29,97+57").expect("chord")])
    }

    #[test]
    fn chords_need_ctrl_alt_or_super_so_typing_is_never_reported() {
        assert!(Chord::parse("1=29,97+57").is_ok());
        assert!(Chord::parse("1=125+42+30").is_ok());
        // Shift with a letter is typing; a key alone is typing.
        assert!(Chord::parse("1=42,54+30").is_err());
        assert!(Chord::parse("1=30").is_err());
        // A group mixing a command modifier with a letter is not a modifier.
        assert!(Chord::parse("1=29,30+57").is_err());
        assert!(Chord::parse("1=29+57+30+31+32+33").is_err());
        assert!(Chord::parse("1=29+768").is_err());
        assert!(Chord::parse("zz=29+57").is_err());
        assert!(parse_chords(&vec!["1=29+57".to_owned(); 9]).is_err());
    }

    #[test]
    fn releasing_either_key_of_a_held_chord_reports_it_once() {
        let mut keys = tracker();
        assert!(keys.key(0, CTRL, 1).is_empty());
        assert!(keys.key(0, SPACE, 1).is_empty());
        assert!(keys.key(0, SPACE, 2).is_empty());
        assert_eq!(keys.key(0, CTRL, 0), [0x2a]);
        assert!(keys.key(0, SPACE, 0).is_empty());

        keys.key(0, CTRL, 1);
        keys.key(0, SPACE, 1);
        assert_eq!(keys.key(0, SPACE, 0), [0x2a]);
        assert!(keys.key(0, CTRL, 0).is_empty());
    }

    #[test]
    fn keys_released_without_the_whole_chord_held_report_nothing() {
        let mut keys = tracker();
        keys.key(0, CTRL, 1);
        keys.key(0, A, 1);
        assert!(keys.key(0, A, 0).is_empty());
        assert!(keys.key(0, CTRL, 0).is_empty());
        keys.key(0, SPACE, 1);
        assert!(keys.key(0, SPACE, 0).is_empty());
    }

    #[test]
    fn either_side_and_any_keyboard_hold_a_modifier() {
        let mut keys = tracker();
        keys.key(0, CTRL, 1);
        keys.key(1, RIGHT_CTRL, 1);
        keys.key(1, SPACE, 1);
        assert!(keys.key(0, CTRL, 0).is_empty());
        assert_eq!(keys.forget(1), [0x2a]);
    }
}
