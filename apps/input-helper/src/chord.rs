//! Tracks a shortcut chord across keyboards.

use std::collections::{HashMap, HashSet};

use crate::{
    codes::{COMMAND_MODIFIERS, MAX_KEY_CODE},
    event::KeyValue,
};

const MAX_GROUPS: usize = 5;
const MAX_ALTERNATIVES: usize = 8;

/// Keys held together, each group satisfied by any one of its key codes (a
/// modifier's left or right key, or a key's positions across keyboard layouts).
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChordTransition {
    Pressed,
    Released,
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

    pub fn can_watch(&self, codes: impl Iterator<Item = u16>) -> bool {
        self.chord
            .as_ref()
            .is_some_and(|chord| chord.held(&codes.collect()))
    }

    pub fn chord_held(&self) -> bool {
        self.held
    }

    /// Auto-repeat changes nothing.
    pub fn key(&mut self, device: u32, code: u16, value: KeyValue) -> Option<ChordTransition> {
        match value {
            KeyValue::Press => {
                self.down.entry(device).or_default().insert(code);
                self.transition()
            }
            KeyValue::Release => {
                if let Some(keys) = self.down.get_mut(&device) {
                    keys.remove(&code);
                }
                self.transition()
            }
            KeyValue::Repeat => None,
        }
    }

    pub fn holds_any(&self, codes: &[u16]) -> bool {
        self.down
            .values()
            .flatten()
            .any(|code| codes.contains(code))
    }

    pub fn forget(&mut self, device: u32) -> Option<ChordTransition> {
        self.down.remove(&device);
        self.transition()
    }

    fn transition(&mut self) -> Option<ChordTransition> {
        let chord = self.chord.as_ref()?;
        let down = self.all_down();
        let held = chord.held(&down);
        let transition = match (self.held, held) {
            (false, true) => Some(ChordTransition::Pressed),
            (true, false) => Some(ChordTransition::Released),
            _ => None,
        };
        self.held = held;
        transition
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        codes::{KEY_LEFTCTRL, KEY_RIGHTCTRL},
        event::KeyValue::{Press, Release, Repeat},
    };

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
        assert!(Chord::parse("29+16,17,18,19,20,21,22,23,24").is_err());
        assert!(Chord::parse("zz=29+57").is_err());
    }

    #[test]
    fn the_same_symbol_can_have_different_positions_in_several_layouts() {
        let chord = Chord::parse("29,97+16,30,45").expect("layout alternatives");
        for code in [16, 30, 45] {
            let mut keys = ChordTracker::new(Some(chord.clone()));
            keys.key(0, KEY_LEFTCTRL, Press);
            keys.key(0, code, Press);
            assert_eq!(keys.key(0, code, Release), Some(ChordTransition::Released));
        }
    }

    #[test]
    fn readable_devices_must_support_every_group_before_monitoring_is_ready() {
        let keys = tracker();
        assert!(!keys.can_watch([272, 273].into_iter()));
        assert!(!keys.can_watch([KEY_LEFTCTRL].into_iter()));
        assert!(keys.can_watch([KEY_RIGHTCTRL, SPACE].into_iter()));
    }

    #[test]
    fn device_arrival_removal_and_reconnection_update_monitoring_and_held_state() {
        let mut keys = tracker();
        let mut supported = HashMap::from([(0, vec![KEY_LEFTCTRL])]);
        assert!(!keys.can_watch(supported.values().flatten().copied()));
        assert_eq!(keys.key(0, KEY_LEFTCTRL, Press), None);

        supported.insert(1, vec![SPACE]);
        assert_eq!(keys.key(1, SPACE, Press), Some(ChordTransition::Pressed));
        assert!(keys.can_watch(supported.values().flatten().copied()));
        assert!(keys.chord_held());
        assert_eq!(keys.key(1, SPACE, Release), Some(ChordTransition::Released));
        assert!(keys.can_watch(supported.values().flatten().copied()));
        assert_eq!(keys.key(1, SPACE, Press), Some(ChordTransition::Pressed));

        supported.remove(&1);
        assert_eq!(keys.forget(1), Some(ChordTransition::Released));
        assert!(!keys.can_watch(supported.values().flatten().copied()));
        assert!(!keys.chord_held());
        assert!(keys.holds_any(&[KEY_LEFTCTRL]));

        supported.insert(2, vec![SPACE]);
        assert_eq!(keys.key(2, SPACE, Press), Some(ChordTransition::Pressed));
        assert!(keys.can_watch(supported.values().flatten().copied()));
        assert!(keys.chord_held());
        assert_eq!(keys.key(2, SPACE, Release), Some(ChordTransition::Released));
    }

    #[test]
    fn dropped_events_clear_held_state_while_the_device_still_supports_the_chord() {
        let mut keys = tracker();
        let supported = [KEY_LEFTCTRL, SPACE];
        keys.key(0, KEY_LEFTCTRL, Press);
        assert_eq!(keys.key(0, SPACE, Press), Some(ChordTransition::Pressed));

        assert_eq!(keys.forget(0), Some(ChordTransition::Released));
        assert!(keys.can_watch(supported.into_iter()));
        assert!(!keys.chord_held());
        assert!(!keys.holds_any(&supported));
        assert_eq!(keys.key(0, SPACE, Repeat), None);

        assert_eq!(keys.key(0, KEY_LEFTCTRL, Press), None);
        assert_eq!(keys.key(0, SPACE, Press), Some(ChordTransition::Pressed));
        assert_eq!(keys.key(0, SPACE, Release), Some(ChordTransition::Released));
    }

    #[test]
    fn paste_modifier_tracking_does_not_need_a_chord() {
        let mut keys = ChordTracker::new(None);
        assert_eq!(keys.key(0, KEY_LEFTCTRL, Press), None);
        assert!(keys.holds_any(&[KEY_LEFTCTRL]));
        assert!(!keys.chord_held());
        assert_eq!(keys.key(0, KEY_LEFTCTRL, Repeat), None);
        assert!(keys.holds_any(&[KEY_LEFTCTRL]));
        assert_eq!(keys.key(1, KEY_RIGHTCTRL, Press), None);
        assert_eq!(keys.key(0, KEY_LEFTCTRL, Release), None);
        assert!(keys.holds_any(&[KEY_LEFTCTRL, KEY_RIGHTCTRL]));
        assert_eq!(keys.forget(0), None);
        assert!(!keys.holds_any(&[KEY_LEFTCTRL]));
        assert_eq!(keys.forget(1), None);
        assert!(!keys.holds_any(&[KEY_LEFTCTRL, KEY_RIGHTCTRL]));
        assert_eq!(keys.key(2, KEY_LEFTCTRL, Press), None);
        assert!(keys.holds_any(&[KEY_LEFTCTRL]));
        assert_eq!(keys.key(2, KEY_LEFTCTRL, Release), None);
        assert!(!keys.holds_any(&[KEY_LEFTCTRL]));
        assert!(!keys.chord_held());
    }

    #[test]
    fn releasing_either_key_of_a_held_chord_reports_it_once() {
        let mut keys = tracker();
        assert_eq!(keys.key(0, KEY_LEFTCTRL, Press), None);
        assert_eq!(keys.key(0, SPACE, Press), Some(ChordTransition::Pressed));
        assert_eq!(keys.key(0, SPACE, Repeat), None);
        assert_eq!(
            keys.key(0, KEY_LEFTCTRL, Release),
            Some(ChordTransition::Released)
        );
        assert_eq!(keys.key(0, SPACE, Release), None);

        keys.key(0, KEY_LEFTCTRL, Press);
        keys.key(0, SPACE, Press);
        assert_eq!(keys.key(0, SPACE, Release), Some(ChordTransition::Released));
        assert_eq!(keys.key(0, KEY_LEFTCTRL, Release), None);
    }

    #[test]
    fn keys_released_without_the_whole_chord_held_report_nothing() {
        let mut keys = tracker();
        keys.key(0, KEY_LEFTCTRL, Press);
        keys.key(0, A, Press);
        assert_eq!(keys.key(0, A, Release), None);
        assert_eq!(keys.key(0, KEY_LEFTCTRL, Release), None);
        keys.key(0, SPACE, Press);
        assert_eq!(keys.key(0, SPACE, Release), None);
    }

    #[test]
    fn either_side_and_any_keyboard_hold_a_modifier() {
        let mut keys = tracker();
        keys.key(0, KEY_LEFTCTRL, Press);
        keys.key(1, KEY_RIGHTCTRL, Press);
        keys.key(1, SPACE, Press);
        assert_eq!(keys.key(0, KEY_LEFTCTRL, Release), None);
        assert_eq!(keys.forget(1), Some(ChordTransition::Released));
    }
}
