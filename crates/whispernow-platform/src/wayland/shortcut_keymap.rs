use std::{fs::File, io, os::unix::fs::FileExt as _};

use wayland_client::{
    Connection, Dispatch, QueueHandle, WEnum,
    protocol::{wl_keyboard, wl_registry, wl_seat},
};
use whispernow_domain::{Shortcut, ShortcutKey, ShortcutModifier};
use xkbcommon::xkb;

// Wayland's XKB keycodes are evdev codes plus 8; KEY_MAX is from linux/input-event-codes.h.
const XKB_OFFSET: u32 = 8;
const KEY_MAX: u16 = 0x2ff;
const KEY_LEFTCTRL: u16 = 29;
const KEY_RIGHTCTRL: u16 = 97;
const KEY_LEFTALT: u16 = 56;
const KEY_RIGHTALT: u16 = 100;
const KEY_LEFTMETA: u16 = 125;
const KEY_RIGHTMETA: u16 = 126;
// The privileged helper's command groups allow only these physical keys.
const COMMAND_MODIFIERS: [u16; 6] = [
    KEY_LEFTCTRL,
    KEY_RIGHTCTRL,
    KEY_LEFTALT,
    KEY_RIGHTALT,
    KEY_LEFTMETA,
    KEY_RIGHTMETA,
];

pub fn shortcut_input_chord(shortcut: &Shortcut) -> io::Result<String> {
    let mut groups: Vec<_> = shortcut
        .modifiers
        .iter()
        .map(|modifier| {
            let names: &[&str] = match modifier {
                ShortcutModifier::Control => &["Control_L", "Control_R"],
                ShortcutModifier::Alt => &["Alt_L", "Alt_R"],
                ShortcutModifier::Shift => &["Shift_L", "Shift_R"],
                ShortcutModifier::Meta => &["Super_L", "Super_R"],
            };
            KeyGroup {
                symbols: names
                    .iter()
                    .map(|name| xkb::keysym_from_name(name, xkb::KEYSYM_NO_FLAGS))
                    .collect(),
                codes: Vec::new(),
                command_modifier: *modifier != ShortcutModifier::Shift,
            }
        })
        .collect();
    let keys = match shortcut.key {
        ShortcutKey::Enter => vec!["Return", "KP_Enter"],
        key => vec![key.keysym()],
    };
    groups.push(KeyGroup {
        symbols: keys
            .into_iter()
            .map(|name| xkb::keysym_from_name(name, xkb::KEYSYM_NO_FLAGS))
            .collect(),
        codes: Vec::new(),
        command_modifier: false,
    });
    let mut state = ShortcutKeymap {
        groups,
        error: None,
    };
    let connection = Connection::connect_to_env().map_err(io::Error::other)?;
    let mut queue = connection.new_event_queue();
    connection.display().get_registry(&queue.handle(), ());
    // Registry, seat capabilities and keyboard keymap arrive in successive roundtrips.
    for _ in 0..3 {
        queue.roundtrip(&mut state).map_err(io::Error::other)?;
    }
    if let Some(error) = state.error {
        return Err(error);
    }
    for group in &mut state.groups {
        if group.command_modifier {
            group.codes.retain(|code| COMMAND_MODIFIERS.contains(code));
        }
    }
    if state.groups.iter().any(|group| group.codes.is_empty()) {
        return Err(io::Error::other(
            "the shortcut has no key codes supported by the release helper",
        ));
    }
    Ok(state
        .groups
        .into_iter()
        .map(|group| {
            let mut codes = group.codes;
            codes.sort_unstable();
            codes.dedup();
            codes
                .into_iter()
                .map(|code| code.to_string())
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect::<Vec<_>>()
        .join("+"))
}

struct ShortcutKeymap {
    groups: Vec<KeyGroup>,
    error: Option<io::Error>,
}

struct KeyGroup {
    symbols: Vec<xkb::Keysym>,
    codes: Vec<u16>,
    command_modifier: bool,
}

impl Dispatch<wl_registry::WlRegistry, ()> for ShortcutKeymap {
    fn event(
        _: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        (): &(),
        _: &Connection,
        queue: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global {
            name,
            interface,
            version,
        } = event
            && interface == "wl_seat"
        {
            registry.bind::<wl_seat::WlSeat, _, _>(name, version.min(7), queue, ());
        }
    }
}

impl Dispatch<wl_seat::WlSeat, ()> for ShortcutKeymap {
    fn event(
        _: &mut Self,
        seat: &wl_seat::WlSeat,
        event: wl_seat::Event,
        (): &(),
        _: &Connection,
        queue: &QueueHandle<Self>,
    ) {
        if let wl_seat::Event::Capabilities {
            capabilities: WEnum::Value(capabilities),
        } = event
            && capabilities.contains(wl_seat::Capability::Keyboard)
        {
            seat.get_keyboard(queue, ());
        }
    }
}

impl Dispatch<wl_keyboard::WlKeyboard, ()> for ShortcutKeymap {
    fn event(
        state: &mut Self,
        _: &wl_keyboard::WlKeyboard,
        event: wl_keyboard::Event,
        (): &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wl_keyboard::Event::Keymap { format, fd, size } = event {
            let result = (|| {
                if format != WEnum::Value(wl_keyboard::KeymapFormat::XkbV1) {
                    return Err(io::Error::other("the compositor keymap is not XKB"));
                }
                let mut bytes = vec![0; usize::try_from(size).map_err(io::Error::other)?];
                // The compositor may share the descriptor's cursor across clients.
                File::from(fd).read_exact_at(&mut bytes, 0)?;
                let text = String::from_utf8(bytes).map_err(io::Error::other)?;
                let context = xkb::Context::new(xkb::CONTEXT_NO_FLAGS);
                let keymap = xkb::Keymap::new_from_string(
                    &context,
                    text.trim_end_matches('\0').to_owned(),
                    xkb::KEYMAP_FORMAT_TEXT_V1,
                    xkb::KEYMAP_COMPILE_NO_FLAGS,
                )
                .ok_or_else(|| io::Error::other("the compositor keymap could not be compiled"))?;
                for group in &mut state.groups {
                    group.codes.extend(key_codes(&keymap, &group.symbols));
                }
                Ok(())
            })();
            if let Err(error) = result {
                state.error = Some(error);
            }
        }
    }
}

fn key_codes(keymap: &xkb::Keymap, symbols: &[xkb::Keysym]) -> Vec<u16> {
    let mut codes = Vec::new();
    keymap.key_for_each(|keymap, key| {
        let Some(code) = key
            .raw()
            .checked_sub(XKB_OFFSET)
            .and_then(|code| u16::try_from(code).ok())
            .filter(|code| (1..=KEY_MAX).contains(code))
        else {
            return;
        };
        let matches = (0..keymap.num_layouts_for_key(key)).any(|layout| {
            (0..keymap.num_levels_for_key(key, layout)).any(|level| {
                keymap
                    .key_get_syms_by_level(key, layout, level)
                    .iter()
                    .any(|symbol| symbols.contains(symbol))
            })
        });
        if matches {
            codes.push(code);
        }
    });
    codes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keymap(layout: &str, variant: &str) -> xkb::Keymap {
        let context = xkb::Context::new(xkb::CONTEXT_NO_FLAGS);
        xkb::Keymap::new_from_names(
            &context,
            "evdev",
            "pc105",
            layout,
            variant,
            Some(String::new()),
            xkb::KEYMAP_COMPILE_NO_FLAGS,
        )
        .expect("keymap")
    }

    #[test]
    fn letters_follow_the_layout_instead_of_qwerty_positions() {
        let a = xkb::keysym_from_name("a", xkb::KEYSYM_NO_FLAGS);
        let q = xkb::keysym_from_name("q", xkb::KEYSYM_NO_FLAGS);
        let french = keymap("fr", "");
        assert_eq!(key_codes(&french, &[a]), [16]);
        assert_eq!(key_codes(&french, &[q]), [30]);
        assert_eq!(key_codes(&keymap("us", "dvorak"), &[q]), [45]);
    }

    #[test]
    fn configured_layouts_and_shifted_digits_are_included() {
        let a = xkb::keysym_from_name("a", xkb::KEYSYM_NO_FLAGS);
        assert_eq!(key_codes(&keymap("us,fr", ","), &[a]), [16, 30]);
        let one = xkb::keysym_from_name("1", xkb::KEYSYM_NO_FLAGS);
        assert_eq!(key_codes(&keymap("fr", ""), &[one]), [2]);
    }

    #[test]
    #[ignore = "requires a Wayland compositor with standard modifier keys"]
    fn the_compositor_supplies_a_keymap_without_a_focused_surface() {
        let chord = shortcut_input_chord(&Shortcut::default()).expect("compositor keymap");
        let groups: Vec<_> = chord.split('+').collect();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0], "29,97");
        assert_eq!(groups[1], "57");
        let shortcut = Shortcut {
            modifiers: [ShortcutModifier::Meta].into_iter().collect(),
            key: ShortcutKey::Space,
        };
        assert_eq!(
            shortcut_input_chord(&shortcut).expect("Super mapping"),
            "125,126+57"
        );
        let shortcut = Shortcut {
            key: ShortcutKey::Enter,
            ..Shortcut::default()
        };
        assert_eq!(
            shortcut_input_chord(&shortcut).expect("Enter mapping"),
            "29,97+28,96"
        );
    }
}
