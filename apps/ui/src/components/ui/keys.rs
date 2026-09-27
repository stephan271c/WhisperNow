use dioxus::prelude::*;
use whispernow_api::Shortcut;

/// One key of a shortcut.
#[component]
pub(crate) fn Kbd(children: Element) -> Element {
    rsx! { kbd { class: "kbd", {children} } }
}

/// A shortcut rendered as one key cap per part, in `shortcut_parts` order.
#[component]
pub(crate) fn ShortcutLabel(shortcut: Shortcut) -> Element {
    rsx! {
        span { class: "shortcut-label",
            for part in shortcut_parts(&shortcut) {
                Kbd { "{part}" }
            }
        }
    }
}

/// The single display form of a shortcut: modifiers in display order, then the key.
pub(crate) fn shortcut_parts(shortcut: &Shortcut) -> Vec<&'static str> {
    shortcut
        .modifiers
        .iter()
        .map(|modifier| modifier.label())
        .chain([shortcut.key.label()])
        .collect()
}

#[cfg(test)]
mod tests {
    use whispernow_api::{ShortcutKey, ShortcutModifier};

    use super::*;

    #[test]
    fn modifiers_render_in_display_order_before_the_key() {
        let shortcut = Shortcut {
            modifiers: [
                ShortcutModifier::Meta,
                ShortcutModifier::Shift,
                ShortcutModifier::Control,
            ]
            .into(),
            key: ShortcutKey::PageDown,
        };
        assert_eq!(
            shortcut_parts(&shortcut),
            ["Ctrl", "Shift", "Meta", "Page Down"]
        );
    }
}
