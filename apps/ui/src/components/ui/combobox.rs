use std::rc::Rc;

use dioxus::prelude::*;

use super::listbox::{OptionList, use_open};

/// How a suggestion matches the typed text, best first.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Match {
    Whole,
    Start,
    /// A word begins after any character that is not a letter or digit, as `nemotron` does in
    /// `nvidia/nemotron-3.5`.
    WordStart,
}

/// `typed` is lowercase; the suggestion is compared lowercase.
fn match_of(suggestion: &str, typed: &str) -> Option<Match> {
    let suggestion = suggestion.to_lowercase();
    if suggestion == typed {
        return Some(Match::Whole);
    }
    if suggestion.starts_with(typed) {
        return Some(Match::Start);
    }
    suggestion
        .char_indices()
        .filter(|(_, character)| !character.is_alphanumeric())
        .any(|(at, separator)| suggestion[at + separator.len_utf8()..].starts_with(typed))
        .then_some(Match::WordStart)
}

/// The suggestions `typed` matches, best match first and otherwise in their order.
fn matching(suggestions: &[String], typed: &str) -> Vec<String> {
    let typed = typed.to_lowercase();
    let mut matches = suggestions
        .iter()
        .filter_map(|suggestion| Some((match_of(suggestion, &typed)?, suggestion)))
        .collect::<Vec<_>>();
    matches.sort_by_key(|(how, _)| *how);
    matches
        .into_iter()
        .map(|(_, suggestion)| suggestion.clone())
        .collect()
}

/// A text field that lists the `suggestions` in which the typed text begins the whole suggestion
/// or one of its words, ignoring case.
///
/// It follows the ARIA editable combobox pattern with list autocomplete: focus stays in the field,
/// which names the highlighted suggestion with `aria-activedescendant`, and suggestion `n` has the
/// id `{id}-option-{n}`. Typing or Down opens the list. Enter or a click puts the highlighted
/// suggestion in the field; Enter with none highlighted is left to the form.
#[component]
pub(crate) fn Combobox(
    #[props(into)] id: String,
    #[props(into)] label: String,
    #[props(into)] value: String,
    suggestions: Vec<String>,
    oninput: EventHandler<String>,
    #[props(into)] placeholder: Option<String>,
    #[props(default)] disabled: bool,
) -> Element {
    let mut open = use_open(disabled);
    let mut active = use_signal(|| None::<usize>);
    let mut field = use_signal(|| None::<Rc<MountedData>>);

    let matches = matching(&suggestions, &value);
    let count = matches.len();
    let expanded = open() && !disabled && count > 0;
    let current = active().filter(|index| expanded && *index < count);
    let mut close = move || {
        open.set(false);
        active.set(None);
    };
    let choose = use_callback({
        let matches = matches.clone();
        move |index: usize| {
            oninput.call(matches[index].clone());
            close();
        }
    });
    let onkeydown = move |event: KeyboardEvent| match event.key() {
        Key::Enter => match current {
            Some(index) => {
                event.prevent_default();
                choose.call(index);
            }
            None => close(),
        },
        Key::Escape if expanded => {
            event.prevent_default();
            event.stop_propagation();
            close();
        }
        Key::Tab => close(),
        key @ (Key::ArrowDown | Key::ArrowUp) => {
            let Some(last) = count.checked_sub(1) else {
                return;
            };
            event.prevent_default();
            let next = match (key, current) {
                (Key::ArrowDown, None) => 0,
                (Key::ArrowDown, Some(index)) => (index + 1).min(last),
                (_, None) => last,
                (_, Some(index)) => index.saturating_sub(1),
            };
            open.set(true);
            active.set(Some(next));
        }
        _ => {}
    };

    rsx! {
        div { class: "select-field",
            input {
                id: "{id}",
                class: "text-input",
                r#type: "text",
                role: "combobox",
                aria_label: "{label}",
                aria_autocomplete: "list",
                aria_expanded: if expanded { "true" } else { "false" },
                aria_controls: expanded.then(|| format!("{id}-listbox")),
                aria_activedescendant: current.map(|index| format!("{id}-option-{index}")),
                // The webview's own autofill list would cover this one.
                autocomplete: "off",
                spellcheck: "false",
                placeholder,
                disabled,
                value: "{value}",
                onmounted: move |event| field.set(Some(event.data())),
                oninput: move |event: FormEvent| {
                    let typed = event.value();
                    open.set(!typed.is_empty());
                    active.set(None);
                    oninput.call(typed);
                },
                onkeydown,
                onblur: move |_| close(),
            }
            if expanded {
                OptionList {
                    id: id.clone(),
                    label: label.clone(),
                    anchor: field,
                    options: matches,
                    active: current,
                    selected: current,
                    onhover: move |index| {
                        if *active.peek() != Some(index) {
                            active.set(Some(index));
                        }
                    },
                    onchoose: choose,
                }
            }
        }
    }
}
