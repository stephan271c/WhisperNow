use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use dioxus::prelude::*;

use super::listbox::{Placement, place, reveal};

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
    let mut open = use_signal(|| false);
    let mut active = use_signal(|| None::<usize>);
    let mut placement = use_signal(|| None::<Placement>);
    let mut field = use_signal(|| None::<Rc<MountedData>>);
    let mut listbox = use_signal(|| None::<Rc<MountedData>>);
    // Mounted rows by suggestion; kept outside the reactive graph because mounting is not a change.
    let rows = use_hook(|| Rc::new(RefCell::new(HashMap::<String, Rc<MountedData>>::new())));
    use_effect(use_reactive!(|disabled| {
        if disabled {
            open.set(false);
            placement.set(None);
        }
    }));

    let matches = matching(&suggestions, &value);
    let expanded = open() && !disabled && !matches.is_empty();
    let current = active().filter(|index| expanded && *index < matches.len());
    let highlighted = current.map(|index| matches[index].clone());
    let mut close = move || {
        open.set(false);
        active.set(None);
        placement.set(None);
        listbox.set(None);
    };
    let choose = use_callback(move |suggestion: String| {
        oninput.call(suggestion);
        close();
    });
    let onkeydown = {
        let rows = rows.clone();
        let matches = matches.clone();
        move |event: KeyboardEvent| match event.key() {
            Key::Enter => match current {
                Some(index) => {
                    event.prevent_default();
                    choose.call(matches[index].clone());
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
                let Some(last) = matches.len().checked_sub(1) else {
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
                let row = rows.borrow().get(&matches[next]).cloned();
                if let (Some(list), Some(row)) = (listbox.peek().clone(), row) {
                    spawn(reveal(list, row));
                }
            }
            _ => {}
        }
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
                ul {
                    id: "{id}-listbox",
                    class: "select-listbox",
                    role: "listbox",
                    aria_label: "{label}",
                    "data-placement": placement().map(Placement::side_name),
                    style: placement().and_then(Placement::room_style),
                    // Keeps focus, and so the highlighted suggestion, in the field.
                    onmousedown: move |event| event.prevent_default(),
                    onmounted: {
                        let rows = rows.clone();
                        move |event: MountedEvent| {
                            let list = event.data();
                            listbox.set(Some(list.clone()));
                            let rows = rows.clone();
                            let highlighted = highlighted.clone();
                            spawn(async move {
                                placement.set(Some(place(field.peek().clone(), list.clone()).await));
                                let row = highlighted
                                    .and_then(|suggestion| rows.borrow().get(&suggestion).cloned());
                                if let Some(row) = row {
                                    reveal(list, row).await;
                                }
                            });
                        }
                    },
                    for (index, suggestion) in matches.into_iter().enumerate() {
                        li {
                            key: "{suggestion}",
                            id: "{id}-option-{index}",
                            class: if Some(index) == current { "select-option active" } else { "select-option" },
                            role: "option",
                            aria_selected: if Some(index) == current { "true" } else { "false" },
                            onmounted: {
                                let rows = rows.clone();
                                let suggestion = suggestion.clone();
                                move |event: MountedEvent| {
                                    rows.borrow_mut().insert(suggestion.clone(), event.data());
                                }
                            },
                            onmousemove: move |_| {
                                if *active.peek() != Some(index) {
                                    active.set(Some(index));
                                }
                            },
                            onclick: {
                                let suggestion = suggestion.clone();
                                move |_| choose.call(suggestion.clone())
                            },
                            span { class: "select-option-label", "{suggestion}" }
                        }
                    }
                }
            }
        }
    }
}
