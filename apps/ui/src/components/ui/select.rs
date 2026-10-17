use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use dioxus::prelude::*;

use super::fields::current_wiring;
use super::focus::focus;
use super::listbox::{Placement, place, reveal};
use super::{Icon, IconName};

/// The next option after `current`, wrapping, whose lowercase label starts with `typed`.
fn typeahead(labels: &[String], current: usize, typed: &str) -> Option<usize> {
    let typed = typed.to_lowercase();
    (1..=labels.len())
        .map(|step| (current + step) % labels.len())
        .find(|index| labels[*index].starts_with(&typed))
}

/// A dropdown of `options`, each (value, label), showing the label of `value`, or `placeholder`
/// while `value` is empty.
///
/// It follows the ARIA select-only combobox pattern: focus stays on the trigger, which names the
/// highlighted option with `aria-activedescendant`, and option `n` has the id `{id}-option-{n}`.
/// Inside a `Field` or `SettingRow` it takes the frame's id and label; outside one it needs an
/// `id` and a `label`.
#[component]
pub(crate) fn Select(
    #[props(into)] value: String,
    options: Vec<(String, String)>,
    onchange: EventHandler<String>,
    #[props(into)] id: Option<String>,
    #[props(into)] label: Option<String>,
    #[props(into)] aria_describedby: Option<String>,
    #[props(into)] placeholder: Option<String>,
    #[props(default)] disabled: bool,
) -> Element {
    let wiring = current_wiring();
    let id = wiring
        .as_ref()
        .map(|wiring| wiring.id.clone())
        .or(id)
        .expect("a Select outside a Field or SettingRow has an id");
    let described_by = wiring
        .as_ref()
        .and_then(|wiring| wiring.described_by.clone())
        .or(aria_describedby);
    let invalid = wiring.as_ref().is_some_and(|wiring| wiring.invalid);
    let mut open = use_signal(|| false);
    let mut active = use_signal(|| 0_usize);
    let mut placement = use_signal(|| None::<Placement>);
    let mut trigger = use_signal(|| None::<Rc<MountedData>>);
    let mut listbox = use_signal(|| None::<Rc<MountedData>>);
    // Mounted options by index; kept outside the reactive graph because mounting is not a change.
    let rows = use_hook(|| Rc::new(RefCell::new(HashMap::<usize, Rc<MountedData>>::new())));
    use_effect(use_reactive!(|disabled| {
        if disabled {
            open.set(false);
            placement.set(None);
        }
    }));
    use_effect({
        let rows = rows.clone();
        move || {
            let index = active();
            if !open() {
                return;
            }
            let list = listbox.peek().clone();
            if let (Some(list), Some(row)) = (list, rows.borrow().get(&index).cloned()) {
                spawn(reveal(list, row));
            }
        }
    });

    let selected = options.iter().position(|(option, _)| *option == value);
    let unset = selected.is_none() && value.is_empty();
    let shown = match selected {
        Some(index) => options[index].1.clone(),
        None if unset => placeholder.unwrap_or_default(),
        None => value.clone(),
    };
    let count = options.len();
    let expanded = open() && !disabled;
    let mut show = move |index: usize| {
        active.set(index);
        open.set(true);
    };
    let mut close = move || {
        open.set(false);
        placement.set(None);
        listbox.set(None);
    };
    let choose = use_callback({
        let options = options.clone();
        move |index: usize| {
            if let Some((option, _)) = options.get(index)
                && *option != value
            {
                onchange.call(option.clone());
            }
            close();
        }
    });
    let labels = options
        .iter()
        .map(|(_, label)| label.to_lowercase())
        .collect::<Vec<_>>();
    let onkeydown = move |event: KeyboardEvent| {
        let Some(last) = count.checked_sub(1) else {
            return;
        };
        let expanded = open();
        let current = if expanded {
            active().min(last)
        } else {
            selected.unwrap_or(0)
        };
        let key = event.key();
        if key == Key::Enter || key == Key::Character(" ".to_owned()) {
            event.prevent_default();
            if expanded {
                choose.call(current);
            } else {
                show(current);
            }
            return;
        }
        if key == Key::Tab {
            close();
            return;
        }
        if key == Key::Escape && expanded {
            event.prevent_default();
            event.stop_propagation();
            close();
            return;
        }
        let shortcut = Modifiers::CONTROL | Modifiers::ALT | Modifiers::META;
        let target = match key {
            Key::ArrowDown if expanded => (current + 1).min(last),
            Key::ArrowUp if expanded => current.saturating_sub(1),
            Key::ArrowDown | Key::ArrowUp => current,
            Key::Home => 0,
            Key::End => last,
            Key::Character(typed) if !event.modifiers().intersects(shortcut) => {
                match typeahead(&labels, current, &typed) {
                    Some(found) => found,
                    None => return,
                }
            }
            _ => return,
        };
        event.prevent_default();
        show(target);
    };

    rsx! {
        div { class: "select-field",
            if let Some(label) = label {
                span { id: "{id}-label", class: "visually-hidden", "{label}" }
            }
            button {
                r#type: "button",
                class: "select",
                id: "{id}",
                role: "combobox",
                aria_haspopup: "listbox",
                aria_expanded: if expanded { "true" } else { "false" },
                aria_controls: expanded.then(|| format!("{id}-listbox")),
                aria_activedescendant: expanded.then(|| format!("{id}-option-{}", active())),
                aria_labelledby: "{id}-label",
                aria_describedby: described_by,
                aria_invalid: invalid.then_some("true"),
                disabled,
                onmounted: move |event| trigger.set(Some(event.data())),
                onclick: move |_| {
                    if open() {
                        close();
                    } else if count > 0 {
                        show(selected.unwrap_or(0));
                        // WebKit on macOS does not focus a clicked button; keys and blur need it.
                        if let Some(button) = trigger.peek().clone() {
                            focus(button);
                        }
                    }
                },
                onkeydown,
                onblur: move |_| close(),
                span { class: if unset { "select-value select-placeholder" } else { "select-value" }, "{shown}" }
                Icon { name: IconName::ChevronDown }
            }
            if expanded {
                ul {
                    id: "{id}-listbox",
                    class: "select-listbox",
                    role: "listbox",
                    aria_labelledby: "{id}-label",
                    "data-placement": placement().map(Placement::side_name),
                    style: placement().and_then(Placement::room_style),
                    // Keeps focus, and so the highlighted option, on the trigger.
                    onmousedown: move |event| event.prevent_default(),
                    onmounted: {
                        let rows = rows.clone();
                        move |event: MountedEvent| {
                            let list = event.data();
                            listbox.set(Some(list.clone()));
                            let rows = rows.clone();
                            spawn(async move {
                                placement.set(Some(place(trigger.peek().clone(), list.clone()).await));
                                let row = rows.borrow().get(&*active.peek()).cloned();
                                if let Some(row) = row {
                                    reveal(list, row).await;
                                }
                            });
                        }
                    },
                    for (index, (option, text)) in options.iter().enumerate() {
                        li {
                            key: "{option}",
                            id: "{id}-option-{index}",
                            class: if index == active() { "select-option active" } else { "select-option" },
                            role: "option",
                            aria_selected: if Some(index) == selected { "true" } else { "false" },
                            onmounted: {
                                let rows = rows.clone();
                                move |event: MountedEvent| {
                                    rows.borrow_mut().insert(index, event.data());
                                }
                            },
                            onmousemove: move |_| {
                                if *active.peek() != index {
                                    active.set(index);
                                }
                            },
                            onclick: move |_| choose.call(index),
                            span { class: "select-option-label", "{text}" }
                            Icon { name: IconName::Check }
                        }
                    }
                }
            }
        }
    }
}
