use std::rc::Rc;

use dioxus::prelude::*;

use super::fields::{FieldWiring, current_wiring};
use super::focus::focus;
use super::listbox::{OptionList, use_open};
use super::{Icon, IconName};

/// The next option after `current`, wrapping, whose lowercase label starts with `typed`.
fn typeahead(labels: &[String], current: usize, typed: &str) -> Option<usize> {
    let typed = typed.to_lowercase();
    (1..=labels.len())
        .map(|step| (current + step) % labels.len())
        .find(|index| labels[*index].starts_with(&typed))
}

/// A dropdown of `options`, each (value, label), showing the label of `value`, or `placeholder`
/// when no option has that value.
///
/// It follows the ARIA select-only combobox pattern: focus stays on the trigger, which names the
/// highlighted option with `aria-activedescendant`, and option `n` has the id `{id}-option-{n}`.
/// It takes the id and label of the `Field` or `SettingRow` around it.
#[component]
pub(crate) fn Select<T: Clone + PartialEq + 'static>(
    value: T,
    options: Vec<(T, String)>,
    onchange: EventHandler<T>,
    #[props(into)] placeholder: Option<String>,
    #[props(default)] disabled: bool,
) -> Element {
    let FieldWiring {
        id,
        described_by,
        invalid,
    } = current_wiring().expect("a Select is inside a Field or SettingRow");
    let mut open = use_open(disabled);
    let mut active = use_signal(|| 0_usize);
    let mut trigger = use_signal(|| None::<Rc<MountedData>>);

    let selected = options.iter().position(|(option, _)| *option == value);
    let shown = selected.map_or_else(
        || placeholder.unwrap_or_default(),
        |index| options[index].1.clone(),
    );
    let count = options.len();
    let expanded = open() && !disabled;
    let mut show = move |index: usize| {
        active.set(index);
        open.set(true);
    };
    let mut close = move || open.set(false);
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
                span { class: if selected.is_none() { "select-value select-placeholder" } else { "select-value" }, "{shown}" }
                Icon { name: IconName::ChevronDown }
            }
            if expanded {
                OptionList {
                    id: id.clone(),
                    labelled_by: format!("{id}-label"),
                    anchor: trigger,
                    options: options.iter().map(|(_, label)| label.clone()).collect::<Vec<_>>(),
                    active: Some(active()),
                    selected,
                    check: true,
                    onhover: move |index| {
                        if *active.peek() != index {
                            active.set(index);
                        }
                    },
                    onchoose: choose,
                }
            }
        }
    }
}
