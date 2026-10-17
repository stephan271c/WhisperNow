use dioxus::prelude::*;

/// How a `Field` or `SettingRow` labels and describes the control rendered inside it.
///
/// The frame owns the ids: the control takes `id`, the label is `{id}-label`, and the
/// description and error are `{id}-description` and `{id}-error`.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct FieldWiring {
    pub(super) id: String,
    pub(super) described_by: Option<String>,
    pub(super) invalid: bool,
}

impl FieldWiring {
    fn new(id: &str, has_description: bool, has_error: bool) -> Self {
        let described_by = [
            has_description.then(|| format!("{id}-description")),
            has_error.then(|| format!("{id}-error")),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" ");
        Self {
            id: id.to_owned(),
            described_by: (!described_by.is_empty()).then_some(described_by),
            invalid: has_error,
        }
    }

    fn label_id(&self) -> String {
        format!("{}-label", self.id)
    }
}

/// Provides the wiring to the control below; it follows the frame's props as they change.
fn use_field_wiring(id: &str, has_description: bool, has_error: bool) {
    let id = id.to_owned();
    let wiring = use_memo(use_reactive!(|(id, has_description, has_error)| {
        FieldWiring::new(&id, has_description, has_error)
    }));
    use_context_provider(|| wiring);
}

pub(super) fn current_wiring() -> Option<FieldWiring> {
    try_use_context::<Memo<FieldWiring>>().map(|wiring| wiring())
}

#[component]
pub(crate) fn Field(
    id: String,
    label: String,
    description: Option<String>,
    error: Option<String>,
    children: Element,
) -> Element {
    use_field_wiring(&id, description.is_some(), error.is_some());
    rsx! {
        div { class: "field",
            label { id: "{id}-label", class: "field-label", r#for: "{id}", "{label}" }
            if let Some(description) = description {
                p { id: "{id}-description", class: "field-description", "{description}" }
            }
            {children}
            if let Some(error) = error {
                p { id: "{id}-error", class: "field-error-message", role: "alert", "{error}" }
            }
        }
    }
}

/// A settings row: title and description on the left, the control on the right.
#[component]
pub(crate) fn SettingRow(
    id: String,
    title: String,
    description: Option<String>,
    error: Option<String>,
    children: Element,
) -> Element {
    use_field_wiring(&id, description.is_some(), error.is_some());
    rsx! {
        div { class: "setting-row",
            div { class: "setting-row-text",
                label { id: "{id}-label", class: "setting-row-title", r#for: "{id}", "{title}" }
                if let Some(description) = description {
                    p { id: "{id}-description", class: "setting-row-description", "{description}" }
                }
                if let Some(error) = error {
                    p { id: "{id}-error", class: "field-error-message", role: "alert", "{error}" }
                }
            }
            div { class: "setting-row-control", {children} }
        }
    }
}

/// A single-line text input. Inside a `Field` or `SettingRow` it takes the frame's id.
#[component]
pub(crate) fn TextInput(
    oninput: EventHandler<FormEvent>,
    #[props(extends = GlobalAttributes, extends = input)] attributes: Vec<Attribute>,
) -> Element {
    let wiring = current_wiring();
    rsx! {
        input {
            class: "text-input",
            id: wiring.as_ref().map(|wiring| wiring.id.clone()),
            aria_describedby: wiring.as_ref().and_then(|wiring| wiring.described_by.clone()),
            aria_invalid: wiring.as_ref().filter(|wiring| wiring.invalid).map(|_| "true"),
            oninput: move |event| oninput.call(event),
            ..attributes,
        }
    }
}

/// What a [`DraftText`] shows instead of its saved `value`.
#[derive(Clone, Debug, PartialEq)]
enum Draft {
    /// Being typed, or refused before it was sent: shown until committed again.
    Typing(String),
    /// Handed to `oncommit`. Shown while the frame reports an error for that change, so a
    /// failed save keeps the typed text for another try.
    Committed(String),
}

/// Free text that saves itself. Typing edits a local draft; leaving the field, or pressing
/// Enter in a single line, calls `oncommit` once with the draft if it differs from `value`.
/// `oncommit` returns whether the change was taken; a refused one stays in the field. Inside a
/// `Field` or `SettingRow` it takes the frame's id, and a taken change that later fails, shown
/// by the frame's error, also keeps its text until the user edits it or presses Escape to go
/// back to the saved value.
#[component]
pub(crate) fn DraftText(
    value: String,
    oncommit: Callback<String, bool>,
    #[props(default)] multiline: bool,
    #[props(extends = GlobalAttributes, extends = input)] attributes: Vec<Attribute>,
) -> Element {
    let wiring = current_wiring();
    let id = wiring.as_ref().map(|wiring| wiring.id.clone());
    let described_by = wiring
        .as_ref()
        .and_then(|wiring| wiring.described_by.clone());
    let has_error = wiring.as_ref().is_some_and(|wiring| wiring.invalid);
    let aria_invalid = has_error.then_some("true");
    let mut draft = use_signal(|| None::<Draft>);
    let saved = value.clone();
    // Once the saved value moves on without an error, a committed draft is stale.
    use_effect(use_reactive!(|(saved, has_error)| {
        let stale =
            matches!(&*draft.peek(), Some(Draft::Committed(text)) if !has_error && *text != saved);
        if stale {
            draft.set(None);
        }
    }));
    let shown = match draft() {
        Some(Draft::Typing(text)) => text,
        Some(Draft::Committed(text)) if has_error => text,
        _ => value.clone(),
    };
    let commit = move |event: FormEvent| {
        let text = event.value();
        if text == value {
            draft.set(None);
        } else if oncommit.call(text.clone()) {
            draft.set(Some(Draft::Committed(text)));
        } else {
            draft.set(Some(Draft::Typing(text)));
        }
    };
    let revert = move |event: KeyboardEvent| {
        if event.key() == Key::Escape {
            draft.set(None);
        }
    };
    if multiline {
        return rsx! {
            textarea {
                class: "text-input",
                id,
                aria_describedby: described_by,
                aria_invalid,
                value: shown,
                oninput: move |event| draft.set(Some(Draft::Typing(event.value()))),
                onchange: commit,
                onkeydown: revert,
                ..attributes,
            }
        };
    }
    rsx! {
        input {
            class: "text-input",
            id,
            aria_describedby: described_by,
            aria_invalid,
            value: shown,
            oninput: move |event| draft.set(Some(Draft::Typing(event.value()))),
            onchange: commit,
            onkeydown: revert,
            ..attributes,
        }
    }
}

/// A choice between a few options shown side by side, one pressed. Inside a `Field` or
/// `SettingRow` the group takes the frame's id and label, and each option's id is `{id}-{value}`;
/// outside one it needs a `label`.
#[component]
pub(crate) fn Segmented(
    value: &'static str,
    options: Vec<(&'static str, &'static str)>,
    onchange: EventHandler<&'static str>,
    label: Option<String>,
    #[props(default)] disabled: bool,
) -> Element {
    let wiring = current_wiring();
    let id = wiring.as_ref().map(|wiring| wiring.id.clone());
    rsx! {
        div {
            class: "segmented",
            id: id.clone(),
            role: "group",
            aria_label: label,
            aria_labelledby: wiring.as_ref().map(FieldWiring::label_id),
            aria_describedby: wiring.as_ref().and_then(|wiring| wiring.described_by.clone()),
            for (option, label) in options {
                button {
                    r#type: "button",
                    class: "segmented-option",
                    id: id.as_ref().map(|id| format!("{id}-{option}")),
                    aria_pressed: if option == value { "true" } else { "false" },
                    disabled,
                    onclick: move |_| if option != value { onchange.call(option) },
                    "{label}"
                }
            }
        }
    }
}

/// An on/off switch. Outside a `Field` or `SettingRow` it needs a `label`.
#[component]
pub(crate) fn Toggle(
    checked: bool,
    onchange: EventHandler<bool>,
    label: Option<String>,
    #[props(default)] disabled: bool,
) -> Element {
    let wiring = current_wiring();
    rsx! {
        button {
            r#type: "button",
            role: "switch",
            class: "toggle",
            id: wiring.as_ref().map(|wiring| wiring.id.clone()),
            aria_checked: if checked { "true" } else { "false" },
            aria_label: label,
            aria_labelledby: wiring.as_ref().map(FieldWiring::label_id),
            aria_describedby: wiring.as_ref().and_then(|wiring| wiring.described_by.clone()),
            disabled,
            onclick: move |_| onchange.call(!checked),
            span { class: "toggle-thumb", aria_hidden: "true" }
        }
    }
}
