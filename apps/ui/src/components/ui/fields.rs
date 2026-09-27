use dioxus::prelude::*;

/// How a `Field` or `SettingRow` labels and describes the control rendered inside it.
///
/// The frame owns the ids: the control takes `id`, the label is `{id}-label`, and the
/// description, error and save status are `{id}-description`, `{id}-error` and `{id}-status`.
#[derive(Clone, Debug, PartialEq)]
struct FieldWiring {
    id: String,
    described_by: Option<String>,
    invalid: bool,
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

fn current_wiring() -> Option<FieldWiring> {
    try_use_context::<Memo<FieldWiring>>().map(|wiring| wiring())
}

/// A stacked label, optional description, control and optional error.
///
/// A control that saves itself passes `status`: the frame then keeps a polite live region for
/// it, so "Saving…" and "Saved" are announced without taking focus. Only `error` is an alert.
#[component]
pub(crate) fn Field(
    id: String,
    label: String,
    description: Option<String>,
    error: Option<String>,
    status: Option<String>,
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
            if let Some(status) = status {
                SaveStatusText { id: id.clone(), status }
            }
            if let Some(error) = error {
                p { id: "{id}-error", class: "field-error-message", role: "alert", "{error}" }
            }
        }
    }
}

/// The live region that reports a control's save status.
#[component]
fn SaveStatusText(id: String, status: String) -> Element {
    rsx! {
        span { id: "{id}-status", class: "save-status", role: "status", aria_live: "polite", "{status}" }
    }
}

/// A settings row: title and description on the left, the control on the right. `status`
/// works as in [`Field`] and is shown beside the control.
#[component]
pub(crate) fn SettingRow(
    id: String,
    title: String,
    description: Option<String>,
    error: Option<String>,
    status: Option<String>,
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
            div { class: "setting-row-control",
                {children}
                if let Some(status) = status {
                    SaveStatusText { id: id.clone(), status }
                }
            }
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
    /// Handed to `oncommit`. Shown while the frame reports that change as refused, so a
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
    let refused = wiring.as_ref().is_some_and(|wiring| wiring.invalid);
    let invalid = refused.then_some("true");
    let mut draft = use_signal(|| None::<Draft>);
    let saved = value.clone();
    // Once the saved value moves on without a refusal, a committed draft is stale.
    use_effect(use_reactive!(|(saved, refused)| {
        let stale =
            matches!(&*draft.peek(), Some(Draft::Committed(text)) if !refused && *text != saved);
        if stale {
            draft.set(None);
        }
    }));
    let shown = match draft() {
        Some(Draft::Typing(text)) => text,
        Some(Draft::Committed(text)) if refused => text,
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
                aria_invalid: invalid,
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
            aria_invalid: invalid,
            value: shown,
            oninput: move |event| draft.set(Some(Draft::Typing(event.value()))),
            onchange: commit,
            onkeydown: revert,
            ..attributes,
        }
    }
}

/// A native select showing `value`. Inside a `Field` or `SettingRow` it takes the frame's id.
#[component]
pub(crate) fn Select(
    #[props(into)] value: String,
    onchange: EventHandler<FormEvent>,
    #[props(extends = GlobalAttributes, extends = select)] attributes: Vec<Attribute>,
    children: Element,
) -> Element {
    let wiring = current_wiring();
    rsx! {
        select {
            class: "select",
            id: wiring.as_ref().map(|wiring| wiring.id.clone()),
            aria_describedby: wiring.as_ref().and_then(|wiring| wiring.described_by.clone()),
            aria_invalid: wiring.as_ref().filter(|wiring| wiring.invalid).map(|_| "true"),
            value,
            onchange: move |event: FormEvent| onchange.call(event),
            ..attributes,
            {children}
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
