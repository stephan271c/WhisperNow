use std::rc::Rc;

use dioxus::prelude::*;
use whispernow_domain::VocabularyReplacement;

use super::copy_button::CopyButton;
use super::settings::SettingsLoading;
use super::ui::focus::take_focus;
use super::ui::{
    Button, ButtonVariant, DraftText, Field, IconButton, IconName, Section, TextInput,
};
use crate::UiContext;
use crate::blocked_reason::DisabledReason;
use crate::management::SettingKey;

/// The add form's id; tests submit it as Enter does.
const ADD_FORM: &str = "vocabulary-add-form";

fn rule_name(entry: &VocabularyReplacement) -> String {
    format!("‘{} → {}’", entry.source, entry.replacement)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum RuleFocus {
    Rule(usize),
    /// No rule is left in view, so the list is gone: the form that adds one.
    AddForm,
}

fn matches(entry: &VocabularyReplacement, query: &str) -> bool {
    entry.source.to_lowercase().contains(query) || entry.replacement.to_lowercase().contains(query)
}

#[component]
pub(super) fn VocabularyPage() -> Element {
    let context = use_context::<UiContext>();
    let settings_blocked = context.settings_blocked();
    let disabled = settings_blocked.is_some();
    let management = context.management;
    let mut search = use_signal(String::new);
    let mut focus = use_signal(|| None::<RuleFocus>);
    let add_form = use_signal(|| None::<Rc<MountedData>>);
    use_effect(move || {
        if focus() == Some(RuleFocus::AddForm) {
            take_focus(add_form);
            focus.set(None);
        }
    });
    let Some(rules) = management
        .read()
        .shown_settings()
        .map(|settings| settings.vocabulary_replacements.clone())
    else {
        return rsx! { SettingsLoading {} };
    };
    let error = management.read().setting_error(SettingKey::Vocabulary);
    let query = search().trim().to_lowercase();
    let filtering = !query.is_empty();
    let total = rules.len();
    let shown = rules
        .iter()
        .enumerate()
        .filter(|(_, entry)| !filtering || matches(entry, &query))
        .map(|(index, entry)| (index, entry.clone()))
        .collect::<Vec<_>>();
    let shown_indices = shown.iter().map(|(index, _)| *index).collect::<Vec<_>>();
    // The shown rule after the deleted one, which has moved up, else the one before it.
    let after_delete =
        |position: usize| match (shown_indices.get(position + 1), position.checked_sub(1)) {
            (Some(next), _) => RuleFocus::Rule(next - 1),
            (None, Some(before)) => RuleFocus::Rule(shown_indices[before]),
            (None, None) => RuleFocus::AddForm,
        };
    rsx! { div { class: "settings-stack",
        DisabledReason { id: "settings-disabled-reason", reason: settings_blocked }
        Section { id: "vocabulary", title: "Vocabulary",
            description: "Rules run from top to bottom.",
            AddRuleForm { disabled, mounted: add_form }
            if let Some(error) = error {
                p { id: "vocabulary-error", class: "field-error-message vocabulary-error", role: "alert", "{error}" }
            }
            if total > 0 {
                Field { id: "vocabulary-search", label: "Search rules",
                    TextInput { r#type: "search", value: "{search}", oninput: move |event: FormEvent| search.set(event.value()) }
                }
            }
            if total == 0 {
                p { class: "section-note", "No rules yet." }
            } else if shown.is_empty() {
                p { class: "section-note", role: "status", "No rules match “{search}”." }
            } else {
                if filtering {
                    p { class: "section-note", role: "status", "Showing {shown.len()} of {total} rules. Clear the search to reorder them." }
                }
                ol { class: "vocabulary-list",
                    for (position, (index, entry)) in shown.into_iter().enumerate() {
                        VocabularyRow { key: "{index}", index, entry, last: index + 1 == total, reorder: !filtering,
                            disabled,
                            focus, after_delete: after_delete(position) }
                    }
                }
            }
        }
        VocabularyTransfer { rules, disabled }
    } }
}

/// The form that adds a rule. Typing re-renders only this form.
#[component]
fn AddRuleForm(disabled: bool, mut mounted: Signal<Option<Rc<MountedData>>>) -> Element {
    let context = use_context::<UiContext>();
    let mut source = use_signal(String::new);
    let mut replacement = use_signal(String::new);
    rsx! {
        form { id: ADD_FORM, class: "vocabulary-add", tabindex: "-1", aria_label: "Add a rule",
            onmounted: move |event| mounted.set(Some(event.data())),
            onsubmit: move |event: FormEvent| {
                event.prevent_default();
                let entry = VocabularyReplacement::new(source().trim(), replacement().trim());
                context.change_vocabulary(move |rules| rules.push(entry));
                source.set(String::new());
                replacement.set(String::new());
            },
            Field { id: "vocabulary-source", label: "Replace this",
                TextInput { value: "{source}", disabled, oninput: move |event: FormEvent| source.set(event.value()) }
            }
            Field { id: "vocabulary-replacement", label: "With this",
                TextInput { value: "{replacement}", disabled, oninput: move |event: FormEvent| replacement.set(event.value()) }
            }
            button { r#type: "submit", id: "vocabulary-add", class: "btn btn-primary btn-md", disabled, "Add rule" }
            p { class: "field-description vocabulary-add-hint", "Press Enter to add." }
        }
    }
}

/// One rule: both texts edit in place; move and delete name the rule they act on. Deleting it
/// moves focus to `after_delete`.
#[component]
fn VocabularyRow(
    index: usize,
    entry: VocabularyReplacement,
    last: bool,
    reorder: bool,
    disabled: bool,
    mut focus: Signal<Option<RuleFocus>>,
    after_delete: RuleFocus,
) -> Element {
    let context = use_context::<UiContext>();
    let mut row = use_signal(|| None::<Rc<MountedData>>);
    use_effect(move || {
        if focus() == Some(RuleFocus::Rule(index)) {
            take_focus(row);
            focus.set(None);
        }
    });
    let name = rule_name(&entry);
    let number = index + 1;
    let change = move |edit: fn(&mut Vec<VocabularyReplacement>, usize)| {
        context.change_vocabulary(move |rules| edit(rules, index));
    };
    let edit = move |edited: VocabularyReplacement| {
        context.change_vocabulary(move |rules| {
            if let Some(rule) = rules.get_mut(index) {
                *rule = edited;
            }
        });
        true
    };
    let replacement = entry.replacement.clone();
    let source = entry.source.clone();
    rsx! {
        li { id: "vocabulary-rule-{number}", class: "vocabulary-row", tabindex: "-1", aria_label: "Rule {number}, {name}",
            onmounted: move |event| row.set(Some(event.data())),
            DraftText { id: "vocabulary-rule-{number}-source", aria_label: "Replace this, rule {number}",
                disabled,
                value: entry.source.clone(),
                oncommit: move |text: String| edit(VocabularyReplacement::new(text.trim(), replacement.clone())),
            }
            span { class: "vocabulary-arrow", aria_hidden: "true", "→" }
            DraftText { id: "vocabulary-rule-{number}-replacement", aria_label: "With this, rule {number}",
                disabled,
                value: entry.replacement.clone(),
                oncommit: move |text: String| edit(VocabularyReplacement::new(source.clone(), text.trim())),
            }
            div { class: "vocabulary-row-actions",
                if reorder {
                    IconButton { icon: IconName::ArrowUp, label: "Move rule {name} up", disabled: disabled || index == 0,
                        onclick: move |_| { change(|rules, index| if let Some(above) = index.checked_sub(1) { rules.swap(above, index) }); } }
                    IconButton { icon: IconName::ArrowDown, label: "Move rule {name} down", disabled: disabled || last,
                        onclick: move |_| {
                            change(|rules, index| if index + 1 < rules.len() { rules.swap(index, index + 1) });
                        } }
                }
                IconButton { id: "vocabulary-rule-{number}-delete", icon: IconName::Trash, label: "Delete rule {name}",
                    disabled,
                    onclick: move |_| {
                        change(|rules, index| if index < rules.len() { rules.remove(index); });
                        focus.set(Some(after_delete));
                    }
                }
            }
        }
    }
}

#[component]
fn VocabularyTransfer(rules: Vec<VocabularyReplacement>, disabled: bool) -> Element {
    let context = use_context::<UiContext>();
    let mut management = context.management;
    let mut pasted = use_signal(String::new);
    let export = use_memo(use_reactive!(|rules| {
        (!rules.is_empty())
            .then(|| serde_json::to_string_pretty(&rules).expect("a list of text pairs serializes"))
    }))();
    rsx! {
        Section { id: "vocabulary-transfer", title: "Import and export",
            description: "A JSON list of {{\"source\": …, \"replacement\": …}} pairs.",
            div { class: "card-actions",
                CopyButton { id: "vocabulary-export",
                    text: export, label: "Copy rules as JSON",
                }
            }
            Field { id: "vocabulary-import", label: "Paste rules to import",
                description: "A rule for text that already has one replaces it.",
                textarea { id: "vocabulary-import", class: "text-input", rows: "4",
                    disabled,
                    aria_describedby: "vocabulary-import-description",
                    value: "{pasted}", oninput: move |event| pasted.set(event.value()) }
            }
            div { class: "card-actions",
                Button { id: "vocabulary-import-submit", variant: ButtonVariant::Secondary,
                    disabled: disabled || pasted().trim().is_empty(),
                    onclick: move |_| match serde_json::from_str::<Vec<VocabularyReplacement>>(pasted().trim()) {
                        Ok(entries) => {
                            context.change_vocabulary(move |rules| import_rules(rules, &entries));
                            pasted.set(String::new());
                        },
                        Err(_) => management.write().reject_setting(
                            SettingKey::Vocabulary,
                            "Paste the JSON that Copy rules as JSON produces",
                        ),
                    },
                    "Import rules"
                }
            }
        }
    }
}

/// Adds imported rules; an imported rule for text that already has one replaces its
/// replacement.
fn import_rules(rules: &mut Vec<VocabularyReplacement>, entries: &[VocabularyReplacement]) {
    for entry in entries {
        match rules.iter_mut().find(|rule| rule.source == entry.source) {
            Some(rule) => rule.replacement.clone_from(&entry.replacement),
            None => rules.push(entry.clone()),
        }
    }
}
