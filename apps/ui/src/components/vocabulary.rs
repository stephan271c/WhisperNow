//! Vocabulary: replacement rules applied to each transcript, in order.

use std::rc::Rc;

use dioxus::prelude::*;
use serde::{Deserialize, Serialize};
use whispernow_api::{Settings, VocabularyReplacement};

use super::actions::use_actions;
use super::configuration::{SettingsBlockedReason, SettingsLoading, status_parts};
use super::ui::focus::{neighbour, take_focus};
use super::ui::{
    Button, ButtonVariant, DraftText, Field, IconButton, IconName, Section, TextInput,
};
use crate::{NoticeScope, SettingKey, UiContext};

/// The add form's id; tests submit it as Enter does.
const ADD_FORM: &str = "vocabulary-add-form";

/// A rule as import and export write it: `{"source": …, "replacement": …}`, named fields a
/// person can read, where settings store a `[source, replacement]` pair.
#[derive(Deserialize, Serialize)]
struct TransferRule {
    source: String,
    replacement: String,
}

/// How a rule is named in the accessible names of its controls.
fn rule_name(entry: &VocabularyReplacement) -> String {
    format!("‘{} → {}’", entry.source, entry.replacement)
}

/// Where focus goes after a rule is deleted.
#[derive(Clone, Copy, Debug, PartialEq)]
enum RuleFocus {
    /// The rule at this position.
    Rule(usize),
    /// No rule is left in view, so the list is gone: the form that adds one.
    AddForm,
}

/// Whether `entry` matches the search `query`, ignoring case.
fn matches(entry: &VocabularyReplacement, query: &str) -> bool {
    let query = query.trim().to_lowercase();
    query.is_empty()
        || entry.source.to_lowercase().contains(&query)
        || entry.replacement.to_lowercase().contains(&query)
}

/// Vocabulary: add, search, edit, reorder, delete, import and export rules.
#[component]
pub(super) fn VocabularyPage() -> Element {
    let actions = use_actions();
    let management = use_context::<UiContext>().management;
    let mut source = use_signal(String::new);
    let mut replacement = use_signal(String::new);
    let mut search = use_signal(String::new);
    let focus = use_signal(|| None::<RuleFocus>);
    let mut add_form = use_signal(|| None::<Rc<MountedData>>);
    use_effect(move || {
        if focus() == Some(RuleFocus::AddForm) {
            take_focus(add_form);
        }
    });
    let Some(settings) = management.read().shown_settings().cloned() else {
        return rsx! { SettingsLoading {} };
    };
    let vocabulary_status = management.read().setting_status(&SettingKey::Vocabulary);
    let (status, error) = status_parts(vocabulary_status);
    let status = status.unwrap_or_default();
    let filtering = !search().trim().is_empty();
    let total = settings.vocabulary_replacements.len();
    let shown = settings
        .vocabulary_replacements
        .iter()
        .cloned()
        .enumerate()
        .filter(|(_, entry)| matches(entry, &search()))
        .collect::<Vec<_>>();
    let shown_indices = shown.iter().map(|(index, _)| *index).collect::<Vec<_>>();
    // The rule focus moves to once the rule at `index` is gone and those after it moved up.
    let after_delete = |position: usize, index: usize| {
        neighbour(&shown_indices, position).map_or(RuleFocus::AddForm, |next| {
            RuleFocus::Rule(if next > index { next - 1 } else { next })
        })
    };
    rsx! { div { class: "settings-stack",
        SettingsBlockedReason {}
        Section { id: "vocabulary", title: "Vocabulary",
            description: "Fix words the model often gets wrong. Rules run from top to bottom, so a later rule sees the text an earlier one produced.",
            form { id: ADD_FORM, class: "vocabulary-add", tabindex: "-1", aria_label: "Add a rule",
                onmounted: move |event| add_form.set(Some(event.data())),
                onsubmit: move |event: FormEvent| {
                    event.prevent_default();
                    let entry = VocabularyReplacement::new(source().trim(), replacement().trim());
                    let add = move |settings: &mut Settings| {
                        settings.vocabulary_replacements.push(entry.clone());
                    };
                    if actions.change_setting(SettingKey::Vocabulary, add) {
                        source.set(String::new());
                        replacement.set(String::new());
                    }
                },
                Field { id: "vocabulary-source", label: "Replace this",
                    TextInput { value: "{source}", oninput: move |event: FormEvent| source.set(event.value()) }
                }
                Field { id: "vocabulary-replacement", label: "With this",
                    TextInput { value: "{replacement}", oninput: move |event: FormEvent| replacement.set(event.value()) }
                }
                div { class: "vocabulary-add-action",
                    button { r#type: "submit", id: "vocabulary-add", class: "btn btn-primary btn-md", "Add rule" }
                    p { class: "field-description", "Press Enter to add." }
                }
            }
            div { class: "vocabulary-feedback",
                span { id: "vocabulary-status", class: "save-status", role: "status", aria_live: "polite", "{status}" }
                if let Some(error) = error {
                    p { id: "vocabulary-error", class: "field-error-message", role: "alert", "{error}" }
                }
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
                            focus, after_delete: after_delete(position, index) }
                    }
                }
            }
        }
        VocabularyTransfer { rules: settings.vocabulary_replacements.clone() }
    } }
}

/// One rule: both texts edit in place; move and delete name the rule they act on. Deleting it
/// moves focus to `after_delete`.
#[component]
fn VocabularyRow(
    index: usize,
    entry: VocabularyReplacement,
    last: bool,
    reorder: bool,
    focus: Signal<Option<RuleFocus>>,
    after_delete: RuleFocus,
) -> Element {
    let actions = use_actions();
    let mut row = use_signal(|| None::<Rc<MountedData>>);
    use_effect(move || {
        if focus() == Some(RuleFocus::Rule(index)) {
            take_focus(row);
        }
    });
    let name = rule_name(&entry);
    let number = index + 1;
    let change = move |edit: fn(&mut Vec<VocabularyReplacement>, usize)| {
        actions.change_setting(SettingKey::Vocabulary, move |settings| {
            edit(&mut settings.vocabulary_replacements, index);
        })
    };
    let edit = move |edited: VocabularyReplacement| {
        actions.change_setting(SettingKey::Vocabulary, move |settings| {
            if let Some(rule) = settings.vocabulary_replacements.get_mut(index) {
                *rule = edited;
            }
        })
    };
    let replacement = entry.replacement.clone();
    let source = entry.source.clone();
    rsx! {
        li { id: "vocabulary-rule-{number}", class: "vocabulary-row", tabindex: "-1", aria_label: "Rule {number}, {name}",
            onmounted: move |event| row.set(Some(event.data())),
            DraftText { id: "vocabulary-rule-{number}-source", aria_label: "Replace this, rule {number}",
                value: entry.source.clone(),
                oncommit: move |text: String| edit(VocabularyReplacement::new(text.trim(), replacement.clone())),
            }
            span { class: "vocabulary-arrow", aria_hidden: "true", "→" }
            DraftText { id: "vocabulary-rule-{number}-replacement", aria_label: "With this, rule {number}",
                value: entry.replacement.clone(),
                oncommit: move |text: String| edit(VocabularyReplacement::new(source.clone(), text.trim())),
            }
            div { class: "vocabulary-row-actions",
                if reorder {
                    IconButton { icon: IconName::ArrowUp, label: "Move rule {name} up", disabled: index == 0,
                        onclick: move |_| { change(|rules, index| if let Some(above) = index.checked_sub(1) { rules.swap(above, index) }); } }
                    IconButton { icon: IconName::ArrowDown, label: "Move rule {name} down", disabled: last,
                        onclick: move |_| {
                            change(|rules, index| if index + 1 < rules.len() { rules.swap(index, index + 1) });
                        } }
                }
                IconButton { id: "vocabulary-rule-{number}-delete", icon: IconName::Trash, label: "Delete rule {name}",
                    onclick: move |_| {
                        let mut focus = focus;
                        if change(|rules, index| if index < rules.len() { rules.remove(index); }) {
                            focus.set(Some(after_delete));
                        }
                    }
                }
            }
        }
    }
}

/// Copying the rules out as JSON, and adding rules pasted as JSON.
#[component]
fn VocabularyTransfer(rules: Vec<VocabularyReplacement>) -> Element {
    let actions = use_actions();
    let mut management = use_context::<UiContext>().management;
    let mut pasted = use_signal(String::new);
    let export = serde_json::to_string_pretty(
        &rules
            .iter()
            .map(|rule| TransferRule {
                source: rule.source.clone(),
                replacement: rule.replacement.clone(),
            })
            .collect::<Vec<_>>(),
    )
    .unwrap_or_default();
    rsx! {
        Section { id: "vocabulary-transfer", title: "Import and export",
            description: "Rules move as JSON, a list of {{\"source\": …, \"replacement\": …}} pairs. JSON keeps commas, quotes and line breaks inside a rule intact, which a spreadsheet format would not.",
            div { class: "card-actions",
                Button { id: "vocabulary-export", disabled: rules.is_empty(),
                    onclick: move |_| actions.copy_text(export.clone(), NoticeScope::Settings),
                    "Copy rules as JSON"
                }
            }
            Field { id: "vocabulary-import", label: "Paste rules to import",
                description: "A pasted rule for text that already has one replaces its replacement.",
                textarea { id: "vocabulary-import", class: "text-input", rows: "4",
                    aria_describedby: "vocabulary-import-description",
                    value: "{pasted}", oninput: move |event| pasted.set(event.value()) }
            }
            div { class: "card-actions",
                Button { id: "vocabulary-import-submit", variant: ButtonVariant::Secondary,
                    disabled: pasted().trim().is_empty(),
                    onclick: move |_| match serde_json::from_str::<Vec<TransferRule>>(pasted().trim()) {
                        Ok(entries) => if actions.change_setting(SettingKey::Vocabulary, move |settings| import_rules(settings, &entries)) {
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
fn import_rules(settings: &mut Settings, entries: &[TransferRule]) {
    let rules = &mut settings.vocabulary_replacements;
    for entry in entries {
        match rules.iter_mut().find(|rule| rule.source == entry.source) {
            Some(rule) => rule.replacement.clone_from(&entry.replacement),
            None => rules.push(VocabularyReplacement::new(
                entry.source.clone(),
                entry.replacement.clone(),
            )),
        }
    }
}
