use std::sync::{Arc, Mutex};

use dioxus::prelude::*;

use super::{
    Banner, Combobox, ConfirmButton, Field, IconButton, IconName, ProgressBar, Select, SettingRow,
    TextInput, Toggle, Tone,
};
use crate::test_support::{Page, click_rendered, element, focused, mount_rendered, tag, text};

/// Renders `app` inside a runtime so its event handlers can be created.
fn render(app: fn() -> Element) -> String {
    let mut dom = VirtualDom::new(app);
    dom.rebuild_in_place();
    dioxus_ssr::render(&dom)
}

#[test]
fn field_describes_its_control_with_the_description_and_error() {
    let html = render(|| {
        rsx! {
            Field { id: "endpoint", label: "API endpoint", description: "HTTPS only.", error: "Not HTTPS.",
                TextInput { value: "http://example.com", oninput: |_| {} }
            }
        }
    });
    let label = tag(&html, "endpoint-label").expect("label");
    assert!(label.starts_with("<label"));
    assert!(label.contains("for=\"endpoint\""));
    let input = tag(&html, "endpoint").expect("element");
    assert!(input.starts_with("<input"));
    assert!(input.contains("aria-describedby=\"endpoint-description endpoint-error\""));
    assert!(input.contains("aria-invalid=\"true\""));
    assert!(html.contains("id=\"endpoint-description\""));
    assert!(
        tag(&html, "endpoint-error")
            .expect("element")
            .contains("role=\"alert\"")
    );

    let valid = render(|| {
        rsx! {
            Field { id: "name", label: "Name",
                TextInput { value: "", oninput: |_| {} }
            }
        }
    });
    let input = tag(&valid, "name").expect("element");
    assert!(!input.contains("aria-describedby"));
    assert!(!input.contains("aria-invalid"));
    assert!(!valid.contains("role=\"alert\""));
}

#[test]
fn setting_row_toggle_is_a_labelled_switch() {
    let html = render(|| {
        rsx! {
            SettingRow { id: "start-login", title: "Start on login", description: "Opens at sign-in.",
                Toggle { checked: true, onchange: |_| {} }
            }
        }
    });
    let switch = element(&html, "role=\"switch\"").expect("element");
    assert!(switch.starts_with("<button"));
    assert!(switch.contains("id=\"start-login\""));
    assert!(switch.contains("aria-checked=\"true\""));
    assert!(switch.contains("aria-labelledby=\"start-login-label\""));
    assert!(switch.contains("aria-describedby=\"start-login-description\""));
    assert!(
        tag(&html, "start-login-label")
            .expect("element")
            .contains("for=\"start-login\"")
    );

    let standalone = render(|| {
        rsx! {
            Toggle { checked: false, label: "Mute", disabled: true, onchange: |_| {} }
        }
    });
    let switch = element(&standalone, "role=\"switch\"").expect("element");
    assert!(switch.contains("aria-checked=\"false\""));
    assert!(switch.contains("aria-label=\"Mute\""));
    assert!(switch.contains("disabled"));
}

#[test]
fn progress_bar_always_has_a_name_and_a_clamped_value() {
    for (downloaded, total, expected) in [
        (2, Some(5), "40"),
        (9, Some(5), "100"),
        (0, Some(0), "0"),
        (0, None, "0"),
        (2, None, "100"),
    ] {
        let html = dioxus_ssr::render_element(
            rsx! { ProgressBar { label: "Model download", downloaded, total } },
        );
        let bar = element(&html, "role=\"progressbar\"").expect("element");
        assert!(bar.contains("aria-label=\"Model download\""));
        assert!(bar.contains(&format!("aria-valuenow=\"{expected}\"")));
        assert!(bar.contains("aria-valuemax=\"100\""));
    }
}

#[test]
fn progress_bar_describes_known_and_unknown_download_sizes() {
    let known = dioxus_ssr::render_element(rsx! {
        div { id: "download-size",
            ProgressBar { label: "Model download", downloaded: 2_048, total: Some(4_096) }
        }
    });
    assert!(known.contains("aria-valuetext=\"2.0 KiB of 4.0 KiB downloaded\""));
    assert_eq!(
        text(&known, "download-size").trim(),
        "2.0 KiB of 4.0 KiB downloaded"
    );

    let unknown = dioxus_ssr::render_element(rsx! {
        div { id: "download-size",
            ProgressBar { label: "Model download", downloaded: 2_048, total: None }
        }
    });
    assert!(unknown.contains("aria-valuetext=\"2.0 KiB downloaded\""));
    assert_eq!(text(&unknown, "download-size").trim(), "2.0 KiB downloaded");
}

#[test]
fn banners_announce_warnings_and_errors_assertively() {
    for (tone, role) in [
        (Tone::Info, "status"),
        (Tone::Success, "status"),
        (Tone::Warning, "alert"),
        (Tone::Error, "alert"),
    ] {
        let html = dioxus_ssr::render_element(rsx! { Banner { tone, p { "message" } } });
        assert!(html.contains(&format!("role=\"{role}\"")), "{tone:?}");
    }
}

#[test]
fn icon_buttons_are_named_by_their_label() {
    let html = render(|| rsx! { IconButton { icon: IconName::Close, label: "Dismiss" } });
    let button = element(&html, "<button").expect("element");
    assert!(button.contains("aria-label=\"Dismiss\""));
    assert!(button.contains("type=\"button\""));
    assert!(
        element(&html, "<svg")
            .expect("element")
            .contains("aria-hidden=\"true\"")
    );
}

#[derive(Clone, Default)]
struct ConfirmLog(Arc<Mutex<usize>>);

impl PartialEq for ConfirmLog {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

#[component]
fn ConfirmSurface(log: ConfirmLog) -> Element {
    rsx! {
        ConfirmButton { id: "delete", label: "Delete prompt", confirm_label: "Delete permanently",
            onconfirm: move |()| *log.0.lock().expect("confirm log") += 1,
        }
    }
}

#[test]
fn confirm_button_acts_only_on_the_second_explicit_press() {
    let log = ConfirmLog::default();
    let mut dom =
        VirtualDom::new_with_props(ConfirmSurface, ConfirmSurfaceProps { log: log.clone() });
    let initial = dom.rebuild_to_vec();

    let armed = click_rendered(&mut dom, &initial, "delete");
    assert_eq!(*log.0.lock().expect("confirm log"), 0);
    let disarmed = click_rendered(&mut dom, &armed, "delete-cancel");
    assert_eq!(*log.0.lock().expect("confirm log"), 0);

    let armed = click_rendered(&mut dom, &disarmed, "delete");
    click_rendered(&mut dom, &armed, "delete-confirm");
    assert_eq!(*log.0.lock().expect("confirm log"), 1);
}

#[test]
fn confirm_button_returns_focus_to_its_trigger_after_either_choice() {
    for choice in ["delete-confirm", "delete-cancel"] {
        let mut dom = VirtualDom::new_with_props(
            ConfirmSurface,
            ConfirmSurfaceProps {
                log: ConfirmLog::default(),
            },
        );
        let initial = dom.rebuild_to_vec();
        mount_rendered(&mut dom, &initial, "delete");
        dom.process_events();
        assert_eq!(focused(), None, "the first render leaves focus alone");

        let armed = click_rendered(&mut dom, &initial, "delete");
        let disarmed = click_rendered(&mut dom, &armed, choice);
        mount_rendered(&mut dom, &disarmed, "delete");
        dom.process_events();
        assert_eq!(
            focused().as_deref(),
            Some("delete"),
            "{choice}: the trigger takes focus back when it is mounted again"
        );
    }
}

#[derive(Clone, Default)]
struct ChoiceLog(Arc<Mutex<Vec<String>>>);

impl PartialEq for ChoiceLog {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl ChoiceLog {
    fn chosen(&self) -> Vec<String> {
        self.0.lock().expect("choice log").clone()
    }
}

#[component]
fn SelectSurface(log: ChoiceLog) -> Element {
    let options = [
        ("system", "Match system"),
        ("light", "Light"),
        ("dark", "Dark"),
    ]
    .map(|(value, label)| (value.to_owned(), label.to_owned()))
    .to_vec();
    rsx! {
        Field { id: "theme", label: "Theme",
            Select { value: "light", options,
                onchange: move |value| log.0.lock().expect("choice log").push(value),
            }
        }
    }
}

fn select_page(log: &ChoiceLog) -> Page {
    let log = log.clone();
    Page::new(|_| VirtualDom::new_with_props(SelectSurface, SelectSurfaceProps { log }))
}

fn highlighted(page: &Page, id: &str) -> Option<String> {
    let html = page.html();
    let trigger = tag(&html, id).expect("trigger");
    trigger
        .split("aria-activedescendant=\"")
        .nth(1)
        .and_then(|rest| rest.split('"').next())
        .map(str::to_owned)
}

#[test]
fn select_is_a_labelled_combobox_that_opens_on_its_value() {
    let log = ChoiceLog::default();
    let mut page = select_page(&log);
    let html = page.html();
    let trigger = tag(&html, "theme").expect("trigger");
    assert!(trigger.starts_with("<button"));
    assert!(trigger.contains("role=\"combobox\""));
    assert!(trigger.contains("aria-labelledby=\"theme-label\""));
    assert!(trigger.contains("aria-expanded=\"false\""));
    assert!(text(&html, "theme").contains("Light"));
    assert!(tag(&html, "theme-listbox").is_none());

    page.click("theme");
    let html = page.html();
    assert!(
        tag(&html, "theme")
            .expect("trigger")
            .contains("aria-expanded=\"true\"")
    );
    assert_eq!(
        highlighted(&page, "theme").as_deref(),
        Some("theme-option-1")
    );
    assert!(
        tag(&html, "theme-option-1")
            .expect("current option")
            .contains("aria-selected=\"true\"")
    );

    page.click("theme-option-2");
    assert_eq!(log.chosen(), ["dark"]);
    assert!(tag(&page.html(), "theme-listbox").is_none());
}

#[test]
fn select_keys_move_the_highlight_and_only_enter_or_a_click_chooses() {
    let log = ChoiceLog::default();
    let mut page = select_page(&log);
    page.key("theme", Key::ArrowDown);
    assert_eq!(
        highlighted(&page, "theme").as_deref(),
        Some("theme-option-1"),
        "the first arrow opens the list without moving"
    );
    page.key("theme", Key::ArrowDown)
        .key("theme", Key::ArrowDown);
    assert_eq!(
        highlighted(&page, "theme").as_deref(),
        Some("theme-option-2")
    );
    page.key("theme", Key::Home);
    assert_eq!(
        highlighted(&page, "theme").as_deref(),
        Some("theme-option-0")
    );
    page.key("theme", Key::Character("d".to_owned()));
    assert_eq!(
        highlighted(&page, "theme").as_deref(),
        Some("theme-option-2")
    );
    page.key("theme", Key::Escape);
    assert_eq!(highlighted(&page, "theme"), None);
    assert!(log.chosen().is_empty());

    page.key("theme", Key::End).key("theme", Key::Enter);
    assert_eq!(log.chosen(), ["dark"]);
    assert_eq!(highlighted(&page, "theme"), None);
}

#[test]
fn choosing_the_current_option_closes_the_list_without_a_change() {
    let log = ChoiceLog::default();
    let mut page = select_page(&log);
    page.click("theme").click("theme-option-1");
    assert!(log.chosen().is_empty());
    assert!(tag(&page.html(), "theme-listbox").is_none());
}

#[component]
fn ComboboxSurface(log: ChoiceLog) -> Element {
    let mut typed = use_signal(String::new);
    let suggestions = [
        "nvidia/nemotron-3.5-super-vl:free",
        "NVIDIA/llama-3.1-nemotron",
        "nousresearch/hermes-4",
        "openai/gpt-6-mini",
        "nemotron-mini-2",
        "nemotron-mini",
    ]
    .map(str::to_owned)
    .to_vec();
    rsx! {
        Combobox { id: "model", label: "Model", value: typed(), suggestions,
            oninput: move |value: String| {
                log.0.lock().expect("choice log").push(value.clone());
                typed.set(value);
            },
        }
    }
}

fn combobox_page(log: &ChoiceLog) -> Page {
    let log = log.clone();
    Page::new(|_| VirtualDom::new_with_props(ComboboxSurface, ComboboxSurfaceProps { log }))
}

fn listed(page: &Page) -> Vec<String> {
    let html = page.html();
    html.match_indices("id=\"model-option-")
        .filter_map(|(at, _)| html[at + "id=\"".len()..].split('"').next())
        .map(|option| text(&html, option))
        .collect()
}

#[test]
fn combobox_lists_the_suggestions_with_a_word_that_starts_with_the_typed_text() {
    let log = ChoiceLog::default();
    let mut page = combobox_page(&log);
    let html = page.html();
    let field = tag(&html, "model").expect("field");
    assert!(field.starts_with("<input"));
    assert!(field.contains("role=\"combobox\""));
    assert!(field.contains("aria-autocomplete=\"list\""));
    assert!(field.contains("aria-expanded=\"false\""));
    assert!(tag(&html, "model-listbox").is_none());

    page.input("model", "nvid");
    assert_eq!(
        listed(&page),
        [
            "nvidia/nemotron-3.5-super-vl:free",
            "NVIDIA/llama-3.1-nemotron"
        ],
        "matched from the start, ignoring case"
    );
    assert!(
        tag(&page.html(), "model")
            .expect("field")
            .contains("aria-expanded=\"true\"")
    );
    page.input("model", "nemotron");
    assert_eq!(
        listed(&page),
        [
            "nemotron-mini-2",
            "nemotron-mini",
            "nvidia/nemotron-3.5-super-vl:free",
            "NVIDIA/llama-3.1-nemotron"
        ],
        "suggestions that start with it come first, then the others in their order"
    );
    page.input("model", "nemotron-mini");
    assert_eq!(listed(&page), ["nemotron-mini", "nemotron-mini-2"]);
    page.input("model", "3.5");
    assert_eq!(listed(&page), ["nvidia/nemotron-3.5-super-vl:free"]);
    page.input("model", "emotron");
    assert!(
        listed(&page).is_empty(),
        "a match inside a word is not offered"
    );
    assert!(tag(&page.html(), "model-listbox").is_none());

    page.input("model", "");
    assert!(tag(&page.html(), "model-listbox").is_none());
    page.key("model", Key::ArrowDown);
    assert_eq!(
        listed(&page).len(),
        6,
        "Down in an empty field lists everything"
    );
}

#[test]
fn combobox_keys_highlight_and_only_enter_on_a_highlight_fills_the_field() {
    let log = ChoiceLog::default();
    let mut page = combobox_page(&log);
    page.input("model", "n").key("model", Key::Enter);
    assert!(tag(&page.html(), "model-listbox").is_none());
    assert_eq!(
        log.chosen(),
        ["n"],
        "Enter without a highlight only closes the list"
    );

    page.input("model", "nv").key("model", Key::ArrowDown);
    assert_eq!(
        highlighted(&page, "model").as_deref(),
        Some("model-option-0")
    );
    page.key("model", Key::ArrowDown)
        .key("model", Key::ArrowDown);
    assert_eq!(
        highlighted(&page, "model").as_deref(),
        Some("model-option-1"),
        "the highlight stops at the last suggestion"
    );
    page.key("model", Key::ArrowUp);
    assert_eq!(
        highlighted(&page, "model").as_deref(),
        Some("model-option-0")
    );
    page.key("model", Key::Enter);
    assert_eq!(
        log.chosen().last().map(String::as_str),
        Some("nvidia/nemotron-3.5-super-vl:free")
    );
    assert!(tag(&page.html(), "model-listbox").is_none());

    page.input("model", "o").key("model", Key::Escape);
    assert!(tag(&page.html(), "model-listbox").is_none());
    assert_eq!(log.chosen().last().map(String::as_str), Some("o"));
}

#[test]
fn clicking_a_suggestion_fills_the_field() {
    let log = ChoiceLog::default();
    let mut page = combobox_page(&log);
    page.input("model", "OPEN").click("model-option-0");
    assert_eq!(log.chosen(), ["OPEN", "openai/gpt-6-mini"]);
    let html = page.html();
    assert!(
        tag(&html, "model")
            .expect("field")
            .contains("value=\"openai/gpt-6-mini\"")
    );
    assert!(tag(&html, "model-listbox").is_none());
}
