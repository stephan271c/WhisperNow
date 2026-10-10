use std::sync::{Arc, Mutex};

use dioxus::prelude::*;

use super::{
    Banner, ConfirmButton, Field, IconButton, IconName, ProgressBar, SettingRow, TextInput, Toggle,
    Tone,
};
use crate::test_support::{click_rendered, element, focused, mount_rendered, tag};

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
    for (value, max, expected) in [(2, 5, "40"), (9, 5, "100"), (0, 0, "0")] {
        let html = dioxus_ssr::render_element(
            rsx! { ProgressBar { label: "Model download", value, max } },
        );
        let bar = element(&html, "role=\"progressbar\"").expect("element");
        assert!(bar.contains("aria-label=\"Model download\""));
        assert!(bar.contains(&format!("aria-valuenow=\"{expected}\"")));
        assert!(bar.contains("aria-valuemax=\"100\""));
    }
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
