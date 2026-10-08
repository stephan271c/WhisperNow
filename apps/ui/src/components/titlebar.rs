use dioxus::prelude::*;

use super::ui::{Icon, IconName};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct WindowChrome {
    /// The window is undecorated, so the page draws the titlebar and its controls.
    pub(super) custom: bool,
    pub(super) maximized: bool,
}

/// Reads the window frame once, then follows maximize changes while the shell is mounted.
/// Natively there is no window, so the shell keeps the native-frame layout.
pub(super) fn use_window_chrome() -> Signal<WindowChrome> {
    let chrome = use_signal(WindowChrome::default);
    #[cfg(target_arch = "wasm32")]
    {
        let mut listener = use_signal(|| None::<crate::client::ListenerOwner>);
        use_future(move || async move {
            let mut chrome = chrome;
            if !crate::client::has_custom_titlebar().await {
                return;
            }
            chrome.write().custom = true;
            // Tauri calls back from JavaScript with no Dioxus scope on the stack.
            let runtime = dioxus::core::Runtime::current();
            let scope = dioxus::core::current_scope_id();
            // Without the listener the maximize button keeps its label; it still toggles.
            if let Ok(owner) = crate::client::listen_maximized(move |maximized| {
                runtime.in_scope(scope, || chrome.write().maximized = maximized);
            })
            .await
            {
                listener.set(Some(owner));
            }
        });
    }
    chrome
}

#[derive(Clone, Copy)]
enum WindowAction {
    Minimize,
    ToggleMaximize,
    Close,
}

#[cfg_attr(
    not(target_arch = "wasm32"),
    expect(unused_variables, reason = "natively there is no window to act on")
)]
fn run(action: WindowAction) {
    // A refused window operation leaves the window as it was, which the user can see.
    #[cfg(target_arch = "wasm32")]
    spawn(async move {
        let _ = match action {
            WindowAction::Minimize => crate::client::minimize_window().await,
            WindowAction::ToggleMaximize => crate::client::toggle_maximize_window().await,
            WindowAction::Close => crate::client::close_window().await,
        };
    });
}

/// The strip above the page: a window drag region holding the window controls. Its buttons
/// are excluded from dragging by Tauri; a double-click elsewhere toggles maximize.
#[component]
pub(super) fn TitleBar(maximized: bool) -> Element {
    let (maximize_icon, maximize_label) = if maximized {
        (IconName::Restore, "Restore")
    } else {
        (IconName::Maximize, "Maximize")
    };
    rsx! {
        div { class: "titlebar", "data-tauri-drag-region": "deep",
            div { class: "window-controls", role: "group", aria_label: "Window",
                button {
                    id: "window-minimize",
                    class: "window-control",
                    r#type: "button",
                    aria_label: "Minimize",
                    title: "Minimize",
                    onclick: move |_| run(WindowAction::Minimize),
                    Icon { name: IconName::Minimize }
                }
                button {
                    id: "window-maximize",
                    class: "window-control",
                    r#type: "button",
                    aria_label: maximize_label,
                    title: maximize_label,
                    onclick: move |_| run(WindowAction::ToggleMaximize),
                    Icon { name: maximize_icon }
                }
                button {
                    id: "window-close",
                    class: "window-control window-close",
                    r#type: "button",
                    aria_label: "Close",
                    title: "Close",
                    onclick: move |_| run(WindowAction::Close),
                    Icon { name: IconName::Close }
                }
            }
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use dioxus::prelude::*;

    use super::{TitleBar, TitleBarProps};

    fn render(maximized: bool) -> String {
        let mut dom = VirtualDom::new_with_props(TitleBar, TitleBarProps { maximized });
        dom.rebuild_in_place();
        dioxus_ssr::render(&dom)
    }

    #[test]
    fn the_whole_strip_drags_the_window_while_its_buttons_stay_clickable() {
        let html = render(false);

        // "deep" makes empty space anywhere in the strip a drag handle; Tauri never starts a
        // drag from a button, so the controls need no opt-out of their own.
        assert!(
            html.contains(r#"class="titlebar" data-tauri-drag-region="deep""#),
            "{html}"
        );
        for (id, label) in [
            ("window-minimize", "Minimize"),
            ("window-maximize", "Maximize"),
            ("window-close", "Close"),
        ] {
            assert!(
                html.contains(&format!(r#"id="{id}""#)),
                "{id} is drawn: {html}"
            );
            assert!(
                html.contains(&format!(r#"aria-label="{label}""#)),
                "{id} is named: {html}"
            );
        }
    }

    #[test]
    fn a_maximized_window_offers_restore_instead_of_maximize() {
        let html = render(true);

        assert!(html.contains(r#"aria-label="Restore""#), "{html}");
        assert!(!html.contains(r#"aria-label="Maximize""#), "{html}");
    }
}
