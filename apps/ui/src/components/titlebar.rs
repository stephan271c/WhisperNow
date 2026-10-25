use dioxus::prelude::*;

use super::ui::{Icon, IconName};
use crate::client::{Platform, WindowAction, WindowFrame, run_window_action};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct WindowChrome {
    pub(super) frame: WindowFrame,
    pub(super) platform: Platform,
    pub(super) maximized: bool,
}

impl WindowChrome {
    pub(super) fn has_strip(self) -> bool {
        self.frame != WindowFrame::Native
    }

    pub(super) fn shell_class(self) -> String {
        let frame = match self.frame {
            WindowFrame::Drawn => " custom-chrome",
            WindowFrame::Overlay => " overlay-chrome",
            WindowFrame::Native => "",
        };
        let maximized = if self.maximized { " maximized" } else { "" };
        format!("wn-app settings-shell{frame}{maximized}")
    }

    pub(super) fn platform_name(self) -> Option<&'static str> {
        self.has_strip().then(|| self.platform.name())
    }
}

/// Reads the window frame once, then follows maximize changes while the shell is mounted.
/// Natively there is no window, so the shell keeps the native-frame layout.
pub(super) fn use_window_chrome() -> Signal<WindowChrome> {
    let chrome = use_signal(WindowChrome::default);
    let mut listener = use_signal(|| None::<crate::client::ListenerOwner>);
    use_future(move || async move {
        let mut chrome = chrome;
        let frame = crate::client::window_frame().await;
        *chrome.write() = WindowChrome {
            frame,
            platform: crate::client::host_platform(),
            maximized: false,
        };
        // The overlay's native buttons track maximize themselves.
        if frame != WindowFrame::Drawn {
            return;
        }
        // Tauri calls back from JavaScript with no Dioxus scope on the stack.
        let runtime = dioxus::core::Runtime::current();
        let scope = dioxus::core::current_scope_id();
        // Without the listener the maximize button keeps its label; it still toggles.
        if let Ok(owner) = crate::client::listen_maximized(move |maximized| {
            // Resizing reports on every frame; only a change may re-render the shell.
            runtime.in_scope(scope, || {
                if chrome.peek().maximized != maximized {
                    chrome.write().maximized = maximized;
                }
            });
        })
        .await
        {
            listener.set(Some(owner));
        }
    });
    chrome
}

/// The strip above the page: a window drag region, holding the window controls when the page
/// draws them. Its buttons are excluded from dragging by Tauri; a double-click elsewhere
/// toggles maximize.
#[component]
pub(super) fn TitleBar(frame: WindowFrame, maximized: bool) -> Element {
    let (maximize_icon, maximize_label) = if maximized {
        (IconName::Restore, "Restore")
    } else {
        (IconName::Maximize, "Maximize")
    };
    rsx! {
        div { class: "titlebar", "data-tauri-drag-region": "deep",
            if frame == WindowFrame::Drawn {
                div { class: "window-controls", role: "group", aria_label: "Window",
                    button {
                        id: "window-minimize",
                        class: "window-control",
                        r#type: "button",
                        aria_label: "Minimize",
                        title: "Minimize",
                        onclick: move |_| run_window_action(WindowAction::Minimize),
                        Icon { name: IconName::Minimize }
                    }
                    button {
                        id: "window-maximize",
                        class: "window-control",
                        r#type: "button",
                        aria_label: maximize_label,
                        title: maximize_label,
                        onclick: move |_| run_window_action(WindowAction::ToggleMaximize),
                        Icon { name: maximize_icon }
                    }
                    button {
                        id: "window-close",
                        class: "window-control window-close",
                        r#type: "button",
                        aria_label: "Close",
                        title: "Close",
                        onclick: move |_| run_window_action(WindowAction::Close),
                        Icon { name: IconName::Close }
                    }
                }
            }
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use dioxus::prelude::*;
    use whispernow_domain::ipc::Phase;

    use super::{TitleBar, TitleBarProps};
    use crate::client::{Platform, WindowFrame, fake};
    use crate::components::AppShell;
    use crate::management::ManagementState;
    use crate::state::UiState;
    use crate::test_support::{Page, snapshot};

    fn render(frame: WindowFrame, maximized: bool) -> String {
        let mut dom = VirtualDom::new_with_props(TitleBar, TitleBarProps { frame, maximized });
        dom.rebuild_in_place();
        dioxus_ssr::render(&dom)
    }

    fn shell_on(frame: WindowFrame, platform: Platform) -> String {
        let mut state = UiState::default();
        state.hydrate(snapshot(Phase::Idle));
        let mut page = Page::host(state, ManagementState::default(), AppShell);
        fake::set_frame(frame, platform);
        page.settle();
        page.html()
    }

    #[test]
    fn the_whole_strip_drags_the_window_while_its_buttons_stay_clickable() {
        let html = render(WindowFrame::Drawn, false);

        // "deep" makes empty space anywhere in the strip a drag handle; Tauri never starts a
        // drag from a button, so the controls need no opt-out of their own.
        assert!(html.contains(r#"data-tauri-drag-region="deep""#), "{html}");
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
        let html = render(WindowFrame::Drawn, true);

        assert!(html.contains(r#"aria-label="Restore""#), "{html}");
        assert!(!html.contains(r#"aria-label="Maximize""#), "{html}");
    }

    #[test]
    fn the_overlay_strip_drags_but_leaves_the_buttons_to_macos() {
        let html = render(WindowFrame::Overlay, false);

        assert!(html.contains(r#"data-tauri-drag-region="deep""#), "{html}");
        assert!(!html.contains("<button"), "{html}");
        assert!(!html.contains("window-controls"), "{html}");
    }

    #[test]
    fn the_shell_marks_its_frame_and_platform_for_the_stylesheet() {
        for (frame, platform, class, name) in [
            (
                WindowFrame::Drawn,
                Platform::Linux,
                "custom-chrome",
                "linux",
            ),
            (
                WindowFrame::Drawn,
                Platform::Windows,
                "custom-chrome",
                "windows",
            ),
            (
                WindowFrame::Overlay,
                Platform::Macos,
                "overlay-chrome",
                "macos",
            ),
        ] {
            let html = shell_on(frame, platform);

            assert!(html.contains(class), "{class}: {html}");
            assert!(
                html.contains(&format!(r#"data-platform="{name}""#)),
                "{html}"
            );
            assert!(html.contains(r#"class="titlebar""#), "{html}");
        }
    }

    #[test]
    fn a_native_frame_gets_no_strip_and_no_marks() {
        let html = shell_on(WindowFrame::Native, Platform::Windows);

        assert!(!html.contains("titlebar"), "{html}");
        assert!(!html.contains("-chrome"), "{html}");
        assert!(!html.contains("data-platform"), "{html}");
    }
}
