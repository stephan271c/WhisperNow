use std::{collections::HashMap, rc::Rc};

use dioxus::prelude::*;
use whispernow_domain::SetupStep;
use whispernow_domain::ipc::{ModelState, UpdatePhase};

use crate::UiContext;
use crate::notice::{Notice, NoticeScope, Tone};
use crate::route::{Route, SettingsRoute, TopLevel, nav_keyboard_target};

mod actions;
mod ai_cleanup;
mod audio;
mod copy_button;
mod data;
mod general;
mod history;
mod home;
mod labels;
mod microphone_test;
mod models;
mod overlay;
mod paste_helper_notice;
mod privacy;
mod settings;
mod setup;
mod shortcut;
mod titlebar;
pub(crate) mod ui;
mod updates;
mod vocabulary;

use ai_cleanup::AiCleanupPage;
use audio::AudioPage;
use data::ClearDataCard;
use general::GeneralPage;
use history::HistoryScreen;
use home::HomePage;
use models::ModelsPage;
pub(crate) use overlay::RecordingBar;
use privacy::PrivacyPage;
use setup::SetupScreen;
use titlebar::{TitleBar, use_window_chrome};
use ui::focus::{focus, take_focus};
use ui::{Banner, BrandMark, Button, ButtonSize, ButtonVariant, Icon, IconButton, IconName};
use updates::AboutPage;
use vocabulary::VocabularyPage;

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) use privacy::cleanup_destination;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

/// The page heading's id; focus moves to it on navigation.
const PAGE_HEADING: &str = "page-heading";

#[derive(Clone, Copy, PartialEq)]
pub(crate) struct Navigator {
    route: Signal<Route>,
    focus_requests: Signal<u64>,
}

impl Navigator {
    pub(crate) const fn new(route: Signal<Route>, focus_requests: Signal<u64>) -> Self {
        Self {
            route,
            focus_requests,
        }
    }

    pub(crate) fn route(self) -> Route {
        (self.route)()
    }

    pub(crate) fn go(mut self, route: Route) {
        self.route.set(route);
        self.focus_requests += 1;
    }
}

#[derive(Clone, Copy, PartialEq)]
struct NavEntry {
    id: &'static str,
    label: &'static str,
    route: Route,
    icon: IconName,
    nested: bool,
}

fn nav_entries(current: Route) -> Vec<NavEntry> {
    let mut entries = Vec::new();
    for top in TopLevel::ALL {
        entries.push(NavEntry {
            id: top.nav_id(),
            label: top.label(),
            route: top.route(),
            icon: match top {
                TopLevel::Home => IconName::Home,
                TopLevel::History => IconName::History,
                TopLevel::Settings => IconName::Gear,
            },
            nested: false,
        });
        if top == TopLevel::Settings && current.top_level() == TopLevel::Settings {
            entries.extend(SettingsRoute::ALL.map(|page| NavEntry {
                id: page.nav_id(),
                label: page.label(),
                route: Route::Settings(page),
                icon: settings_icon(page),
                nested: true,
            }));
        }
    }
    entries
}

const fn settings_icon(page: SettingsRoute) -> IconName {
    match page {
        SettingsRoute::General => IconName::Sliders,
        SettingsRoute::Audio => IconName::Mic,
        SettingsRoute::SpeechModels => IconName::Wave,
        SettingsRoute::AiCleanup => IconName::Sparkles,
        SettingsRoute::Vocabulary => IconName::Book,
        SettingsRoute::PrivacyData => IconName::Shield,
        SettingsRoute::AboutUpdates => IconName::Info,
    }
}

/// Whether `entry` marks the current page. On a Settings page the sub-page link carries
/// the mark, so exactly one link is current.
fn is_current(entry: &NavEntry, current: Route) -> bool {
    if entry.nested {
        entry.route == current
    } else {
        current.top_level() == entry.route.top_level() && current.top_level() != TopLevel::Settings
    }
}

fn nav_class(entry: &NavEntry, current: Route) -> &'static str {
    if is_current(entry, current) {
        "nav-item selected"
    } else if !entry.nested && current.top_level() == entry.route.top_level() {
        "nav-item open"
    } else {
        "nav-item"
    }
}

#[component]
pub(crate) fn AppShell() -> Element {
    let context = use_context::<UiContext>();
    let navigator = context.navigator;
    let mut heading = use_signal(|| None::<Rc<MountedData>>);
    use_effect(move || {
        if (navigator.focus_requests)() == 0 {
            return;
        }
        crate::client::reset_settings_scroll();
        take_focus(heading);
    });
    let chrome = use_window_chrome()();
    let route = navigator.route();
    let toast = context
        .management
        .read()
        .notice
        .clone()
        .filter(|notice| !route.hosts(notice.scope));
    rsx! {
        div {
            class: chrome.shell_class(),
            "data-platform": chrome.platform_name(),
            a { class: "skip-link", href: "#main-content", "Skip to content" }
            aside { class: "settings-sidebar",
                header {
                    class: "brand",
                    "data-tauri-drag-region": chrome.has_strip().then_some("deep"),
                    span { class: "brand-mark", BrandMark {} }
                    strong { "WhisperNow" }
                }
                Sidebar { route }
                SidebarFooter {}
            }
            main {
                id: "main-content",
                class: "settings-content",
                tabindex: "-1",
                aria_labelledby: PAGE_HEADING,
                header { class: "page-header",
                    h1 {
                        id: PAGE_HEADING,
                        tabindex: "-1",
                        onmounted: move |event| heading.set(Some(event.data())),
                        "{route.title()}"
                    }
                }
                section { class: "toast-region", aria_label: "Notifications",
                    if let Some(notice) = toast {
                        NoticeView { notice, inline: false }
                    }
                }
                Page { route }
            }
            // Last in the document so keyboard focus reaches the page before the window controls.
            if chrome.has_strip() {
                TitleBar { frame: chrome.frame, maximized: chrome.maximized }
            }
        }
    }
}

#[component]
fn Sidebar(route: Route) -> Element {
    let context = use_context::<UiContext>();
    let navigator = context.navigator;
    let entries = nav_entries(route);
    let count = entries.len();
    // Mounted links by id; mounting is not a change, so nothing subscribes to it.
    let mut links = use_signal(HashMap::<&'static str, Rc<MountedData>>::new);
    let ids = entries.iter().map(|entry| entry.id).collect::<Vec<_>>();
    let mut focused = use_signal(|| 0_usize);
    rsx! {
        nav { aria_label: "Main",
            onkeydown: move |event: KeyboardEvent| {
                let Some(target) = nav_keyboard_target(&event.key(), focused(), count) else {
                    return;
                };
                event.prevent_default();
                let link = ids.get(target).and_then(|id| links.peek().get(id).cloned());
                if let Some(link) = link {
                    focus(link);
                }
            },
            ul { class: "nav-list",
                for (index, entry) in entries.into_iter().enumerate() {
                    li { key: "{entry.id}", class: if entry.nested { "nav-sub-item" } else { "nav-top-item" },
                        button {
                            r#type: "button",
                            id: entry.id,
                            class: nav_class(&entry, route),
                            aria_current: if is_current(&entry, route) { "page" } else { "false" },
                            onmounted: move |event: MountedEvent| {
                                links.write().insert(entry.id, event.data());
                            },
                            onfocus: move |_| focused.set(index),
                            onclick: move |_| navigator.go(entry.route),
                            Icon { name: entry.icon }
                            span { "{entry.label}" }
                        }
                    }
                }
            }
        }
    }
}

#[component]
fn SidebarFooter() -> Element {
    let context = use_context::<UiContext>();
    let navigator = context.navigator;
    let state = context.state;
    let model = use_memo(move || {
        state
            .read()
            .snapshot
            .as_ref()
            .map(|snapshot| (snapshot.selected_model_id.clone(), snapshot.model_state))
    })();
    let update = use_memo(move || {
        update_summary(
            state
                .read()
                .update_snapshot
                .as_ref()
                .map(|update| &update.phase),
        )
    })();
    let management = context.management.read();
    // Setup has chosen no model before its model step.
    let unchosen = management
        .unfinished_setup_step()
        .is_some_and(|step| step < SetupStep::Model);
    let model_name = model
        .as_ref()
        .filter(|_| !unchosen)
        .map(|(id, _)| management.model_display_name(id));
    let model_state = model.map(|(_, state)| state);
    let dot = match model_state {
        Some(ModelState::Ready) => "status-dot",
        Some(ModelState::Failed) => "status-dot error",
        _ => "status-dot waiting",
    };
    let state_label = if unchosen {
        ""
    } else {
        model_state.map_or("Connecting…", labels::model_state)
    };
    // Hidden, not removed: a live region must stay mounted to announce the change to Ready.
    let state_class = if unchosen || model_state == Some(ModelState::Ready) {
        "visually-hidden"
    } else {
        "model-status-state"
    };
    rsx! {
        footer { class: "sidebar-footer",
            div { class: "model-status",
                if !unchosen {
                    span { class: dot, aria_hidden: "true" }
                }
                if let Some(name) = model_name {
                    span { class: "model-status-name", "{name}" }
                }
                p { class: state_class, role: "status", aria_live: "polite", aria_atomic: "true", "{state_label}" }
            }
            if let Some(update) = update {
                Button {
                    id: "footer-update",
                    variant: ButtonVariant::Ghost,
                    size: ButtonSize::Sm,
                    onclick: move |_| navigator.go(Route::Settings(SettingsRoute::AboutUpdates)),
                    "{update}"
                }
            }
        }
    }
}

const fn update_summary(phase: Option<&UpdatePhase>) -> Option<&'static str> {
    match phase {
        Some(UpdatePhase::ReadyToRestart { .. }) => Some("Update ready to install"),
        Some(UpdatePhase::ManualInstallAvailable { .. }) => Some("Update available"),
        Some(UpdatePhase::Downloading { .. }) => Some("Downloading update"),
        Some(UpdatePhase::Error { .. }) => Some("Update needs attention"),
        _ => None,
    }
}

#[component]
fn Page(route: Route) -> Element {
    let settings_warnings = use_context::<UiContext>()
        .management
        .read()
        .settings
        .as_ref()
        .map(|view| view.warnings.clone())
        .unwrap_or_default();
    match route {
        Route::Home => rsx! { HomePage {} },
        Route::Setup => rsx! { SetupScreen {} },
        Route::History => rsx! {
            NoticeRegion { scope: NoticeScope::History }
            HistoryScreen {}
        },
        Route::Settings(page) => rsx! {
            for warning in settings_warnings {
                Banner { tone: Tone::Warning, p { "{warning}" } }
            }
            NoticeRegion { scope: NoticeScope::Settings }
            match page {
                SettingsRoute::General => rsx! { GeneralPage {} },
                SettingsRoute::Audio => rsx! { AudioPage {} },
                SettingsRoute::SpeechModels => rsx! { ModelsPage {} },
                SettingsRoute::AiCleanup => rsx! { AiCleanupPage {} },
                SettingsRoute::Vocabulary => rsx! { VocabularyPage {} },
                SettingsRoute::PrivacyData => rsx! {
                    div { class: "settings-stack", PrivacyPage {} ClearDataCard {} }
                },
                SettingsRoute::AboutUpdates => rsx! { AboutPage {} },
            }
        },
    }
}

#[component]
pub(super) fn NoticeRegion(scope: NoticeScope) -> Element {
    let context = use_context::<UiContext>();
    let navigator = context.navigator;
    let notice = context
        .management
        .read()
        .notice
        .clone()
        .filter(|notice| notice.scope == scope && navigator.route().hosts(scope));
    rsx! {
        div { class: "notice-region",
            if let Some(notice) = notice {
                NoticeView { notice, inline: true }
            }
        }
    }
}

#[component]
fn NoticeView(notice: Notice, inline: bool) -> Element {
    let context = use_context::<UiContext>();
    let mut management = context.management;
    let id = notice.id;
    use_resource(use_reactive!(|id| async move {
        crate::client::delay(Notice::AUTO_DISMISS_MS).await;
        management.write().dismiss_notice(id);
    }));
    let placement = if inline {
        "notice-inline"
    } else {
        "notice-toast"
    };
    rsx! {
        div { id: "notice", class: "toast {notice.tone.class()} {placement}", role: notice.tone.role(),
            span { class: "toast-icon", Icon { name: notice.tone.icon() } }
            p { class: "toast-message", "{notice.message}" }
            IconButton {
                id: "notice-dismiss",
                icon: IconName::Close,
                label: "Dismiss notification",
                size: ButtonSize::Sm,
                onclick: move |_| management.write().dismiss_notice(id),
            }
        }
    }
}
