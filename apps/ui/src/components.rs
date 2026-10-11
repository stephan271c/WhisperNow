use std::{cell::RefCell, collections::HashMap, rc::Rc};

use dioxus::prelude::*;
use whispernow_domain::ModelAvailability;
use whispernow_domain::ipc::{ModelState, UpdatePhase};

use crate::{
    Notice, NoticeScope, Route, SettingsRoute, TopLevel, UiContext, UiOperation,
    nav_keyboard_target,
};

mod actions;
mod ai_cleanup;
mod configuration;
mod data;
mod history;
mod home;
mod labels;
mod models;
mod overlay;
mod setup;
mod shared;
mod titlebar;
pub(crate) mod ui;
mod updates;
mod vocabulary;

use actions::DisabledReason;
use ai_cleanup::AiCleanupPage;
use configuration::{AudioPage, GeneralPage, PrivacyPage};
use data::ClearDataCard;
use history::HistoryScreen;
use home::HomePage;
use models::ModelsPage;
pub use overlay::RecordingBar;
use setup::SetupScreen;
use shared::{PasteHelperNotice, ShortcutCapture};
use titlebar::{TitleBar, use_window_chrome};
use ui::focus::{focus, take_focus};
use ui::{BrandMark, Button, ButtonSize, ButtonVariant, Icon, IconButton, IconName};
use updates::AboutPage;
use vocabulary::VocabularyPage;

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) use configuration::cleanup_destination;

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
    #[cfg_attr(
        not(any(test, target_arch = "wasm32")),
        expect(dead_code, reason = "called by the browser's window root and by tests")
    )]
    pub(crate) const fn new(route: Signal<Route>, focus_requests: Signal<u64>) -> Self {
        Self {
            route,
            focus_requests,
        }
    }

    pub(crate) fn route(self) -> Route {
        (self.route)()
    }

    /// Shows `route` and moves keyboard focus to its heading.
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

/// The sidebar links in order; Settings pages are listed while Settings is open.
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
pub fn AppShell() -> Element {
    let context = use_context::<UiContext>();
    let navigator = context.navigator;
    let mut heading = use_signal(|| None::<Rc<MountedData>>);
    use_effect(move || {
        if (navigator.focus_requests)() == 0 {
            return;
        }
        #[cfg(target_arch = "wasm32")]
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
            class: "wn-app settings-shell",
            class: if chrome.custom { "custom-chrome" },
            class: if chrome.maximized { "maximized" },
            a { class: "skip-link", href: "#main-content", "Skip to content" }
            aside { class: "settings-sidebar",
                header {
                    class: "brand",
                    "data-tauri-drag-region": chrome.custom.then_some("deep"),
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
            if chrome.custom {
                TitleBar { maximized: chrome.maximized }
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
    // Mounted links by id; kept outside the reactive graph because mounting is not a change.
    let links = use_hook(|| Rc::new(RefCell::new(HashMap::<&'static str, Rc<MountedData>>::new())));
    let ids = entries.iter().map(|entry| entry.id).collect::<Vec<_>>();
    let mut focused = use_signal(|| 0_usize);
    rsx! {
        nav { aria_label: "Main",
            onkeydown: {
                let links = links.clone();
                move |event: KeyboardEvent| {
                let key = event.key().to_string();
                let Some(target) = nav_keyboard_target(&key, focused(), count) else {
                    return;
                };
                event.prevent_default();
                let link = ids.get(target).and_then(|id| links.borrow().get(id).cloned());
                if let Some(link) = link {
                    focus(link);
                }
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
                            onmounted: {
                                let links = links.clone();
                                move |event: MountedEvent| {
                                    links.borrow_mut().insert(entry.id, event.data());
                                }
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
    let state = context.state.read();
    let management = context.management.read();
    let selected = state.snapshot.as_ref().map(|snapshot| {
        (
            snapshot.selected_model_id.as_str().to_owned(),
            snapshot.model_state,
        )
    });
    let installed = management
        .speech_models()
        .filter(|model| model.availability != ModelAvailability::NotInstalled)
        .map(|model| (model.id.clone(), model.display_name.clone()))
        .collect::<Vec<_>>();
    let (model_name, model_state) = selected.as_ref().map_or_else(
        || ("Connecting…".to_owned(), None),
        |(id, model_state)| (management.model_display_name(id), Some(*model_state)),
    );
    let dot = match model_state {
        Some(ModelState::Ready) => "status-dot",
        Some(ModelState::Failed) => "status-dot error",
        _ => "status-dot waiting",
    };
    let state_label = model_state.map_or("Connecting", labels::model_state);
    let switch_blocked = context
        .blocked(UiOperation::ModelSwitch)
        .or(context.settings_blocked());
    let update = update_summary(state.update_snapshot.as_ref().map(|update| &update.phase));
    let selected_id = selected.map(|(id, _)| id).unwrap_or_default();
    rsx! {
        footer { class: "sidebar-footer",
            p { class: "model-status", role: "status", aria_live: "polite", aria_atomic: "true",
                span { class: dot, aria_hidden: "true" }
                span { class: "model-status-name", "{model_name}" }
                span { class: "model-status-state", "{state_label}" }
            }
            if installed.len() > 1 {
                label { class: "visually-hidden", r#for: "footer-model", "Speech model" }
                select {
                    id: "footer-model",
                    class: "select footer-model",
                    value: "{selected_id}",
                    disabled: switch_blocked.is_some(),
                    aria_describedby: switch_blocked.map(|_| "footer-model-disabled-reason"),
                    onchange: move |event| {
                        let value = event.value();
                        let id = context.management.read().speech_models()
                            .find(|model| model.id.as_str() == value && model.availability != ModelAvailability::NotInstalled)
                            .map(|model| model.id.clone());
                        if let Some(id) = id {
                            context.switch_model(id);
                        }
                    },
                    for (id, name) in installed {
                        option { value: id.as_str(), selected: id.as_str() == selected_id, "{name}" }
                    }
                }
                DisabledReason { id: "footer-model-disabled-reason", reason: switch_blocked }
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
            span { class: "app-version", "Version {env!(\"CARGO_PKG_VERSION\")}" }
        }
    }
}

/// A short footer line for update phases that need the user.
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
    match route {
        Route::Home => rsx! { HomePage {} },
        Route::Setup => rsx! { SetupScreen {} },
        Route::History => rsx! {
            NoticeRegion { scope: NoticeScope::History }
            HistoryScreen {}
        },
        Route::Settings(page) => rsx! {
            NoticeRegion { scope: NoticeScope::Settings }
            SettingsPage { page }
        },
    }
}

/// One Settings section. Each control saves itself, so there is no page-wide Save.
#[component]
fn SettingsPage(page: SettingsRoute) -> Element {
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
    let auto_dismisses = notice.auto_dismisses();
    use_effect(use_reactive!(|id, auto_dismisses| {
        if auto_dismisses {
            spawn(async move {
                crate::client::delay(Notice::AUTO_DISMISS_MS).await;
                management.write().dismiss_notice(id);
            });
        }
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
