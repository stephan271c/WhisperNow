mod admission;
mod client;
mod components;
mod management;
mod notice;
mod overlay;
mod presentation;
mod route;
mod state;

use admission::{BlockedReason, UiOperation, blocked};
use components::Navigator;
// The two windows' roots; `app` picks one.
pub use components::{AppShell, RecordingBar};
use management::{HistoryCopied, ManagementState, SaveStatus, SettingKey, SetupEdit, SetupTest};
use notice::{Notice, NoticeAction, NoticeScope};
use overlay::OverlayView;
use presentation::{HomeStatus, HomeStatusKind};
use route::{Route, SettingsRoute, TopLevel, nav_keyboard_target};
use state::{EventStreamHealth, UiState};

#[cfg(target_arch = "wasm32")]
use client::{ListenerOwner, current_window_label};

use dioxus::prelude::*;

#[derive(Clone, Copy, PartialEq)]
struct UiContext {
    state: Signal<UiState>,
    management: Signal<ManagementState>,
    navigator: Navigator,
}

#[cfg(target_arch = "wasm32")]
pub fn app() -> Element {
    let mut state = use_signal(UiState::default);
    let management = use_signal(ManagementState::default);
    let navigator = Navigator::new(use_signal(|| Route::Home), use_signal(|| 0));
    let context = use_context_provider(|| UiContext {
        state,
        management,
        navigator,
    });
    let actions = components::UiActions::new(context);
    // Follows a theme change at once; the recording bar uses theme-independent inverse colours.
    use_effect(move || client::apply_theme(management.read().theme_attribute().map(str::to_owned)));
    let mut listeners = use_signal(|| None::<ListenerOwner>);
    let window_label = use_signal(current_window_label);
    use_future(move || async move {
        let recording_bar = window_label() == "recording-bar";
        let mut event_state = state;
        // Tauri calls the listener from JavaScript with no Dioxus scope on the stack, so the
        // callback re-enters this component's scope before `spawn` looks it up.
        let runtime = dioxus::core::Runtime::current();
        let scope = dioxus::core::current_scope_id();
        match client::listen_all(&whispernow_api::EVENT_NAMES, move |event| {
            runtime.in_scope(scope, || {
                if recording_bar {
                    event_state.write().apply_event(event);
                } else {
                    actions.apply_event(event);
                }
            });
        })
        .await
        {
            Ok(owner) => {
                listeners.set(Some(owner));
                state.write().mark_event_stream_listening();
            }
            Err(error) => state
                .write()
                .report_event_stream_failure(error.api_error().cloned()),
        }

        if recording_bar {
            hydrate_recording_bar(state).await;
            return;
        }

        match client::get_update_state().await {
            Ok(snapshot) => state.write().hydrate_update(snapshot),
            Err(error) => state
                .write()
                .report_update_command_failure(error.api_error().map(|api| api.summary.clone())),
        }
        actions.refresh_all();
    });

    rsx! {
        // Both windows share the tokens and primitives; the overlay remaps them to inverse colours.
        link { rel: "stylesheet", href: "/tokens.css" }
        link { rel: "stylesheet", href: "/ui.css" }
        if window_label() == "recording-bar" {
            link { rel: "stylesheet", href: "/overlay.css" }
            RecordingBar {}
        } else {
            link { rel: "stylesheet", href: "/settings.css" }
            AppShell {}
        }
    }
}

/// Installs the recording bar's first snapshot; a newer event wins over it.
#[cfg(target_arch = "wasm32")]
async fn hydrate_recording_bar(mut state: Signal<UiState>) {
    match client::get_state().await {
        Ok(snapshot) => state.write().hydrate(snapshot),
        Err(error) => state
            .write()
            .report_client_error(error.api_error().cloned()),
    }
}

#[cfg(test)]
mod test_support;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

#[cfg(all(test, target_arch = "wasm32"))]
mod browser_tests;
