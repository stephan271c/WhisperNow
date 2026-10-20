mod blocked_reason;
mod client;
mod components;
mod home_status;
mod management;
mod notice;
mod operations;
mod overlay;
mod route;
mod state;

use blocked_reason::BlockedReason;
use components::Navigator;
use operations::UiOperation;
// The two windows' roots; `app` picks one.
pub use components::{AppShell, RecordingBar};
use home_status::{HomeStatus, HomeStatusKind};
use management::{Copied, CopyTarget, ManagementState, SettingKey, SetupTest};
use notice::{Notice, NoticeScope};
use overlay::OverlayView;
use route::{Route, SettingsRoute, TopLevel, nav_keyboard_target};
use state::UiState;

use client::{ListenerOwner, current_window_label};

use dioxus::prelude::*;

#[derive(Clone, Copy, PartialEq)]
struct UiContext {
    state: Signal<UiState>,
    management: Signal<ManagementState>,
    navigator: Navigator,
}

impl UiContext {
    async fn hydrate_state(mut self) {
        match client::call("get_state").await {
            Ok(snapshot) => self.state.write().hydrate(snapshot),
            Err(error) => self.state.write().report_client_error(error),
        }
    }
}

pub fn app() -> Element {
    let mut state = use_signal(|| UiState {
        recording_bar: current_window_label() == "recording-bar",
        ..UiState::default()
    });
    let recording_bar = state.peek().recording_bar;
    let management = use_signal(ManagementState::default);
    let navigator = Navigator::new(use_signal(|| Route::Home), use_signal(|| 0));
    let context = use_context_provider(|| UiContext {
        state,
        management,
        navigator,
    });
    // Follows a theme change at once; the recording bar uses theme-independent inverse colours.
    use_effect(move || client::apply_theme(management.read().theme_attribute()));
    let mut listeners = use_signal(|| None::<ListenerOwner>);
    use_future(move || async move {
        let mut event_state = state;
        // Tauri calls the listener from JavaScript with no Dioxus scope on the stack, so the
        // callback re-enters this component's scope before `spawn` looks it up.
        let runtime = dioxus::core::Runtime::current();
        let scope = dioxus::core::current_scope_id();
        match client::listen_events(move |event| {
            runtime.in_scope(scope, || {
                if recording_bar {
                    event_state.write().apply_event(event);
                } else {
                    context.apply_event(event);
                }
            });
        })
        .await
        {
            Ok(owner) => {
                listeners.set(Some(owner));
                state.write().event_stream_failed = false;
            }
            Err(error) => state.write().report_event_stream_failure(error),
        }

        if recording_bar {
            context.hydrate_state().await;
            return;
        }

        let answer = client::call("get_update_state").await;
        if state.peek().update_snapshot.is_none() {
            match answer {
                Ok(snapshot) => state.write().apply_update_state(snapshot),
                Err(error) => state.write().report_update_command_failure(error.summary),
            }
        }
        context.refresh_all();
    });

    rsx! {
        // Both windows share the tokens and primitives; the overlay remaps them to inverse colours.
        link { rel: "stylesheet", href: "/tokens.css" }
        link { rel: "stylesheet", href: "/ui.css" }
        if recording_bar {
            link { rel: "stylesheet", href: "/overlay.css" }
            RecordingBar {}
        } else {
            link { rel: "stylesheet", href: "/settings.css" }
            AppShell {}
        }
    }
}

#[cfg(test)]
mod test_fixtures;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod test_support;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

#[cfg(all(test, target_arch = "wasm32"))]
mod browser_tests;
