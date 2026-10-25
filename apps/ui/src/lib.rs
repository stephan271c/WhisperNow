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
use client::{ListenerOwner, current_window_label};
use components::{AppShell, Navigator, RecordingBar};
use dioxus::prelude::*;
use management::ManagementState;
use route::Route;
use state::UiState;

#[derive(Clone, Copy, PartialEq)]
struct UiContext {
    state: Signal<UiState>,
    management: Signal<ManagementState>,
    navigator: Navigator,
    /// Apart from `state` so a level reading every 50 ms re-renders only the meter that shows it.
    audio_rms: Signal<f32>,
    /// Many controls check these; a snapshot that changes neither re-renders none of them.
    connected: Memo<bool>,
    phase_reason: Memo<Option<BlockedReason>>,
}

impl UiContext {
    pub(crate) fn provide(state: Signal<UiState>, management: Signal<ManagementState>) -> Self {
        let navigator = Navigator::new(use_signal(|| Route::Home), use_signal(|| 0));
        let audio_rms = use_signal(|| 0.0);
        let connected = use_memo(move || state.read().snapshot.is_some());
        let phase_reason = use_memo(move || blocked_reason::phase_reason(&state.read()));
        use_context_provider(|| Self {
            state,
            management,
            navigator,
            audio_rms,
            connected,
            phase_reason,
        })
    }
}

pub fn app() -> Element {
    let mut state = use_signal(|| UiState {
        recording_bar: current_window_label() == "recording-bar",
        ..UiState::default()
    });
    let recording_bar = state.peek().recording_bar;
    let management = use_signal(ManagementState::default);
    let context = UiContext::provide(state, management);
    // Follows a theme change at once; the recording bar uses theme-independent inverse colours.
    let theme = use_memo(move || management.read().theme_attribute());
    use_effect(move || client::apply_theme(theme()));
    use_effect(client::track_scroll_activity);
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
            Err(_) => state.write().event_stream_failed = true,
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
