use std::sync::mpsc;

use tauri::{Emitter as _, Manager as _};
use whispernow_domain::ipc::{AppState, EventPayload};

use super::Gateway;
use crate::overlay::RecordingOverlay;

/// Starts the thread that forwards every published event, in publication order. It ends when
/// the last event sender is dropped.
pub(crate) fn forward_events(
    app: tauri::AppHandle,
    mut overlay: RecordingOverlay,
    receiver: mpsc::Receiver<EventPayload>,
) -> std::io::Result<()> {
    std::thread::Builder::new()
        .name("whispernow-events".to_owned())
        .spawn(move || {
            let mut shortcuts_enabled = true;
            for event in receiver {
                if let EventPayload::StateChanged { state } = &event {
                    observe(&app, &mut overlay, state, &mut shortcuts_enabled);
                }
                emit(&app, event);
            }
        })
        .map(drop)
}

fn observe(
    app: &tauri::AppHandle,
    overlay: &mut RecordingOverlay,
    state: &AppState,
    shortcuts_enabled: &mut bool,
) {
    overlay.observe(state);
    let Some(app_state) = app.try_state::<Gateway>() else {
        return;
    };
    app_state.tray.apply_snapshot(state);
    app_state.observe_dictation(state.phase.is_busy());
    let enabled = state.phase.shortcuts_enabled();
    if enabled != *shortcuts_enabled {
        *shortcuts_enabled = enabled;
        let app = app.clone();
        tauri::async_runtime::spawn_blocking(move || {
            app.state::<Gateway>().sync_shortcuts();
        });
    }
    if app_state
        .activation
        .release_deferred_activation(&state.phase)
    {
        crate::show_settings_window(app);
    }
}

pub(crate) fn emit(app: &tauri::AppHandle, event: EventPayload) {
    let event_name = event.event_name();
    if app.emit(event_name, event).is_err() {
        tracing::error!(event_name, "desktop event emission failed");
    }
}
