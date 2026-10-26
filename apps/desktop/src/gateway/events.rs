use std::sync::mpsc;

use tauri::{Emitter as _, Manager as _};
use whispernow_domain::ipc::{AUDIO_LEVEL_EVENT, AppState, EVENT, EventPayload};

use super::Gateway;
use crate::overlay::RecordingOverlay;
use crate::services::Published;

pub(crate) fn forward_events(
    app: tauri::AppHandle,
    mut overlay: RecordingOverlay,
    receiver: mpsc::Receiver<Published>,
) -> std::io::Result<()> {
    std::thread::Builder::new()
        .name("whispernow-events".to_owned())
        .spawn(move || {
            let mut shortcuts_enabled = true;
            for published in receiver {
                match published {
                    Published::Event(event) => {
                        if let EventPayload::StateChanged { state } = &event {
                            observe(&app, &mut overlay, state, &mut shortcuts_enabled);
                        }
                        emit(&app, EVENT, event);
                    }
                    Published::Level(level) => emit(&app, AUDIO_LEVEL_EVENT, level),
                }
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
    if let Some(action) = overlay.observe(state) {
        crate::overlay::apply(app, action);
    }
    let gateway = app.state::<Gateway>();
    gateway.tray.apply_snapshot(state);
    gateway.observe_dictation(state.phase.is_busy());
    let enabled = state.phase.shortcuts_enabled();
    if enabled != *shortcuts_enabled {
        *shortcuts_enabled = enabled;
        let app = app.clone();
        tauri::async_runtime::spawn_blocking(move || {
            app.state::<Gateway>().sync_shortcuts();
        });
    }
    if gateway.activation.release_deferred_activation(&state.phase) {
        crate::show_settings_window(app);
    }
}

fn emit(app: &tauri::AppHandle, name: &str, payload: impl serde::Serialize + Clone) {
    if let Err(error) = app.emit(name, payload) {
        tracing::error!(%error, "desktop event emission failed");
    }
}
