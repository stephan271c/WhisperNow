use tauri::Manager as _;
use whispernow_domain::WindowGeometry;

use super::{
    MonitorArea, OVERLAY_LOGICAL_SIZE, OverlayAction, monitor_under_cursor, overlay_origin,
};
use crate::RECORDING_BAR_WINDOW;

pub(crate) fn apply(app: &tauri::AppHandle, action: OverlayAction) {
    match action {
        OverlayAction::Show => crate::on_window(app, RECORDING_BAR_WINDOW, place_and_show),
        OverlayAction::Hide => crate::hide_recording_bar(app),
        OverlayAction::Notify(message) => notify(message),
    }
}

#[cfg(target_os = "linux")]
fn notify(message: String) {
    use ashpd::desktop::notification::{Notification, NotificationProxy, Priority};
    tauri::async_runtime::spawn(async move {
        let sent = async {
            NotificationProxy::new()
                .await?
                .add_notification(
                    "whispernow-dictation",
                    Notification::new("WhisperNow")
                        .body(message.as_str())
                        .priority(Priority::High),
                )
                .await
        };
        if let Err(error) = sent.await {
            tracing::warn!(%error, "dictation failure notification could not be shown");
        }
    });
}

/// Only native Wayland sessions ask for notifications.
#[cfg(not(target_os = "linux"))]
fn notify(_message: String) {}

fn place_and_show(window: &tauri::WebviewWindow) {
    let app = window.app_handle();
    let primary = app
        .primary_monitor()
        .ok()
        .flatten()
        .map(|monitor| monitor_area(&monitor).bounds);
    let mut monitors: Vec<MonitorArea> = app
        .available_monitors()
        .unwrap_or_default()
        .iter()
        .map(monitor_area)
        .collect();
    // The primary monitor is the fallback when the cursor position is unknown.
    monitors.sort_by_key(|monitor| Some(monitor.bounds) != primary);
    let cursor = app
        .cursor_position()
        .ok()
        .map(|cursor| (cursor.x, cursor.y));
    if let Some(monitor) = monitor_under_cursor(cursor, &monitors) {
        let (x, y) = overlay_origin(monitor, OVERLAY_LOGICAL_SIZE);
        // Move first so the logical size is applied at the destination monitor's scale.
        let placed = window
            .set_position(tauri::PhysicalPosition::new(x, y))
            .and_then(|()| {
                window.set_size(tauri::LogicalSize::new(
                    OVERLAY_LOGICAL_SIZE.0,
                    OVERLAY_LOGICAL_SIZE.1,
                ))
            });
        if let Err(error) = placed {
            tracing::warn!(%error, "recording overlay could not be positioned");
        }
    }
    if let Err(error) = window.show() {
        tracing::warn!(%error, "recording overlay could not be shown");
    }
}

pub(crate) fn monitor_area(monitor: &tauri::window::Monitor) -> MonitorArea {
    let work_area = monitor.work_area();
    MonitorArea {
        bounds: WindowGeometry {
            x: monitor.position().x,
            y: monitor.position().y,
            width: monitor.size().width,
            height: monitor.size().height,
        },
        work_area: WindowGeometry {
            x: work_area.position.x,
            y: work_area.position.y,
            width: work_area.size.width,
            height: work_area.size.height,
        },
        scale_factor: monitor.scale_factor(),
    }
}
