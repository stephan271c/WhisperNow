use tauri::Manager as _;

use super::{
    MonitorArea, OVERLAY_LOGICAL_SIZE, OverlayWindow, PhysicalRect, monitor_under_cursor,
    overlay_origin,
};
use crate::RECORDING_BAR_WINDOW;

pub(crate) struct TauriOverlayWindow {
    app: tauri::AppHandle,
}

impl TauriOverlayWindow {
    pub(crate) const fn new(app: tauri::AppHandle) -> Self {
        Self { app }
    }
}

impl OverlayWindow for TauriOverlayWindow {
    fn show(&self) {
        let app = self.app.clone();
        if self
            .app
            .run_on_main_thread(move || place_and_show(&app))
            .is_err()
        {
            tracing::warn!("recording overlay show could not be scheduled");
        }
    }

    fn hide(&self) {
        crate::hide_recording_bar(&self.app);
    }

    #[cfg(target_os = "linux")]
    fn notify(&self, message: String) {
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
            if sent.await.is_err() {
                tracing::warn!("dictation failure notification could not be shown");
            }
        });
    }

    /// Only native Wayland sessions ask for notifications.
    #[cfg(not(target_os = "linux"))]
    fn notify(&self, _message: String) {}
}

/// Runs on the event-loop thread.
fn place_and_show(app: &tauri::AppHandle) {
    let Some(window) = app.get_webview_window(RECORDING_BAR_WINDOW) else {
        return;
    };
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
        if placed.is_err() {
            tracing::warn!("recording overlay could not be positioned");
        }
    }
    if window.show().is_err() {
        tracing::warn!("recording overlay could not be shown");
    }
}

pub(crate) fn monitor_area(monitor: &tauri::window::Monitor) -> MonitorArea {
    let work_area = monitor.work_area();
    MonitorArea {
        bounds: PhysicalRect {
            x: monitor.position().x,
            y: monitor.position().y,
            width: monitor.size().width,
            height: monitor.size().height,
        },
        work_area: PhysicalRect {
            x: work_area.position.x,
            y: work_area.position.y,
            width: work_area.size.width,
            height: work_area.size.height,
        },
        scale_factor: monitor.scale_factor(),
    }
}
