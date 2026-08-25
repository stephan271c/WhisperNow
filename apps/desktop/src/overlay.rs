//! The recording overlay: the recording-bar window, shown while a dictation runs and while its
//! failure waits to be dismissed.
//!
//! The pill renders the published state snapshot itself; this module only decides where the
//! window is and whether it shows. Placement and visibility are pure functions so they can be
//! tested without a window; [`OverlayWindow`] is the only native boundary.

use std::sync::{Arc, Mutex};

use whispernow_api::{
    ApiErrorDto, AppStateDto, OverlayPosition, OverlayStyle, PhaseDto, RecoveryActionDto,
};

mod tauri_window;

pub(crate) use tauri_window::{TauriOverlayWindow, monitor_area};

/// Logical size of the transparent overlay window; the pill grows inside it.
pub(crate) const OVERLAY_LOGICAL_SIZE: (f64, f64) = (360.0, 132.0);
/// Logical distance between the overlay and the work-area edge.
const EDGE_MARGIN: f64 = 48.0;
/// A notification has no actions, so it points to History, where the text can be copied.
const NOT_PASTED_NOTIFICATION: &str =
    "The text was not pasted. Open History in WhisperNow to copy it.";

/// Where the overlay is presented for a phase.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OverlayPresentation {
    None,
    Window,
    /// Native Wayland cannot keep an always-on-top window; a failure becomes an OS
    /// notification instead.
    Notification,
}

/// The native side of the overlay. Every method only queues work (a main-thread window change
/// or a notification), so the overlay may call it under its own lock.
pub(crate) trait OverlayWindow: Send + Sync {
    /// Places the window on the monitor under the cursor and shows it without focus.
    fn show(&self, position: OverlayPosition);
    fn hide(&self);
    fn notify(&self, message: String);
}

/// Physical monitor geometry. The work area excludes docks and panels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MonitorArea {
    pub(crate) bounds: PhysicalRect,
    pub(crate) work_area: PhysicalRect,
    pub(crate) scale_factor: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PhysicalRect {
    pub(crate) x: i32,
    pub(crate) y: i32,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl PhysicalRect {
    /// Whether the two rectangles share any area.
    pub(crate) fn overlaps(&self, other: &Self) -> bool {
        let (left, top) = (i64::from(self.x), i64::from(self.y));
        let (other_left, other_top) = (i64::from(other.x), i64::from(other.y));
        left < other_left + i64::from(other.width)
            && top < other_top + i64::from(other.height)
            && other_left < left + i64::from(self.width)
            && other_top < top + i64::from(self.height)
    }

    fn contains(&self, (x, y): (f64, f64)) -> bool {
        let left = f64::from(self.x);
        let top = f64::from(self.y);
        x >= left
            && x < left + f64::from(self.width)
            && y >= top
            && y < top + f64::from(self.height)
    }
}

/// The monitor containing the cursor. Without a cursor position (native Wayland) or when it is
/// outside every monitor, the first monitor is used; callers list the primary monitor first.
pub(crate) fn monitor_under_cursor(
    cursor: Option<(f64, f64)>,
    monitors: &[MonitorArea],
) -> Option<&MonitorArea> {
    cursor
        .and_then(|cursor| {
            monitors
                .iter()
                .find(|monitor| monitor.bounds.contains(cursor))
        })
        .or_else(|| monitors.first())
}

/// Physical top-left corner that centres a window of `logical_size` horizontally in the work
/// area, against its top or bottom edge. Converting with the target monitor's own scale factor
/// keeps placement correct on mixed-DPI setups.
#[allow(
    clippy::cast_possible_truncation,
    reason = "rounded window dimensions and margins are far inside i32"
)]
pub(crate) fn overlay_origin(
    monitor: &MonitorArea,
    position: OverlayPosition,
    logical_size: (f64, f64),
) -> (i32, i32) {
    let scale = monitor.scale_factor;
    let width = (logical_size.0 * scale).round() as i32;
    let height = (logical_size.1 * scale).round() as i32;
    let margin = (EDGE_MARGIN * scale).round() as i32;
    let work = monitor.work_area;
    let work_width = i32::try_from(work.width).unwrap_or(i32::MAX);
    let work_height = i32::try_from(work.height).unwrap_or(i32::MAX);
    let x = work.x + (work_width - width) / 2;
    let y = match position {
        OverlayPosition::Top => work.y + margin,
        OverlayPosition::Bottom => work.y + work_height - height - margin,
    };
    (x, y)
}

/// Where a phase is presented under the chosen style, given whether the session can keep a
/// window floating above others (not native Wayland).
pub(crate) const fn overlay_presentation(
    style: OverlayStyle,
    phase: &PhaseDto,
    can_float: bool,
) -> OverlayPresentation {
    match (style, phase, can_float) {
        (OverlayStyle::Off, _, _)
        | (_, PhaseDto::Idle | PhaseDto::Paused | PhaseDto::Updating, _) => {
            OverlayPresentation::None
        }
        (_, _, true) => OverlayPresentation::Window,
        (_, PhaseDto::Error { .. }, false) => OverlayPresentation::Notification,
        (_, _, false) => OverlayPresentation::None,
    }
}

/// The message a failure notification shows.
fn notification(error: &ApiErrorDto) -> String {
    if error.recovery == Some(RecoveryActionDto::CopyManually) {
        NOT_PASTED_NOTIFICATION.to_owned()
    } else {
        error.summary.clone()
    }
}

#[derive(Default)]
struct Presented {
    window_shown: bool,
    notified: bool,
}

/// Shows and hides the recording bar as snapshots arrive. Its lock is a leaf, and the window
/// layer only queues work under it, which keeps window changes in snapshot order.
pub(crate) struct RecordingOverlay {
    window: Arc<dyn OverlayWindow>,
    can_float: bool,
    presented: Mutex<Presented>,
}

impl RecordingOverlay {
    pub(crate) fn new(window: Arc<dyn OverlayWindow>, can_float: bool) -> Self {
        Self {
            window,
            can_float,
            presented: Mutex::default(),
        }
    }

    /// Follows a published state snapshot.
    pub(crate) fn observe(&self, snapshot: &AppStateDto) {
        let presentation =
            overlay_presentation(snapshot.overlay_style, &snapshot.phase, self.can_float);
        let mut presented = self.presented.lock().expect("overlay lock poisoned");
        let show = presentation == OverlayPresentation::Window;
        if show && !presented.window_shown {
            self.window.show(snapshot.overlay_position);
        } else if !show && presented.window_shown {
            self.window.hide();
        }
        presented.window_shown = show;
        let notify = presentation == OverlayPresentation::Notification;
        if notify
            && !presented.notified
            && let PhaseDto::Error { error } = &snapshot.phase
        {
            self.window.notify(notification(error));
        }
        presented.notified = notify;
    }
}

#[cfg(test)]
mod tests;
