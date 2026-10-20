//! The pill renders the published state snapshot itself; this module only decides where the
//! window is and whether it shows.

use whispernow_domain::ipc::{AppState, ErrorCategory, Phase};
use whispernow_domain::{AppError, WindowGeometry};

mod tauri_window;

pub(crate) use tauri_window::{apply, monitor_area};

/// Logical size of the transparent overlay window; the pill grows inside it.
pub(crate) const OVERLAY_LOGICAL_SIZE: (f64, f64) = (360.0, 132.0);
/// Logical distance between the overlay and the work-area edge.
const EDGE_MARGIN: f64 = 48.0;
/// A notification has no actions, so it points to History, where the text can be copied.
const NOT_PASTED_NOTIFICATION: &str =
    "The text was not pasted. Open History in WhisperNow to copy it.";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OverlayPresentation {
    None,
    Window,
    /// Native Wayland cannot keep an always-on-top window; a failure becomes an OS
    /// notification instead.
    Notification,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum OverlayAction {
    Show,
    Hide,
    Notify(String),
}

/// Physical monitor geometry. The work area excludes docks and panels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MonitorArea {
    pub(crate) bounds: WindowGeometry,
    pub(crate) work_area: WindowGeometry,
    pub(crate) scale_factor: f64,
}

pub(crate) fn overlaps(first: &WindowGeometry, second: &WindowGeometry) -> bool {
    let (left, top) = (i64::from(first.x), i64::from(first.y));
    let (other_left, other_top) = (i64::from(second.x), i64::from(second.y));
    left < other_left + i64::from(second.width)
        && top < other_top + i64::from(second.height)
        && other_left < left + i64::from(first.width)
        && other_top < top + i64::from(first.height)
}

fn contains(rect: &WindowGeometry, (x, y): (f64, f64)) -> bool {
    let left = f64::from(rect.x);
    let top = f64::from(rect.y);
    x >= left && x < left + f64::from(rect.width) && y >= top && y < top + f64::from(rect.height)
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
                .find(|monitor| contains(&monitor.bounds, cursor))
        })
        .or_else(|| monitors.first())
}

/// Physical top-left corner that centres a window of `logical_size` horizontally in the work
/// area, against its bottom edge. Converting with the target monitor's own scale factor
/// keeps placement correct on mixed-DPI setups.
#[allow(
    clippy::cast_possible_truncation,
    reason = "rounded window dimensions and margins are far inside i32"
)]
pub(crate) fn overlay_origin(monitor: &MonitorArea, logical_size: (f64, f64)) -> (i32, i32) {
    let scale = monitor.scale_factor;
    let width = (logical_size.0 * scale).round() as i32;
    let height = (logical_size.1 * scale).round() as i32;
    let margin = (EDGE_MARGIN * scale).round() as i32;
    let work = monitor.work_area;
    let work_width = i32::try_from(work.width).unwrap_or(i32::MAX);
    let work_height = i32::try_from(work.height).unwrap_or(i32::MAX);
    let x = work.x + (work_width - width) / 2;
    (x, work.y + work_height - height - margin)
}

/// Where a phase is presented, given whether the session can keep a window floating above
/// others (not native Wayland).
pub(crate) const fn overlay_presentation(phase: &Phase, can_float: bool) -> OverlayPresentation {
    match (phase, can_float) {
        (Phase::Idle | Phase::Paused | Phase::ClearingData | Phase::Updating, _) => {
            OverlayPresentation::None
        }
        (_, true) => OverlayPresentation::Window,
        (Phase::Error, false) => OverlayPresentation::Notification,
        (_, false) => OverlayPresentation::None,
    }
}

fn notification(error: &AppError) -> String {
    if error.category == ErrorCategory::Output {
        NOT_PASTED_NOTIFICATION.to_owned()
    } else {
        error.summary.clone()
    }
}

pub(crate) struct RecordingOverlay {
    can_float: bool,
    presented: OverlayPresentation,
}

impl RecordingOverlay {
    pub(crate) const fn new(can_float: bool) -> Self {
        Self {
            can_float,
            presented: OverlayPresentation::None,
        }
    }

    /// The action `snapshot` calls for; none while the presentation is unchanged.
    pub(crate) fn observe(&mut self, snapshot: &AppState) -> Option<OverlayAction> {
        let presentation = overlay_presentation(&snapshot.phase, self.can_float);
        let previous = std::mem::replace(&mut self.presented, presentation);
        if presentation == previous {
            return None;
        }
        match presentation {
            OverlayPresentation::Window => Some(OverlayAction::Show),
            OverlayPresentation::None => {
                (previous == OverlayPresentation::Window).then_some(OverlayAction::Hide)
            }
            OverlayPresentation::Notification => snapshot
                .error
                .as_ref()
                .map(|error| OverlayAction::Notify(notification(error))),
        }
    }
}

#[cfg(test)]
mod tests;
