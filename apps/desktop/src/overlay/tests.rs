use whispernow_domain::ipc::{AppState, ErrorCategory, Phase};
use whispernow_domain::{AppError, WindowGeometry};

use crate::tests::{recording, snapshot};

use super::{
    MonitorArea, OverlayAction, RecordingOverlay, monitor_under_cursor, overlaps, overlay_origin,
};

fn failed(category: ErrorCategory) -> AppState {
    AppState {
        error: Some(AppError {
            category,
            code: "paste_failed".to_owned(),
            summary: "The text was not pasted; copy it instead".to_owned(),
        }),
        ..snapshot(Phase::Error)
    }
}

#[test]
fn a_dictation_shows_the_window_once_until_it_ends() {
    let mut overlay = RecordingOverlay::new(true);
    let actions: Vec<_> = [
        recording(),
        Phase::Processing { cleaning_up: false },
        Phase::Processing { cleaning_up: true },
        Phase::Idle,
    ]
    .into_iter()
    .map(|phase| overlay.observe(&snapshot(phase)))
    .collect();
    assert_eq!(
        actions,
        [
            Some(OverlayAction::Show),
            None,
            None,
            Some(OverlayAction::Hide)
        ]
    );
}

#[test]
fn a_failure_stays_until_the_phase_leaves_it() {
    let mut overlay = RecordingOverlay::new(true);
    assert_eq!(
        overlay.observe(&snapshot(recording())),
        Some(OverlayAction::Show)
    );
    assert_eq!(overlay.observe(&failed(ErrorCategory::Output)), None);
    assert_eq!(
        overlay.observe(&snapshot(Phase::Idle)),
        Some(OverlayAction::Hide)
    );
}

#[test]
fn native_wayland_notifies_each_failure_once_instead_of_showing_a_window() {
    let mut overlay = RecordingOverlay::new(false);
    assert_eq!(overlay.observe(&snapshot(recording())), None);
    assert_eq!(
        overlay.observe(&failed(ErrorCategory::Output)),
        Some(OverlayAction::Notify(
            "The text was not pasted. Open History in WhisperNow to copy it.".to_owned()
        ))
    );
    assert_eq!(overlay.observe(&failed(ErrorCategory::Output)), None);
    assert_eq!(overlay.observe(&snapshot(Phase::Idle)), None);
    assert_eq!(
        overlay.observe(&failed(ErrorCategory::General)),
        Some(OverlayAction::Notify(
            "The text was not pasted; copy it instead".to_owned()
        ))
    );
}

fn monitor(x: i32, width: u32, scale_factor: f64) -> MonitorArea {
    MonitorArea {
        bounds: WindowGeometry {
            x,
            y: 0,
            width,
            height: 1_080,
        },
        work_area: WindowGeometry {
            x,
            y: 0,
            width,
            height: 1_040,
        },
        scale_factor,
    }
}

#[test]
fn the_overlay_uses_the_monitor_under_the_cursor_or_the_primary_monitor() {
    let primary = monitor(0, 1_920, 1.0);
    let secondary = monitor(1_920, 3_840, 2.0);
    let monitors = [primary, secondary];
    assert_eq!(
        monitor_under_cursor(Some((2_500.0, 400.0)), &monitors),
        Some(&secondary)
    );
    assert_eq!(
        monitor_under_cursor(Some((100.0, 400.0)), &monitors),
        Some(&primary)
    );
    assert_eq!(monitor_under_cursor(None, &monitors), Some(&primary));
    assert_eq!(
        monitor_under_cursor(Some((-50.0, 400.0)), &monitors),
        Some(&primary)
    );
    assert_eq!(monitor_under_cursor(None, &[]), None);
}

#[test]
fn placement_is_centred_on_the_bottom_edge_at_the_monitor_scale() {
    let size = (360.0, 132.0);
    assert_eq!(
        overlay_origin(&monitor(0, 1_920, 1.0), size),
        (780, 1_040 - 132 - 48)
    );
    // A 200% monitor to the right of the primary one: physical sizes double.
    assert_eq!(
        overlay_origin(&monitor(1_920, 3_840, 2.0), size),
        (1_920 + (3_840 - 720) / 2, 1_040 - 264 - 96)
    );
}

#[test]
fn rectangles_overlap_only_when_they_share_area() {
    let rect = WindowGeometry {
        x: 0,
        y: 0,
        width: 100,
        height: 100,
    };
    let at = |x, y| WindowGeometry { x, y, ..rect };
    assert!(overlaps(&rect, &at(99, 99)));
    assert!(overlaps(&rect, &at(-99, 0)));
    assert!(!overlaps(&rect, &at(100, 0)));
    assert!(!overlaps(&rect, &at(0, -100)));
}
