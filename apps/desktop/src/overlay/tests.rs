use std::sync::{Arc, Mutex};
use whispernow_domain::AppError;

use whispernow_domain::ipc::{ErrorCategory, Phase};

use crate::tests::{recording, snapshot};

use super::{
    MonitorArea, OverlayPresentation, OverlayWindow, PhysicalRect, RecordingOverlay,
    monitor_under_cursor, overlay_origin, overlay_presentation,
};

#[derive(Default)]
struct FakeWindow(Mutex<Vec<String>>);

impl FakeWindow {
    fn calls(&self) -> Vec<String> {
        self.0.lock().expect("window log").clone()
    }
}

impl OverlayWindow for FakeWindow {
    fn show(&self) {
        self.0.lock().expect("window log").push("show".to_owned());
    }

    fn hide(&self) {
        self.0.lock().expect("window log").push("hide".to_owned());
    }

    fn notify(&self, message: String) {
        self.0
            .lock()
            .expect("window log")
            .push(format!("notify {message}"));
    }
}

fn failed(category: ErrorCategory) -> Phase {
    Phase::Error {
        error: AppError {
            category,
            code: "paste_failed".to_owned(),
            summary: "The text was not pasted; copy it instead".to_owned(),
        },
    }
}

fn overlay(can_float: bool) -> (RecordingOverlay, Arc<FakeWindow>) {
    let window = Arc::new(FakeWindow::default());
    (RecordingOverlay::new(window.clone(), can_float), window)
}

#[test]
fn a_dictation_shows_the_window_once_until_it_ends() {
    let (overlay, window) = overlay(true);
    for phase in [
        recording(),
        Phase::Processing { cleaning_up: false },
        Phase::Processing { cleaning_up: true },
        Phase::Idle,
    ] {
        overlay.observe(&snapshot(phase));
    }
    assert_eq!(window.calls(), ["show", "hide"]);
}

#[test]
fn a_failure_stays_until_the_phase_leaves_it() {
    let (overlay, window) = overlay(true);
    overlay.observe(&snapshot(recording()));
    overlay.observe(&snapshot(failed(ErrorCategory::Output)));
    assert_eq!(window.calls(), ["show"]);
    overlay.observe(&snapshot(Phase::Idle));
    assert_eq!(window.calls(), ["show", "hide"]);
}

#[test]
fn native_wayland_notifies_each_failure_once_instead_of_showing_a_window() {
    let (overlay, window) = overlay(false);
    overlay.observe(&snapshot(recording()));
    overlay.observe(&snapshot(failed(ErrorCategory::Output)));
    overlay.observe(&snapshot(failed(ErrorCategory::Output)));
    overlay.observe(&snapshot(Phase::Idle));
    overlay.observe(&snapshot(failed(ErrorCategory::General)));
    assert_eq!(
        window.calls(),
        [
            "notify The text was not pasted. Open History in WhisperNow to copy it.",
            "notify The text was not pasted; copy it instead",
        ]
    );
}

#[test]
fn visibility_policy_covers_every_phase() {
    for (phase, window, wayland) in [
        (
            Phase::Idle,
            OverlayPresentation::None,
            OverlayPresentation::None,
        ),
        (
            Phase::Paused,
            OverlayPresentation::None,
            OverlayPresentation::None,
        ),
        (
            Phase::Updating,
            OverlayPresentation::None,
            OverlayPresentation::None,
        ),
        (
            recording(),
            OverlayPresentation::Window,
            OverlayPresentation::None,
        ),
        (
            Phase::Processing { cleaning_up: false },
            OverlayPresentation::Window,
            OverlayPresentation::None,
        ),
        (
            failed(ErrorCategory::General),
            OverlayPresentation::Window,
            OverlayPresentation::Notification,
        ),
    ] {
        assert_eq!(overlay_presentation(&phase, true), window);
        assert_eq!(overlay_presentation(&phase, false), wayland);
    }
}

fn monitor(x: i32, width: u32, scale_factor: f64) -> MonitorArea {
    MonitorArea {
        bounds: PhysicalRect {
            x,
            y: 0,
            width,
            height: 1_080,
        },
        work_area: PhysicalRect {
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
    let rect = PhysicalRect {
        x: 0,
        y: 0,
        width: 100,
        height: 100,
    };
    let at = |x, y| PhysicalRect { x, y, ..rect };
    assert!(rect.overlaps(&at(99, 99)));
    assert!(rect.overlaps(&at(-99, 0)));
    assert!(!rect.overlaps(&at(100, 0)));
    assert!(!rect.overlaps(&at(0, -100)));
}
