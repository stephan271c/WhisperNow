use std::sync::Mutex;

use tauri::{
    image::Image,
    menu::{Menu, MenuItem, PredefinedMenuItem},
    tray::{MouseButton, MouseButtonState, TrayIcon, TrayIconBuilder, TrayIconEvent},
};
use whispernow_domain::ipc::{AppState, ModelState, Phase};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TrayAction {
    Settings,
    CheckForUpdates,
    CopyLastTranscript,
    CancelRecording,
    /// Pauses, or resumes when dictation is paused at the time of the click.
    TogglePause,
    Quit,
}

impl TrayAction {
    const ALL: [Self; 6] = [
        Self::Settings,
        Self::CheckForUpdates,
        Self::CopyLastTranscript,
        Self::CancelRecording,
        Self::TogglePause,
        Self::Quit,
    ];

    const fn menu_id(self) -> &'static str {
        match self {
            Self::Settings => "tray-settings",
            Self::CheckForUpdates => "tray-check-updates",
            Self::CopyLastTranscript => "tray-copy-last",
            Self::CancelRecording => "tray-cancel",
            Self::TogglePause => "tray-pause",
            Self::Quit => "tray-quit",
        }
    }

    pub(crate) fn from_menu_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|action| action.menu_id() == id)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TrayState {
    Loading,
    Idle,
    Recording,
    RecordingError,
    Processing,
    Paused,
    Updating,
    ClearingData,
    Error,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TrayPresentation {
    pub(crate) status_text: &'static str,
    pub(crate) tooltip: &'static str,
}

impl TrayState {
    pub(crate) const fn presentation(self) -> TrayPresentation {
        match self {
            Self::Loading => TrayPresentation {
                status_text: "Status: Loading…",
                tooltip: "WhisperNow — Loading",
            },
            Self::Idle => TrayPresentation {
                status_text: "Status: Ready",
                tooltip: "WhisperNow — Ready",
            },
            Self::Recording => TrayPresentation {
                status_text: "Status: Recording",
                tooltip: "WhisperNow — Recording",
            },
            Self::RecordingError => TrayPresentation {
                status_text: "Status: Recording — error; open WhisperNow",
                tooltip: "WhisperNow — Recording (error)",
            },
            Self::Processing => TrayPresentation {
                status_text: "Status: Processing",
                tooltip: "WhisperNow — Processing",
            },
            Self::Paused => TrayPresentation {
                status_text: "Status: Paused",
                tooltip: "WhisperNow — Paused",
            },
            Self::Updating => TrayPresentation {
                status_text: "Status: Installing an update",
                tooltip: "WhisperNow — Installing an update",
            },
            Self::ClearingData => TrayPresentation {
                status_text: "Status: Clearing application data",
                tooltip: "WhisperNow — Clearing application data",
            },
            Self::Error => TrayPresentation {
                status_text: "Status: Error — open WhisperNow",
                tooltip: "WhisperNow — Error",
            },
        }
    }
}

pub(crate) fn tray_state(snapshot: &AppState) -> TrayState {
    let failed = snapshot.error.is_some() || snapshot.model_state == ModelState::Failed;
    let loading = matches!(
        snapshot.model_state,
        ModelState::Downloading | ModelState::Loading
    );
    match snapshot.phase {
        Phase::Recording { .. } if failed => TrayState::RecordingError,
        _ if failed => TrayState::Error,
        Phase::Paused => TrayState::Paused,
        Phase::Updating => TrayState::Updating,
        Phase::ClearingData => TrayState::ClearingData,
        Phase::Recording { .. } => TrayState::Recording,
        Phase::Processing { .. } => TrayState::Processing,
        Phase::Idle | Phase::Error if loading => TrayState::Loading,
        Phase::Idle | Phase::Error => TrayState::Idle,
    }
}

/// Everything the native tray shows. It is derived from snapshots alone and
/// applied as a whole on the main thread, and only when it changes, so
/// repeated snapshots never touch the native icon.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TrayDesired {
    pub(crate) state: TrayState,
    pub(crate) pause: PauseItem,
    pub(crate) copy_last: bool,
    pub(crate) cancel: bool,
}

impl TrayDesired {
    const LOADING: Self = Self {
        state: TrayState::Loading,
        pause: PauseItem::Unavailable,
        copy_last: false,
        cancel: false,
    };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PauseItem {
    Pause,
    Resume,
    Unavailable,
}

pub(crate) fn tray_desired(snapshot: &AppState, copy_capable: bool) -> TrayDesired {
    TrayDesired {
        state: tray_state(snapshot),
        pause: match snapshot.phase {
            Phase::ClearingData | Phase::Updating => PauseItem::Unavailable,
            Phase::Paused => PauseItem::Resume,
            _ => PauseItem::Pause,
        },
        copy_last: copy_capable && snapshot.last_transcript.is_some(),
        cancel: snapshot.phase.is_busy(),
    }
}

#[derive(Clone)]
struct NativeTray {
    icon: TrayIcon,
    status: MenuItem<tauri::Wry>,
    copy_last: MenuItem<tauri::Wry>,
    cancel: MenuItem<tauri::Wry>,
    pause: MenuItem<tauri::Wry>,
}

impl NativeTray {
    /// Runs on the main thread.
    fn render(&self, desired: TrayDesired) {
        let presentation = desired.state.presentation();
        let _ = self.status.set_text(presentation.status_text);
        let _ = self.copy_last.set_enabled(desired.copy_last);
        let _ = self.cancel.set_enabled(desired.cancel);
        let _ = self.pause.set_text(match desired.pause {
            PauseItem::Pause | PauseItem::Unavailable => "Pause",
            PauseItem::Resume => "Resume",
        });
        let _ = self
            .pause
            .set_enabled(desired.pause != PauseItem::Unavailable);
        let _ = self.icon.set_tooltip(Some(presentation.tooltip));
        let _ = self.icon.set_icon(Some(state_icon(desired.state)));
    }
}

pub(crate) struct TrayController {
    native: Option<NativeTray>,
    desired: Mutex<TrayDesired>,
    copy_capable: bool,
}

impl TrayController {
    pub(crate) fn new(app: &tauri::App, copy_capable: bool) -> tauri::Result<Self> {
        let presentation = TrayState::Loading.presentation();
        let status = menu_item(app, "tray-status", presentation.status_text, false)?;
        let item =
            |action: TrayAction, text, enabled| menu_item(app, action.menu_id(), text, enabled);
        let copy_last = item(
            TrayAction::CopyLastTranscript,
            "Copy Last Transcript",
            false,
        )?;
        let cancel = item(TrayAction::CancelRecording, "Cancel Recording", false)?;
        let pause = item(TrayAction::TogglePause, "Pause", false)?;
        let settings = item(TrayAction::Settings, "Open WhisperNow", true)?;
        let check_updates = item(TrayAction::CheckForUpdates, "Check for Updates", true)?;
        let quit = item(TrayAction::Quit, "Quit WhisperNow", true)?;
        let menu = Menu::with_items(
            app,
            &[
                &status,
                &PredefinedMenuItem::separator(app)?,
                &copy_last,
                &cancel,
                &pause,
                &PredefinedMenuItem::separator(app)?,
                &settings,
                &check_updates,
                &PredefinedMenuItem::separator(app)?,
                &quit,
            ],
        )?;
        let icon = TrayIconBuilder::with_id("whispernow-tray")
            .menu(&menu)
            .icon(state_icon(TrayState::Loading))
            .tooltip(presentation.tooltip)
            .show_menu_on_left_click(false)
            .on_menu_event(|app, event| {
                if let Some(action) = TrayAction::from_menu_id(event.id().0.as_str()) {
                    crate::handle_tray_action(app, action);
                }
            })
            .on_tray_icon_event(|tray, event| {
                if matches!(
                    event,
                    TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } | TrayIconEvent::DoubleClick {
                        button: MouseButton::Left,
                        ..
                    }
                ) {
                    crate::handle_tray_action(tray.app_handle(), TrayAction::Settings);
                }
            })
            .build(app)?;
        Ok(Self::with(
            Some(NativeTray {
                icon,
                status,
                copy_last,
                cancel,
                pause,
            }),
            copy_capable,
        ))
    }

    #[cfg(test)]
    pub(crate) fn headless(copy_capable: bool) -> Self {
        Self::with(None, copy_capable)
    }

    fn with(native: Option<NativeTray>, copy_capable: bool) -> Self {
        Self {
            native,
            desired: Mutex::new(TrayDesired::LOADING),
            copy_capable,
        }
    }

    /// Derives the tray from a published snapshot. Snapshots arrive in
    /// publication order, so the latest one always wins.
    pub(crate) fn apply_snapshot(&self, snapshot: &AppState) {
        self.apply(tray_desired(snapshot, self.copy_capable));
    }

    /// Records the desired tray and, when it differs from the last one,
    /// schedules one fire-and-forget render on the main thread. Renders are
    /// scheduled under the `desired` lock, so they run in the order applied.
    fn apply(&self, desired: TrayDesired) {
        let mut current = self.desired.lock().expect("tray desired lock poisoned");
        if *current == desired {
            return;
        }
        *current = desired;
        self.render(desired);
    }

    fn render(&self, desired: TrayDesired) {
        let Some(native) = self.native.clone() else {
            return;
        };
        let app = native.icon.app_handle().clone();
        let _ = app.run_on_main_thread(move || native.render(desired));
    }
}

fn menu_item(
    app: &tauri::App,
    id: &str,
    text: &str,
    enabled: bool,
) -> tauri::Result<MenuItem<tauri::Wry>> {
    MenuItem::with_id(app, id, text, enabled, None::<&str>)
}

/// Every state is the app's key with a different face, so status does not depend on
/// color: an empty key while loading, updating or clearing data, the logo's bars when idle, a solid key while recording,
/// dots while processing, a pause sign when paused and an exclamation mark on error.
///
/// Drawn at 32 px on a 2 px grid so it stays sharp on 2x displays, with 4x4 supersampled edges.
fn state_icon(state: TrayState) -> Image<'static> {
    const SIDE: u16 = 32;
    const SAMPLES: u16 = 4;
    let side = usize::from(SIDE);
    let mut pixels = vec![0_u8; side * side * 4];
    for py in 0..SIDE {
        for px in 0..SIDE {
            let mut hits = 0_u16;
            for sy in 0..SAMPLES {
                for sx in 0..SAMPLES {
                    let x = f32::from(px) + (f32::from(sx) + 0.5) / f32::from(SAMPLES);
                    let y = f32::from(py) + (f32::from(sy) + 0.5) / f32::from(SAMPLES);
                    hits += u16::from(covers(state, x, y));
                }
            }
            if hits > 0 {
                let alpha = u8::try_from(255 * hits / (SAMPLES * SAMPLES)).unwrap_or(u8::MAX);
                let offset = (usize::from(py) * side + usize::from(px)) * 4;
                pixels[offset..offset + 4].copy_from_slice(&[245, 245, 245, alpha]);
            }
        }
    }
    Image::new_owned(pixels, u32::from(SIDE), u32::from(SIDE))
}

/// Whether the point lies inside the state's shape, in 32-unit icon coordinates.
fn covers(state: TrayState, x: f32, y: f32) -> bool {
    // The key: 28-unit body with 3-unit walls and a 6-unit bottom lip; the face is centred
    // on `FACE_Y`.
    const FACE_Y: f32 = 14.5;
    let key = rounded_rect(x, y, [2.0, 2.0, 30.0, 30.0], 7.0);
    let rim = key && !rounded_rect(x, y, [5.0, 5.0, 27.0, 24.0], 4.0);
    let bars = pill(x, y, 9.0, FACE_Y, 4.0, 6.0)
        || pill(x, y, 16.0, FACE_Y, 4.0, 12.0)
        || pill(x, y, 23.0, FACE_Y, 4.0, 8.0);
    let exclamation = pill(x, y, 16.0, 11.5, 4.0, 10.0) || dot(x, y, 16.0, 20.5, 2.25);
    match state {
        TrayState::Loading | TrayState::Updating | TrayState::ClearingData => rim,
        TrayState::Idle => rim || bars,
        TrayState::Recording => key && !bars,
        TrayState::RecordingError => key && !exclamation,
        TrayState::Processing => {
            rim || [9.5, 16.0, 22.5]
                .iter()
                .any(|&cx| dot(x, y, cx, FACE_Y, 2.25))
        }
        TrayState::Paused => {
            rim || pill(x, y, 12.0, FACE_Y, 4.0, 12.0) || pill(x, y, 20.0, FACE_Y, 4.0, 12.0)
        }
        TrayState::Error => rim || exclamation,
    }
}

fn rounded_rect(x: f32, y: f32, [left, top, right, bottom]: [f32; 4], radius: f32) -> bool {
    if x < left || x > right || y < top || y > bottom {
        return false;
    }
    let nearest_x = x.clamp(left + radius, right - radius);
    let nearest_y = y.clamp(top + radius, bottom - radius);
    dot(x, y, nearest_x, nearest_y, radius)
}

/// A vertical bar with round ends.
fn pill(x: f32, y: f32, center_x: f32, center_y: f32, width: f32, height: f32) -> bool {
    let (half_w, half_h) = (width / 2.0, height / 2.0);
    rounded_rect(
        x,
        y,
        [
            center_x - half_w,
            center_y - half_h,
            center_x + half_w,
            center_y + half_h,
        ],
        half_w,
    )
}

fn dot(x: f32, y: f32, center_x: f32, center_y: f32, radius: f32) -> bool {
    (x - center_x).powi(2) + (y - center_y).powi(2) <= radius * radius
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::tests::{recording, snapshot};

    #[test]
    fn every_tray_action_is_found_by_its_own_menu_id() {
        for action in TrayAction::ALL {
            assert_eq!(TrayAction::from_menu_id(action.menu_id()), Some(action));
        }
        assert_eq!(TrayAction::from_menu_id("tray-status"), None);
    }

    fn transcription_failed() -> whispernow_domain::AppError {
        whispernow_domain::AppError {
            category: whispernow_domain::ErrorCategory::Recognition,
            code: "transcription_failed".to_owned(),
            summary: "The recording could not be transcribed".to_owned(),
        }
    }

    #[test]
    fn tray_state_and_actions_follow_the_phase() {
        for (phase, background_error, expected) in [
            (Phase::Idle, false, TrayState::Idle),
            (recording(), false, TrayState::Recording),
            (
                Phase::Processing { cleaning_up: true },
                false,
                TrayState::Processing,
            ),
            (Phase::Paused, false, TrayState::Paused),
            (Phase::Updating, false, TrayState::Updating),
            (Phase::ClearingData, false, TrayState::ClearingData),
            (Phase::Idle, true, TrayState::Error),
            (Phase::Error, true, TrayState::Error),
            (recording(), true, TrayState::RecordingError),
            (Phase::Paused, true, TrayState::Error),
        ] {
            let mut snapshot = snapshot(phase.clone());
            if background_error {
                snapshot.error = Some(transcription_failed());
            }
            assert_eq!(tray_desired(&snapshot, true).state, expected, "{phase:?}");
        }
        let mut loading = snapshot(Phase::Idle);
        loading.model_state = ModelState::Loading;
        assert_eq!(tray_desired(&loading, true).state, TrayState::Loading);
        let updating = tray_desired(&snapshot(Phase::Updating), true);
        assert_eq!(updating.pause, PauseItem::Unavailable);
        assert_eq!(
            updating.state.presentation().status_text,
            "Status: Installing an update"
        );
        assert_eq!(
            tray_desired(&snapshot(Phase::ClearingData), true).pause,
            PauseItem::Unavailable
        );
        assert_eq!(
            tray_desired(&snapshot(Phase::Paused), true).pause,
            PauseItem::Resume
        );
    }

    #[test]
    fn copy_last_transcript_is_offered_when_there_is_text() {
        let copied = AppState {
            last_transcript: Some("hello".to_owned()),
            ..snapshot(Phase::Idle)
        };
        assert!(tray_desired(&copied, true).copy_last);
    }

    #[test]
    fn a_session_that_cannot_copy_keeps_copy_last_transcript_disabled() {
        let copied = AppState {
            last_transcript: Some("hello".to_owned()),
            ..snapshot(Phase::Idle)
        };
        let able = TrayController::headless(true);
        able.apply_snapshot(&copied);
        assert!(able.desired.lock().expect("desired").copy_last);

        let unable = TrayController::headless(false);
        unable.apply_snapshot(&copied);
        let desired = *unable.desired.lock().expect("desired");
        assert!(!desired.copy_last);
        assert_eq!(desired.state, TrayState::Idle, "nothing became an error");
    }

    #[test]
    fn every_tray_face_is_distinct() {
        let states = [
            TrayState::Loading,
            TrayState::Idle,
            TrayState::Recording,
            TrayState::RecordingError,
            TrayState::Processing,
            TrayState::Paused,
            TrayState::Error,
        ];
        let icons: Vec<_> = states.map(state_icon).into_iter().collect();
        for (state, icon) in states.iter().zip(&icons) {
            assert!(
                icon.rgba().chunks(4).any(|pixel| pixel[3] == 255),
                "{state:?} draws nothing opaque"
            );
        }
        for (i, first) in icons.iter().enumerate() {
            for (j, second) in icons.iter().enumerate().skip(i + 1) {
                assert_ne!(
                    first.rgba(),
                    second.rgba(),
                    "{:?} and {:?} look the same",
                    states[i],
                    states[j]
                );
            }
        }
    }
}
