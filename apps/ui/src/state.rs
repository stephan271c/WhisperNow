use whispernow_domain::AppError;

use whispernow_domain::ipc::{AppState, EventPayload, Phase, UpdateState};

use crate::OverlayView;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct UiState {
    pub snapshot: Option<AppState>,
    pub update_snapshot: Option<UpdateState>,
    pub audio_rms: f32,
    pub error: Option<AppError>,
    pub update_command_error: Option<String>,
    pub connection_failed: bool,
    pub event_stream_failed: bool,
    pub overlay: OverlayView,
    pub recording_bar: bool,
}

impl UiState {
    /// Installs a snapshot read at startup unless an event already brought a newer one.
    pub(crate) fn hydrate(&mut self, state: AppState) {
        if self.snapshot.is_none() {
            self.apply_state(state);
        }
        self.connection_failed = false;
    }

    pub fn apply_event(&mut self, event: EventPayload) {
        match event {
            EventPayload::StateChanged { state } => self.apply_state(state),
            EventPayload::AudioLevel {
                rms, elapsed_ms, ..
            } => {
                if self
                    .snapshot
                    .as_ref()
                    .is_some_and(|state| matches!(state.phase, Phase::Recording { .. }))
                {
                    self.audio_rms = rms;
                    if self.recording_bar {
                        self.overlay.apply_level(rms, elapsed_ms);
                    }
                }
            }
            EventPayload::UpdateChanged { state } => self.apply_update_state(state),
            // The main window's listener follows navigation requests.
            EventPayload::NavigationRequested { .. } | EventPayload::HistoryChanged => {}
        }
    }

    pub fn apply_update_state(&mut self, state: UpdateState) {
        self.update_command_error = None;
        self.update_snapshot = Some(state);
    }

    fn apply_state(&mut self, state: AppState) {
        if self.recording_bar {
            if self
                .snapshot
                .as_ref()
                .is_none_or(|old| old.phase != state.phase)
            {
                self.overlay.action_error = None;
            }
            if let Phase::Recording { session, .. } = state.phase {
                self.overlay.start_recording(session);
            }
        }
        if !matches!(state.phase, Phase::Recording { .. }) {
            self.audio_rms = 0.0;
        }
        // A failed dictation is the current problem until it is dismissed.
        self.error = match &state.phase {
            Phase::Error { error } => Some(error.clone()),
            _ => state.error.clone(),
        };
        self.snapshot = Some(state);
    }

    pub fn report_client_error(&mut self, error: AppError) {
        if error.code == "desktop_unreachable" {
            self.connection_failed = true;
        } else {
            self.error = Some(error);
        }
    }

    pub fn report_update_command_failure(&mut self, summary: String) {
        self.update_command_error = Some(summary);
    }

    pub fn report_event_stream_failure(&mut self, error: AppError) {
        self.event_stream_failed = true;
        self.error = Some(error);
    }
}
