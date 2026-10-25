use whispernow_domain::ipc::{AppState, EventPayload, Phase, UpdateState};

use crate::overlay::OverlayView;

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct UiState {
    pub(crate) snapshot: Option<AppState>,
    pub(crate) update_snapshot: Option<UpdateState>,
    pub(crate) update_command_error: Option<String>,
    pub(crate) connection_failed: bool,
    pub(crate) event_stream_failed: bool,
    pub(crate) overlay: OverlayView,
    pub(crate) recording_bar: bool,
}

impl UiState {
    /// Installs a snapshot read at startup unless an event already brought a newer one.
    pub(crate) fn hydrate(&mut self, state: AppState) {
        if self.snapshot.is_none() {
            self.apply_state(state);
        }
        self.connection_failed = false;
    }

    #[must_use]
    pub(crate) fn is_recording(&self) -> bool {
        self.snapshot
            .as_ref()
            .is_some_and(|state| matches!(state.phase, Phase::Recording { .. }))
    }

    pub(crate) fn apply_event(&mut self, event: EventPayload) {
        match event {
            EventPayload::StateChanged { state } => self.apply_state(state),
            EventPayload::AudioLevel {
                rms, elapsed_ms, ..
            } => {
                if self.recording_bar && self.is_recording() {
                    self.overlay.apply_level(rms, elapsed_ms);
                }
            }
            EventPayload::UpdateChanged { state } => self.apply_update_state(state),
            // The main window's listener follows navigation requests.
            EventPayload::NavigationRequested { .. }
            | EventPayload::HistoryChanged
            | EventPayload::ModelsChanged => {}
        }
    }

    pub(crate) fn apply_update_state(&mut self, state: UpdateState) {
        self.update_command_error = None;
        self.update_snapshot = Some(state);
    }

    pub(crate) fn apply_state(&mut self, state: AppState) {
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
        self.snapshot = Some(state);
    }

    /// A state read that fails before any snapshot arrived means the desktop is out of reach.
    pub(crate) fn report_state_read_failure(&mut self) {
        if self.snapshot.is_none() {
            self.connection_failed = true;
        }
    }

    pub(crate) fn report_update_command_failure(&mut self, summary: String) {
        self.update_command_error = Some(summary);
    }
}
