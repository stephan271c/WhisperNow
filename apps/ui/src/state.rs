use whispernow_domain::ipc::{AppState, EventPayload, Phase, UpdateState};

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct UiState {
    pub(crate) snapshot: Option<AppState>,
    pub(crate) update_snapshot: Option<UpdateState>,
    pub(crate) update_command_error: Option<String>,
    pub(crate) connection_failed: bool,
    pub(crate) event_stream_failed: bool,
    /// Why the last overlay action failed; cleared by the next phase.
    pub(crate) overlay_action_error: Option<String>,
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
    pub(crate) fn recording_session(&self) -> Option<u64> {
        match self.snapshot.as_ref()?.phase {
            Phase::Recording { session, .. } => Some(session),
            _ => None,
        }
    }

    pub(crate) fn apply_event(&mut self, event: EventPayload) {
        match event {
            EventPayload::StateChanged { state } => self.apply_state(state),
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
        if self.recording_bar
            && self
                .snapshot
                .as_ref()
                .is_none_or(|old| old.phase != state.phase)
        {
            self.overlay_action_error = None;
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
