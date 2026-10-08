use std::fmt;

use whispernow_api::{
    ApiErrorDto, AppStateDto, EventPayload, ModelStateDto, PhaseDto, UpdateStateDto,
};

use crate::OverlayView;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum EventStreamHealth {
    #[default]
    Connecting,
    Listening,
    Failed,
}

#[derive(Clone, Default, PartialEq)]
pub struct UiState {
    pub snapshot: Option<AppStateDto>,
    pub update_snapshot: Option<UpdateStateDto>,
    pub audio_rms: f32,
    pub error: Option<ApiErrorDto>,
    pub update_command_error: Option<String>,
    pub connection_failed: bool,
    pub event_stream_health: EventStreamHealth,
    pub overlay: OverlayView,
}

impl fmt::Debug for UiState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("UiState")
            .field("snapshot", &self.snapshot)
            .field("update_snapshot", &self.update_snapshot)
            .field("audio_rms", &self.audio_rms)
            .field("error", &self.error)
            .field("update_command_error", &self.update_command_error)
            .field("connection_failed", &self.connection_failed)
            .field("event_stream_health", &self.event_stream_health)
            .field("overlay_phase", &self.overlay.phase)
            .finish()
    }
}

impl UiState {
    /// Installs a snapshot read at startup unless an event already brought a newer one.
    pub(crate) fn hydrate(&mut self, state: AppStateDto) {
        if self.snapshot.is_none() {
            self.apply_state(state);
        }
        self.connection_failed = false;
    }

    pub fn hydrate_update(&mut self, state: UpdateStateDto) {
        self.apply_update_state(state);
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
                    .is_some_and(|state| matches!(state.phase, PhaseDto::Recording { .. }))
                {
                    self.audio_rms = rms.clamp(0.0, 1.0);
                }
                self.overlay.apply_level(rms, elapsed_ms);
            }
            EventPayload::UpdateChanged { state } => self.apply_update_state(state),
            // The main window's listener follows navigation requests.
            EventPayload::NavigationRequested { .. } | EventPayload::HistoryChanged => {}
        }
    }

    fn apply_update_state(&mut self, state: UpdateStateDto) {
        self.update_command_error = None;
        self.update_snapshot = Some(state);
    }

    fn apply_state(&mut self, state: AppStateDto) {
        self.overlay.install(&state);
        if !matches!(state.phase, PhaseDto::Recording { .. }) {
            self.audio_rms = 0.0;
        }
        // A failed dictation is the current problem until it is dismissed.
        self.error = match &state.phase {
            PhaseDto::Error { error } => Some(error.clone()),
            _ => state.error.clone(),
        };
        self.snapshot = Some(state);
    }

    #[must_use]
    pub fn is_loading_model(&self) -> bool {
        self.snapshot.as_ref().is_some_and(|snapshot| {
            matches!(
                snapshot.model_state,
                ModelStateDto::Downloading | ModelStateDto::Loading
            )
        })
    }

    pub fn report_client_error(&mut self, error: Option<ApiErrorDto>) {
        if let Some(error) = error {
            self.error = Some(error);
        } else {
            self.connection_failed = true;
        }
    }

    pub fn report_update_command_failure(&mut self, summary: Option<String>) {
        self.update_command_error =
            Some(summary.unwrap_or_else(|| "The update service could not be reached".to_owned()));
    }

    #[cfg_attr(
        not(any(test, target_arch = "wasm32")),
        expect(dead_code, reason = "called by the browser's window root and by tests")
    )]
    pub fn mark_event_stream_listening(&mut self) {
        self.event_stream_health = EventStreamHealth::Listening;
    }

    #[cfg_attr(
        not(any(test, target_arch = "wasm32")),
        expect(dead_code, reason = "called by the browser's window root and by tests")
    )]
    pub fn report_event_stream_failure(&mut self, error: Option<ApiErrorDto>) {
        self.event_stream_health = EventStreamHealth::Failed;
        if let Some(error) = error {
            self.error = Some(error);
        }
    }
}
