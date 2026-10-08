use std::{
    collections::BTreeMap,
    sync::{Mutex, MutexGuard, mpsc},
};

use whispernow_api::{
    ActiveDownloadDto, ApiErrorDto, AppStateDto, ErrorCategoryDto, EventPayload, ModelStateDto,
    OverlayPosition, OverlayStyle, PhaseDto, RecordingMode, SetupTestResultDto, TranscriptResponse,
};
use whispernow_core::CancellationToken;
use whispernow_domain::ModelId;

/// Sending never blocks, so a snapshot can be sent while the state lock is held and arrives
/// in change order.
#[derive(Clone)]
pub(crate) struct Events(mpsc::Sender<EventPayload>);

impl Events {
    pub(crate) fn channel() -> (Self, mpsc::Receiver<EventPayload>) {
        let (sender, receiver) = mpsc::channel();
        (Self(sender), receiver)
    }

    pub(crate) fn publish(&self, payload: EventPayload) {
        let _closed = self.0.send(payload);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RecordingTrigger {
    Hold,
    Toggle,
    /// The first-run test dictation: its transcript is shown in setup only, so it is never
    /// pasted, and neither it nor its audio is kept in history.
    SetupTest,
}

impl RecordingTrigger {
    pub(crate) const fn is_setup_test(self) -> bool {
        matches!(self, Self::SetupTest)
    }
}

/// Every surface derives its view from this one value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum Phase {
    Idle,
    Recording {
        trigger: RecordingTrigger,
        session: u64,
    },
    Processing {
        trigger: RecordingTrigger,
        cleaning_up: bool,
    },
    Paused,
    Updating,
    /// The latest dictation failed or was not delivered.
    Error(ApiErrorDto),
}

impl Phase {
    pub(crate) fn dto(&self) -> PhaseDto {
        match self {
            Self::Idle => PhaseDto::Idle,
            Self::Recording { trigger, session } => PhaseDto::Recording {
                mode: match trigger {
                    RecordingTrigger::Hold => RecordingMode::Hold,
                    RecordingTrigger::Toggle | RecordingTrigger::SetupTest => RecordingMode::Toggle,
                },
                session: *session,
            },
            Self::Processing { cleaning_up, .. } => PhaseDto::Processing {
                cleaning_up: *cleaning_up,
            },
            Self::Paused => PhaseDto::Paused,
            Self::Updating => PhaseDto::Updating,
            Self::Error(error) => PhaseDto::Error {
                error: error.clone(),
            },
        }
    }

    pub(crate) const fn is_busy(&self) -> bool {
        matches!(self, Self::Recording { .. } | Self::Processing { .. })
    }

    /// Refuses work that needs the desktop idle: nothing dictating, paused or updating.
    pub(crate) fn require_idle(&self) -> Result<(), ApiErrorDto> {
        match self {
            Self::Idle | Self::Error(_) => Ok(()),
            Self::Recording { .. } => Err(conflict(
                "workflow_busy",
                "Stop the current recording first",
            )),
            Self::Processing { .. } => Err(conflict(
                "workflow_busy",
                "Wait for the current transcription to finish",
            )),
            Self::Paused => Err(conflict("application_paused", "Resume WhisperNow first")),
            Self::Updating => Err(conflict(
                "update_installing",
                "An application update is being installed",
            )),
        }
    }
}

fn conflict(code: &str, summary: &str) -> ApiErrorDto {
    ApiErrorDto::new(ErrorCategoryDto::General, code, summary)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ModelStatus {
    Unloaded,
    Loading,
    Ready(ModelId),
    Failed,
}

/// Why the dictation in flight was asked to end early, which decides what history keeps.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CancelReason {
    /// The user cancelled; the audio is kept as a cancelled entry.
    User,
    /// Pausing; the audio is kept as a failed entry and the phase becomes `Paused`.
    Pause,
    /// Closing; the audio is kept as a failed entry.
    Shutdown,
}

pub(crate) struct ActiveDownload {
    pub(crate) progress: ActiveDownloadDto,
    pub(crate) cancellation: CancellationToken,
}

pub(crate) struct State {
    pub(crate) phase: Phase,
    pub(crate) selected_model_id: ModelId,
    pub(crate) model: ModelStatus,
    pub(crate) last_transcript: Option<TranscriptResponse>,
    pub(crate) last_error: Option<ApiErrorDto>,
    /// How setup's latest test dictation ended, kept apart from `last_transcript` and
    /// `last_error`.
    pub(crate) setup_test: Option<SetupTestResultDto>,
    pub(crate) active_downloads: BTreeMap<ModelId, ActiveDownload>,
    pub(crate) download_failures: usize,
    pub(crate) setup_complete: bool,
    pub(crate) overlay_style: OverlayStyle,
    pub(crate) overlay_position: OverlayPosition,
    /// Set with `cancellation` when the dictation in flight is asked to end early.
    pub(crate) cancel: Option<CancelReason>,
    /// The dictation in flight's cancellation; the worker replaces it for each recording.
    pub(crate) cancellation: CancellationToken,
}

impl State {
    pub(crate) fn new(selected_model_id: ModelId, setup_complete: bool) -> Self {
        Self {
            phase: Phase::Idle,
            selected_model_id,
            model: ModelStatus::Unloaded,
            last_transcript: None,
            last_error: None,
            setup_test: None,
            active_downloads: BTreeMap::new(),
            download_failures: 0,
            setup_complete,
            overlay_style: OverlayStyle::default(),
            overlay_position: OverlayPosition::default(),
            cancel: None,
            cancellation: CancellationToken::default(),
        }
    }

    fn model_state(&self) -> ModelStateDto {
        if !self.active_downloads.is_empty() {
            return ModelStateDto::Downloading;
        }
        match &self.model {
            ModelStatus::Ready(model_id) if *model_id == self.selected_model_id => {
                ModelStateDto::Ready
            }
            ModelStatus::Loading => ModelStateDto::Loading,
            ModelStatus::Failed => ModelStateDto::Failed,
            ModelStatus::Unloaded | ModelStatus::Ready(_) if self.download_failures > 0 => {
                ModelStateDto::Failed
            }
            ModelStatus::Unloaded | ModelStatus::Ready(_) => ModelStateDto::Unloaded,
        }
    }

    pub(crate) fn dto(&self) -> AppStateDto {
        AppStateDto {
            phase: self.phase.dto(),
            selected_model_id: self.selected_model_id.clone(),
            model_state: self.model_state(),
            last_transcript: self.last_transcript.clone(),
            error: self.last_error.clone(),
            active_downloads: self
                .active_downloads
                .values()
                .map(|download| download.progress.clone())
                .collect(),
            setup_test: self.setup_test.clone(),
            overlay_style: self.overlay_style,
            overlay_position: self.overlay_position,
        }
    }
}

/// Each change publishes its snapshot before the lock is released.
pub(crate) struct StateCell {
    value: Mutex<State>,
    events: Events,
}

impl StateCell {
    pub(crate) fn new(state: State, events: Events) -> Self {
        Self {
            value: Mutex::new(state),
            events,
        }
    }

    pub(crate) fn refuse_while_updating(&self) -> Result<(), ApiErrorDto> {
        let phase = &self.lock().phase;
        if *phase == Phase::Updating {
            return phase.require_idle();
        }
        Ok(())
    }

    pub(crate) fn lock(&self) -> MutexGuard<'_, State> {
        self.value.lock().expect("application state lock poisoned")
    }

    pub(crate) const fn events(&self) -> &Events {
        &self.events
    }

    pub(crate) fn snapshot(&self) -> AppStateDto {
        self.lock().dto()
    }

    /// Applies one change and, when it succeeds, publishes the resulting snapshot.
    pub(crate) fn update<R>(
        &self,
        change: impl FnOnce(&mut State) -> Result<R, ApiErrorDto>,
    ) -> Result<R, ApiErrorDto> {
        let mut state = self.lock();
        let result = change(&mut state)?;
        self.events
            .publish(EventPayload::StateChanged { state: state.dto() });
        Ok(result)
    }

    pub(crate) fn apply(&self, change: impl FnOnce(&mut State)) {
        let mut state = self.lock();
        change(&mut state);
        self.events
            .publish(EventPayload::StateChanged { state: state.dto() });
    }

    pub(crate) fn publish(&self) {
        self.apply(|_| {});
    }

    /// Ends a failed dictation's `Error` phase once the user has dealt with it. Any other phase
    /// is left alone, so a dismissal that raced a new dictation does nothing.
    pub(crate) fn dismiss_error(&self) {
        let mut state = self.lock();
        if matches!(state.phase, Phase::Error(_)) {
            state.phase = Phase::Idle;
            state.last_error = None;
            self.events
                .publish(EventPayload::StateChanged { state: state.dto() });
        }
    }

    pub(crate) fn publish_error(&self, error: ApiErrorDto) {
        self.apply(|state| state.last_error = Some(error));
    }
}
