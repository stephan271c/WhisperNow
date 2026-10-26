use std::{
    collections::BTreeMap,
    ops::Deref,
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
};
use whispernow_domain::AppError;

use whispernow_core::CancellationToken;
use whispernow_domain::ipc::{
    ActiveDownload, AppState, AudioLevel, CleanupDevice, EventPayload, ModelState, Phase,
    SetupTestResult,
};
use whispernow_domain::{DownloadPhase, ModelId};

/// What the event pump emits, each under its own event name.
#[derive(Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "every slot fits an event anyway; boxing would allocate per snapshot"
)]
pub(crate) enum Published {
    Event(EventPayload),
    Level(AudioLevel),
}

/// Sending never blocks, so a snapshot can be sent while the state lock is held and arrives
/// in change order.
#[derive(Clone)]
pub(crate) struct Events {
    sender: mpsc::Sender<Published>,
    pump_loss_logged: Arc<AtomicBool>,
}

impl Events {
    pub(crate) fn channel() -> (Self, mpsc::Receiver<Published>) {
        let (sender, receiver) = mpsc::channel();
        let events = Self {
            sender,
            pump_loss_logged: Arc::default(),
        };
        (events, receiver)
    }

    /// Logs the first event the stopped pump could not take: audio levels would repeat it
    /// many times a second.
    pub(crate) fn publish(&self, payload: EventPayload) {
        self.send(Published::Event(payload));
    }

    pub(crate) fn level(&self, level: AudioLevel) {
        self.send(Published::Level(level));
    }

    fn send(&self, published: Published) {
        if self.sender.send(published).is_err()
            && !self.pump_loss_logged.swap(true, Ordering::Relaxed)
        {
            tracing::error!("the event pump stopped; the interface receives no more updates");
        }
    }

    pub(crate) fn history_changed(&self) {
        self.publish(EventPayload::HistoryChanged);
    }

    pub(crate) fn models_changed(&self) {
        self.publish(EventPayload::ModelsChanged);
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum LoadedModel {
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

pub(crate) struct DownloadTask {
    pub(crate) progress: ActiveDownload,
    pub(crate) cancellation: CancellationToken,
}

impl DownloadTask {
    pub(crate) fn new(
        model_id: ModelId,
        total_bytes: u64,
        cancellation: CancellationToken,
    ) -> Self {
        Self {
            progress: ActiveDownload {
                model_id,
                phase: DownloadPhase::Downloading,
                downloaded_bytes: 0,
                total_bytes,
                bytes_per_second: None,
            },
            cancellation,
        }
    }
}

pub(crate) struct State {
    pub(crate) phase: Phase,
    pub(crate) selected_model_id: ModelId,
    pub(crate) model: LoadedModel,
    pub(crate) last_transcript: Option<String>,
    pub(crate) last_error: Option<AppError>,
    /// How setup's latest test dictation ended, kept apart from `last_transcript` and
    /// `last_error`.
    pub(crate) setup_test: Option<SetupTestResult>,
    pub(crate) cleanup_device: Option<CleanupDevice>,
    pub(crate) active_downloads: BTreeMap<ModelId, DownloadTask>,
    pub(crate) failed_download: Option<ModelId>,
    /// Set with `cancellation` when the dictation in flight is asked to end early.
    pub(crate) cancel: Option<CancelReason>,
    /// The dictation in flight's cancellation; the worker replaces it for each recording.
    pub(crate) cancellation: CancellationToken,
}

impl State {
    pub(crate) fn new(selected_model_id: ModelId) -> Self {
        Self {
            phase: Phase::Idle,
            selected_model_id,
            model: LoadedModel::Unloaded,
            last_transcript: None,
            last_error: None,
            setup_test: None,
            cleanup_device: None,
            active_downloads: BTreeMap::new(),
            failed_download: None,
            cancel: None,
            cancellation: CancellationToken::default(),
        }
    }

    pub(crate) fn request_cancel(&mut self, reason: CancelReason) {
        self.cancel = Some(reason);
        self.cancellation.cancel();
    }

    /// Ends the dictation in flight in `next`, or in `Paused` when a pause asked it to end.
    pub(crate) fn end_dictation(&mut self, next: Phase) {
        let reason = self.cancel.take();
        self.phase = if reason == Some(CancelReason::Pause) {
            Phase::Paused
        } else {
            next
        };
    }

    pub(crate) fn require_quiescent(&self) -> Result<(), AppError> {
        self.phase.require_idle()?;
        self.require_no_downloads()
    }

    pub(crate) fn enter(&mut self, phase: Phase) -> Result<Phase, AppError> {
        self.require_quiescent()?;
        Ok(std::mem::replace(&mut self.phase, phase))
    }

    pub(crate) fn require_not_in_maintenance(&self) -> Result<(), AppError> {
        match self.phase {
            Phase::ClearingData | Phase::Updating => self.phase.require_idle(),
            _ => Ok(()),
        }
    }

    pub(crate) fn require_no_downloads(&self) -> Result<(), AppError> {
        if self.active_downloads.is_empty() {
            Ok(())
        } else {
            Err(AppError::new(
                whispernow_domain::ErrorCategory::General,
                "model_transition_busy",
                "Wait for model downloads to finish",
            ))
        }
    }

    fn model_state(&self) -> ModelState {
        if self.active_downloads.contains_key(&self.selected_model_id) {
            return ModelState::Downloading;
        }
        match &self.model {
            LoadedModel::Ready(model_id) if *model_id == self.selected_model_id => {
                ModelState::Ready
            }
            LoadedModel::Loading => ModelState::Loading,
            LoadedModel::Failed => ModelState::Failed,
            LoadedModel::Unloaded | LoadedModel::Ready(_)
                if self.failed_download.as_ref() == Some(&self.selected_model_id) =>
            {
                ModelState::Failed
            }
            LoadedModel::Unloaded | LoadedModel::Ready(_) => ModelState::Unloaded,
        }
    }

    pub(crate) fn snapshot(&self) -> AppState {
        AppState {
            phase: self.phase.clone(),
            selected_model_id: self.selected_model_id.clone(),
            model_state: self.model_state(),
            last_transcript: self.last_transcript.clone(),
            error: self.last_error.clone(),
            active_downloads: self
                .active_downloads
                .values()
                .map(|download| download.progress.clone())
                .collect(),
            failed_download: self.failed_download.clone(),
            setup_test: self.setup_test.clone(),
            cleanup_device: self.cleanup_device.clone(),
        }
    }
}

/// Read access to the state; [`StateCell::apply`] and [`StateCell::update`] change it.
pub(crate) struct StateGuard<'a>(MutexGuard<'a, State>);

impl Deref for StateGuard<'_> {
    type Target = State;

    fn deref(&self) -> &State {
        &self.0
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

    pub(crate) fn refuse_during_maintenance(&self) -> Result<(), AppError> {
        self.lock().require_not_in_maintenance()
    }

    pub(crate) fn lock(&self) -> StateGuard<'_> {
        StateGuard(self.locked())
    }

    fn locked(&self) -> MutexGuard<'_, State> {
        self.value.lock().expect("application state lock poisoned")
    }

    pub(crate) const fn events(&self) -> &Events {
        &self.events
    }

    pub(crate) fn snapshot(&self) -> AppState {
        self.lock().snapshot()
    }

    /// Publishes only when `change` succeeds.
    pub(crate) fn update<R>(
        &self,
        change: impl FnOnce(&mut State) -> Result<R, AppError>,
    ) -> Result<R, AppError> {
        let mut state = self.locked();
        let result = change(&mut state)?;
        self.events.publish(EventPayload::StateChanged {
            state: state.snapshot(),
        });
        Ok(result)
    }

    pub(crate) fn apply(&self, change: impl FnOnce(&mut State)) {
        let mut state = self.locked();
        change(&mut state);
        self.events.publish(EventPayload::StateChanged {
            state: state.snapshot(),
        });
    }

    pub(crate) fn publish(&self) {
        self.apply(|_| {});
    }

    /// Ends a failed dictation's `Error` phase once the user has dealt with it. Any other phase
    /// is left alone, so a dismissal that raced a new dictation does nothing.
    pub(crate) fn dismiss_error(&self) {
        let mut state = self.locked();
        if state.phase == Phase::Error {
            state.phase = Phase::Idle;
            state.last_error = None;
            self.events.publish(EventPayload::StateChanged {
                state: state.snapshot(),
            });
        }
    }

    /// A failed dictation's error stays until it is dismissed, so a later error does not replace it.
    pub(crate) fn publish_error(&self, error: AppError) {
        self.apply(|state| {
            if state.phase != Phase::Error {
                state.last_error = Some(error);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use whispernow_domain::ErrorCategory;

    use super::*;

    fn published_phases(published: &mpsc::Receiver<Published>) -> Vec<Phase> {
        published
            .try_iter()
            .filter_map(|event| match event {
                Published::Event(EventPayload::StateChanged { state }) => Some(state.phase),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn every_change_to_the_state_publishes_its_snapshot() {
        let (events, published) = Events::channel();
        let cell = StateCell::new(State::new(ModelId::default()), events);

        cell.apply(|state| state.phase = Phase::Paused);
        assert_eq!(published_phases(&published), [Phase::Paused]);

        cell.update(|state| {
            state.phase = Phase::Idle;
            Ok(())
        })
        .expect("update");
        assert_eq!(published_phases(&published), [Phase::Idle]);

        cell.update::<()>(|_| Err(AppError::new(ErrorCategory::General, "refused", "Refused")))
            .expect_err("refused update");
        assert!(published_phases(&published).is_empty());

        cell.publish_error(AppError::new(ErrorCategory::General, "failed", "Failed"));
        cell.apply(|state| state.phase = Phase::Error);
        cell.dismiss_error();
        assert_eq!(
            published_phases(&published),
            [Phase::Idle, Phase::Error, Phase::Idle]
        );
        assert!(cell.lock().last_error.is_none());
    }

    #[test]
    fn the_model_state_follows_only_the_selected_models_download() {
        let selected = ModelId::default();
        let other = ModelId::new("lfm25-cleanup").expect("model id");
        let mut state = State::new(selected.clone());
        state.model = LoadedModel::Ready(selected.clone());

        state.active_downloads.insert(
            other.clone(),
            DownloadTask::new(other.clone(), 1_024, CancellationToken::default()),
        );
        assert_eq!(state.model_state(), ModelState::Ready);
        state.active_downloads.clear();
        state.failed_download = Some(other);
        assert_eq!(state.model_state(), ModelState::Ready);

        state.model = LoadedModel::Unloaded;
        assert_eq!(state.model_state(), ModelState::Unloaded);
        state.active_downloads.insert(
            selected.clone(),
            DownloadTask::new(selected.clone(), 1_024, CancellationToken::default()),
        );
        assert_eq!(state.model_state(), ModelState::Downloading);
        state.active_downloads.clear();
        state.failed_download = Some(selected);
        assert_eq!(state.model_state(), ModelState::Failed);
    }
}
