//! One worker thread owns the recognizer and audio capture and runs every dictation, model load
//! and microphone test in the order they were asked for.

use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
use whispernow_domain::AppError;

use whispernow_core::{
    CancellationToken, ResolvedModel,
    asr::AsrEngine,
    audio::{
        AudioCapture, AudioDeviceChoice, AudioInput, AudioInputError, AudioSink, TARGET_SAMPLE_RATE,
    },
    workflow::{
        AudioSegment, PipelineFailureKind, PreparedTranscription, SpeechSegmenter, VoiceActivity,
        has_microphone_audio, prepare_segmented,
    },
};
use whispernow_domain::ipc::{ErrorCategory, EventPayload, MicrophoneTestResponse};
use whispernow_domain::{
    EntryStatus, HistoryLimit, ModelId, NewHistoryEntry, UtcMillis, VocabularyReplacement,
};
use whispernow_platform::ModelStore;

use super::{Phase, RecordingTrigger};
use super::{
    capture_start_error,
    history::HistoryStore,
    microphone_test,
    models::installed,
    output::TranscriptOutput,
    settings::SettingsCoordinator,
    shutting_down,
    state::{CancelReason, LoadedModel, StateCell},
};

mod capture;
mod cleanup;
mod outcome;

use capture::{CapturedSpeech, RecordingSink};
pub(crate) use cleanup::{AiCleanup, CleanupPlan};
use outcome::{
    DecodeFailure, Outcome, capture_duration_millis, cleanup_fallback_warning, interruption_warning,
};

const MAX_RECORDING_SAMPLES: usize = TARGET_SAMPLE_RATE as usize * 60 * 30;

#[derive(Clone)]
pub(crate) struct DictationSettings {
    pub(crate) model_id: ModelId,
    pub(crate) vocabulary: Vec<VocabularyReplacement>,
    pub(crate) cleanup: Option<CleanupPlan>,
    pub(crate) history_limit: HistoryLimit,
    /// Whether a successful dictation keeps its audio in history.
    pub(crate) keep_audio: bool,
}

pub(crate) struct RecordingSettings {
    pub(crate) input_device: AudioDeviceChoice,
    pub(crate) dictation: DictationSettings,
}

pub(crate) struct DictationParts {
    pub(crate) state: Arc<StateCell>,
    pub(crate) settings: Arc<SettingsCoordinator>,
    pub(crate) history: Arc<HistoryStore>,
    pub(crate) audio: Arc<dyn AudioInput>,
    pub(crate) models: Arc<ModelStore>,
    pub(crate) voice_activity: Arc<dyn Fn() -> Box<dyn VoiceActivity> + Send + Sync>,
    pub(crate) cleanup: Arc<AiCleanup>,
    pub(crate) output: Arc<dyn TranscriptOutput>,
    /// Builds the recognizer, and a fresh one after a native panic.
    pub(crate) engine: Box<dyn Fn() -> Box<dyn AsrEngine> + Send>,
}

type Reply<T> = mpsc::Sender<Result<T, AppError>>;

pub(super) enum Message {
    Start {
        trigger: RecordingTrigger,
        reply: Reply<u64>,
    },
    Stop {
        session: u64,
    },
    /// A recording shortcut went down: it ends the toggle recording it started, or starts one.
    ShortcutPressed {
        trigger: RecordingTrigger,
    },
    /// The hold shortcut came up: it ends the recording it started.
    HoldReleased,
    /// The dictation in flight was asked to end early; `State::cancel` says why.
    Cancel,
    /// The recording `session` found another speech segment.
    Segments {
        session: u64,
    },
    LoadModel {
        model: Box<ResolvedModel>,
        reply: Reply<()>,
    },
    UnloadModel {
        reply: Reply<()>,
    },
    MicTest {
        device: AudioDeviceChoice,
        duration: Duration,
        reply: Reply<MicrophoneTestResponse>,
    },
    BeginUpdate {
        reply: Reply<()>,
    },
    Shutdown,
}

pub(crate) struct Dictation {
    state: Arc<StateCell>,
    sender: mpsc::Sender<Message>,
    stopping: Arc<AtomicBool>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

pub(super) struct DataClear<'a> {
    state: &'a StateCell,
    previous_phase: Phase,
}

impl Drop for DataClear<'_> {
    fn drop(&mut self) {
        self.state
            .apply(|state| state.phase = self.previous_phase.clone());
    }
}

impl Dictation {
    pub(crate) fn spawn(parts: DictationParts) -> Result<Self, AppError> {
        let (sender, receiver) = mpsc::channel();
        let state = Arc::clone(&parts.state);
        let stopping = Arc::new(AtomicBool::new(false));
        let engine = (parts.engine)();
        let worker = Worker {
            parts,
            engine,
            sender: sender.clone(),
            stopping: Arc::clone(&stopping),
            next_session: 0,
            recording: None,
            deferred_update: None,
        };
        let worker = thread::Builder::new()
            .name("whispernow-dictation".to_owned())
            .spawn(move || worker.run(&receiver))
            .map_err(|_| {
                AppError::new(
                    ErrorCategory::General,
                    "worker_spawn_failed",
                    "The dictation worker could not be started",
                )
            })?;
        Ok(Self {
            state,
            sender,
            stopping,
            worker: Mutex::new(Some(worker)),
        })
    }

    fn send(&self, message: Message) -> Result<(), AppError> {
        self.sender.send(message).map_err(|_| worker_stopped())
    }

    fn ask<T>(&self, message: impl FnOnce(Reply<T>) -> Message) -> Result<T, AppError> {
        if self.stopping.load(Ordering::Acquire) {
            return Err(shutting_down());
        }
        let (reply, answer) = mpsc::channel();
        self.send(message(reply))?;
        answer.recv().map_err(|_| worker_stopped())?
    }

    /// Starts capturing a dictation and returns its session. A start while the previous
    /// dictation is transcribed and pasted, a fraction of a second, waits for it in the
    /// worker's queue, so a quick re-press is not lost; one during a slower cleanup is refused.
    pub(crate) fn start(&self, trigger: RecordingTrigger) -> Result<u64, AppError> {
        {
            let state = self.state.lock();
            if !matches!(
                state.phase,
                Phase::Processing {
                    cleaning_up: false,
                    ..
                }
            ) {
                state.phase.require_idle()?;
            }
        }
        self.ask(|reply| Message::Start { trigger, reply })
    }

    /// A recording shortcut went down. The worker decides from its recording whether this
    /// starts one or ends the toggle recording it started. A press during a slower cleanup,
    /// a pause or an update is dropped.
    pub(crate) fn shortcut_pressed(&self, trigger: RecordingTrigger) -> Result<(), AppError> {
        let accepted = matches!(
            self.state.lock().phase,
            Phase::Idle
                | Phase::Error { .. }
                | Phase::Recording { .. }
                | Phase::Processing {
                    cleaning_up: false,
                    ..
                }
        );
        if accepted {
            self.send(Message::ShortcutPressed { trigger })
        } else {
            Ok(())
        }
    }

    pub(crate) fn hold_released(&self) -> Result<(), AppError> {
        self.send(Message::HoldReleased)
    }

    pub(crate) fn stop(&self) -> Result<(), AppError> {
        let Phase::Recording { session, .. } = self.state.lock().phase else {
            return Err(not_recording());
        };
        self.send(Message::Stop { session })
    }

    /// Ends the dictation in flight early, keeping its audio in history.
    pub(crate) fn cancel(&self) -> Result<(), AppError> {
        {
            let mut state = self.state.lock();
            if !matches!(
                state.phase,
                Phase::Recording { .. } | Phase::Processing { .. }
            ) {
                return Err(not_recording());
            }
            state.cancel = Some(CancelReason::User);
            state.cancellation.cancel();
        }
        self.send(Message::Cancel)
    }

    /// Pauses dictation. A dictation in flight ends early, its audio is kept, and the phase
    /// becomes `Paused` when it has.
    pub(crate) fn pause(&self) -> Result<(), AppError> {
        let busy = self.state.update(|state| match state.phase {
            Phase::Idle | Phase::Error { .. } => {
                state.phase = Phase::Paused;
                Ok(false)
            }
            Phase::Recording { .. } | Phase::Processing { .. } => {
                state.cancel = Some(CancelReason::Pause);
                state.cancellation.cancel();
                Ok(true)
            }
            Phase::Paused => Err(AppError::new(
                ErrorCategory::General,
                "application_already_paused",
                "The application is already paused",
            )),
            Phase::ClearingData | Phase::Updating => state.phase.require_idle().map(|()| false),
        })?;
        if busy {
            self.send(Message::Cancel)?;
        }
        Ok(())
    }

    pub(crate) fn resume(&self) -> Result<(), AppError> {
        self.state.update(|state| {
            if state.phase != Phase::Paused {
                return Err(AppError::new(
                    ErrorCategory::General,
                    "application_not_paused",
                    "The application is not paused",
                ));
            }
            state.phase = Phase::Idle;
            Ok(())
        })
    }

    /// A model never changes during a dictation.
    pub(crate) fn load_model(&self, model: ResolvedModel) -> Result<(), AppError> {
        self.ask(|reply| Message::LoadModel {
            model: Box::new(model),
            reply,
        })
    }

    pub(super) fn begin_data_clear(&self) -> Result<DataClear<'_>, AppError> {
        let previous_phase = self.state.update(|state| {
            state.phase.require_idle()?;
            state.require_no_downloads()?;
            Ok(std::mem::replace(&mut state.phase, Phase::ClearingData))
        })?;
        let clear = DataClear {
            state: &self.state,
            previous_phase,
        };
        // Reserve admission before waiting: startup and queued starts read settings on
        // this worker, so their settings lock must stay available until the reply.
        self.ask(|reply| Message::UnloadModel { reply })?;
        Ok(clear)
    }

    /// Listens to `device` for `duration` and reports its levels; no audio is kept.
    pub(crate) fn test_microphone(
        &self,
        device: AudioDeviceChoice,
        duration: Duration,
    ) -> Result<MicrophoneTestResponse, AppError> {
        self.ask(|reply| Message::MicTest {
            device,
            duration,
            reply,
        })
    }

    /// Waits until no dictation is in flight, then enters `Updating`.
    pub(crate) fn begin_update(&self) -> Result<(), AppError> {
        self.ask(|reply| Message::BeginUpdate { reply })
    }

    /// Leaves `Updating` after an update could not be installed.
    pub(crate) fn end_update(&self) {
        self.state.apply(|state| {
            if state.phase == Phase::Updating {
                state.phase = Phase::Idle;
            }
        });
    }

    /// Ends the dictation in flight, keeping its audio, and stops the worker after it.
    pub(crate) fn request_shutdown(&self) {
        self.stopping.store(true, Ordering::Release);
        {
            let mut state = self.state.lock();
            if state.phase.is_busy() {
                state.cancel = Some(CancelReason::Shutdown);
                state.cancellation.cancel();
            }
        }
        let _worker_stopped = self.sender.send(Message::Shutdown);
    }

    /// Waits for the worker to finish after [`Self::request_shutdown`].
    pub(crate) fn join(&self) {
        let worker = self
            .worker
            .lock()
            .expect("dictation worker lock poisoned")
            .take();
        if let Some(worker) = worker {
            let _worker_panicked = worker.join();
        }
    }
}

impl Drop for Dictation {
    fn drop(&mut self) {
        self.request_shutdown();
        self.join();
    }
}

/// One recording from the start of capture until its processing ends.
struct Recording {
    session: u64,
    trigger: RecordingTrigger,
    /// Taken when capture stops; dropping it stops capture too.
    capture: Option<Box<dyn AudioCapture>>,
    sink: Arc<RecordingSink>,
    started_at: UtcMillis,
    settings: DictationSettings,
    cancellation: CancellationToken,
    model: ResolvedModel,
    /// The text of each segment decoded while capture continued.
    decoded: Vec<String>,
    /// A segment failed to decode while recording; the retry after stopping decodes them all.
    live_failed: bool,
}

impl Recording {
    /// Stops capture, draining queued audio into the sink.
    fn stop_capture(&mut self) -> Result<(), AudioInputError> {
        self.capture.take().map_or(Ok(()), AudioCapture::stop)
    }
}

struct Worker {
    parts: DictationParts,
    engine: Box<dyn AsrEngine>,
    sender: mpsc::Sender<Message>,
    stopping: Arc<AtomicBool>,
    next_session: u64,
    recording: Option<Recording>,
    deferred_update: Option<Reply<()>>,
}

impl Worker {
    fn run(mut self, receiver: &mpsc::Receiver<Message>) {
        self.preload_model();
        for message in receiver {
            match message {
                Message::Start { trigger, reply } => match self.start(trigger) {
                    Ok(recording) => {
                        let _caller_left = reply.send(Ok(recording.session));
                        self.prepare_model(recording);
                    }
                    Err(error) => {
                        let _caller_left = reply.send(Err(error));
                    }
                },
                Message::Stop { session } => self.stop(session),
                Message::ShortcutPressed { trigger } => self.shortcut_pressed(trigger),
                Message::HoldReleased => {
                    if let Some(recording) = &self.recording
                        && recording.trigger == RecordingTrigger::Hold
                    {
                        self.stop(recording.session);
                    }
                }
                Message::Cancel => self.cancel(),
                Message::Segments { session } => self.decode_live(session),
                Message::LoadModel { model, reply } => {
                    let _caller_left = reply.send(self.load_model(&model));
                }
                Message::UnloadModel { reply } => {
                    self.unload();
                    let _caller_left = reply.send(Ok(()));
                }
                Message::MicTest {
                    device,
                    duration,
                    reply,
                } => {
                    let _caller_left = reply.send(self.test_microphone(device, duration));
                }
                Message::BeginUpdate { reply } => self.begin_update(reply),
                Message::Shutdown => break,
            }
        }
        self.cancel();
        self.unload();
    }

    fn start(&mut self, trigger: RecordingTrigger) -> Result<Recording, AppError> {
        self.parts.state.lock().phase.require_idle()?;
        let RecordingSettings {
            input_device: device,
            dictation: settings,
        } = if trigger.is_setup_test() {
            self.parts.settings.setup_test_settings()?
        } else {
            self.parts.settings.recording_settings()?
        };
        let model = installed(self.parts.models.as_ref(), &settings.model_id)?;
        self.next_session += 1;
        let session = self.next_session;
        let cancellation = CancellationToken::default();
        let sink = Arc::new(RecordingSink::new(
            session,
            self.sender.clone(),
            self.parts.state.events().clone(),
            MAX_RECORDING_SAMPLES,
            SpeechSegmenter::new((self.parts.voice_activity)()),
        ));
        let previous_phase = self.parts.state.update(|state| {
            state.phase.require_idle()?;
            let previous_phase =
                std::mem::replace(&mut state.phase, Phase::Recording { trigger, session });
            state.cancel = None;
            state.cancellation = cancellation.clone();
            Ok(previous_phase)
        })?;
        let capture = match self
            .parts
            .audio
            .start(device, Arc::clone(&sink) as Arc<dyn AudioSink>)
        {
            Ok(capture) => capture,
            Err(error) => {
                self.parts.state.apply(|state| {
                    state.phase = if state.cancel == Some(CancelReason::Pause) {
                        Phase::Paused
                    } else {
                        previous_phase
                    };
                    state.cancel = None;
                });
                return Err(capture_start_error(&error));
            }
        };
        self.parts.state.apply(|state| {
            state.last_error = None;
            state.download_failures = 0;
            // A setup test reports to setup alone and leaves the last transcript.
            if trigger.is_setup_test() {
                state.setup_test = None;
            } else {
                state.last_transcript = None;
            }
        });
        let recording = Recording {
            session,
            trigger,
            capture: Some(capture),
            sink,
            started_at: UtcMillis::now(),
            settings,
            cancellation,
            model,
            decoded: Vec::new(),
            live_failed: false,
        };
        tracing::info!(?trigger, "recording started");
        Ok(recording)
    }

    fn shortcut_pressed(&mut self, trigger: RecordingTrigger) {
        if let Some(recording) = &self.recording {
            if trigger == RecordingTrigger::Toggle && recording.trigger == trigger {
                self.stop(recording.session);
            }
            return;
        }
        if self.parts.state.lock().phase.require_idle().is_err() {
            return;
        }
        match self.start(trigger) {
            Ok(recording) => self.prepare_model(recording),
            Err(error) => self.parts.state.publish_error(error),
        }
    }

    /// Loads the selected model at startup. Loading takes seconds, and a first dictation that
    /// loaded it would have its stop wait behind the load. A model that is not installed or
    /// fails to load is left for the first dictation to report.
    fn preload_model(&mut self) {
        let Ok(settings) = self.parts.settings.transaction().load() else {
            return;
        };
        if let Ok(model) = installed(self.parts.models.as_ref(), &settings.model_id) {
            let _reported_by_dictation = self.ensure_loaded(&model);
        }
    }

    /// Loads the recording's model while capture runs; a model that cannot load ends it.
    fn prepare_model(&mut self, recording: Recording) {
        if let Err(error) = self.ensure_loaded(&recording.model) {
            recording.cancellation.cancel();
            self.finish(recording.trigger, Outcome::Failed(error));
        } else {
            self.recording = Some(recording);
        }
    }

    fn stop(&mut self, session: u64) {
        let Some(mut recording) = self
            .recording
            .take_if(|recording| recording.session == session)
        else {
            return;
        };
        let trigger = recording.trigger;
        self.parts.state.apply(|state| {
            state.phase = Phase::Processing { cleaning_up: false };
        });
        let outcome = self.process(&mut recording);
        self.finish(trigger, outcome);
        tracing::info!(?trigger, "recording processing completed");
    }

    /// Ends the recording in flight after a cancel, keeping its audio as the reason asks.
    fn cancel(&mut self) {
        let Some(mut recording) = self.recording.take() else {
            return;
        };
        recording.cancellation.cancel();
        if recording.stop_capture().is_err() {
            tracing::warn!("audio capture could not be stopped cleanly");
        }
        let CapturedSpeech { samples, .. } = recording.sink.take();
        let outcome = self.cancelled(&recording, &samples, has_microphone_audio(&samples));
        self.finish(recording.trigger, outcome);
        tracing::info!(trigger = ?recording.trigger, "recording cancelled");
    }

    fn process(&mut self, recording: &mut Recording) -> Outcome {
        let stopped = recording.stop_capture();
        let CapturedSpeech { samples, segments } = recording.sink.take();
        let samples = Arc::<[f32]>::from(samples);
        let has_audio = has_microphone_audio(&samples);
        if stopped.is_err() {
            return self.failed_with_audio(
                recording,
                &samples,
                has_audio,
                AppError::new(
                    ErrorCategory::Audio,
                    "audio_stop",
                    "Audio capture could not be stopped",
                ),
            );
        }
        if recording.cancellation.is_cancelled() {
            return self.cancelled(recording, &samples, has_audio);
        }
        if !has_audio {
            return self.pipeline_failure(
                recording,
                &samples,
                has_audio,
                PipelineFailureKind::NoAudio,
            );
        }
        // Every segment decodes or the dictation fails: a transcript never has a silent gap.
        let decoded = match self.decode_rest(recording, &samples, &segments) {
            Ok(decoded) => decoded,
            Err(DecodeFailure::Cancelled) => return self.cancelled(recording, &samples, has_audio),
            Err(DecodeFailure::Failed) => {
                tracing::warn!("speech recognition failed; retrying once on the same audio");
                match self.decode_all(recording, &samples, &segments) {
                    Ok(decoded) => decoded,
                    Err(DecodeFailure::Cancelled) => {
                        return self.cancelled(recording, &samples, has_audio);
                    }
                    Err(DecodeFailure::Failed) => {
                        return self.pipeline_failure(
                            recording,
                            &samples,
                            has_audio,
                            PipelineFailureKind::RecognitionFailed,
                        );
                    }
                }
            }
        };
        let settings = recording.settings.clone();
        let mut prepared = match prepare_segmented(&decoded, &settings.vocabulary) {
            Ok(prepared) => prepared,
            Err(kind) => return self.pipeline_failure(recording, &samples, has_audio, kind),
        };
        let cleanup_model = if let Some(plan) = &settings.cleanup {
            self.parts.state.apply(|state| {
                state.phase = Phase::Processing { cleaning_up: true };
            });
            self.parts
                .cleanup
                .apply(&mut prepared, plan, &recording.cancellation)
        } else {
            None
        };
        if recording.cancellation.is_cancelled() {
            return self.cancelled(recording, &samples, has_audio);
        }
        if recording.trigger.is_setup_test() {
            return Outcome::Heard(prepared.final_text);
        }
        self.deliver(recording, &samples, prepared, cleanup_model)
    }

    /// Saves the transcript to history, pastes it, and records whether the paste worked.
    fn deliver(
        &self,
        recording: &Recording,
        samples: &[f32],
        prepared: PreparedTranscription,
        cleanup_model: Option<String>,
    ) -> Outcome {
        let history = self.parts.history.insert(
            NewHistoryEntry {
                created_at: recording.started_at,
                duration_millis: Some(capture_duration_millis(samples)),
                raw_text: Some(prepared.raw_text.clone()),
                cleaned_text: (prepared.final_text != prepared.raw_text)
                    .then(|| prepared.final_text.clone()),
                status: EntryStatus::Ok,
                error: None,
                asr_model_id: Some(recording.settings.model_id.clone()),
                llm_model: cleanup_model,
            },
            samples,
            recording.settings.keep_audio,
            recording.settings.history_limit,
        );
        self.history_changed();
        let notice = self
            .parts
            .output
            .output(&prepared.final_text, &recording.cancellation);
        let (delivery_failure, delivery_error) = notice.failure();
        let history_warning = self.parts.history.take_warning();
        if let (Some(reason), Ok(entry)) = (delivery_failure, &history) {
            match self.parts.history.mark_not_delivered(*entry, reason) {
                Ok(()) => self.history_changed(),
                Err(error) => {
                    tracing::warn!(code = error.code, "the paste failure was not recorded");
                }
            }
        }
        let cleanup_warning = cleanup_fallback_warning(&prepared);
        let warning = history
            .as_ref()
            .err()
            .cloned()
            .or_else(|| recording.sink.interruption().and_then(interruption_warning))
            .or(cleanup_warning)
            .or(history_warning);
        Outcome::Delivered {
            transcript: prepared.final_text,
            failure: delivery_error,
            warning,
        }
    }

    fn history_changed(&self) {
        self.parts
            .state
            .events()
            .publish(EventPayload::HistoryChanged);
    }

    /// Decodes the segments a pause ended while capture continues.
    fn decode_live(&mut self, session: u64) {
        let Some(mut recording) = self.recording.take() else {
            return;
        };
        if recording.session == session
            && !recording.live_failed
            && !recording.cancellation.is_cancelled()
        {
            for samples in recording.sink.segments_after(recording.decoded.len()) {
                let Ok(text) = self.recognize(&samples, &recording.cancellation) else {
                    recording.live_failed = true;
                    break;
                };
                recording.decoded.push(text);
            }
        }
        self.recording = Some(recording);
    }

    /// Decodes the segments the live decoder has not reached yet.
    fn decode_rest(
        &mut self,
        recording: &mut Recording,
        samples: &[f32],
        segments: &[AudioSegment],
    ) -> Result<Vec<String>, DecodeFailure> {
        if recording.live_failed {
            return Err(DecodeFailure::Failed);
        }
        let mut decoded = std::mem::take(&mut recording.decoded);
        for segment in segments.iter().skip(decoded.len()) {
            decoded.push(self.recognize(&samples[segment.range()], &recording.cancellation)?);
        }
        Ok(decoded)
    }

    /// Decodes every segment again, reloading a recognizer that failed natively.
    fn decode_all(
        &mut self,
        recording: &Recording,
        samples: &[f32],
        segments: &[AudioSegment],
    ) -> Result<Vec<String>, DecodeFailure> {
        if recording.cancellation.is_cancelled() {
            return Err(DecodeFailure::Cancelled);
        }
        self.ensure_loaded(&recording.model)
            .map_err(|_| DecodeFailure::Failed)?;
        segments
            .iter()
            .map(|segment| self.recognize(&samples[segment.range()], &recording.cancellation))
            .collect()
    }

    fn recognize(
        &mut self,
        samples: &[f32],
        cancellation: &CancellationToken,
    ) -> Result<String, DecodeFailure> {
        if cancellation.is_cancelled() {
            return Err(DecodeFailure::Cancelled);
        }
        let started = Instant::now();
        let result = catch_unwind(AssertUnwindSafe(|| self.engine.transcribe(samples)));
        tracing::info!(
            stage = "segment_asr",
            duration_ms = started.elapsed().as_secs_f64() * 1_000.0,
            sample_count = samples.len(),
            outcome = if matches!(result, Ok(Ok(_))) {
                "completed"
            } else {
                "failed"
            },
            "batch pipeline stage completed"
        );
        match result {
            Err(_) => {
                tracing::warn!("the speech recognizer panicked; it is rebuilt before the retry");
                self.replace_engine();
                Err(DecodeFailure::Failed)
            }
            Ok(_) if cancellation.is_cancelled() => Err(DecodeFailure::Cancelled),
            Ok(Ok(result)) => Ok(result),
            Ok(Err(_)) => Err(DecodeFailure::Failed),
        }
    }

    /// Makes `model` the resident model. A failed load leaves the previous model resident.
    fn ensure_loaded(&mut self, model: &ResolvedModel) -> Result<(), AppError> {
        let requested = &model.definition.id;
        if self.engine.loaded_model_id() == Some(requested) {
            if self.parts.state.lock().model != LoadedModel::Ready(requested.clone()) {
                self.set_model(LoadedModel::Ready(requested.clone()));
            }
            return Ok(());
        }
        self.set_model(LoadedModel::Loading);
        let engine = self.engine.as_mut();
        let loaded = catch_unwind(AssertUnwindSafe(|| engine.load(model)));
        match loaded {
            Ok(Ok(())) => {
                self.set_model(LoadedModel::Ready(requested.clone()));
                Ok(())
            }
            Ok(Err(_)) => {
                self.set_model(LoadedModel::Failed);
                Err(model_load_error())
            }
            Err(_) => {
                self.replace_engine();
                self.set_model(LoadedModel::Failed);
                Err(model_load_error())
            }
        }
    }

    fn load_model(&mut self, model: &ResolvedModel) -> Result<(), AppError> {
        if self.recording.is_some() {
            return Err(AppError::new(
                ErrorCategory::General,
                "workflow_busy",
                "The speech model cannot change during a recording",
            ));
        }
        self.ensure_loaded(model)
    }

    fn unload(&mut self) {
        let engine = self.engine.as_mut();
        if catch_unwind(AssertUnwindSafe(|| engine.unload())).is_err() {
            self.replace_engine();
        }
        self.set_model(LoadedModel::Unloaded);
    }

    fn replace_engine(&mut self) {
        let failed = std::mem::replace(&mut self.engine, (self.parts.engine)());
        let _drop_panicked = catch_unwind(AssertUnwindSafe(move || drop(failed)));
    }

    fn set_model(&self, model: LoadedModel) {
        self.parts.state.apply(|state| state.model = model);
    }

    fn test_microphone(
        &self,
        device: AudioDeviceChoice,
        duration: Duration,
    ) -> Result<MicrophoneTestResponse, AppError> {
        self.parts.state.lock().phase.require_idle()?;
        microphone_test::run(self.parts.audio.as_ref(), device, duration, || {
            self.stopping.load(Ordering::Acquire)
        })
    }

    fn begin_update(&mut self, reply: Reply<()>) {
        if self.recording.is_some() {
            self.deferred_update = Some(reply);
            return;
        }
        let begun = self.parts.state.update(|state| {
            state.phase.require_idle()?;
            state.require_no_downloads()?;
            state.phase = Phase::Updating;
            Ok(())
        });
        let _caller_left = reply.send(begun);
    }
}

fn model_load_error() -> AppError {
    AppError::new(
        ErrorCategory::Recognition,
        "model_load",
        "The speech model could not be loaded",
    )
}

fn worker_stopped() -> AppError {
    AppError::new(
        ErrorCategory::General,
        "worker_stopped",
        "The dictation worker stopped",
    )
}

fn not_recording() -> AppError {
    AppError::new(
        ErrorCategory::General,
        "recording_conflict",
        "No recording is in progress",
    )
}

#[cfg(test)]
pub(crate) mod tests;
