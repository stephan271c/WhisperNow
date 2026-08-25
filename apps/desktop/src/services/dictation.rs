//! One worker thread owns the recognizer and audio capture and runs every dictation, model load
//! and microphone test in the order they were asked for.
//!
//! Callers change the shared [`Phase`] only where no engine or capture work is involved:
//! pausing or resuming an idle desktop, and asking the dictation in flight to end early. Every
//! other transition happens on the worker.

use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use whispernow_api::{
    ApiErrorDto, ErrorCategoryDto, EventPayload, MicrophoneTestResponse, TranscriptResponse,
};
use whispernow_core::{
    CancellationToken, ResolvedModel,
    asr::AsrEngine,
    audio::{
        AudioCapture, AudioDeviceChoice, AudioInput, AudioInputError, AudioSink,
        StartCaptureRequest,
    },
    workflow::{
        AudioSegment, PipelineFailureKind, PreparedTranscription, SpeechSegmenter,
        VoiceActivitySource, apply_vocabulary_replacements, has_microphone_audio,
        join_segment_texts, prepare_segmented, recognize_segment,
    },
};
use whispernow_domain::{
    EntryStatus, HistoryLimit, ModelId, NewHistoryEntry, UtcMillis, VocabularyReplacement,
};
use whispernow_platform::ModelStore;

use super::RecordingTrigger;
use super::{
    capture_start_error, device_choice,
    history::HistoryStore,
    microphone_test,
    output::TranscriptOutput,
    settings::SettingsCoordinator,
    shutting_down,
    state::{CancelReason, ModelStatus, Phase, StateCell},
};

mod capture;
mod cleanup;
mod outcome;

use capture::{CapturedSpeech, RecordingSink};
pub(crate) use cleanup::{AiCleanup, CleanupPlan};
use outcome::{
    DecodeFailure, Outcome, capture_duration_millis, cleanup_fallback_warning, delivery_error,
    delivery_failure, interruption_warning, text_output_status,
};

/// Thirty minutes of 16 kHz audio.
const MAX_RECORDING_SAMPLES: usize = 16_000 * 60 * 30;

/// The non-secret settings one dictation is transcribed and cleaned up with, read when it
/// starts. The provider API key is read only when an LLM call is prepared.
#[derive(Clone)]
pub(crate) struct DictationSettings {
    pub(crate) model_id: ModelId,
    pub(crate) vocabulary: Vec<VocabularyReplacement>,
    pub(crate) cleanup: Option<CleanupPlan>,
    /// Total time one AI cleanup may take, including any retries.
    pub(crate) llm_timeout: Duration,
    pub(crate) history_limit: HistoryLimit,
    /// Whether a successful dictation keeps its audio in history.
    pub(crate) keep_audio: bool,
}

/// Everything a recording reads from the settings when it starts.
pub(crate) struct RecordingSettings {
    pub(crate) input_device: Option<String>,
    pub(crate) dictation: DictationSettings,
}

/// What the worker needs to run dictations.
pub(crate) struct DictationParts {
    pub(crate) state: Arc<StateCell>,
    pub(crate) settings: Arc<SettingsCoordinator>,
    pub(crate) history: Arc<HistoryStore>,
    pub(crate) audio: Arc<dyn AudioInput>,
    pub(crate) models: Arc<ModelStore>,
    pub(crate) voice_activity: Arc<dyn VoiceActivitySource>,
    pub(crate) llm: Arc<AiCleanup>,
    pub(crate) output: Arc<dyn TranscriptOutput>,
    /// Builds the recognizer, and a fresh one after a native panic.
    pub(crate) engine: Box<dyn Fn() -> Box<dyn AsrEngine> + Send>,
    pub(crate) max_recording_samples: usize,
}

impl DictationParts {
    pub(crate) const MAX_RECORDING_SAMPLES: usize = MAX_RECORDING_SAMPLES;
}

type Reply<T> = mpsc::Sender<Result<T, ApiErrorDto>>;

/// The work the worker does, in the order it was asked for.
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
    /// Enters `Updating` once no dictation is in flight.
    BeginUpdate {
        reply: Reply<()>,
    },
    Shutdown,
}

/// The handle every command uses to reach the dictation worker.
pub(crate) struct Dictation {
    state: Arc<StateCell>,
    sender: mpsc::Sender<Message>,
    stopping: Arc<AtomicBool>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl Dictation {
    pub(crate) fn spawn(parts: DictationParts) -> Result<Self, ApiErrorDto> {
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
                ApiErrorDto::new(
                    ErrorCategoryDto::General,
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

    fn send(&self, message: Message) -> Result<(), ApiErrorDto> {
        self.sender.send(message).map_err(|_| worker_stopped())
    }

    /// Sends one message and waits for the worker's answer.
    fn ask<T>(&self, message: impl FnOnce(Reply<T>) -> Message) -> Result<T, ApiErrorDto> {
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
    pub(crate) fn start(&self, trigger: RecordingTrigger) -> Result<u64, ApiErrorDto> {
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
    pub(crate) fn shortcut_pressed(&self, trigger: RecordingTrigger) -> Result<(), ApiErrorDto> {
        let accepted = matches!(
            self.state.lock().phase,
            Phase::Idle
                | Phase::Error(_)
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

    /// The hold shortcut came up; it ends the recording it started.
    pub(crate) fn hold_released(&self) -> Result<(), ApiErrorDto> {
        self.send(Message::HoldReleased)
    }

    /// Asks the worker to stop capturing and process the recording.
    pub(crate) fn stop(&self) -> Result<(), ApiErrorDto> {
        let Phase::Recording { session, .. } = self.state.lock().phase else {
            return Err(not_recording());
        };
        self.send(Message::Stop { session })
    }

    /// Ends the dictation in flight early, keeping its audio in history.
    pub(crate) fn cancel(&self) -> Result<(), ApiErrorDto> {
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
    pub(crate) fn pause(&self) -> Result<(), ApiErrorDto> {
        let busy = self.state.update(|state| match state.phase {
            Phase::Idle | Phase::Error(_) => {
                state.phase = Phase::Paused;
                Ok(false)
            }
            Phase::Recording { .. } | Phase::Processing { .. } => {
                state.cancel = Some(CancelReason::Pause);
                state.cancellation.cancel();
                Ok(true)
            }
            Phase::Paused => Err(ApiErrorDto::new(
                ErrorCategoryDto::General,
                "application_already_paused",
                "The application is already paused",
            )),
            Phase::Updating => state.phase.require_idle().map(|()| false),
        })?;
        if busy {
            self.send(Message::Cancel)?;
        }
        Ok(())
    }

    pub(crate) fn resume(&self) -> Result<(), ApiErrorDto> {
        self.state.update(|state| {
            if state.phase != Phase::Paused {
                return Err(ApiErrorDto::new(
                    ErrorCategoryDto::General,
                    "application_not_paused",
                    "The application is not paused",
                ));
            }
            state.phase = Phase::Idle;
            Ok(())
        })
    }

    /// Loads `model` into the recognizer. A model never changes during a dictation.
    pub(crate) fn load_model(&self, model: ResolvedModel) -> Result<(), ApiErrorDto> {
        self.ask(|reply| Message::LoadModel {
            model: Box::new(model),
            reply,
        })
    }

    pub(crate) fn unload_model(&self) -> Result<(), ApiErrorDto> {
        self.ask(|reply| Message::UnloadModel { reply })
    }

    /// Listens to `device` for `duration` and reports its levels; no audio is kept.
    pub(crate) fn test_microphone(
        &self,
        device: AudioDeviceChoice,
        duration: Duration,
    ) -> Result<MicrophoneTestResponse, ApiErrorDto> {
        self.ask(|reply| Message::MicTest {
            device,
            duration,
            reply,
        })
    }

    /// Waits until no dictation is in flight, then enters `Updating`.
    pub(crate) fn begin_update(&self) -> Result<(), ApiErrorDto> {
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
            .unwrap_or_else(PoisonError::into_inner)
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
                Message::Start { trigger, reply } => {
                    let started = self.start(trigger);
                    let admitted = started.is_ok();
                    let _caller_left = reply.send(started);
                    if admitted {
                        self.prepare_model();
                    }
                }
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

    fn start(&mut self, trigger: RecordingTrigger) -> Result<u64, ApiErrorDto> {
        let RecordingSettings {
            input_device,
            dictation: settings,
        } = if trigger.is_setup_test() {
            self.parts.settings.setup_test_settings()?
        } else {
            self.parts.settings.recording_settings()?
        };
        let device = device_choice(input_device.as_deref());
        let model = resolve_model(self.parts.models.as_ref(), &settings.model_id)?;
        self.next_session += 1;
        let session = self.next_session;
        let cancellation = CancellationToken::default();
        let sink = Arc::new(RecordingSink::new(
            session,
            self.sender.clone(),
            self.parts.state.events().clone(),
            self.parts.max_recording_samples,
            SpeechSegmenter::new(self.parts.voice_activity.create()),
        ));
        let capture = self
            .parts
            .audio
            .start(
                StartCaptureRequest { device },
                Arc::clone(&sink) as Arc<dyn AudioSink>,
            )
            .map_err(|error| capture_start_error(&error))?;
        // A refused start drops the capture, which stops it.
        self.parts.state.update(|state| {
            state.phase.require_idle()?;
            state.phase = Phase::Recording { trigger, session };
            state.cancel = None;
            state.cancellation = cancellation.clone();
            state.last_error = None;
            state.download_failures = 0;
            // A setup test reports to setup alone and leaves the last transcript.
            if trigger.is_setup_test() {
                state.setup_test = None;
            } else {
                state.last_transcript = None;
            }
            Ok(())
        })?;
        self.recording = Some(Recording {
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
        });
        tracing::info!(?trigger, "recording started");
        Ok(session)
    }

    /// A second press of the toggle shortcut ends its recording; a press while idle starts
    /// one. A press during another recording, a pause or an update is dropped.
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
            Ok(_) => self.prepare_model(),
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
        if let Ok(model) = resolve_model(self.parts.models.as_ref(), &settings.model_id) {
            let _reported_by_dictation = self.ensure_loaded(&model);
        }
    }

    /// Loads the recording's model while capture runs; a model that cannot load ends it.
    fn prepare_model(&mut self) {
        let Some(model) = self
            .recording
            .as_ref()
            .map(|recording| recording.model.clone())
        else {
            return;
        };
        if let Err(error) = self.ensure_loaded(&model)
            && let Some(recording) = self.recording.take()
        {
            recording.cancellation.cancel();
            self.finish(recording.trigger, Outcome::Failed(error));
        }
    }

    fn stop(&mut self, session: u64) {
        if self
            .recording
            .as_ref()
            .is_none_or(|recording| recording.session != session)
        {
            return;
        }
        let Some(mut recording) = self.recording.take() else {
            return;
        };
        let trigger = recording.trigger;
        self.parts.state.apply(|state| {
            state.phase = Phase::Processing {
                trigger,
                cleaning_up: false,
            };
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
        let outcome = self.cancelled(&recording, &samples);
        self.finish(recording.trigger, outcome);
        tracing::info!(trigger = ?recording.trigger, "recording cancelled");
    }

    fn process(&mut self, recording: &mut Recording) -> Outcome {
        let stopped = recording.stop_capture();
        let CapturedSpeech { samples, segments } = recording.sink.take();
        let samples = Arc::<[f32]>::from(samples);
        if stopped.is_err() {
            return self.failed_with_audio(
                recording,
                &samples,
                ApiErrorDto::new(
                    ErrorCategoryDto::Audio,
                    "audio_stop",
                    "Audio capture could not be stopped",
                ),
            );
        }
        if recording.cancellation.is_cancelled() {
            return self.cancelled(recording, &samples);
        }
        if !has_microphone_audio(&samples) {
            return self.pipeline_failure(recording, &samples, PipelineFailureKind::NoAudio);
        }
        if segments.is_empty() {
            return self.pipeline_failure(recording, &samples, PipelineFailureKind::NoSpeech);
        }
        // Every segment decodes or the dictation fails: a transcript never has a silent gap.
        let decoded = match self.decode_rest(recording, &samples, &segments) {
            Ok(decoded) => decoded,
            Err(DecodeFailure::Cancelled) => return self.cancelled(recording, &samples),
            Err(DecodeFailure::Failed) => {
                tracing::warn!("speech recognition failed; retrying once on the same audio");
                match self.decode_all(recording, &samples, &segments) {
                    Ok(decoded) => decoded,
                    Err(DecodeFailure::Cancelled) => return self.cancelled(recording, &samples),
                    Err(DecodeFailure::Failed) => {
                        return self.pipeline_failure(
                            recording,
                            &samples,
                            PipelineFailureKind::RecognitionFailed,
                        );
                    }
                }
            }
        };
        let settings = recording.settings.clone();
        let mut prepared =
            match prepare_segmented(recording.started_at, &decoded, &settings.vocabulary) {
                Ok(prepared) => prepared,
                Err(kind) => return self.pipeline_failure(recording, &samples, kind),
            };
        if settings.cleanup.is_some() {
            let trigger = recording.trigger;
            self.parts.state.apply(|state| {
                state.phase = Phase::Processing {
                    trigger,
                    cleaning_up: true,
                };
            });
        }
        let llm_model = self.parts.llm.apply(
            &mut prepared,
            settings.cleanup.as_ref(),
            settings.llm_timeout,
            &recording.cancellation,
        );
        if recording.cancellation.is_cancelled() {
            return self.cancelled(recording, &samples);
        }
        if recording.trigger.is_setup_test() {
            return Outcome::Heard(prepared.final_text);
        }
        self.deliver(recording, &samples, prepared, llm_model)
    }

    /// Saves the transcript to history, pastes it, and records whether the paste worked.
    fn deliver(
        &self,
        recording: &Recording,
        samples: &[f32],
        prepared: PreparedTranscription,
        llm_model: Option<String>,
    ) -> Outcome {
        let history = self.parts.history.insert(
            NewHistoryEntry {
                created_at: prepared.timestamp,
                duration_millis: Some(capture_duration_millis(samples)),
                raw_text: Some(prepared.raw_text.clone()),
                cleaned_text: (prepared.final_text != prepared.raw_text)
                    .then(|| prepared.final_text.clone()),
                status: EntryStatus::Ok,
                error: None,
                asr_model_id: Some(recording.settings.model_id.clone()),
                llm_model,
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
        let history_warning = self.parts.history.take_warning();
        if let (Some(reason), Ok(entry)) = (delivery_failure(notice), &history) {
            match self.parts.history.mark_not_delivered(entry.id, reason) {
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
            .map(|error| ApiErrorDto::new(ErrorCategoryDto::General, &error.code, &error.summary))
            .or_else(|| recording.sink.interruption().and_then(interruption_warning))
            .or_else(|| cleanup_warning.clone())
            .or(history_warning);
        Outcome::Delivered {
            transcript: TranscriptResponse {
                raw_text: prepared.raw_text,
                final_text: prepared.final_text,
                text_output_status: Some(text_output_status(notice)),
                cleanup_failed: cleanup_warning.is_some(),
                history_save_failed: history.is_err(),
            },
            failure: delivery_error(notice),
            warning,
        }
    }

    fn history_changed(&self) {
        self.parts
            .state
            .events()
            .publish(EventPayload::HistoryChanged);
    }

    /// Decodes the segments a pause ended while capture continues, and publishes the text so
    /// far to the recording bar.
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
                self.publish_partial(&recording);
            }
        }
        self.recording = Some(recording);
    }

    /// Publishes the text committed so far. The event carries transcript text, so the desktop
    /// delivers it only to the recording bar.
    fn publish_partial(&self, recording: &Recording) {
        let joined = join_segment_texts(recording.decoded.iter().map(String::as_str));
        self.parts
            .state
            .events()
            .publish(EventPayload::PartialTranscript {
                session: recording.session,
                committed_text: apply_vocabulary_replacements(
                    &joined,
                    &recording.settings.vocabulary,
                ),
            });
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
        let engine = self.engine.as_mut();
        let result = catch_unwind(AssertUnwindSafe(|| {
            recognize_segment(samples, engine, cancellation)
        }));
        match result {
            Err(_) => {
                tracing::warn!("the speech recognizer panicked; it is rebuilt before the retry");
                self.replace_engine();
                Err(DecodeFailure::Failed)
            }
            Ok(_) if cancellation.is_cancelled() => Err(DecodeFailure::Cancelled),
            Ok(Ok(result)) => Ok(result),
            Ok(Err(PipelineFailureKind::Cancelled)) => Err(DecodeFailure::Cancelled),
            Ok(Err(_)) => Err(DecodeFailure::Failed),
        }
    }

    /// Makes `model` the resident model. A failed load leaves the previous model resident.
    fn ensure_loaded(&mut self, model: &ResolvedModel) -> Result<(), ApiErrorDto> {
        let requested = &model.definition.id;
        if self.engine.loaded_model_id() == Some(requested) {
            if self.parts.state.lock().model != ModelStatus::Ready(requested.clone()) {
                self.set_model(ModelStatus::Ready(requested.clone()));
            }
            return Ok(());
        }
        self.set_model(ModelStatus::Loading);
        let engine = self.engine.as_mut();
        let loaded = catch_unwind(AssertUnwindSafe(|| engine.load(model)));
        match loaded {
            Ok(Ok(())) => {
                self.set_model(ModelStatus::Ready(requested.clone()));
                Ok(())
            }
            Ok(Err(_)) => {
                self.set_model(ModelStatus::Failed);
                Err(model_load_error())
            }
            Err(_) => {
                self.replace_engine();
                self.set_model(ModelStatus::Failed);
                Err(model_load_error())
            }
        }
    }

    fn load_model(&mut self, model: &ResolvedModel) -> Result<(), ApiErrorDto> {
        if self.recording.is_some() {
            return Err(ApiErrorDto::new(
                ErrorCategoryDto::General,
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
        self.set_model(ModelStatus::Unloaded);
    }

    /// Replaces an engine that panicked.
    fn replace_engine(&mut self) {
        let failed = std::mem::replace(&mut self.engine, (self.parts.engine)());
        let _drop_panicked = catch_unwind(AssertUnwindSafe(move || drop(failed)));
    }

    fn set_model(&self, model: ModelStatus) {
        self.parts.state.apply(|state| state.model = model);
    }

    fn test_microphone(
        &self,
        device: AudioDeviceChoice,
        duration: Duration,
    ) -> Result<MicrophoneTestResponse, ApiErrorDto> {
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
            if !state.active_downloads.is_empty() {
                return Err(ApiErrorDto::new(
                    ErrorCategoryDto::General,
                    "model_transition_busy",
                    "Wait for the model download to finish",
                ));
            }
            state.phase = Phase::Updating;
            Ok(())
        });
        let _caller_left = reply.send(begun);
    }
}

fn resolve_model(models: &ModelStore, model_id: &ModelId) -> Result<ResolvedModel, ApiErrorDto> {
    models
        .resolve(model_id)
        .map_err(|_| {
            ApiErrorDto::new(
                ErrorCategoryDto::Model,
                "model_resolve",
                "The selected model could not be inspected",
            )
        })?
        .ok_or_else(|| {
            ApiErrorDto::new(
                ErrorCategoryDto::Model,
                "model_not_installed",
                "Install the selected model before transcribing",
            )
        })
}

fn model_load_error() -> ApiErrorDto {
    ApiErrorDto::new(
        ErrorCategoryDto::Recognition,
        "model_load",
        "The speech model could not be loaded",
    )
}

fn worker_stopped() -> ApiErrorDto {
    ApiErrorDto::new(
        ErrorCategoryDto::General,
        "worker_stopped",
        "The dictation worker stopped",
    )
}

fn not_recording() -> ApiErrorDto {
    ApiErrorDto::new(
        ErrorCategoryDto::General,
        "recording_conflict",
        "No recording is in progress",
    )
}

#[cfg(test)]
mod tests;
