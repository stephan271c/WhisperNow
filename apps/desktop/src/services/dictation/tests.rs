//! Dictation scenarios run on the real worker with a scripted recognizer, fake capture and a
//! temporary history.

use std::{
    collections::VecDeque,
    fs,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

use whispernow_api::{ErrorCategoryDto, RecoveryActionDto, TextOutputStatusDto};
use whispernow_core::{
    CancellationToken, ResolvedModel,
    asr::{AsrEngine, AsrError},
    audio::{
        AudioBlock, AudioCapture, AudioDeviceChoice, AudioDeviceDescriptor, AudioInput,
        AudioInputError, AudioSink, StartCaptureRequest,
    },
    llm::{LlmError, LlmFinishReason, LlmFuture, LlmProvider, LlmRequest, LlmResponse, LlmService},
    workflow::{VoiceActivity, VoiceActivitySource},
};
use whispernow_domain::{
    CLEAN_UP_PROMPT_ID, EntryStatus, HistoryEntry, HistoryFilter, HistoryPageRequest, ModelId,
    ModelRegistry, Settings,
};
use whispernow_platform::{AppPaths, MemoryCredentialStore, ModelStore, SettingsStore};

use super::{AiCleanup, Dictation, DictationParts};
use crate::services::{
    Events, RecordingTrigger,
    history::HistoryStore,
    output::{OutputNotice, TranscriptOutput},
    settings::SettingsCoordinator,
    state::{ModelStatus, Phase, State, StateCell},
};

const TRANSCRIPT: &str = "hello from the worker";

/// What the scripted recognizer does next.
#[derive(Default)]
struct Script {
    /// Decodes that fail before one succeeds.
    failures: AtomicUsize,
    /// Texts the next decodes return, before they fall back to [`TRANSCRIPT`].
    texts: Mutex<VecDeque<&'static str>>,
    decodes: AtomicUsize,
    /// When set, the next decode reports that it started and waits to be released.
    hold: Mutex<Option<(mpsc::Sender<()>, mpsc::Receiver<()>)>>,
}

struct ScriptedAsr {
    script: Arc<Script>,
    loaded: Option<ModelId>,
}

impl AsrEngine for ScriptedAsr {
    fn loaded_model_id(&self) -> Option<&ModelId> {
        self.loaded.as_ref()
    }

    fn load(&mut self, model: &ResolvedModel) -> Result<(), AsrError> {
        self.loaded = Some(model.definition.id.clone());
        Ok(())
    }

    fn transcribe(&mut self, _samples: &[f32]) -> Result<String, AsrError> {
        self.script.decodes.fetch_add(1, Ordering::SeqCst);
        let hold = self.script.hold.lock().expect("hold").take();
        if let Some((entered, release)) = hold {
            entered.send(()).expect("test waits for the decode");
            release.recv().expect("test releases the decode");
        }
        if self
            .script
            .failures
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |left| {
                left.checked_sub(1)
            })
            .is_ok()
        {
            return Err(AsrError::TranscriptionFailed);
        }
        let text = self.script.texts.lock().expect("texts").pop_front();
        Ok(text.unwrap_or(TRANSCRIPT).to_owned())
    }

    fn unload(&mut self) {
        self.loaded = None;
    }
}

/// Capture that delivers its audio as soon as it starts.
struct FakeAudio {
    audio: Vec<f32>,
}

impl AudioInput for FakeAudio {
    fn list_input_devices(&self) -> Result<Vec<AudioDeviceDescriptor>, AudioInputError> {
        Ok(vec![AudioDeviceDescriptor {
            choice: AudioDeviceChoice::SystemDefault,
            display_name: "Test microphone".to_owned(),
            is_current_default: true,
        }])
    }

    fn start(
        &self,
        _request: StartCaptureRequest,
        sink: Arc<dyn AudioSink>,
    ) -> Result<Box<dyn AudioCapture>, AudioInputError> {
        sink.samples(AudioBlock::new(self.audio.clone()))
            .map_err(|_| AudioInputError::Backend {
                operation: "deliver test audio",
            })?;
        Ok(Box::new(FakeCapture))
    }
}

struct FakeCapture;

impl AudioCapture for FakeCapture {
    fn stop(self: Box<Self>) -> Result<(), AudioInputError> {
        Ok(())
    }
}

struct FakeOutput {
    notice: OutputNotice,
    pasted: Mutex<Vec<String>>,
}

impl TranscriptOutput for FakeOutput {
    fn output(&self, text: &str, _cancellation: &CancellationToken) -> OutputNotice {
        self.pasted.lock().expect("pasted").push(text.to_owned());
        self.notice
    }
}

/// A frame is speech when it is loud.
struct Loudness;

impl VoiceActivitySource for Loudness {
    fn create(&self) -> Box<dyn VoiceActivity> {
        Box::new(Loudness)
    }
}

impl VoiceActivity for Loudness {
    fn is_speech(&mut self, frame: &[f32]) -> bool {
        frame.iter().any(|sample| sample.abs() > 0.01)
    }
}

/// An LLM provider that answers with `reply`, or fails while it is unset.
#[derive(Default)]
struct FakeProvider {
    reply: Mutex<Option<&'static str>>,
    requests: Mutex<Vec<LlmRequest>>,
}

impl LlmService for FakeProvider {
    fn complete(&self, request: LlmRequest) -> LlmFuture<'_> {
        self.requests.lock().expect("requests").push(request);
        let reply = *self.reply.lock().expect("reply");
        Box::pin(async move {
            reply
                .map(|text| LlmResponse {
                    text: text.to_owned(),
                    finish_reason: LlmFinishReason::Completed,
                })
                .ok_or(LlmError::InvalidRequest)
        })
    }
}

struct Fixture {
    // Declared first so the worker stops before the temporary files go.
    dictation: Dictation,
    state: Arc<StateCell>,
    settings: Arc<SettingsCoordinator>,
    history: Arc<HistoryStore>,
    output: Arc<FakeOutput>,
    script: Arc<Script>,
    provider: Arc<FakeProvider>,
    model: ResolvedModel,
    _root: tempfile::TempDir,
}

impl Fixture {
    fn new() -> Self {
        Self::with_output(OutputNotice::PastedAndRestored)
    }

    fn with_output(notice: OutputNotice) -> Self {
        Self::build(notice, vec![0.25; 16_000])
    }

    fn with_audio(audio: Vec<f32>) -> Self {
        Self::build(OutputNotice::PastedAndRestored, audio)
    }

    fn build(notice: OutputNotice, audio: Vec<f32>) -> Self {
        let root = tempfile::tempdir().expect("temporary root");
        let paths = AppPaths::from_base_dirs(root.path().join("config"), root.path().join("data"));
        let bundled = root.path().join("bundled");
        let model_id = ModelId::default();
        let registry = ModelRegistry::builtin();
        let definition = registry.get(model_id.as_str()).expect("default model");
        for artifact in definition.layout.required_artifacts() {
            let path = bundled.join(model_id.as_str()).join(artifact);
            fs::create_dir_all(path.parent().expect("artifact parent")).expect("model directory");
            fs::write(path, b"fixture").expect("model artifact");
        }
        let persistence = Arc::new(SettingsStore::with_store(
            paths.clone(),
            Arc::new(MemoryCredentialStore::default()),
        ));
        let saved = Settings {
            keep_audio: true,
            ..Settings::default()
        };
        persistence.save(&saved).expect("save settings");
        let history = Arc::new(HistoryStore::open(paths.clone()));
        let state = Arc::new(StateCell::new(
            State::new(model_id.clone(), true),
            Events::channel().0,
        ));
        let settings = Arc::new(SettingsCoordinator::new(
            Arc::clone(&persistence),
            Arc::clone(&state),
        ));
        let models = Arc::new(ModelStore::new(paths, vec![bundled]));
        let model = models
            .resolve(&model_id)
            .expect("resolve")
            .expect("bundled model");
        let output = Arc::new(FakeOutput {
            notice,
            pasted: Mutex::new(Vec::new()),
        });
        let script = Arc::new(Script::default());
        let provider = Arc::new(FakeProvider::default());
        let engine_script = Arc::clone(&script);
        let dictation = Dictation::spawn(DictationParts {
            state: Arc::clone(&state),
            settings: Arc::clone(&settings),
            history: Arc::clone(&history),
            audio: Arc::new(FakeAudio { audio }),
            models,
            voice_activity: Arc::new(Loudness),
            llm: Arc::new(AiCleanup {
                service: Arc::clone(&provider) as Arc<dyn LlmService>,
                runtime: Arc::new(
                    tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .expect("runtime"),
                ),
                api_keys: Arc::clone(&persistence),
            }),
            output: output.clone(),
            engine: Box::new(move || {
                Box::new(ScriptedAsr {
                    script: Arc::clone(&engine_script),
                    loaded: None,
                })
            }),
            max_recording_samples: DictationParts::MAX_RECORDING_SAMPLES,
        })
        .expect("dictation worker");
        Self {
            dictation,
            state,
            settings,
            history,
            output,
            script,
            provider,
            model,
            _root: root,
        }
    }

    fn start(&self) {
        self.dictation
            .start(RecordingTrigger::Toggle)
            .expect("start recording");
    }

    /// Waits until the worker has handled every message sent before this one.
    fn flush(&self) {
        let _busy_or_loaded = self.dictation.load_model(self.model.clone());
    }

    fn recording_trigger(&self) -> Option<RecordingTrigger> {
        match self.state.lock().phase {
            Phase::Recording { trigger, .. } => Some(trigger),
            _ => None,
        }
    }

    /// Makes the next decode wait; returns the signal that it started and its release.
    fn hold_next_decode(&self) -> (mpsc::Receiver<()>, mpsc::Sender<()>) {
        let (entered, started) = mpsc::channel();
        let (release, released) = mpsc::channel();
        *self.script.hold.lock().expect("hold") = Some((entered, released));
        (started, release)
    }

    /// Waits until no dictation is in flight and returns the phase it ended in.
    fn settled(&self) -> Phase {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let phase = self.state.lock().phase.clone();
            if !phase.is_busy() {
                return phase;
            }
            assert!(Instant::now() < deadline, "dictation never settled");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn history(&self) -> Vec<HistoryEntry> {
        self.history
            .list(&HistoryPageRequest {
                before: None,
                limit: 100,
                filter: HistoryFilter::default(),
            })
            .expect("list history")
            .entries
    }

    fn pasted(&self) -> Vec<String> {
        self.output.pasted.lock().expect("pasted").clone()
    }
}

fn error_code(phase: &Phase) -> &str {
    match phase {
        Phase::Error(error) => &error.code,
        other => panic!("expected an error phase, got {other:?}"),
    }
}

#[test]
fn a_dictation_is_transcribed_saved_and_pasted() {
    let fixture = Fixture::new();
    fixture.start();
    fixture.dictation.stop().expect("stop");

    assert_eq!(fixture.settled(), Phase::Idle);
    assert_eq!(fixture.pasted(), [TRANSCRIPT]);
    let state = fixture.state.lock();
    let transcript = state.last_transcript.as_ref().expect("last transcript");
    assert_eq!(transcript.final_text, TRANSCRIPT);
    assert_eq!(
        transcript.text_output_status,
        Some(TextOutputStatusDto::PastedAndRestored)
    );
    assert!(state.last_error.is_none());
    drop(state);
    let history = fixture.history();
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].status, EntryStatus::Ok);
    assert_eq!(history[0].raw_text.as_deref(), Some(TRANSCRIPT));
    assert!(history[0].has_audio);
}

#[test]
fn a_blank_segment_is_left_out_of_the_transcript() {
    let speech = vec![0.25; 24_000];
    let audio = [speech.as_slice(), &vec![0.0; 16_000], &speech].concat();
    let fixture = Fixture::with_audio(audio);
    fixture
        .script
        .texts
        .lock()
        .expect("texts")
        .extend(["", "second words"]);
    fixture.start();
    fixture.dictation.stop().expect("stop");

    assert_eq!(fixture.settled(), Phase::Idle);
    assert_eq!(fixture.pasted(), ["second words"]);
    assert_eq!(fixture.script.decodes.load(Ordering::SeqCst), 2);
}

#[test]
fn a_hold_shortcut_records_from_its_press_until_its_release() {
    let fixture = Fixture::new();
    fixture
        .dictation
        .shortcut_pressed(RecordingTrigger::Hold)
        .expect("press");
    fixture
        .dictation
        .shortcut_pressed(RecordingTrigger::Hold)
        .expect("a second press");
    fixture.flush();
    assert_eq!(fixture.recording_trigger(), Some(RecordingTrigger::Hold));

    fixture.dictation.hold_released().expect("release");
    assert_eq!(fixture.settled(), Phase::Idle);
    assert_eq!(fixture.pasted(), [TRANSCRIPT]);
    assert_eq!(fixture.history().len(), 1);
}

#[test]
fn a_shortcut_ends_only_the_recording_it_started() {
    let fixture = Fixture::new();
    fixture
        .dictation
        .shortcut_pressed(RecordingTrigger::Hold)
        .expect("hold press");
    fixture
        .dictation
        .shortcut_pressed(RecordingTrigger::Toggle)
        .expect("toggle press");
    fixture.flush();
    assert_eq!(fixture.recording_trigger(), Some(RecordingTrigger::Hold));

    fixture.dictation.hold_released().expect("hold release");
    assert_eq!(fixture.settled(), Phase::Idle);

    fixture
        .dictation
        .shortcut_pressed(RecordingTrigger::Toggle)
        .expect("toggle press");
    fixture
        .dictation
        .shortcut_pressed(RecordingTrigger::Hold)
        .expect("hold press");
    fixture.dictation.hold_released().expect("hold release");
    fixture.flush();
    assert_eq!(fixture.recording_trigger(), Some(RecordingTrigger::Toggle));
    fixture
        .dictation
        .shortcut_pressed(RecordingTrigger::Toggle)
        .expect("second toggle press");
    assert_eq!(fixture.settled(), Phase::Idle);
    assert_eq!(fixture.pasted(), [TRANSCRIPT, TRANSCRIPT]);
}

#[test]
fn cancelling_a_recording_keeps_its_audio_as_a_cancelled_entry() {
    let fixture = Fixture::new();
    fixture.start();
    fixture.dictation.cancel().expect("cancel");

    assert_eq!(fixture.settled(), Phase::Idle);
    assert!(fixture.pasted().is_empty());
    assert_eq!(fixture.script.decodes.load(Ordering::SeqCst), 0);
    let history = fixture.history();
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].status, EntryStatus::Cancelled);
    assert!(history[0].has_audio);
}

#[test]
fn cancelling_during_processing_pastes_nothing() {
    let fixture = Fixture::new();
    let (started, release) = fixture.hold_next_decode();
    fixture.start();
    fixture.dictation.stop().expect("stop");
    started.recv().expect("decode started");
    assert!(matches!(
        fixture.state.lock().phase,
        Phase::Processing { .. }
    ));
    fixture.dictation.cancel().expect("cancel");
    release.send(()).expect("release decode");

    assert_eq!(fixture.settled(), Phase::Idle);
    assert!(fixture.pasted().is_empty());
    assert!(fixture.state.lock().last_transcript.is_none());
    let history = fixture.history();
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].status, EntryStatus::Cancelled);
}

#[test]
fn a_start_during_transcription_waits_and_records_next() {
    let fixture = Fixture::new();
    let (started, release) = fixture.hold_next_decode();
    fixture.start();
    fixture.dictation.stop().expect("stop");
    started.recv().expect("decode started");

    std::thread::scope(|scope| {
        let next = scope.spawn(|| fixture.dictation.start(RecordingTrigger::Toggle));
        release.send(()).expect("release decode");
        next.join()
            .expect("start thread")
            .expect("the re-press starts");
    });

    assert_eq!(fixture.pasted(), [TRANSCRIPT]);
    assert!(matches!(
        fixture.state.lock().phase,
        Phase::Recording { .. }
    ));
}

#[test]
fn a_failed_decode_is_retried_once_on_the_same_audio() {
    let fixture = Fixture::new();
    fixture.script.failures.store(1, Ordering::SeqCst);
    fixture.start();
    fixture.dictation.stop().expect("stop");

    assert_eq!(fixture.settled(), Phase::Idle);
    assert_eq!(fixture.script.decodes.load(Ordering::SeqCst), 2);
    assert_eq!(fixture.pasted(), [TRANSCRIPT]);
    assert_eq!(fixture.history()[0].status, EntryStatus::Ok);
}

#[test]
fn a_decode_that_fails_twice_keeps_the_audio_as_a_failed_entry() {
    let fixture = Fixture::new();
    fixture.script.failures.store(2, Ordering::SeqCst);
    fixture.start();
    fixture.dictation.stop().expect("stop");

    let phase = fixture.settled();
    assert_eq!(error_code(&phase), "transcription_failed");
    assert!(fixture.pasted().is_empty());
    let history = fixture.history();
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].status, EntryStatus::Failed);
    assert!(history[0].raw_text.is_none());
    assert!(history[0].has_audio);

    fixture.state.dismiss_error();
    let state = fixture.state.lock();
    assert_eq!(state.phase, Phase::Idle);
    assert!(
        state.last_error.is_none(),
        "a dismissed failure stops showing"
    );
}

impl Fixture {
    fn enable_cleanup(&self) {
        let transaction = self.settings.transaction();
        let mut saved = transaction.load().expect("load settings");
        saved.active_prompt_id = Some("fix_grammar".to_owned());
        saved.provider_settings_mut(LlmProvider::OpenAi).model = "gpt-test".to_owned();
        transaction.save(&saved).expect("save settings");
    }

    fn cleanup_models(&self) -> Vec<String> {
        let requests = self.provider.requests.lock().expect("requests");
        requests
            .iter()
            .map(|request| request.config.model.clone())
            .collect()
    }
}

#[test]
fn a_cleaned_dictation_pastes_and_saves_the_cleaned_text() {
    let fixture = Fixture::new();
    fixture.enable_cleanup();
    *fixture.provider.reply.lock().expect("reply") = Some("Cleaned words.");
    fixture.start();
    fixture.dictation.stop().expect("stop");

    assert_eq!(fixture.settled(), Phase::Idle);
    assert_eq!(fixture.pasted(), ["Cleaned words."]);
    assert_eq!(fixture.cleanup_models(), ["gpt-test"]);
    let state = fixture.state.lock();
    let transcript = state.last_transcript.as_ref().expect("last transcript");
    assert_eq!(transcript.raw_text, TRANSCRIPT);
    assert_eq!(transcript.final_text, "Cleaned words.");
    assert!(!transcript.cleanup_failed);
    assert!(state.last_error.is_none());
    drop(state);
    let history = fixture.history();
    assert_eq!(history[0].status, EntryStatus::Ok);
    assert_eq!(history[0].raw_text.as_deref(), Some(TRANSCRIPT));
    assert_eq!(history[0].cleaned_text.as_deref(), Some("Cleaned words."));
    assert_eq!(history[0].llm_model.as_deref(), Some("gpt-test"));
}

#[test]
fn a_dictation_cleaned_by_the_local_model_drafts_from_the_transcript() {
    let fixture = Fixture::new();
    let saved = {
        let transaction = fixture.settings.transaction();
        let mut saved = transaction.load().expect("load settings");
        saved.active_prompt_id = Some(CLEAN_UP_PROMPT_ID.to_owned());
        saved.llm_provider = LlmProvider::LocalCleanup;
        transaction.save(&saved).expect("save settings");
        saved
    };
    *fixture.provider.reply.lock().expect("reply") = Some("Hello from the worker.");
    fixture.start();
    fixture.dictation.stop().expect("stop");

    assert_eq!(fixture.settled(), Phase::Idle);
    assert_eq!(fixture.pasted(), ["Hello from the worker."]);
    let requests = fixture.provider.requests.lock().expect("requests");
    assert_eq!(requests[0].config.provider, LlmProvider::LocalCleanup);
    assert!(
        requests[0].decoding.draft.is_some(),
        "the local model drafts from the transcript"
    );
    drop(requests);
    let history = fixture.history();
    assert_eq!(
        history[0].cleaned_text.as_deref(),
        Some("Hello from the worker.")
    );
    assert_eq!(
        history[0].llm_model.as_deref(),
        Some(saved.cleanup_model_id())
    );
}

#[test]
fn a_failed_cleanup_pastes_the_recognized_text_and_says_so() {
    let fixture = Fixture::new();
    fixture.enable_cleanup();
    fixture.start();
    fixture.dictation.stop().expect("stop");

    assert_eq!(fixture.settled(), Phase::Idle);
    assert_eq!(fixture.pasted(), [TRANSCRIPT]);
    let state = fixture.state.lock();
    assert!(
        state
            .last_transcript
            .as_ref()
            .is_some_and(|transcript| transcript.cleanup_failed)
    );
    assert_eq!(
        state.last_error.as_ref().map(|error| error.code.as_str()),
        Some("cleanup_failed")
    );
    drop(state);
    let history = fixture.history();
    assert_eq!(history[0].status, EntryStatus::Ok);
    assert!(history[0].llm_model.is_none());
}

#[test]
fn a_failed_paste_is_recorded_and_offers_a_copy() {
    let fixture = Fixture::with_output(OutputNotice::PasteFailed);
    fixture.start();
    fixture.dictation.stop().expect("stop");

    let phase = fixture.settled();
    assert_eq!(error_code(&phase), "paste_failed");
    let Phase::Error(error) = &phase else {
        unreachable!("checked above")
    };
    assert_eq!(error.recovery, Some(RecoveryActionDto::CopyManually));
    assert_eq!(
        fixture
            .state
            .lock()
            .last_transcript
            .as_ref()
            .map(|transcript| transcript.final_text.clone()),
        Some(TRANSCRIPT.to_owned())
    );
    let history = fixture.history();
    assert_eq!(history[0].raw_text.as_deref(), Some(TRANSCRIPT));
    assert_eq!(
        history[0].error.as_deref(),
        Some("The text could not be pasted")
    );
}

#[test]
fn the_model_cannot_change_during_a_recording() {
    let fixture = Fixture::new();
    fixture.start();

    let refused = fixture
        .dictation
        .load_model(fixture.model.clone())
        .expect_err("a recording keeps its model");
    assert_eq!(refused.code, "workflow_busy");
    assert_eq!(refused.category, ErrorCategoryDto::General);

    fixture.dictation.stop().expect("stop");
    assert_eq!(fixture.settled(), Phase::Idle);
    fixture
        .dictation
        .load_model(fixture.model.clone())
        .expect("an idle recognizer loads the model");
}

#[test]
fn the_selected_model_loads_before_the_first_dictation() {
    let fixture = Fixture::new();
    let ready = ModelStatus::Ready(fixture.model.definition.id.clone());
    let deadline = Instant::now() + Duration::from_secs(10);
    while fixture.state.lock().model != ready {
        assert!(Instant::now() < deadline, "the model was not preloaded");
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn closing_during_processing_keeps_the_audio_and_stops_the_worker() {
    let fixture = Fixture::new();
    let (started, release) = fixture.hold_next_decode();
    fixture.start();
    fixture.dictation.stop().expect("stop");
    started.recv().expect("decode started");
    fixture.dictation.request_shutdown();
    release.send(()).expect("release decode");
    fixture.dictation.join();

    assert_eq!(fixture.state.lock().phase, Phase::Idle);
    assert!(fixture.pasted().is_empty());
    let history = fixture.history();
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].status, EntryStatus::Failed);
    assert_eq!(
        history[0].error.as_deref(),
        Some("WhisperNow closed before this dictation was transcribed")
    );
    assert!(history[0].has_audio);
    let refused = fixture
        .dictation
        .start(RecordingTrigger::Toggle)
        .expect_err("a closed worker starts nothing");
    assert_eq!(refused.code, "application_shutting_down");
}

#[test]
fn a_recording_without_speech_pastes_and_saves_nothing() {
    for (audio, code) in [(0.005, "no_speech"), (0.0, "no_microphone_audio")] {
        let fixture = Fixture::with_audio(vec![audio; 16_000]);
        fixture.start();
        fixture.dictation.stop().expect("stop");

        let phase = fixture.settled();
        assert_eq!(error_code(&phase), code);
        assert!(fixture.pasted().is_empty(), "{code}");
        assert!(fixture.history().is_empty(), "{code}");
        assert_eq!(fixture.script.decodes.load(Ordering::SeqCst), 0, "{code}");
    }
}

#[test]
fn a_paste_without_permission_points_to_the_settings() {
    let fixture = Fixture::with_output(OutputNotice::PermissionDenied);
    fixture.start();
    fixture.dictation.stop().expect("stop");

    let Phase::Error(error) = fixture.settled() else {
        panic!("a refused paste is an error")
    };
    assert_eq!(error.category, ErrorCategoryDto::Permission);
    assert_eq!(error.recovery, Some(RecoveryActionDto::OpenSettings));
    assert_eq!(
        fixture
            .state
            .lock()
            .last_transcript
            .as_ref()
            .and_then(|transcript| transcript.text_output_status),
        Some(TextOutputStatusDto::PasteFailed)
    );
}
