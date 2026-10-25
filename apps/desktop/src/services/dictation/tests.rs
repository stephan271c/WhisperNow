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

use whispernow_core::{
    CancellationToken, ResolvedModel,
    asr::{AsrEngine, AsrError},
    audio::{AudioInput, TARGET_SAMPLE_RATE},
    llm::{LlmError, LlmFinishReason, LlmFuture, LlmRequest, LlmResponse, LlmService},
    workflow::VoiceActivity,
};
use whispernow_domain::ipc::SetApiKeyRequest;
use whispernow_domain::{
    ApiKey, AppError, EntryStatus, ErrorCategory, HistoryEntry, HistoryFilter, HistoryPageRequest,
    LlmProvider, ModelId, ModelRegistry, SettingChange, Settings,
};
use whispernow_platform::{AppPaths, MemoryCredentialStore, ModelStore};

use super::{AiCleanup, Dictation, DictationParts, Message};

mod clear_data;
mod lifecycle;
use crate::services::{
    Phase, RecordingTrigger,
    history::HistoryStore,
    output::{OutputNotice, TranscriptOutput},
    settings::SavedSettings,
    state::{LoadedModel, StateCell},
    test_support::{FakeAudioInput, SettingsHarness, install_artifacts},
};

const TRANSCRIPT: &str = "hello from the worker";

#[derive(Default)]
struct Script {
    /// Decodes that fail before one succeeds.
    failures: AtomicUsize,
    /// Texts the next decodes return, before they fall back to [`TRANSCRIPT`].
    texts: Mutex<VecDeque<&'static str>>,
    decodes: AtomicUsize,
    /// When set, the next decode reports that it started and waits to be released.
    hold: Mutex<Option<(mpsc::Sender<()>, mpsc::Receiver<()>)>>,
    hold_load: Mutex<Option<(mpsc::Sender<()>, mpsc::Receiver<()>)>>,
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
        if let Some((entered, release)) = self.script.hold_load.lock().expect("load hold").take() {
            entered.send(()).expect("test waits for model load");
            release
                .recv_timeout(Duration::from_secs(10))
                .expect("test releases model load");
        }
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

struct Loudness;

impl VoiceActivity for Loudness {
    fn poll_speech(&mut self, frame: &[f32]) -> bool {
        frame.iter().any(|sample| sample.abs() > 0.01)
    }
}

/// An LLM provider that answers with `reply`, or while it is unset fails with `failure`
/// (`InvalidRequest` when that is unset too).
#[derive(Default)]
struct FakeProvider {
    reply: Mutex<Option<&'static str>>,
    failure: Mutex<Option<LlmError>>,
    requests: Mutex<Vec<LlmRequest>>,
}

impl LlmService for FakeProvider {
    fn complete(&self, request: LlmRequest) -> LlmFuture<'_> {
        self.requests.lock().expect("requests").push(request);
        let reply = *self.reply.lock().expect("reply");
        let failure = self
            .failure
            .lock()
            .expect("failure")
            .unwrap_or(LlmError::InvalidRequest);
        Box::pin(async move {
            reply
                .map(|text| LlmResponse {
                    text: text.to_owned(),
                    finish_reason: LlmFinishReason::Completed,
                })
                .ok_or(failure)
        })
    }
}

pub(crate) struct Fixture {
    // Declared first so the worker stops before the temporary files go.
    pub(crate) dictation: Dictation,
    pub(crate) state: Arc<StateCell>,
    settings: Arc<SavedSettings>,
    history: Arc<HistoryStore>,
    output: Arc<FakeOutput>,
    script: Arc<Script>,
    provider: Arc<FakeProvider>,
    credentials: Arc<MemoryCredentialStore>,
    model: ResolvedModel,
    paths: AppPaths,
    detectors_built: Arc<AtomicUsize>,
    root: tempfile::TempDir,
}

impl Fixture {
    pub(crate) fn new() -> Self {
        Self::with_output(OutputNotice::Pasted)
    }

    fn with_output(notice: OutputNotice) -> Self {
        let speech = vec![0.25; TARGET_SAMPLE_RATE as usize];
        Self::with_script(
            notice,
            Arc::new(FakeAudioInput::recording(speech)),
            Arc::new(Script::default()),
        )
    }

    fn with_audio(audio: Vec<f32>) -> Self {
        Self::with_script(
            OutputNotice::Pasted,
            Arc::new(FakeAudioInput::recording(audio)),
            Arc::new(Script::default()),
        )
    }

    fn with_script(notice: OutputNotice, audio: Arc<dyn AudioInput>, script: Arc<Script>) -> Self {
        let saved = Settings {
            keep_audio: true,
            ..Settings::default()
        };
        let SettingsHarness {
            root,
            paths,
            credentials,
            persistence,
            state,
            settings,
        } = SettingsHarness::new(saved.clone());
        let model_id = ModelId::default();
        let registry = ModelRegistry::builtin();
        install_artifacts(
            &paths,
            registry.get(model_id.as_str()).expect("default model"),
        );
        persistence.save(&saved).expect("save settings");
        let history = Arc::new(HistoryStore::open(paths.clone()).store);
        let models = Arc::new(ModelStore::new(paths.clone()));
        let model = models
            .resolve(&model_id)
            .expect("resolve")
            .expect("downloaded model");
        let output = Arc::new(FakeOutput {
            notice,
            pasted: Mutex::new(Vec::new()),
        });
        let provider = Arc::new(FakeProvider::default());
        let engine_script = Arc::clone(&script);
        let detectors_built = Arc::new(AtomicUsize::new(0));
        let built = Arc::clone(&detectors_built);
        let dictation = Dictation::spawn(DictationParts {
            state: Arc::clone(&state),
            settings: Arc::clone(&settings),
            history: Arc::clone(&history),
            audio,
            models,
            voice_activity: Arc::new(move || {
                built.fetch_add(1, Ordering::SeqCst);
                Box::new(Loudness)
            }),
            cleanup: Arc::new(AiCleanup {
                service: Arc::clone(&provider) as Arc<dyn LlmService>,
                api_keys: Arc::clone(&persistence),
            }),
            output: output.clone(),
            engine: Box::new(move || {
                Box::new(ScriptedAsr {
                    script: Arc::clone(&engine_script),
                    loaded: None,
                })
            }),
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
            credentials,
            model,
            paths,
            detectors_built,
            root,
        }
    }

    fn start(&self) {
        self.dictation
            .start(RecordingTrigger::Toggle)
            .expect("start recording");
    }

    /// Waits until the worker has handled every message sent before this one.
    pub(crate) fn flush(&self) {
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

    fn settled_error(&self) -> AppError {
        assert_eq!(self.settled(), Phase::Error);
        self.state
            .snapshot()
            .error
            .expect("a failed dictation shows its error")
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
                limit: 100,
                filter: HistoryFilter::default(),
            })
            .expect("list history")
            .entries
    }

    fn pasted(&self) -> Vec<String> {
        self.output.pasted.lock().expect("pasted").clone()
    }

    fn remove_models(&self) {
        fs::remove_dir_all(self.paths.models_dir()).expect("remove the models");
    }

    fn queued_start_error(&self) -> AppError {
        let (reply, answer) = mpsc::channel();
        self.dictation
            .sender
            .send(Message::Start {
                trigger: RecordingTrigger::Toggle,
                reply,
            })
            .expect("queue the start");
        answer
            .recv_timeout(Duration::from_secs(10))
            .expect("start replied")
            .expect_err("start refused")
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
    assert_eq!(transcript, TRANSCRIPT);
    assert!(state.last_error.is_none());
    drop(state);
    let history = fixture.history();
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].status, EntryStatus::Ok);
    assert_eq!(history[0].raw_text.as_deref(), Some(TRANSCRIPT));
    assert!(history[0].audio_kept);
}

#[test]
fn each_recording_takes_a_fresh_detector_built_while_idle() {
    let fixture = Fixture::new();
    let built = || fixture.detectors_built.load(Ordering::SeqCst);
    fixture.flush();
    assert_eq!(built(), 1, "built at startup");

    fixture.start();
    assert_eq!(built(), 1, "the start takes the one built");
    fixture.dictation.stop().expect("stop");
    assert_eq!(fixture.settled(), Phase::Idle);
    fixture.flush();
    assert_eq!(built(), 2, "the next one is built once the recording ends");

    fixture.start();
    fixture.dictation.cancel().expect("cancel");
    fixture.settled();
    fixture.flush();
    assert_eq!(built(), 3);
}

#[test]
fn a_dictation_keeps_no_audio_with_keep_audio_off() {
    let fixture = Fixture::new();
    fixture
        .settings
        .update(
            &SettingChange::KeepAudio(false),
            &FakeAudioInput::recording(Vec::new()),
        )
        .expect("keep audio off");
    fixture.start();
    fixture.dictation.stop().expect("stop");

    assert_eq!(fixture.settled(), Phase::Idle);
    let history = fixture.history();
    assert_eq!(history.len(), 1);
    assert!(!history[0].audio_kept);
}

#[test]
fn vocabulary_that_empties_a_transcript_clears_the_last_transcript() {
    let fixture = Fixture::new();
    fixture
        .state
        .apply(|state| state.last_transcript = Some("earlier dictation".to_owned()));
    fixture
        .settings
        .edit(|settings| {
            settings.vocabulary_replacements = vec![whispernow_domain::VocabularyReplacement::new(
                TRANSCRIPT, " ",
            )];
            Ok(())
        })
        .expect("save vocabulary");
    fixture.start();
    fixture.dictation.stop().expect("stop");

    assert_eq!(fixture.settled(), Phase::Idle);
    assert!(fixture.state.lock().last_transcript.is_none());
    let history = fixture.history();
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].raw_text.as_deref(), Some(TRANSCRIPT));
    assert_eq!(history[0].cleaned_text.as_deref(), Some(" "));
}

#[test]
fn a_blank_segment_is_left_out_of_the_transcript() {
    let speech = vec![0.25; 24_000];
    let audio = [
        speech.as_slice(),
        &vec![0.0; 3 * TARGET_SAMPLE_RATE as usize],
        &speech,
    ]
    .concat();
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
    assert!(history[0].audio_kept);
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
fn a_start_while_paused_is_refused_even_when_the_model_is_missing() {
    let fixture = Fixture::new();
    fixture.flush();
    fixture.remove_models();
    fixture.dictation.pause().expect("pause");

    let refused = fixture
        .dictation
        .start(RecordingTrigger::Toggle)
        .expect_err("paused");
    assert_eq!(refused.code, "application_paused");
    fixture
        .dictation
        .shortcut_pressed(RecordingTrigger::Toggle)
        .expect("a dropped press");
    fixture.flush();
    let snapshot = fixture.state.snapshot();
    assert_eq!(snapshot.phase, Phase::Paused);
    assert!(snapshot.error.is_none());
}

#[test]
fn a_queued_start_that_reaches_a_paused_worker_reports_a_missing_model_before_the_pause() {
    let fixture = Fixture::new();
    fixture.flush();
    fixture.dictation.pause().expect("pause");

    assert_eq!(fixture.queued_start_error().code, "application_paused");
    fixture.remove_models();
    assert_eq!(fixture.queued_start_error().code, "model_not_installed");
    assert_eq!(fixture.state.snapshot().phase, Phase::Paused);
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

    assert_eq!(fixture.settled_error().code, "transcription_failed");
    assert!(fixture.pasted().is_empty());
    let history = fixture.history();
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].status, EntryStatus::Failed);
    assert!(history[0].raw_text.is_none());
    assert!(history[0].audio_kept);

    fixture.state.dismiss_error();
    let state = fixture.state.lock();
    assert_eq!(state.phase, Phase::Idle);
    assert!(
        state.last_error.is_none(),
        "a dismissed failure stops showing"
    );
}

#[test]
fn a_later_error_does_not_replace_a_failed_dictations_error() {
    let fixture = Fixture::new();
    fixture.script.failures.store(2, Ordering::SeqCst);
    fixture.start();
    fixture.dictation.stop().expect("stop");
    assert_eq!(fixture.settled_error().code, "transcription_failed");

    fixture.state.publish_error(AppError::new(
        ErrorCategory::General,
        "another_error",
        "Another error",
    ));
    assert_eq!(
        fixture.state.snapshot().error.expect("error").code,
        "transcription_failed"
    );

    fixture.state.dismiss_error();
    fixture.state.publish_error(AppError::new(
        ErrorCategory::General,
        "another_error",
        "Another error",
    ));
    assert_eq!(
        fixture.state.snapshot().error.expect("error").code,
        "another_error"
    );
}

impl Fixture {
    fn enable_cleanup(&self) {
        self.settings
            .edit(|settings| {
                settings.cleanup_enabled = true;
                settings
                    .provider_settings_mut(LlmProvider::OpenAi)
                    .choose_model("gpt-test".to_owned());
                Ok(())
            })
            .expect("save settings");
    }

    fn cleanup_models(&self) -> Vec<String> {
        let requests = self.provider.requests.lock().expect("requests");
        requests
            .iter()
            .map(|request| match request {
                LlmRequest::Local { model, .. } => model.as_str().to_owned(),
                LlmRequest::Remote { target, .. } => target.model.clone(),
            })
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
    assert_eq!(transcript, "Cleaned words.");
    assert!(state.last_error.is_none());
    drop(state);
    let history = fixture.history();
    assert_eq!(history[0].status, EntryStatus::Ok);
    assert_eq!(history[0].raw_text.as_deref(), Some(TRANSCRIPT));
    assert_eq!(history[0].cleaned_text.as_deref(), Some("Cleaned words."));
}

#[test]
fn a_saved_remote_api_key_reaches_the_cleanup_request() {
    let fixture = Fixture::new();
    fixture.enable_cleanup();
    fixture
        .settings
        .set_api_key(&SetApiKeyRequest {
            provider: LlmProvider::OpenAi,
            key: ApiKey::new("test-provider-key").expect("fixture key"),
        })
        .expect("save key");
    *fixture.provider.reply.lock().expect("reply") = Some("Cleaned with the saved key.");
    fixture.start();
    fixture.dictation.stop().expect("stop");

    assert_eq!(fixture.settled(), Phase::Idle);
    assert_eq!(fixture.pasted(), ["Cleaned with the saved key."]);
    let requests = fixture.provider.requests.lock().expect("requests");
    assert_eq!(requests.len(), 1);
    let LlmRequest::Remote {
        target, api_key, ..
    } = &requests[0]
    else {
        panic!("remote cleanup request");
    };
    assert_eq!(target.provider.provider(), LlmProvider::OpenAi);
    assert_eq!(
        api_key.as_ref().map(ApiKey::expose_secret),
        Some("test-provider-key")
    );
}

#[test]
fn a_dictation_cleaned_by_the_local_model_drafts_from_the_transcript() {
    let fixture = Fixture::new();
    let saved = fixture
        .settings
        .edit(|settings| {
            settings.cleanup_enabled = true;
            settings.llm_provider = LlmProvider::LocalCleanup;
            settings.vocabulary_replacements = vec![whispernow_domain::VocabularyReplacement::new(
                "worker",
                "WhisperNow team",
            )];
            Ok(settings.clone())
        })
        .expect("save settings");
    *fixture.provider.reply.lock().expect("reply") = Some("Hello from the WhisperNow team.");
    fixture.start();
    fixture.dictation.stop().expect("stop");

    assert_eq!(fixture.settled(), Phase::Idle);
    assert_eq!(fixture.pasted(), ["Hello from the WhisperNow team."]);
    let requests = fixture.provider.requests.lock().expect("requests");
    assert_eq!(requests.len(), 1);
    let LlmRequest::Local {
        model,
        transcript,
        draft,
        ..
    } = &requests[0]
    else {
        panic!("local cleanup request");
    };
    assert_eq!(*model, saved.cleanup_model);
    assert_eq!(transcript, "hello from the WhisperNow team");
    assert_eq!(draft.as_str(), "Hello from the WhisperNow team");
    drop(requests);
    let history = fixture.history();
    assert_eq!(history[0].raw_text.as_deref(), Some(TRANSCRIPT));
    assert_eq!(
        history[0].cleaned_text.as_deref(),
        Some("Hello from the WhisperNow team.")
    );
}

#[test]
fn an_incomplete_cleanup_provider_keeps_the_transcript_without_calling_the_provider() {
    let fixture = Fixture::new();
    let audio = FakeAudioInput::recording(Vec::new());
    fixture
        .settings
        .update(&SettingChange::CleanupEnabled(true), &audio)
        .expect("enable cleanup");
    let view = fixture
        .settings
        .update(&SettingChange::ActiveProvider(LlmProvider::Custom), &audio)
        .expect("save incomplete provider");
    assert_eq!(view.warnings.len(), 1);

    fixture.start();
    fixture.dictation.stop().expect("stop");
    assert_eq!(fixture.settled(), Phase::Idle);
    assert_eq!(fixture.pasted(), [TRANSCRIPT]);
    assert!(fixture.cleanup_models().is_empty());
    assert!(fixture.state.lock().last_error.is_none());
    let history = fixture.history();
    assert_eq!(history[0].raw_text.as_deref(), Some(TRANSCRIPT));
    assert_eq!(history[0].cleaned_text, None);
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
    assert_eq!(state.last_transcript.as_deref(), Some(TRANSCRIPT));
    assert_eq!(
        state.last_error.as_ref().map(|error| error.code.as_str()),
        Some("cleanup_failed")
    );
    let category = state.last_error.as_ref().map(|error| error.category);
    drop(state);
    assert_eq!(category, Some(ErrorCategory::Cleanup));
    let history = fixture.history();
    assert_eq!(history[0].status, EntryStatus::Ok);
    assert_eq!(fixture.cleanup_models(), ["gpt-test"]);
}

#[test]
fn a_missing_cleanup_model_is_named_in_the_warning() {
    let fixture = Fixture::new();
    fixture.enable_cleanup();
    *fixture.provider.failure.lock().expect("failure") = Some(LlmError::ModelNotFound);
    fixture.start();
    fixture.dictation.stop().expect("stop");

    assert_eq!(fixture.settled(), Phase::Idle);
    assert_eq!(fixture.pasted(), [TRANSCRIPT]);
    let warning = fixture
        .state
        .lock()
        .last_error
        .clone()
        .expect("cleanup warning");
    assert_eq!(warning.code, "cleanup_model_not_found");
    assert_eq!(
        warning.summary,
        "AI cleanup couldn't find the model \"gpt-test\"; check its name in Settings → AI cleanup"
    );
    assert_eq!(warning.category, ErrorCategory::Cleanup);
}

#[test]
fn a_failed_paste_is_recorded_and_points_to_history() {
    let fixture = Fixture::with_output(OutputNotice::PasteFailed);
    fixture.start();
    fixture.dictation.stop().expect("stop");

    let error = fixture.settled_error();
    assert_eq!(error.code, "paste_failed");
    assert_eq!(error.category, ErrorCategory::Output);

    assert_eq!(
        fixture.state.lock().last_transcript.clone(),
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
    assert_eq!(refused.category, ErrorCategory::General);

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
    let ready = LoadedModel::Ready(fixture.model.definition.id.clone());
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
    assert!(history[0].audio_kept);
    let refused = fixture
        .dictation
        .start(RecordingTrigger::Toggle)
        .expect_err("a closed worker starts nothing");
    assert_eq!(refused.code, "application_shutting_down");
}

#[test]
fn a_panicking_worker_moves_the_phase_to_error_and_refuses_later_work() {
    let speech = vec![0.25; TARGET_SAMPLE_RATE as usize];
    let audio = FakeAudioInput::recording(speech).before_start(|| panic!("capture panicked"));
    let fixture = Fixture::with_script(
        OutputNotice::Pasted,
        Arc::new(audio),
        Arc::new(Script::default()),
    );

    let refused = fixture
        .dictation
        .start(RecordingTrigger::Toggle)
        .expect_err("the worker stopped before it replied");
    assert_eq!(refused.code, "worker_stopped");
    assert_eq!(fixture.settled_error().code, "worker_panicked");

    let later = fixture
        .dictation
        .start(RecordingTrigger::Toggle)
        .expect_err("no worker is left");
    assert_eq!(later.code, "worker_stopped");
    fixture.state.dismiss_error();
    assert_eq!(fixture.state.snapshot().phase, Phase::Idle);
}

#[test]
fn a_recording_without_speech_pastes_and_saves_nothing() {
    for (audio, code) in [(0.005, "no_speech"), (0.0, "no_microphone_audio")] {
        let fixture = Fixture::with_audio(vec![audio; TARGET_SAMPLE_RATE as usize]);
        fixture.start();
        fixture.dictation.stop().expect("stop");

        assert_eq!(fixture.settled_error().code, code);
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

    assert_eq!(
        fixture.settled_error().category,
        ErrorCategory::Settings,
        "a refused paste is an error"
    );

    assert_eq!(
        fixture.state.lock().last_transcript.as_deref(),
        Some(TRANSCRIPT)
    );
}

#[test]
fn a_setup_test_reports_its_transcript_without_paste_history_or_a_last_transcript() {
    let fixture = Fixture::new();
    fixture
        .state
        .apply(|state| state.last_transcript = Some("Earlier dictation".to_owned()));
    fixture
        .dictation
        .start(RecordingTrigger::SetupTest)
        .expect("start setup test");
    assert!(matches!(
        fixture.state.snapshot().phase,
        Phase::Recording {
            trigger: RecordingTrigger::SetupTest,
            ..
        }
    ));
    fixture.dictation.stop().expect("stop setup test");
    assert_eq!(fixture.settled(), Phase::Idle);
    let state = fixture.state.snapshot();
    assert_eq!(
        state.setup_test,
        Some(whispernow_domain::ipc::SetupTestResult::Heard {
            text: TRANSCRIPT.to_owned()
        })
    );
    assert_eq!(state.last_transcript.as_deref(), Some("Earlier dictation"));
    assert!(state.error.is_none());
    assert!(fixture.pasted().is_empty());
    assert!(fixture.history().is_empty());
}
