use std::process::{Child, Command};

use whispernow_core::audio::AudioInputError;
use whispernow_platform::SettingsStore;

use super::lifecycle::{test_backend, test_backend_before_preload};
use super::*;
use crate::services::{Backend, test_support::shut_down};
use whispernow_domain::{
    ClearDataCategory, HistoryLimit, HistoryRetention, NewHistoryEntry, UtcMillis,
};

const WAIT: Duration = Duration::from_secs(10);

fn retained_entry(backend: &Backend) -> (HistoryEntry, Vec<u8>) {
    let id = backend
        .history
        .insert(
            NewHistoryEntry {
                created_at: UtcMillis::now(),
                duration_millis: 1_000,
                raw_text: Some("keep this dictation".to_owned()),
                cleaned_text: None,
                status: EntryStatus::Ok,
                error: None,
                asr_model_id: ModelId::default(),
            },
            Some(&[0.25; 160]),
            HistoryLimit::Unlimited,
            HistoryRetention::Forever,
        )
        .expect("retained dictation");
    let entry = backend
        .history_page(&HistoryPageRequest::first_page(HistoryFilter::default()))
        .expect("history")
        .entries
        .into_iter()
        .find(|entry| entry.id == id)
        .expect("retained dictation");
    let audio = backend.history.audio(id).expect("retained audio");
    (entry, audio)
}

fn assert_retained(backend: &Backend, entry: &HistoryEntry, audio: &[u8]) {
    let page = backend
        .history_page(&HistoryPageRequest::first_page(HistoryFilter::default()))
        .expect("history");
    assert_eq!(page.entries.as_slice(), std::slice::from_ref(entry));
    assert_eq!(backend.history.audio(entry.id).expect("audio"), audio);
}

#[test]
fn clearing_settings_resets_memory_before_the_next_edit_even_when_the_file_is_missing() {
    use whispernow_domain::{SettingChange, ThemePreference, VocabularyReplacement};

    for missing_file in [false, true] {
        let (backend, _root) = test_backend(Fixture::new());
        backend
            .update_settings(&SettingChange::Vocabulary(vec![
                VocabularyReplacement::new("teh", "the"),
            ]))
            .expect("save vocabulary");
        if missing_file {
            fs::remove_file(backend.paths.settings_file()).expect("external deletion");
        }

        backend
            .clear_data(&[ClearDataCategory::Settings])
            .expect("clear settings");
        assert_eq!(backend.settings.snapshot(), Settings::default());
        backend
            .update_settings(&SettingChange::Theme(ThemePreference::Dark))
            .expect("edit after clearing");
        let expected = Settings {
            theme: ThemePreference::Dark,
            ..Settings::default()
        };
        assert_eq!(backend.settings.snapshot(), expected);
        let persisted = SettingsStore::with_store(
            backend.paths.clone(),
            Arc::new(MemoryCredentialStore::default()),
        );
        assert_eq!(
            persisted.load().expect("reload saved settings").settings,
            expected
        );
        shut_down(&backend);
    }
}

#[test]
fn empty_clear_requests_preserve_history_and_audio() {
    let (backend, _root) = test_backend(Fixture::new());
    let (entry, audio) = retained_entry(&backend);
    fs::create_dir_all(backend.paths.python_logs_dir()).expect("logs directory");
    let log = backend.paths.python_logs_dir().join("app.log");
    fs::write(&log, b"log").expect("log");
    let settings = fs::read(backend.paths.settings_file()).expect("saved settings");

    let result = backend.clear_data(&[]).expect("clear selected data");

    assert!(result.items.is_empty());
    assert!(!result.restart_required);
    assert!(log.exists());
    assert_eq!(
        fs::read(backend.paths.settings_file()).expect("settings"),
        settings
    );
    assert!(backend.paths.models_dir().exists());
    assert_eq!(backend.state.snapshot().phase, Phase::Idle);
    assert_retained(&backend, &entry, &audio);
    shut_down(&backend);
}

fn select_another_speech_model(backend: &Backend) -> ModelId {
    let registry = ModelRegistry::builtin();
    let model = registry
        .speech_models()
        .find(|model| model.id != ModelId::default())
        .expect("another speech model");
    install_artifacts(&backend.paths, model);
    backend
        .select_model(model.id.clone())
        .expect("select model");
    model.id.clone()
}

#[test]
fn clearing_only_models_keeps_history_and_the_selected_model_in_state_and_settings() {
    let (backend, _root) = test_backend(Fixture::new());
    let (entry, audio) = retained_entry(&backend);
    let selected = select_another_speech_model(&backend);

    let result = backend
        .clear_data(&[ClearDataCategory::Models])
        .expect("clear models");

    assert!(result.restart_required);
    assert!(!backend.paths.models_dir().exists());
    assert_eq!(backend.settings.snapshot().model_id, selected);
    assert_eq!(backend.state.snapshot().selected_model_id, selected);
    assert_retained(&backend, &entry, &audio);
    shut_down(&backend);
}

#[test]
fn clearing_settings_resets_the_selected_model_in_state_and_settings() {
    let (backend, _root) = test_backend(Fixture::new());
    select_another_speech_model(&backend);

    backend
        .clear_data(&[ClearDataCategory::Settings])
        .expect("clear settings");

    assert_eq!(backend.settings.snapshot().model_id, ModelId::default());
    assert_eq!(
        backend.state.snapshot().selected_model_id,
        ModelId::default()
    );
    shut_down(&backend);
}

#[test]
fn history_deletion_removes_rows_and_audio_and_keeps_other_categories() {
    let (backend, _root) = test_backend(Fixture::new());
    let (entry, _) = retained_entry(&backend);
    fs::write(backend.paths.migrated_history_file(), b"Python history").expect("migrated history");
    let unreadable = backend.paths.unreadable_history_file("20261015T120000Z");
    fs::write(&unreadable, b"legacy").expect("malformed history backup");
    fs::create_dir_all(backend.paths.python_logs_dir()).expect("logs directory");
    let settings = fs::read(backend.paths.settings_file()).expect("saved settings");

    let result = backend
        .clear_data(&[ClearDataCategory::History])
        .expect("clear history data");

    assert_eq!(result.items.len(), 1);
    assert!(
        result
            .items
            .iter()
            .all(|item| item.location.category == ClearDataCategory::History && item.deleted)
    );
    assert!(!result.restart_required);
    assert!(backend.paths.history_database_file().exists());
    assert!(!backend.paths.recordings_dir().exists());
    assert!(!backend.paths.migrated_history_file().exists());
    assert!(!unreadable.exists());
    assert!(backend.paths.python_logs_dir().exists());
    assert!(backend.paths.models_dir().exists());
    assert_eq!(
        fs::read(backend.paths.settings_file()).expect("settings"),
        settings
    );
    let page = backend
        .history_page(&HistoryPageRequest::first_page(HistoryFilter::default()))
        .expect("new history");
    assert!(page.entries.is_empty());
    assert!(backend.history.audio(entry.id).is_err());
    let later = backend
        .history
        .insert(
            NewHistoryEntry {
                created_at: UtcMillis::now(),
                duration_millis: 1_000,
                raw_text: Some("after clear".to_owned()),
                cleaned_text: None,
                status: EntryStatus::Ok,
                error: None,
                asr_model_id: ModelId::default(),
            },
            Some(&[0.25; 160]),
            HistoryLimit::Unlimited,
            HistoryRetention::Forever,
        )
        .expect("dictation after clear");
    assert!(later > entry.id);
    assert!(backend.history.audio(later).is_ok());
    shut_down(&backend);
}

#[test]
fn clearing_every_category_reports_files_and_unavailable_credentials_separately() {
    use crate::services::clear_data::credential_store_location;

    for available in [true, false] {
        let fixture = Fixture::new();
        let credentials = Arc::clone(&fixture.credentials);
        let (backend, _root) = test_backend(fixture);
        let (entry, _) = retained_entry(&backend);
        backend
            .settings
            .set_api_key(&SetApiKeyRequest {
                provider: LlmProvider::OpenAi,
                key: ApiKey::new("saved-key").expect("key"),
            })
            .expect("save key");
        fs::create_dir_all(backend.paths.python_logs_dir()).expect("logs directory");
        credentials.set_available(available);

        let result = backend
            .clear_data(&ClearDataCategory::ALL)
            .expect("clear all data");

        assert!(result.restart_required);
        for item in &result.items {
            assert_eq!(
                item.deleted,
                item.location != credential_store_location() || available
            );
        }
        assert!(
            result
                .items
                .iter()
                .any(|item| item.location == credential_store_location())
        );
        assert!(!backend.paths.settings_file().exists());
        assert!(backend.paths.history_database_file().exists());
        assert!(!backend.paths.recordings_dir().exists());
        assert!(!backend.paths.models_dir().exists());
        assert!(!backend.paths.python_logs_dir().exists());
        let warning = backend.state.snapshot().error.map(|error| error.code);
        assert_eq!(
            warning.as_deref(),
            (!available).then_some("credential_store_unavailable")
        );
        credentials.set_available(true);
        assert_eq!(
            backend
                .settings
                .api_key(LlmProvider::OpenAi)
                .expect("read key")
                .is_some(),
            !available
        );
        assert!(backend.history.audio(entry.id).is_err());
        shut_down(&backend);
    }
}

#[cfg(unix)]
#[test]
fn a_recordings_deletion_failure_keeps_history_recoverable() {
    use std::os::unix::fs::PermissionsExt as _;

    let (backend, _root) = test_backend(Fixture::new());
    let (entry, audio) = retained_entry(&backend);
    let recordings = backend.paths.recordings_dir();
    let original_permissions = fs::metadata(&recordings).expect("recordings").permissions();
    fs::set_permissions(&recordings, fs::Permissions::from_mode(0o500))
        .expect("read-only recordings");

    let result = backend.clear_data(&[ClearDataCategory::History]);
    fs::set_permissions(&recordings, original_permissions).expect("restore recordings permissions");
    let result = result.expect("partial deletion report");

    assert_eq!(result.items.len(), 1);
    assert_eq!(
        result.items[0].location.category,
        ClearDataCategory::History
    );
    assert!(!result.items[0].deleted);
    assert!(recordings.exists());
    assert!(backend.paths.history_database_file().exists());
    assert!(!result.restart_required);
    assert_eq!(backend.state.snapshot().phase, Phase::Idle);
    assert_retained(&backend, &entry, &audio);
    shut_down(&backend);
}

#[test]
fn inspection_failure_preserves_history_and_restores_the_previous_phase() {
    let (backend, _root) = test_backend(Fixture::new());
    let (entry, audio) = retained_entry(&backend);
    let saved_config = backend.paths.config_dir().with_extension("kept");
    fs::rename(backend.paths.config_dir(), &saved_config).expect("move configuration aside");
    fs::write(backend.paths.config_dir(), b"not a directory").expect("obstruct inspection");
    backend.state.apply(|state| {
        state.phase = Phase::Error;
        state.last_error = Some(whispernow_domain::AppError::new(
            ErrorCategory::Output,
            "paste_failed",
            "Copy from History",
        ));
    });

    let result = backend.clear_data(&[ClearDataCategory::History]);

    fs::remove_file(backend.paths.config_dir()).expect("remove obstruction");
    fs::rename(saved_config, backend.paths.config_dir()).expect("restore configuration");
    assert_eq!(
        result.expect_err("inspection refused").code,
        "clear_data_inspect"
    );
    let snapshot = backend.state.snapshot();
    assert_eq!(snapshot.phase, Phase::Error);
    assert_eq!(
        snapshot.error.map(|error| error.code),
        Some("paste_failed".to_owned())
    );
    assert_retained(&backend, &entry, &audio);
    shut_down(&backend);
}

#[test]
fn a_clear_that_resets_the_last_error_ends_the_failed_phase() {
    let (backend, _root) = test_backend(Fixture::new());
    backend.state.apply(|state| {
        state.phase = Phase::Error;
        state.last_error = Some(whispernow_domain::AppError::new(
            ErrorCategory::Output,
            "paste_failed",
            "Copy from History",
        ));
    });

    backend.clear_data(&[]).expect("clear selected data");

    let snapshot = backend.state.snapshot();
    assert_eq!(snapshot.phase, Phase::Idle);
    assert_eq!(snapshot.error, None);
    shut_down(&backend);
}

#[test]
fn clear_data_refuses_paused_and_active_recordings_without_removing_history() {
    for paused in [true, false] {
        let (backend, _root) = test_backend(Fixture::new());
        let (entry, audio) = retained_entry(&backend);
        if paused {
            backend.dictation.pause().expect("pause");
        } else {
            backend
                .dictation
                .start(RecordingTrigger::Toggle)
                .expect("recording");
        }
        let phase = backend.state.snapshot().phase;

        let error = backend
            .clear_data(&[ClearDataCategory::History])
            .expect_err("busy desktop");

        assert_eq!(
            error.code,
            if paused {
                "application_paused"
            } else {
                "workflow_busy"
            }
        );
        assert_eq!(backend.state.snapshot().phase, phase);
        assert_retained(&backend, &entry, &audio);
        shut_down(&backend);
    }
}

#[test]
fn a_failed_worker_wait_restores_idle_and_keeps_history() {
    let (backend, _root) = test_backend(Fixture::new());
    let (entry, audio) = retained_entry(&backend);
    backend.dictation.request_shutdown();
    backend.dictation.join();

    let error = backend
        .clear_data(&[ClearDataCategory::History])
        .expect_err("stopped worker");

    assert_eq!(error.code, "application_shutting_down");
    assert_eq!(backend.state.snapshot().phase, Phase::Idle);
    assert_retained(&backend, &entry, &audio);
    shut_down(&backend);
}

#[test]
fn data_clear_reserves_admission_and_restores_it_after_the_guard_is_released() {
    let fixture = Fixture::new();
    fixture.flush();
    let clear = fixture
        .dictation
        .begin_data_clear()
        .expect("quiescent worker");
    assert_eq!(fixture.state.snapshot().phase, Phase::ClearingData);
    assert_eq!(
        fixture
            .dictation
            .start(RecordingTrigger::Toggle)
            .expect_err("reserved admission")
            .code,
        "application_data_clearing"
    );
    assert_eq!(
        fixture
            .dictation
            .begin_update()
            .expect_err("clear blocks update")
            .code,
        "application_data_clearing"
    );
    assert_eq!(
        fixture
            .dictation
            .pause()
            .expect_err("clear blocks pause")
            .code,
        "application_data_clearing"
    );
    drop(clear);
    assert_eq!(fixture.state.snapshot().phase, Phase::Idle);
    fixture.start();
}

#[test]
fn a_pause_during_failed_capture_start_remains_paused() {
    let (entered, starting) = mpsc::channel();
    let (release, released) = mpsc::channel();
    let released = Mutex::new(released);
    let fixture = Fixture::with_script(
        OutputNotice::Pasted,
        Arc::new(
            FakeAudioInput::refusing(AudioInputError::Backend {
                operation: "start capture",
            })
            .before_start(move || {
                entered.send(()).expect("capture-start observer");
                released
                    .lock()
                    .expect("capture release")
                    .recv_timeout(WAIT)
                    .expect("capture released");
            }),
        ),
        Arc::new(Script::default()),
    );
    let mut release = ReleaseOnDrop(Some(release));
    fixture.flush();
    let (reply, answer) = mpsc::channel();
    fixture
        .dictation
        .sender
        .send(super::super::Message::Start {
            trigger: RecordingTrigger::Toggle,
            reply,
        })
        .expect("request capture");
    starting.recv_timeout(WAIT).expect("capture started");

    fixture.dictation.pause().expect("pause acknowledged");
    release.release();

    assert_eq!(
        answer
            .recv_timeout(WAIT)
            .expect("start replied")
            .expect_err("capture failed")
            .code,
        "audio_start"
    );
    fixture.flush();
    assert_eq!(fixture.state.snapshot().phase, Phase::Paused);
}

pub(super) struct ReleaseOnDrop(pub(super) Option<mpsc::Sender<()>>);

impl ReleaseOnDrop {
    pub(super) fn release(&mut self) {
        if let Some(sender) = self.0.take() {
            let _test_left = sender.send(());
        }
    }
}

impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        self.release();
    }
}

struct TestChild(Child);

impl Drop for TestChild {
    fn drop(&mut self) {
        let _already_exited = self.0.kill();
        let _reaped = self.0.wait();
    }
}

#[test]
fn startup_and_a_queued_start_do_not_deadlock_clear_data() {
    const CHILD_FLAG: &str = "WHISPERNOW_CLEAR_DATA_TEST_CHILD";
    if std::env::var_os(CHILD_FLAG).is_some() {
        run_startup_overlap();
        return;
    }
    let temp = tempfile::tempdir().expect("child temporary root");
    let mut child = TestChild(
        Command::new(std::env::current_exe().expect("test executable"))
            .args(["--exact", "services::dictation::tests::clear_data::startup_and_a_queued_start_do_not_deadlock_clear_data", "--nocapture"])
            .env(CHILD_FLAG, "1")
            .env("TMPDIR", temp.path())
            .env("TMP", temp.path())
            .env("TEMP", temp.path())
            .spawn()
            .expect("start isolated concurrency test"),
    );
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        if let Some(status) = child.0.try_wait().expect("wait for concurrency test") {
            assert!(
                status.success(),
                "isolated concurrency test failed: {status}"
            );
            break;
        }
        assert!(
            Instant::now() < deadline,
            "clear-data worker/settings wait deadlocked"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn run_startup_overlap() {
    let script = Arc::new(Script::default());
    let (entered, loading) = mpsc::channel();
    let (release, released) = mpsc::channel();
    *script.hold_load.lock().expect("load hold") = Some((entered, released));
    let fixture = Fixture::with_script(
        OutputNotice::Pasted,
        Arc::new(FakeAudioInput::recording(Vec::new())),
        script,
    );
    let mut release = ReleaseOnDrop(Some(release));
    loading
        .recv_timeout(WAIT)
        .expect("startup entered model load");
    let (backend, _root) = test_backend_before_preload(fixture);
    let backend = Arc::new(backend);
    let (start_reply, start_answer) = mpsc::channel();
    backend
        .dictation
        .sender
        .send(super::super::Message::Start {
            trigger: RecordingTrigger::Toggle,
            reply: start_reply,
        })
        .expect("queue start before clear");
    let clearing = Arc::clone(&backend);
    let (finished, cleared) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let result = clearing.clear_data(&[ClearDataCategory::Settings]);
        let _test_left = finished.send(result);
    });
    let deadline = Instant::now() + WAIT;
    while backend.state.snapshot().phase != Phase::ClearingData {
        assert!(Instant::now() < deadline, "clear did not reserve admission");
        std::thread::sleep(Duration::from_millis(5));
    }
    release.release();
    let start = start_answer
        .recv_timeout(WAIT)
        .expect("queued start replied");
    assert_eq!(
        start.expect_err("queued start refused").code,
        "application_data_clearing"
    );
    let result = cleared
        .recv_timeout(WAIT)
        .expect("clear completed")
        .expect("clear settings");
    assert!(result.restart_required);
    assert!(result.items.iter().all(|item| item.deleted));
    worker.join().expect("clear thread");
    assert_eq!(backend.state.snapshot().phase, Phase::Idle);
    assert!(!backend.paths.settings_file().exists());
    shut_down(&backend);
}
