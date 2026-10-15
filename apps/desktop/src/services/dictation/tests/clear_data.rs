use std::process::{Child, Command};

use super::lifecycle::{native_services, native_services_before_preload};
use super::*;
use crate::services::NativeServices;
use whispernow_domain::{ClearDataCategory, HistoryLimit, NewHistoryEntry, UtcMillis};

const WAIT: Duration = Duration::from_secs(10);

fn retained_entry(services: &NativeServices) -> (HistoryEntry, Vec<u8>) {
    let id = services
        .history
        .insert(
            NewHistoryEntry {
                created_at: UtcMillis::now(),
                duration_millis: Some(1_000),
                raw_text: Some("keep this dictation".to_owned()),
                cleaned_text: None,
                status: EntryStatus::Ok,
                error: None,
                asr_model_id: Some(ModelId::default()),
            },
            &[0.25; 160],
            true,
            HistoryLimit::Unlimited,
        )
        .expect("retained dictation");
    let entry = services
        .history(&HistoryPageRequest::first_page(HistoryFilter::default()))
        .expect("history")
        .entries
        .into_iter()
        .find(|entry| entry.id == id)
        .expect("retained dictation");
    let audio = services.history.audio(id).expect("retained audio");
    (entry, audio)
}

fn assert_retained(services: &NativeServices, entry: &HistoryEntry, audio: &[u8]) {
    let page = services
        .history(&HistoryPageRequest::first_page(HistoryFilter::default()))
        .expect("history");
    assert_eq!(page.entries.as_slice(), std::slice::from_ref(entry));
    assert_eq!(services.history.audio(entry.id).expect("audio"), audio);
}

#[test]
fn clearing_settings_resets_memory_before_the_next_edit_even_when_the_file_is_missing() {
    use whispernow_domain::{ThemePreference, VocabularyReplacement};

    for missing_file in [false, true] {
        let (services, _root) = native_services(Fixture::new());
        let mut edited = services.settings.load();
        edited.vocabulary_replacements = vec![VocabularyReplacement::new("teh", "the")];
        services.update_settings(edited).expect("save vocabulary");
        if missing_file {
            fs::remove_file(services.paths.settings_file()).expect("external deletion");
        }

        services
            .clear_data(&[ClearDataCategory::Settings])
            .expect("clear settings");
        assert_eq!(services.settings.load(), Settings::default());
        let mut edited = services.settings.load();
        edited.theme = ThemePreference::Dark;
        services
            .update_settings(edited)
            .expect("edit after clearing");
        let expected = Settings {
            theme: ThemePreference::Dark,
            ..Settings::default()
        };
        assert_eq!(services.settings.load(), expected);
        let persisted = SettingsStore::with_store(
            services.paths.clone(),
            Arc::new(MemoryCredentialStore::default()),
        );
        assert_eq!(
            persisted.load().expect("reload saved settings").settings,
            expected
        );
        services.request_shutdown();
        services.shutdown();
    }
}

#[test]
fn empty_clear_requests_preserve_history_and_audio() {
    let (services, _root) = native_services(Fixture::new());
    let (entry, audio) = retained_entry(&services);
    fs::create_dir_all(services.paths.python_logs_dir()).expect("logs directory");
    let log = services.paths.python_logs_dir().join("app.log");
    fs::write(&log, b"log").expect("log");
    let cuda = services.paths.cuda_pack_dir().join("native-library");
    fs::create_dir_all(cuda.parent().expect("CUDA directory")).expect("CUDA directory");
    fs::write(&cuda, b"CUDA data").expect("CUDA file");
    let settings = fs::read(services.paths.settings_file()).expect("saved settings");

    let result = services.clear_data(&[]).expect("clear selected data");

    assert!(result.items.is_empty());
    assert!(!result.restart_required);
    assert!(log.exists());
    assert_eq!(
        fs::read(services.paths.settings_file()).expect("settings"),
        settings
    );
    assert!(services.paths.models_dir().exists());
    assert_eq!(fs::read(&cuda).expect("CUDA file preserved"), b"CUDA data");
    assert_eq!(services.state.snapshot().phase, Phase::Idle);
    assert_retained(&services, &entry, &audio);
    services.request_shutdown();
    services.shutdown();
}

#[test]
fn clearing_only_cuda_model_data_reports_removal_and_restart_and_preserves_history() {
    let (services, _root) = native_services(Fixture::new());
    let (entry, audio) = retained_entry(&services);
    fs::remove_dir_all(services.paths.models_dir()).expect("no speech model data");
    let cuda = services.paths.cuda_pack_dir();
    fs::create_dir_all(&cuda).expect("CUDA directory");
    fs::write(cuda.join("native-library"), b"CUDA data").expect("CUDA file");
    let settings = fs::read(services.paths.settings_file()).expect("saved settings");
    let locations = services.data_locations().expect("data locations");
    assert!(locations.iter().any(|location| {
        location.category == ClearDataCategory::Models && location.path == cuda
    }));

    let result = services
        .clear_data(&[ClearDataCategory::Models])
        .expect("clear models");

    assert_eq!(result.items.len(), 1);
    assert_eq!(result.items[0].category, ClearDataCategory::Models);
    assert_eq!(result.items[0].path, cuda.to_string_lossy());
    assert!(result.items[0].deleted);
    assert!(result.restart_required);
    assert!(!cuda.exists());
    assert_eq!(
        fs::read(services.paths.settings_file()).expect("settings kept"),
        settings
    );
    assert_eq!(services.state.snapshot().phase, Phase::Idle);
    assert_retained(&services, &entry, &audio);
    services.request_shutdown();
    services.shutdown();
}

#[test]
fn history_deletion_removes_rows_and_audio_and_keeps_other_categories() {
    let (services, _root) = native_services(Fixture::new());
    let (entry, _) = retained_entry(&services);
    fs::write(services.paths.migrated_history_file(), b"Python history").expect("migrated history");
    let unreadable = services.paths.unreadable_history_file("20261015T120000Z");
    fs::write(&unreadable, b"legacy").expect("malformed history backup");
    fs::create_dir_all(services.paths.python_logs_dir()).expect("logs directory");
    let settings = fs::read(services.paths.settings_file()).expect("saved settings");

    let result = services
        .clear_data(&[ClearDataCategory::History])
        .expect("clear history data");

    assert_eq!(result.items.len(), 1);
    assert!(
        result
            .items
            .iter()
            .all(|item| item.category == ClearDataCategory::History && item.deleted)
    );
    assert!(!result.restart_required);
    assert!(services.paths.history_database_file().exists());
    assert!(!services.paths.recordings_dir().exists());
    assert!(!services.paths.migrated_history_file().exists());
    assert!(!unreadable.exists());
    assert!(services.paths.python_logs_dir().exists());
    assert!(services.paths.models_dir().exists());
    assert_eq!(
        fs::read(services.paths.settings_file()).expect("settings"),
        settings
    );
    let page = services
        .history(&HistoryPageRequest::first_page(HistoryFilter::default()))
        .expect("new history");
    assert!(page.entries.is_empty());
    assert!(services.history.audio(entry.id).is_err());
    let later = services
        .history
        .insert(
            NewHistoryEntry {
                created_at: UtcMillis::now(),
                duration_millis: Some(1_000),
                raw_text: Some("after clear".to_owned()),
                cleaned_text: None,
                status: EntryStatus::Ok,
                error: None,
                asr_model_id: None,
            },
            &[0.25; 160],
            true,
            HistoryLimit::Unlimited,
        )
        .expect("dictation after clear");
    assert!(later > entry.id);
    assert!(services.history.audio(later).is_ok());
    services.request_shutdown();
    services.shutdown();
}

#[test]
fn clearing_every_category_reports_files_and_unavailable_credentials_separately() {
    use crate::services::clear_data::CREDENTIAL_STORE_CLEAR_TARGET;

    for available in [true, false] {
        let fixture = Fixture::new();
        let credentials = Arc::clone(&fixture.credentials);
        let (services, _root) = native_services(fixture);
        let (entry, _) = retained_entry(&services);
        services
            .settings
            .set_api_key(&SetApiKeyRequest {
                provider: LlmProvider::OpenAi,
                key: ApiKey::new("saved-key").expect("key"),
            })
            .expect("save key");
        fs::create_dir_all(services.paths.python_logs_dir()).expect("logs directory");
        credentials.set_available(available);

        let result = services
            .clear_data(&ClearDataCategory::ALL)
            .expect("clear all data");

        assert!(result.restart_required);
        for item in &result.items {
            assert_eq!(
                item.deleted,
                item.path != CREDENTIAL_STORE_CLEAR_TARGET || available
            );
        }
        assert!(
            result
                .items
                .iter()
                .any(|item| item.path == CREDENTIAL_STORE_CLEAR_TARGET)
        );
        assert!(!services.paths.settings_file().exists());
        assert!(services.paths.history_database_file().exists());
        assert!(!services.paths.recordings_dir().exists());
        assert!(!services.paths.models_dir().exists());
        assert!(!services.paths.python_logs_dir().exists());
        let warning = services.state.snapshot().error.map(|error| error.code);
        assert_eq!(
            warning.as_deref(),
            (!available).then_some("credential_store_unavailable")
        );
        credentials.set_available(true);
        assert_eq!(
            services
                .settings
                .api_keys()
                .api_key(LlmProvider::OpenAi)
                .expect("read key")
                .is_some(),
            !available
        );
        assert!(services.history.audio(entry.id).is_err());
        services.request_shutdown();
        services.shutdown();
    }
}

#[cfg(unix)]
#[test]
fn a_recordings_deletion_failure_keeps_history_recoverable() {
    use std::os::unix::fs::PermissionsExt as _;

    let (services, _root) = native_services(Fixture::new());
    let (entry, audio) = retained_entry(&services);
    let recordings = services.paths.recordings_dir();
    let original_permissions = fs::metadata(&recordings).expect("recordings").permissions();
    fs::set_permissions(&recordings, fs::Permissions::from_mode(0o500))
        .expect("read-only recordings");

    let result = services.clear_data(&[ClearDataCategory::History]);
    fs::set_permissions(&recordings, original_permissions).expect("restore recordings permissions");
    let result = result.expect("partial deletion report");

    assert_eq!(result.items.len(), 1);
    assert_eq!(result.items[0].category, ClearDataCategory::History);
    assert!(!result.items[0].deleted);
    assert!(recordings.exists());
    assert!(services.paths.history_database_file().exists());
    assert!(!result.restart_required);
    assert_eq!(services.state.snapshot().phase, Phase::Idle);
    assert_retained(&services, &entry, &audio);
    services.request_shutdown();
    services.shutdown();
}

#[test]
fn inspection_failure_preserves_history_and_restores_the_previous_phase() {
    let (services, _root) = native_services(Fixture::new());
    let (entry, audio) = retained_entry(&services);
    let saved_config = services.paths.config_dir().with_extension("kept");
    fs::rename(services.paths.config_dir(), &saved_config).expect("move configuration aside");
    fs::write(services.paths.config_dir(), b"not a directory").expect("obstruct inspection");
    let previous_phase = Phase::Error {
        error: whispernow_domain::AppError::new(
            ErrorCategory::Output,
            "paste_failed",
            "Copy from History",
        ),
    };
    services
        .state
        .apply(|state| state.phase = previous_phase.clone());

    let result = services.clear_data(&[ClearDataCategory::History]);

    fs::remove_file(services.paths.config_dir()).expect("remove obstruction");
    fs::rename(saved_config, services.paths.config_dir()).expect("restore configuration");
    assert_eq!(
        result.expect_err("inspection refused").code,
        "clear_data_inspect"
    );
    assert_eq!(services.state.snapshot().phase, previous_phase);
    assert_retained(&services, &entry, &audio);
    services.request_shutdown();
    services.shutdown();
}

#[test]
fn clear_data_refuses_paused_and_active_recordings_without_removing_history() {
    for paused in [true, false] {
        let (services, _root) = native_services(Fixture::new());
        let (entry, audio) = retained_entry(&services);
        if paused {
            services.dictation.pause().expect("pause");
        } else {
            services
                .dictation
                .start(RecordingTrigger::Toggle)
                .expect("recording");
        }
        let phase = services.state.snapshot().phase;

        let error = services
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
        assert_eq!(services.state.snapshot().phase, phase);
        assert_retained(&services, &entry, &audio);
        services.request_shutdown();
        services.shutdown();
    }
}

#[test]
fn a_failed_worker_wait_restores_idle_and_keeps_history() {
    let (services, _root) = native_services(Fixture::new());
    let (entry, audio) = retained_entry(&services);
    services.dictation.request_shutdown();
    services.dictation.join();

    let error = services
        .clear_data(&[ClearDataCategory::History])
        .expect_err("stopped worker");

    assert_eq!(error.code, "application_shutting_down");
    assert_eq!(services.state.snapshot().phase, Phase::Idle);
    assert_retained(&services, &entry, &audio);
    services.request_shutdown();
    services.shutdown();
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
    struct FailingCaptureStart {
        entered: mpsc::Sender<()>,
        released: Mutex<mpsc::Receiver<()>>,
    }

    impl AudioInput for FailingCaptureStart {
        fn list_input_devices(&self) -> Result<Vec<AudioDevice>, AudioInputError> {
            Ok(Vec::new())
        }

        fn start(
            &self,
            _: AudioDeviceChoice,
            _: Arc<dyn AudioSink>,
        ) -> Result<Box<dyn AudioCapture>, AudioInputError> {
            self.entered.send(()).expect("capture-start observer");
            self.released
                .lock()
                .expect("capture release")
                .recv_timeout(WAIT)
                .expect("capture released");
            Err(AudioInputError::Backend {
                operation: "start capture",
            })
        }
    }

    let (entered, starting) = mpsc::channel();
    let (release, released) = mpsc::channel();
    let fixture = Fixture::with_script(
        OutputNotice::Pasted,
        Arc::new(FailingCaptureStart {
            entered,
            released: Mutex::new(released),
        }),
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
        Arc::new(FakeAudio { audio: Vec::new() }),
        script,
    );
    let mut release = ReleaseOnDrop(Some(release));
    loading
        .recv_timeout(WAIT)
        .expect("startup entered model load");
    let (services, _root) = native_services_before_preload(fixture);
    let services = Arc::new(services);
    let (start_reply, start_answer) = mpsc::channel();
    services
        .dictation
        .sender
        .send(super::super::Message::Start {
            trigger: RecordingTrigger::Toggle,
            reply: start_reply,
        })
        .expect("queue start before clear");
    let clearing = Arc::clone(&services);
    let (finished, cleared) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let result = clearing.clear_data(&[ClearDataCategory::Settings]);
        let _test_left = finished.send(result);
    });
    let deadline = Instant::now() + WAIT;
    while services.state.snapshot().phase != Phase::ClearingData {
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
    assert_eq!(services.state.snapshot().phase, Phase::Idle);
    assert!(!services.paths.settings_file().exists());
    services.request_shutdown();
    services.shutdown();
}
